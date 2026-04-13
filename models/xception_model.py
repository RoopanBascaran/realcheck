from transformers import ViTImageProcessor, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import cv2
import os
import io
import base64
import json
import re
import torch
import logging

logger = logging.getLogger(__name__)

PRIMARY_MODEL = 'Nahrawy/AIorNot'
SECOND_MODEL = 'Ateeqq/ai-vs-human-image-detector'
SECOND_MODEL_THRESHOLD = 0.95  # Only override if very confident


class AIDetector:
    def __init__(self):
        # Load primary model (Nahrawy)
        logger.info(f'Loading primary model: {PRIMARY_MODEL}')
        self.processor = ViTImageProcessor.from_pretrained(PRIMARY_MODEL)
        self.model = AutoModelForImageClassification.from_pretrained(PRIMARY_MODEL)
        self.model.eval()
        logger.info('Primary model loaded')

        # Load second model (Ateeqq) for catching modern AI content
        try:
            logger.info(f'Loading second model: {SECOND_MODEL}')
            self.processor2 = AutoImageProcessor.from_pretrained(SECOND_MODEL)
            self.model2 = AutoModelForImageClassification.from_pretrained(SECOND_MODEL)
            self.model2.eval()
            self.has_second_model = True
            logger.info('Second model loaded')
        except Exception as e:
            logger.warning(f'Could not load second model: {e}')
            self.has_second_model = False

    def _score_frame(self, frame):
        """Score a single frame with primary model. Returns AI probability."""
        img = Image.fromarray(frame)
        inputs = self.processor(images=img, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

        labels = self.model.config.id2label
        ai_score = 0.0
        for idx, label_name in labels.items():
            if 'ai' in label_name.lower() or 'fake' in label_name.lower() or 'artificial' in label_name.lower():
                ai_score = probs[idx].item()
                break
        return ai_score

    def _score_frame_second(self, frame):
        """Score a single frame with second model. Returns AI probability."""
        if not self.has_second_model:
            return 0.0
        img = Image.fromarray(frame)
        inputs = self.processor2(images=img, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model2(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

        labels = self.model2.config.id2label
        ai_score = 0.0
        for idx, label_name in labels.items():
            if 'ai' in label_name.lower() or 'fake' in label_name.lower():
                ai_score = probs[idx].item()
                break
        return ai_score

    def _encode_frame_base64(self, frame):
        """Encode a numpy RGB frame to base64 JPEG."""
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _score_frames_groq_vision(self, frames):
        """Send frames to Groq vision model for AI detection. Returns dict or None."""
        groq_key = os.environ.get('GROQ_API_KEY')
        if not groq_key:
            return None

        # Pick 1 frame from the middle (minimizes token usage for free tier)
        n = len(frames)
        selected = [frames[n // 2]]

        # Build image content — no model scores to avoid biasing Groq
        image_content = [
            {
                "type": "text",
                "text": (
                    "This is a frame extracted from a video. "
                    "Analyze them for signs of AI generation: unnatural textures, "
                    "lighting inconsistencies, morphing artifacts, anatomical errors, "
                    "warping, repetitive patterns, too-smooth skin, weird hands/fingers, "
                    "or any visual glitches. "
                    "Respond with ONLY a JSON object, no other text: "
                    '{"verdict": "AI" or "Real", "confidence": 0.0-1.0, "reason": "brief explanation"}'
                )
            }
        ]
        for f in selected:
            b64 = self._encode_frame_base64(f)
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        try:
            import requests as req
            resp = req.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {groq_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                    'messages': [
                        {
                            'role': 'system',
                            'content': (
                                'You are an expert at detecting AI-generated images and videos. '
                                'You analyze visual artifacts to determine authenticity.'
                            )
                        },
                        {'role': 'user', 'content': image_content}
                    ],
                    'max_tokens': 100,
                    'temperature': 0.1,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(f'Groq vision call failed: {resp.status_code} {resp.text[:200]}')
                return None

            text = resp.json()['choices'][0]['message']['content'].strip()
            # Strip markdown code blocks if present
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            result = json.loads(text)
            verdict = result.get('verdict', '').lower()
            confidence = float(result.get('confidence', 0.5))
            reason = result.get('reason', '')
            logger.info(f'Groq vision: verdict={verdict}, confidence={confidence:.2f}, reason={reason}')
            return {'verdict': verdict, 'confidence': confidence, 'reason': reason}

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f'Groq vision response parse failed: {e}')
            return None
        except Exception as e:
            logger.warning(f'Groq vision call failed: {type(e).__name__}: {e}')
            return None

    def classify(self, video_path):
        frames = extract_frames(video_path, max_frames=20)
        if not frames:
            return 'Real'

        # Primary model scores
        scores = [self._score_frame(frame) for frame in frames]
        avg_score = np.mean(scores)
        ai_frames = sum(1 for s in scores if s > 0.5)
        max_ai_score = max(scores)

        # Second model scores (for catching modern AI that primary misses)
        second_avg = 0.0
        if self.has_second_model:
            scores2 = [self._score_frame_second(frame) for frame in frames]
            second_avg = np.mean(scores2)
            logger.info(
                f'Second model: avg={second_avg:.3f}, max={max(scores2):.3f}'
            )

        logger.info(
            f'Primary model: avg={avg_score:.3f}, max={max_ai_score:.3f}, '
            f'ai_frames={ai_frames}/{len(scores)}'
        )

        # Groq vision model (third opinion)
        groq_says_ai = False
        groq_confidence = 0.0
        groq_result = self._score_frames_groq_vision(frames)
        if groq_result:
            groq_says_ai = groq_result['verdict'] == 'ai'
            groq_confidence = groq_result['confidence']

        # Decision logic with three models:
        # Local models detect AI well but give false positives on real videos.
        # Groq vision is great at confirming real videos but misses AI.
        # Strategy: local models propose, Groq gets final say as tie-breaker.
        groq_says_real = groq_result is not None and not groq_says_ai
        groq_available = groq_result is not None

        # Step 1: Local models make initial call
        # Second model (Ateeqq) scores ~1.0 on almost everything, so only let it
        # override when primary model is at least borderline (> 0.4).
        # If primary strongly says Real (< 0.4), second model alone can't override.
        second_model_triggered = False
        local_says_ai = False
        if avg_score > 0.5:
            local_says_ai = True
        elif second_avg > SECOND_MODEL_THRESHOLD and avg_score > 0.4:
            local_says_ai = True
            second_model_triggered = True
            logger.info(f'Second model override: {second_avg:.3f} > {SECOND_MODEL_THRESHOLD}, primary={avg_score:.3f}')

        # Step 2: Groq vision as tie-breaker
        # Groq is good at confirming real videos.
        # Second model (Ateeqq) is unreliable — scores ~1.0 on almost everything.
        # Trust Groq's "Real" when:
        #   - Primary has high avg but also high frame consensus (>=85% frames flagged AI)
        #     This pattern = beauty filter / compression artifact false positive
        #   - Second model actually says Real (< 0.5) — rare but meaningful signal
        ai_frame_ratio = ai_frames / len(scores) if scores else 0
        high_frame_consensus = ai_frame_ratio >= 0.85
        primary_very_strong = avg_score > 0.9 and high_frame_consensus
        second_says_real = second_avg < 0.5  # Rare but strong signal since second model usually scores high
        # Trust Groq when: second model genuinely says Real, OR
        # primary has high frames but not extreme confidence (beauty filter pattern)
        groq_can_override = second_says_real or (high_frame_consensus and not primary_very_strong)
        if groq_available:
            if local_says_ai and groq_says_real and groq_confidence >= 0.8 and groq_can_override:
                # Groq corrects false positive
                base_result = 'Real'
                logger.info(
                    f'Groq override: primary={avg_score:.3f}, frames={ai_frames}/{len(scores)}, '
                    f'second={second_avg:.3f}, Groq says Real ({groq_confidence:.2f}) — trusting Groq'
                )
            elif not local_says_ai and groq_says_ai and groq_confidence >= 0.8:
                # Groq catches AI that local models missed
                base_result = 'AI-generated'
                logger.info(f'Groq detected AI that local models missed ({groq_confidence:.2f})')
            elif local_says_ai and groq_says_ai:
                # All agree on AI
                base_result = 'AI-generated'
                logger.info('All models agree: AI-generated')
            else:
                # Default to local model decision
                base_result = 'AI-generated' if local_says_ai else 'Real'
        else:
            # Groq unavailable — fall back to local models only
            base_result = 'AI-generated' if local_says_ai else 'Real'
            logger.info('Groq unavailable, using local models only')

        # Dynamic feedback threshold based on model agreement
        # Only count second model if primary is also borderline (same logic as Step 1)
        models_saying_ai = sum([
            avg_score > 0.5,
            second_avg > 0.5 and avg_score > 0.4,
            groq_says_ai and groq_confidence >= 0.7,
        ])
        if models_saying_ai >= 3:
            fb_threshold = 0.95
        elif models_saying_ai >= 2:
            fb_threshold = 0.9
        else:
            fb_threshold = 0.7
        logger.info(f'Models saying AI: {models_saying_ai}/3, feedback threshold: {fb_threshold}')

        try:
            from models.feedback_trainer import predict_with_feedback_model
            feature_vectors = self.extract_features(video_path)
            if feature_vectors is not None:
                fb_result = predict_with_feedback_model(feature_vectors)
                if fb_result is not None:
                    fb_label, fb_confidence = fb_result
                    logger.info(f'Feedback model: {fb_label} ({fb_confidence:.2f}), Base: {base_result} ({avg_score:.3f})')
                    if fb_confidence > fb_threshold:
                        # Don't let feedback model override to AI when primary clearly says Real
                        if fb_label == 'AI-generated' and avg_score < 0.4:
                            logger.info(f'Feedback override blocked: primary too low ({avg_score:.3f}) for AI')
                        else:
                            return fb_label
        except Exception as e:
            logger.warning(f'Feedback model check failed: {e}')

        return base_result

    def extract_features(self, video_path):
        """Extract feature vectors from video frames for feedback training."""
        frames = extract_frames(video_path, max_frames=8)
        if not frames:
            return None

        all_features = []
        for frame in frames:
            img = Image.fromarray(frame)
            inputs = self.processor(images=img, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            feature = hidden.mean(dim=1).squeeze().numpy()
            all_features.append(feature)

        return np.array(all_features)


def extract_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frames

    interval = max(1, total_frames // max_frames)
    frame_idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1
    cap.release()
    return frames


def load_model():
    return AIDetector()
