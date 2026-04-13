from transformers import ViTImageProcessor, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import cv2
import os
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

        # Decision logic:
        # 1. If primary model says AI (avg > 0.5) → AI-generated
        # 2. If second model is very confident (avg > 0.95) → AI-generated
        # 3. Otherwise → Real
        second_model_triggered = False
        if avg_score > 0.5:
            base_result = 'AI-generated'
        elif second_avg > SECOND_MODEL_THRESHOLD:
            base_result = 'AI-generated'
            second_model_triggered = True
            logger.info(f'Second model override: {second_avg:.3f} > {SECOND_MODEL_THRESHOLD}')
        else:
            base_result = 'Real'

        # Check if feedback-trained classifier is available
        # Skip feedback override when second model triggered — it's a strong signal
        # that the feedback model hasn't been trained on yet
        if second_model_triggered:
            logger.info('Skipping feedback model — second model detection is authoritative')
            return base_result

        try:
            from models.feedback_trainer import predict_with_feedback_model
            feature_vectors = self.extract_features(video_path)
            if feature_vectors is not None:
                fb_result = predict_with_feedback_model(feature_vectors)
                if fb_result is not None:
                    fb_label, fb_confidence = fb_result
                    logger.info(f'Feedback model: {fb_label} ({fb_confidence:.2f}), Base: {base_result} ({avg_score:.3f})')
                    if fb_confidence > 0.7:
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
