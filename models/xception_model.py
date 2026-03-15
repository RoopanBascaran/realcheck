from transformers import ViTImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import cv2
import os
import torch
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = 'Nahrawy/AIorNot'


class AIDetector:
    def __init__(self):
        logger.info(f'Loading model: {MODEL_NAME}')
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        logger.info('Model loaded successfully')

    def classify(self, video_path):
        frames = extract_frames(video_path)
        if not frames:
            return 'Real'

        scores = []
        for frame in frames:
            img = Image.fromarray(frame)
            inputs = self.processor(images=img, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

            # Get label mapping
            labels = self.model.config.id2label
            ai_score = 0.0
            for idx, label_name in labels.items():
                if 'ai' in label_name.lower() or 'fake' in label_name.lower() or 'artificial' in label_name.lower():
                    ai_score = probs[idx].item()
                    break

            scores.append(ai_score)

        avg_score = np.mean(scores)
        logger.info(f'Classification: avg_score={avg_score:.3f} from {len(scores)} frames')
        base_result = 'AI-generated' if avg_score > 0.5 else 'Real'

        # Check if feedback-trained classifier is available
        try:
            from models.feedback_trainer import predict_with_feedback_model
            feature_vectors = self.extract_features(video_path)
            if feature_vectors is not None:
                fb_result = predict_with_feedback_model(feature_vectors)
                if fb_result is not None:
                    fb_label, fb_confidence = fb_result
                    logger.info(f'Feedback model: {fb_label} ({fb_confidence:.2f}), Base model: {base_result} ({avg_score:.3f})')
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
            # Use last hidden state, averaged over patches, as feature vector
            hidden = outputs.hidden_states[-1]  # (1, num_patches, hidden_dim)
            feature = hidden.mean(dim=1).squeeze().numpy()  # (hidden_dim,)
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
