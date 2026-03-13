from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'xception_ai_detector.h5')


def build_model():
    """Build Xception-based binary classifier for AI vs Real detection."""
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    # Freeze early layers, fine-tune last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model


class XceptionModel:
    def __init__(self):
        self.model = build_model()
        if os.path.exists(WEIGHTS_PATH):
            self.model.load_weights(WEIGHTS_PATH)
            print(f'Loaded trained weights from {WEIGHTS_PATH}')
        else:
            print('WARNING: No trained weights found. Predictions will be random.')
            print(f'Run "python models/train.py" to train the model first.')

    def preprocess_video(self, video_path):
        frames = extract_frames(video_path)
        processed_frames = []
        for frame in frames:
            img = cv2.resize(frame, (299, 299))
            img_array = np.array(img, dtype=np.float32)
            img_array = preprocess_input(img_array)
            processed_frames.append(img_array)
        return np.array(processed_frames)

    def predict(self, video_data):
        predictions = self.model.predict(video_data, verbose=0)
        avg_score = np.mean(predictions)
        return avg_score

    def classify(self, video_path):
        video_data = self.preprocess_video(video_path)
        if len(video_data) == 0:
            return 'Real'
        score = self.predict(video_data)
        return 'AI-generated' if score > 0.5 else 'Real'


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
    return XceptionModel()


def preprocess_video(video_path):
    frames = extract_frames(video_path)
    processed_frames = []
    for frame in frames:
        img = cv2.resize(frame, (299, 299))
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        processed_frames.append(img_array)
    return np.array(processed_frames)


def predict(model, video_data):
    return model.predict(video_data)
