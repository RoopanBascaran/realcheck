from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import pickle

UPLOAD_FOLDER = 'static/uploads'

class FeedbackTrainer:
    def __init__(self, feedback_data_path='feedback_data.pkl'):
        self.model = LogisticRegression(max_iter=1000)
        self.feedback_data_path = feedback_data_path
        self.feedback_data = self.load_feedback_data()
        self._feature_extractor = None

    def _get_feature_extractor(self):
        if self._feature_extractor is None:
            from tensorflow.keras.applications import Xception
            from tensorflow.keras.models import Model
            base_model = Xception(weights='imagenet', include_top=False, pooling='avg')
            self._feature_extractor = base_model
        return self._feature_extractor

    def load_feedback_data(self):
        if os.path.exists(self.feedback_data_path):
            with open(self.feedback_data_path, 'rb') as f:
                return pickle.load(f)
        return []

    def collect_feedback(self, video_id, user_feedback):
        self.feedback_data.append({'video_id': video_id, 'feedback': user_feedback})
        self.save_feedback_data()

    def save_feedback_data(self):
        with open(self.feedback_data_path, 'wb') as f:
            pickle.dump(self.feedback_data, f)

    def retrain_model(self):
        if len(self.feedback_data) < 2:
            return

        features = []
        labels = []
        for feedback in self.feedback_data:
            video_id = feedback['video_id']
            user_feedback = feedback['feedback']
            feature_vector = self.extract_features(video_id)
            if feature_vector is None:
                continue
            features.append(feature_vector)
            labels.append(1 if user_feedback == 'AI-generated' else 0)

        if len(features) < 2 or len(set(labels)) < 2:
            return

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'Model retrained with accuracy: {accuracy:.2f}')

    def extract_features(self, video_id):
        from models.xception_model import preprocess_video
        video_path = os.path.join(UPLOAD_FOLDER, video_id)
        if not os.path.exists(video_path):
            return None
        frames = preprocess_video(video_path)
        if len(frames) == 0:
            return None
        extractor = self._get_feature_extractor()
        frame_features = extractor.predict(frames)
        return np.mean(frame_features, axis=0)


_trainer = None

def _get_trainer():
    global _trainer
    if _trainer is None:
        _trainer = FeedbackTrainer()
    return _trainer


def collect_feedback(video_id, user_feedback):
    _get_trainer().collect_feedback(video_id, user_feedback)


def retrain_model():
    _get_trainer().retrain_model()
