import numpy as np
import os
import json
import logging
import threading

logger = logging.getLogger(__name__)

FEEDBACK_DIR = 'feedback_data'
FEEDBACK_INDEX = os.path.join(FEEDBACK_DIR, 'index.json')

# Use a separate HF Dataset repo for data (not the Space repo, to avoid rebuild)
HF_DATA_REPO = os.environ.get('HF_DATA_REPO', 'Karen-AI/realcheck-data')
HF_TOKEN = os.environ.get('HF_TOKEN')

WEIGHTS_PATH = os.path.join('models', 'weights', 'xception_ai_detector.h5')


def _ensure_dirs():
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)


def _upload_to_hf(local_path, repo_path):
    """Upload a file to the HF dataset repo for persistence (won't trigger Space rebuild)."""
    if not HF_TOKEN:
        logger.warning('HF_TOKEN not set, skipping upload')
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=HF_DATA_REPO,
            repo_type='dataset',
            commit_message=f'Auto-save: {repo_path}'
        )
        logger.info(f'Uploaded {repo_path} to HF dataset repo')
        return True
    except Exception as e:
        logger.error(f'HF upload failed for {repo_path}: {e}')
        return False


def _download_from_hf(repo_path, local_path):
    """Download a file from HF dataset repo."""
    if not HF_TOKEN:
        return False
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=repo_path,
            repo_type='dataset',
            token=HF_TOKEN,
        )
        # hf_hub_download returns the cached path — copy to our local path
        import shutil
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(downloaded, local_path)
        logger.info(f'Downloaded {repo_path} -> {local_path}')
        return True
    except Exception as e:
        logger.debug(f'HF download skipped for {repo_path}: {e}')
        return False


def restore_from_hf():
    """Download feedback data and classifier from HF dataset repo on startup.
    Does NOT import TensorFlow — only downloads files."""
    _ensure_dirs()

    _download_from_hf('index.json', FEEDBACK_INDEX)
    _download_from_hf('features.npz', os.path.join(FEEDBACK_DIR, 'features.npz'))
    _download_from_hf('classifier.pkl', os.path.join(FEEDBACK_DIR, 'classifier.pkl'))

    count = get_feedback_count()
    if count > 0:
        logger.info(f'Restored {count} feedback entries from HF')


def _load_index():
    _ensure_dirs()
    if os.path.exists(FEEDBACK_INDEX):
        with open(FEEDBACK_INDEX, 'r') as f:
            return json.load(f)
    return []


def _save_index(data):
    _ensure_dirs()
    with open(FEEDBACK_INDEX, 'w') as f:
        json.dump(data, f, indent=2)


def _load_features():
    """Load saved feature vectors."""
    path = os.path.join(FEEDBACK_DIR, 'features.npz')
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return dict(data)
    return {}


def _save_features(features_dict):
    """Save feature vectors to disk."""
    path = os.path.join(FEEDBACK_DIR, 'features.npz')
    np.savez_compressed(path, **features_dict)


def _persist_to_hf():
    """Upload all data to HF in background (non-blocking)."""
    def _do_upload():
        _upload_to_hf(FEEDBACK_INDEX, 'index.json')
        features_path = os.path.join(FEEDBACK_DIR, 'features.npz')
        if os.path.exists(features_path):
            _upload_to_hf(features_path, 'features.npz')
        if os.path.exists(WEIGHTS_PATH):
            _upload_to_hf(WEIGHTS_PATH, 'xception_ai_detector.h5')

    thread = threading.Thread(target=_do_upload, daemon=True)
    thread.start()


def save_video_features(video_path, video_id):
    """Extract Xception features from video and save as compact feature vectors."""
    from models.xception_model import extract_frames
    from tensorflow.keras.applications.xception import preprocess_input
    from tensorflow.keras.applications import Xception
    import cv2

    _ensure_dirs()
    frames = extract_frames(video_path, max_frames=8)
    if not frames:
        return False

    # Preprocess frames
    processed = []
    for frame in frames:
        img = cv2.resize(frame, (299, 299))
        img = np.array(img, dtype=np.float32)
        img = preprocess_input(img)
        processed.append(img)

    processed = np.array(processed)

    # Extract compact feature vectors (2048-dim) instead of saving full frames
    extractor = Xception(weights='imagenet', include_top=False, pooling='avg')
    feature_vectors = extractor.predict(processed, verbose=0)  # shape: (N, 2048)

    # Save to features dict (each video is ~65KB instead of ~17MB)
    features_dict = _load_features()
    features_dict[video_id] = feature_vectors
    _save_features(features_dict)

    logger.info(f'Saved {len(frames)} feature vectors for video {video_id}')
    return True


def collect_feedback(video_id, label):
    """Store feedback: label is 'AI-generated' or 'Real'."""
    index = _load_index()
    for entry in index:
        if entry['video_id'] == video_id:
            entry['label'] = label
            _save_index(index)
            logger.info(f'Updated feedback for {video_id}: {label}')
            count = len([e for e in index if 'label' in e])
            return count
    index.append({'video_id': video_id, 'label': label})
    _save_index(index)
    count = len([e for e in index if 'label' in e])
    logger.info(f'Collected feedback #{count} for {video_id}: {label}')
    return count


def get_feedback_count():
    """Return number of labeled samples."""
    index = _load_index()
    return len([e for e in index if 'label' in e])


def retrain_model_async(callback=None):
    """Run retraining in a background thread so it doesn't block the webhook."""
    def _do_train():
        try:
            success = _retrain_model()
            if callback:
                callback(success)
        except Exception as e:
            logger.error(f'Retraining failed: {e}')
            if callback:
                callback(False)

    thread = threading.Thread(target=_do_train, daemon=True)
    thread.start()


def _retrain_model():
    """Fine-tune a logistic regression on Xception features (lightweight)."""
    from sklearn.linear_model import LogisticRegression
    import pickle

    index = _load_index()
    labeled = [e for e in index if 'label' in e]

    if len(labeled) < 5:
        logger.info(f'Not enough feedback to train ({len(labeled)}/5)')
        return False

    features_dict = _load_features()

    all_features = []
    all_labels = []

    for entry in labeled:
        video_id = entry['video_id']
        if video_id not in features_dict:
            logger.warning(f'No features found for {video_id}, skipping')
            continue

        label = 1 if entry['label'] == 'AI-generated' else 0
        vectors = features_dict[video_id]  # shape: (N, 2048)
        # Average frame features into one vector per video
        avg_vector = np.mean(vectors, axis=0)
        all_features.append(avg_vector)
        all_labels.append(label)

    if len(all_features) < 5:
        logger.info('Not enough valid samples to train')
        return False

    # Check we have both classes
    unique_labels = set(all_labels)
    if len(unique_labels) < 2:
        logger.info(f'Need both AI-generated and Real samples, only have: {unique_labels}')
        return False

    X = np.array(all_features)
    y = np.array(all_labels)

    logger.info(f'Training on {len(X)} videos')
    logger.info(f'Label distribution: {sum(y)} AI-generated, {len(y) - sum(y)} Real')

    # Train a lightweight classifier on top of Xception features
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)

    # Save classifier
    clf_path = os.path.join(FEEDBACK_DIR, 'classifier.pkl')
    with open(clf_path, 'wb') as f:
        pickle.dump(clf, f)

    # Calculate training accuracy
    accuracy = clf.score(X, y)
    logger.info(f'Classifier trained with accuracy: {accuracy:.2f}')

    # Persist everything to HF
    _upload_to_hf(clf_path, 'classifier.pkl')
    _persist_to_hf()

    return True


def predict_with_feedback_model(feature_vector):
    """Predict using the feedback-trained classifier if available."""
    import pickle

    clf_path = os.path.join(FEEDBACK_DIR, 'classifier.pkl')
    if not os.path.exists(clf_path):
        return None

    with open(clf_path, 'rb') as f:
        clf = pickle.load(f)

    avg_features = np.mean(feature_vector, axis=0).reshape(1, -1)
    prediction = clf.predict(avg_features)[0]
    probability = clf.predict_proba(avg_features)[0]

    label = 'AI-generated' if prediction == 1 else 'Real'
    confidence = max(probability)
    logger.info(f'Feedback model prediction: {label} ({confidence:.2f})')
    return label, confidence
