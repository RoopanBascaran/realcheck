"""
Training script for AI vs Real video detection.

Dataset structure:
    dataset/
    ├── real/
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    └── ai_generated/
        ├── video1.mp4
        ├── video2.mp4
        └── ...

Usage:
    python models/train.py --dataset dataset/ --epochs 10 --batch_size 16
"""

import argparse
import os
import sys
import numpy as np
from glob import glob

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from models.xception_model import build_model, extract_frames, preprocess_video


def load_dataset(dataset_path):
    """Load video paths and labels from dataset directory."""
    real_dir = os.path.join(dataset_path, 'real')
    ai_dir = os.path.join(dataset_path, 'ai_generated')

    video_exts = ('*.mp4', '*.mov', '*.avi', '*.mkv')

    real_videos = []
    ai_videos = []
    for ext in video_exts:
        real_videos.extend(glob(os.path.join(real_dir, ext)))
        ai_videos.extend(glob(os.path.join(ai_dir, ext)))

    print(f'Found {len(real_videos)} real videos')
    print(f'Found {len(ai_videos)} AI-generated videos')

    if len(real_videos) == 0 or len(ai_videos) == 0:
        print('ERROR: Need videos in both dataset/real/ and dataset/ai_generated/')
        sys.exit(1)

    paths = real_videos + ai_videos
    labels = [0] * len(real_videos) + [1] * len(ai_videos)

    return paths, labels


def extract_all_frames(video_paths, labels, max_frames=16):
    """Extract frames from all videos, returning per-frame data."""
    all_frames = []
    all_labels = []

    for i, (path, label) in enumerate(zip(video_paths, labels)):
        print(f'  Processing {i+1}/{len(video_paths)}: {os.path.basename(path)}', end='\r')
        frames = preprocess_video(path)
        if len(frames) == 0:
            continue
        for frame in frames:
            all_frames.append(frame)
            all_labels.append(label)

    print()
    return np.array(all_frames), np.array(all_labels)


def train(dataset_path, epochs=10, batch_size=16, learning_rate=1e-4):
    """Train the AI vs Real detection model."""

    # Load dataset
    print('\n--- Loading dataset ---')
    paths, labels = load_dataset(dataset_path)

    # Split into train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f'Train: {len(train_paths)} videos, Val: {len(val_paths)} videos')

    # Extract frames
    print('\n--- Extracting training frames ---')
    X_train, y_train = extract_all_frames(train_paths, train_labels)
    print(f'Training frames: {len(X_train)}')

    print('\n--- Extracting validation frames ---')
    X_val, y_val = extract_all_frames(val_paths, val_labels)
    print(f'Validation frames: {len(X_val)}')

    # Build model
    print('\n--- Building model ---')
    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Prepare weights directory
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, 'xception_ai_detector.h5')

    # Callbacks
    callbacks = [
        ModelCheckpoint(weights_path, monitor='val_accuracy', save_best_only=True,
                        save_weights_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    ]

    # Train
    print('\n--- Training ---')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # Final evaluation
    print('\n--- Evaluation ---')
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f'Validation accuracy: {val_acc:.4f}')
    print(f'Validation loss: {val_loss:.4f}')
    print(f'\nWeights saved to: {weights_path}')

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AI vs Real video detector')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    args = parser.parse_args()

    train(args.dataset, args.epochs, args.batch_size, args.lr)
