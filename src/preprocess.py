"""
preprocess.py — Frame extraction and dataset loading utilities.
"""

import os
import cv2
import numpy as np


def extract_frames(video_path, img_size=64, seq_len=8):
    """
    Extracts `seq_len` evenly-spaced frames from a video file.

    Args:
        video_path (str): Path to the video file (.mp4 / .avi / .mov).
        img_size   (int): Resize each frame to (img_size x img_size).
        seq_len    (int): Number of frames to extract.

    Returns:
        np.ndarray of shape (seq_len, img_size, img_size, 3), float32 in [0, 1].
        Returns None if the video cannot be read.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    # evenly spaced indices across full video duration
    frame_indices = set(np.linspace(0, total_frames - 1, seq_len, dtype=int))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in frame_indices:
            try:
                frame = cv2.resize(frame, (img_size, img_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype("float32") / 255.0
                frames.append(frame)
            except Exception:
                cap.release()
                return None

        frame_count += 1

    cap.release()

    if not frames:
        return None

    # pad with last frame if fewer frames were captured than expected
    while len(frames) < seq_len:
        frames.append(frames[-1])

    return np.array(frames[:seq_len])


def load_dataset(dataset_path, classes, img_size=64, seq_len=8):
    """
    Loads all videos from dataset_path/<class>/ folders.

    Args:
        dataset_path (str): Root folder containing one subfolder per class.
        classes     (list): List of class names, e.g. ["NonViolence", "Violence"].
        img_size     (int): Frame resize dimension.
        seq_len      (int): Frames per video.

    Returns:
        X (np.ndarray): shape (N, seq_len, img_size, img_size, 3), dtype float16.
        y (np.ndarray): shape (N,), integer labels.
    """
    X, y = [], []
    skipped = 0

    for label, cls in enumerate(classes):
        folder = os.path.join(dataset_path, cls)

        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".mp4", ".avi", ".mov"))]
        print(f"{cls}: {len(files)} videos found")

        for file in files:
            frames = extract_frames(os.path.join(folder, file), img_size, seq_len)
            if frames is not None:
                X.append(frames)
                y.append(label)
            else:
                skipped += 1

    print(f"Loaded: {len(X)} videos | Skipped: {skipped}")
    return np.array(X, dtype="float16"), np.array(y)
