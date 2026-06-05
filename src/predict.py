"""
predict.py — Single-video inference with visualisation.

Usage:
    python src/predict.py --video path/to/video.mp4 --model violence_detection_model.h5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from preprocess import extract_frames

CLASSES = ["NonViolence", "Violence"]


def predict_video(video_path, model, classes=CLASSES, show_frames=True):
    """
    Predicts the class of a single video.

    Args:
        video_path  (str): Path to the video file.
        model            : Loaded Keras model.
        classes    (list): Class label names.
        show_frames (bool): Whether to display extracted frames.

    Returns:
        (label: str, confidence: float) tuple.
    """
    frames = extract_frames(video_path)

    if frames is None:
        print("Could not load video.")
        return None, None

    # ── Visualise extracted frames ────────────────────────────
    if show_frames:
        fig, axes = plt.subplots(1, len(frames), figsize=(14, 3))
        for i, (ax, frame) in enumerate(zip(axes, frames)):
            ax.imshow(frame)
            ax.axis("off")
            ax.set_title(f"F{i + 1}", fontsize=8)
        plt.suptitle("Extracted Frames", fontsize=11)
        plt.tight_layout()
        plt.show()

    # ── Predict ───────────────────────────────────────────────
    input_tensor = np.expand_dims(frames, axis=0).astype("float32")
    probs = model.predict(input_tensor, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    confidence = float(probs[predicted_idx])

    print("\n" + "=" * 40)
    for cls, prob in zip(classes, probs):
        bar = "█" * int(prob * 20)
        print(f"  {cls:15s}: {prob * 100:5.2f}%  {bar}")
    print("=" * 40)
    print(f"  Prediction : {classes[predicted_idx]}")
    print(f"  Confidence : {confidence * 100:.2f}%")
    print("=" * 40 + "\n")

    return classes[predicted_idx], confidence


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Violence Detection — Single Video Inference")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="violence_detection_model.h5", help="Path to .h5 model file")
    parser.add_argument("--no-frames", action="store_true", help="Skip frame visualisation")
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model = load_model(args.model)

    predict_video(args.video, model, show_frames=not args.no_frames)
