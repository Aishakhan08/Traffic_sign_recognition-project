"""Predict traffic signs from image files or a live webcam.

Usage:
    python predict.py path/to/image.png [more.png ...]
    python predict.py --webcam            (press q to quit)
"""
import argparse
import json

import cv2
import numpy as np
from tensorflow import keras

from config import IMG_SIZE, LABELS_PATH, MODEL_PATH
from data_loader import preprocess_image


def load():
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run `python train.py` first.")
    classes = json.loads(LABELS_PATH.read_text())
    return keras.models.load_model(MODEL_PATH), classes


def predict_top_k(model, classes, rgb_image, k=3):
    probs = model.predict(rgb_image[None, ...], verbose=0)[0]
    top = np.argsort(probs)[::-1][:k]
    return [(classes[i], float(probs[i])) for i in top]


def run_files(paths):
    model, classes = load()
    for path in paths:
        result = predict_top_k(model, classes, preprocess_image(path))
        print(f"\n{path}")
        for name, p in result:
            print(f"  {name:<16s} {p * 100:6.2f}%")


def run_webcam(camera=0):
    model, classes = load()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam.")
    print("Hold a sign inside the green box. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        side = min(h, w) // 2
        x1, y1 = w // 2 - side // 2, h // 2 - side // 2
        roi = frame[y1:y1 + side, x1:x1 + side]
        name, prob = predict_top_k(model, classes, preprocess_image(roi), 1)[0]
        cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {prob * 100:.0f}%", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Traffic Sign Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="image file(s) to classify")
    ap.add_argument("--webcam", action="store_true")
    args = ap.parse_args()
    if args.webcam:
        run_webcam()
    elif args.images:
        run_files(args.images)
    else:
        ap.print_help()
