"""Evaluate the trained model on the test set (report + confusion matrix).

Usage:
    python evaluate.py
    python evaluate.py --data_dir dataset
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from config import DATASET_DIR, LABELS_PATH, MODEL_PATH, OUTPUTS_DIR
from data_loader import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DATASET_DIR))
    args = ap.parse_args()

    classes = json.loads(LABELS_PATH.read_text())
    model = keras.models.load_model(MODEL_PATH)
    d = load_dataset(args.data_dir)
    X, y = d["X_test"], d["y_test"]
    if d["classes"] != classes:
        raise SystemExit("Class list of this dataset differs from the trained model.")

    pred = np.argmax(model.predict(X, verbose=0), axis=1)
    acc = float(np.mean(pred == y))
    print(f"Test accuracy: {acc * 100:.2f}%  ({int((pred != y).sum())} errors / {len(y)})\n")
    print(classification_report(y, pred, target_names=classes, digits=3))

    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (accuracy {acc * 100:.1f}%)")
    thresh = cm.max() / 2 if cm.max() else 1
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im)
    fig.tight_layout()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out = OUTPUTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=130)
    print(f"Confusion matrix saved to {out}")


if __name__ == "__main__":
    main()
