"""Train the traffic sign classifier.

Usage:
    python train.py
    python train.py --data_dir dataset --epochs 40 --batch_size 64
    python train.py --data_dir "C:/data/GTSRB/Train"      # real GTSRB (flat folders)
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (DATASET_DIR, LABELS_PATH, MODEL_PATH, MODELS_DIR,
                    OUTPUTS_DIR, SEED)
from data_loader import load_dataset
from model import build_model


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history["val_" + key], label="validation")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DATASET_DIR))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("Loading dataset ...")
    d = load_dataset(args.data_dir)
    classes = d["classes"]
    print(f"{len(classes)} classes | train {len(d['X_train'])} | "
          f"val {len(d['X_val'])} | test {len(d['X_test'])}")

    LABELS_PATH.write_text(json.dumps(classes, indent=2))

    model = build_model(len(classes))
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy",
            save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        d["X_train"], d["y_train"],
        validation_data=(d["X_val"], d["y_val"]),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=2,
    )

    plot_history(history, OUTPUTS_DIR / "training_curves.png")
    (OUTPUTS_DIR / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals]
                    for k, vals in history.history.items()}, indent=2))

    if len(d["X_test"]):
        loss, acc = model.evaluate(d["X_test"], d["y_test"], verbose=0)
        print(f"\nTest accuracy: {acc * 100:.2f}%  (loss {loss:.4f})")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
