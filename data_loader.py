"""Load image datasets from class-per-folder directory layouts.

Supported layouts
  1) dataset/{train,val,test}/<class_name>/*.png      (bundled dataset)
  2) <folder>/<class_name>/*.png                      (flat, e.g. GTSRB 'Train'
     folder with sub-folders 0..42) -> split automatically 70/15/15
"""
from pathlib import Path

import cv2
import numpy as np

from config import IMAGE_EXTENSIONS, IMG_SIZE, SEED


def _class_dirs(root):
    return sorted(d.name for d in Path(root).iterdir() if d.is_dir())


def _load_folder(root, classes):
    X, y = [], []
    for idx, cls in enumerate(classes):
        cls_dir = Path(root) / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            X.append(img)
            y.append(idx)
    return np.asarray(X, dtype=np.uint8), np.asarray(y, dtype=np.int64)


def load_dataset(data_dir):
    """Return dict with X_train, y_train, X_val, y_val, X_test, y_test, classes."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}\n"
            "Run `python generate_dataset.py` first."
        )

    if (data_dir / "train").is_dir():
        classes = _class_dirs(data_dir / "train")
        out = {"classes": classes}
        for split in ("train", "val", "test"):
            if (data_dir / split).is_dir():
                X, y = _load_folder(data_dir / split, classes)
            else:
                X = np.zeros((0, IMG_SIZE, IMG_SIZE, 3), np.uint8)
                y = np.zeros((0,), np.int64)
            out[f"X_{split}"], out[f"y_{split}"] = X, y
        return out

    # flat layout -> stratified split
    from sklearn.model_selection import train_test_split

    classes = _class_dirs(data_dir)
    X, y = _load_folder(data_dir, classes)
    if len(X) == 0:
        raise RuntimeError(f"No images found in {data_dir}")
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return {"classes": classes,
            "X_train": X_tr, "y_train": y_tr,
            "X_val": X_va, "y_val": y_va,
            "X_test": X_te, "y_test": y_te}


def preprocess_image(path_or_array):
    """Read an image path (or BGR ndarray) -> RGB uint8 array (IMG_SIZE, IMG_SIZE, 3)."""
    if isinstance(path_or_array, (str, Path)):
        img = cv2.imread(str(path_or_array))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path_or_array}")
    else:
        img = path_or_array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
