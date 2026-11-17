"""Generate a synthetic traffic-sign image dataset.

Each sign is drawn with OpenCV, placed on a random cluttered background and
then augmented (rotation, scale, shift, brightness/contrast, blur, noise).

Usage:
    python generate_dataset.py
    python generate_dataset.py --out dataset --train 500 --val 100 --test 100
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from config import CLASS_NAMES, DATASET_DIR, IMG_SIZE, SEED, SPLIT_SIZES

S = 128  # drawing canvas size
C = (64, 64)  # canvas center
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
RED = (40, 30, 200)      # BGR
BLUE = (190, 90, 20)     # BGR
YELLOW = (0, 200, 250)   # BGR


# ----------------------------------------------------------------- helpers
def shrink(pts, factor):
    pts = np.array(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    return (c + (pts - c) * factor).astype(np.int32)


def put_text_centered(img, text, center, height, color, thickness=4):
    font = cv2.FONT_HERSHEY_DUPLEX
    (_, h), _ = cv2.getTextSize(text, font, 1.0, thickness)
    scale = height / h
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (int(center[0] - w / 2), int(center[1] + h / 2))
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


# ------------------------------------------------------------ sign drawing
def draw_speed(img, number):
    cv2.circle(img, C, 54, RED, -1, cv2.LINE_AA)
    cv2.circle(img, C, 42, WHITE, -1, cv2.LINE_AA)
    put_text_centered(img, str(number), C, 32, BLACK, 4)


def draw_stop(img):
    ang = np.deg2rad(22.5 + 45 * np.arange(8))
    pts = np.stack([C[0] + 56 * np.cos(ang), C[1] + 56 * np.sin(ang)], 1).astype(np.int32)
    cv2.fillPoly(img, [pts], WHITE, cv2.LINE_AA)
    cv2.fillPoly(img, [shrink(pts, 0.92)], RED, cv2.LINE_AA)
    put_text_centered(img, "STOP", C, 24, WHITE, 3)


def draw_yield(img):
    outer = np.array([(8, 24), (120, 24), (64, 118)], dtype=np.int32)
    cv2.fillPoly(img, [outer], RED, cv2.LINE_AA)
    cv2.fillPoly(img, [shrink(outer, 0.6)], WHITE, cv2.LINE_AA)


def draw_no_entry(img):
    cv2.circle(img, C, 54, RED, -1, cv2.LINE_AA)
    cv2.rectangle(img, (26, 54), (102, 74), WHITE, -1)


def draw_warning(img):
    outer = np.array([(64, 8), (122, 108), (6, 108)], dtype=np.int32)
    cv2.fillPoly(img, [outer], RED, cv2.LINE_AA)
    cv2.fillPoly(img, [shrink(outer, 0.68)], WHITE, cv2.LINE_AA)
    put_text_centered(img, "!", (64, 76), 34, BLACK, 6)


def draw_arrow_sign(img, direction):
    cv2.circle(img, C, 54, BLUE, -1, cv2.LINE_AA)
    cv2.circle(img, C, 54, WHITE, 3, cv2.LINE_AA)
    ends = {
        "up": ((64, 100), (64, 26)),
        "right": ((26, 64), (102, 64)),
        "left": ((102, 64), (26, 64)),
    }
    a, b = ends[direction]
    cv2.arrowedLine(img, a, b, WHITE, 13, cv2.LINE_AA, tipLength=0.5)


def draw_priority(img):
    pts = np.array([(64, 6), (122, 64), (64, 122), (6, 64)], dtype=np.int32)
    cv2.fillPoly(img, [pts], BLACK, cv2.LINE_AA)
    cv2.fillPoly(img, [shrink(pts, 0.93)], WHITE, cv2.LINE_AA)
    cv2.fillPoly(img, [shrink(pts, 0.72)], YELLOW, cv2.LINE_AA)


def draw_sign(img, name):
    if name.startswith("speed_"):
        draw_speed(img, int(name.split("_")[1]))
    elif name == "stop":
        draw_stop(img)
    elif name == "yield":
        draw_yield(img)
    elif name == "no_entry":
        draw_no_entry(img)
    elif name == "warning":
        draw_warning(img)
    elif name == "ahead_only":
        draw_arrow_sign(img, "up")
    elif name == "turn_right":
        draw_arrow_sign(img, "right")
    elif name == "turn_left":
        draw_arrow_sign(img, "left")
    elif name == "priority_road":
        draw_priority(img)
    else:
        raise ValueError(f"Unknown class: {name}")


# -------------------------------------------------------------- background
def make_background(rng):
    c1 = rng.integers(0, 256, 3).astype(np.float32)
    c2 = rng.integers(0, 256, 3).astype(np.float32)
    t = np.linspace(0, 1, S, dtype=np.float32)[:, None, None]
    grad = c1 * (1 - t) + c2 * t                       # (S,1,3)
    bg = np.repeat(grad, S, axis=1)                    # (S,S,3)
    if rng.random() < 0.5:
        bg = bg.transpose(1, 0, 2).copy()
    bg = bg.astype(np.uint8)
    for _ in range(rng.integers(2, 7)):
        x1, y1 = rng.integers(0, S, 2)
        x2, y2 = rng.integers(0, S, 2)
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        cv2.rectangle(bg, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    return cv2.GaussianBlur(bg, (9, 9), 0)


# ------------------------------------------------------------ augmentation
def augment(img, rng):
    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.75, 1.05)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), angle, scale)
    M[:, 2] += rng.uniform(-8, 8, 2)
    img = cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_REFLECT)

    img = img.astype(np.float32) * rng.uniform(0.5, 1.4) + rng.uniform(-40, 40)
    img = np.clip(img, 0, 255)
    if rng.random() < 0.4:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    img = img + rng.normal(0, rng.uniform(0, 12), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def make_image(name, rng):
    img = make_background(rng)
    draw_sign(img, name)
    return augment(img, rng)


# -------------------------------------------------------------------- main
def save_sample_grid(out_dir, rng):
    tiles = []
    for name in CLASS_NAMES:
        row = [make_image(name, rng) for _ in range(8)]
        tiles.append(np.hstack(row))
    cv2.imwrite(str(Path(out_dir) / "dataset_samples.png"), np.vstack(tiles))


def main():
    p = argparse.ArgumentParser(description="Generate synthetic traffic sign dataset")
    p.add_argument("--out", default=str(DATASET_DIR))
    p.add_argument("--train", type=int, default=SPLIT_SIZES["train"])
    p.add_argument("--val", type=int, default=SPLIT_SIZES["val"])
    p.add_argument("--test", type=int, default=SPLIT_SIZES["test"])
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(args.seed)
    sizes = {"train": args.train, "val": args.val, "test": args.test}

    for split, n in sizes.items():
        for name in CLASS_NAMES:
            d = out / split / name
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                cv2.imwrite(str(d / f"{name}_{i:04d}.png"), make_image(name, rng))
        print(f"{split}: {n} images x {len(CLASS_NAMES)} classes")

    save_sample_grid(out, rng)
    total = sum(sizes.values()) * len(CLASS_NAMES)
    print(f"Done. {total} images written to {out}")


if __name__ == "__main__":
    main()
