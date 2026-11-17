"""Central configuration for the Traffic Sign Recognition project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODELS_DIR / "traffic_sign_model.keras"
LABELS_PATH = MODELS_DIR / "labels.json"

IMG_SIZE = 48          # images are resized to IMG_SIZE x IMG_SIZE
SEED = 42

# Classes of the bundled (generated) dataset.
CLASS_NAMES = [
    "speed_20", "speed_30", "speed_50", "speed_60", "speed_80",
    "stop", "yield", "no_entry", "warning",
    "ahead_only", "turn_right", "turn_left", "priority_road",
]

# Images per class for each split when generating the dataset.
SPLIT_SIZES = {"train": 300, "val": 60, "test": 60}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm"}
