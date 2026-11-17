# Traffic Sign Recognition using Deep Learning

A complete CNN-based traffic sign classifier (TensorFlow/Keras + OpenCV) with a
bundled dataset, training, evaluation, prediction, webcam and GUI apps.

## Project structure

```
traffic_sign_recognition/
├── dataset/                 # bundled dataset (13 classes, 48x48 PNG)
│   ├── train/<class>/*.png  # 300 images per class
│   ├── val/<class>/*.png    # 60 images per class
│   ├── test/<class>/*.png   # 60 images per class
│   └── dataset_samples.png  # preview of the data
├── config.py                # paths, image size, class list
├── generate_dataset.py      # (re)generate the synthetic dataset
├── data_loader.py           # loads image folders -> numpy arrays
├── model.py                 # CNN architecture
├── train.py                 # training (saves model + curves)
├── evaluate.py              # test accuracy, report, confusion matrix
├── predict.py               # predict images or use the webcam
├── app_gui.py               # tkinter desktop GUI
├── requirements.txt
├── run_all.bat / run_all.sh # one-click full pipeline
├── models/                  # trained model + labels.json (after training)
└── outputs/                 # curves, confusion matrix (after training)
```

## Classes (13)

speed_20, speed_30, speed_50, speed_60, speed_80, stop, yield, no_entry,
warning, ahead_only, turn_right, turn_left, priority_road

## Setup

Python 3.10 - 3.12 is recommended for TensorFlow.

```bash
python -m venv venv
# Windows:   venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python train.py                 # train (about a few minutes on CPU)
python evaluate.py              # test report + outputs/confusion_matrix.png
python predict.py dataset/test/stop/stop_0000.png
python predict.py --webcam      # live camera, press q to quit
python app_gui.py               # GUI
```

Or run everything at once: `run_all.bat` (Windows) / `./run_all.sh` (Linux/Mac).

## Model

Rescaling -> light augmentation (rotation, zoom, shift, contrast; no flips,
since flips would swap turn-left / turn-right) -> 3 blocks of
[Conv-BN-ReLU x2, MaxPool, Dropout] with 32/64/128 filters ->
GlobalAveragePooling -> Dense(128) -> Dropout -> Softmax.
Adam optimizer, ReduceLROnPlateau, EarlyStopping, best-model checkpointing.

## About the dataset

The bundled dataset is **synthetic**: signs are drawn with OpenCV on random
backgrounds with rotation, scale, shift, brightness, blur and noise. It lets the
project run immediately without downloads. Regenerate or enlarge it with:

```bash
python generate_dataset.py --train 500 --val 100 --test 100
```

## Using the real GTSRB dataset (43 classes)

1. Download GTSRB (e.g. from Kaggle, "GTSRB - German Traffic Sign Recognition
   Benchmark"). Use the `Train` folder that contains sub-folders `0` ... `42`.
2. Train directly on it (it is split 70/15/15 automatically):

```bash
python train.py --data_dir "path/to/GTSRB/Train" --epochs 40
python evaluate.py --data_dir "path/to/GTSRB/Train"
```

Class folder names become the labels (saved in `models/labels.json`).
Any folder of the form `<class_name>/*.png` works as a custom dataset.

## Notes

- Models trained on synthetic data will not be reliable on real photos; use
  GTSRB (above) for real-world use.
- `predict.py --webcam` classifies the centre square of the camera frame.
