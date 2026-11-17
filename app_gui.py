"""Simple desktop GUI: choose an image and see the predicted traffic sign.

Usage:
    python app_gui.py
"""
import json
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk
from tensorflow import keras

from config import IMG_SIZE, LABELS_PATH, MODEL_PATH


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Traffic Sign Recognition")
        self.geometry("420x520")
        self.resizable(False, False)

        if not MODEL_PATH.exists():
            messagebox.showerror("Model missing", "Run `python train.py` first.")
            self.destroy()
            return
        self.model = keras.models.load_model(MODEL_PATH)
        self.classes = json.loads(LABELS_PATH.read_text())

        tk.Label(self, text="Traffic Sign Recognition",
                 font=("Arial", 16, "bold")).pack(pady=10)
        self.canvas = tk.Label(self, width=256, height=256, bg="#dddddd")
        self.canvas.pack(pady=5)
        tk.Button(self, text="Choose image...", command=self.choose,
                  font=("Arial", 12)).pack(pady=10)
        self.result = tk.Label(self, text="", font=("Arial", 13), justify="left")
        self.result.pack(pady=5)

    def choose(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.ppm")])
        if not path:
            return
        img = Image.open(path).convert("RGB")
        shown = img.resize((256, 256))
        self.photo = ImageTk.PhotoImage(shown)
        self.canvas.configure(image=self.photo, width=256, height=256)

        arr = np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS), dtype=np.uint8)
        probs = self.model.predict(arr[None, ...], verbose=0)[0]
        top = np.argsort(probs)[::-1][:3]
        self.result.configure(text="\n".join(
            f"{self.classes[i]:<16s} {probs[i] * 100:5.1f}%" for i in top))


if __name__ == "__main__":
    App().mainloop()
