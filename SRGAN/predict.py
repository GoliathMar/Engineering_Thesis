#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")


from model import build_generator

parser = argparse.ArgumentParser()
parser.add_argument("model_weights", help="Ścieżka do pliku .h5")
parser.add_argument("input_dir", help="Katalog z obrazami Low-Res")
parser.add_argument("output_dir", help="Gdzie zapisać wyniki")
args = parser.parse_args()

MODEL_PATH = Path(args.model_weights).resolve()
IN_DIR = Path(args.input_dir).resolve()
OUT_DIR = Path(args.output_dir).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading generator: {MODEL_PATH}")


gen = build_generator(scale=2)


try:
    gen.load_weights(str(MODEL_PATH))
except ValueError as e:
    print("\nError: Incorrect weights")
    raise e

print("Model loaded. Starting inference...")

pngs = sorted(IN_DIR.rglob("*.png"))
for src in pngs:
    #Loading [0, 255] -> [0, 1]
    img = Image.open(src).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0


    lr = arr * 2.0 - 1.0
    lr = np.expand_dims(lr, axis=0)


    sr = gen.predict(lr, verbose=0)[0]


    sr = (sr + 1.0) * 127.5
    sr = np.clip(sr, 0, 255).astype(np.uint8)

    #Save
    out_img = Image.fromarray(sr)
    dst = OUT_DIR / src.name
    out_img.save(dst)
    print(f"Saved: {dst.name}")

print("Done.")