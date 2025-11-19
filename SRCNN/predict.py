import sys
from os import listdir, makedirs
from os.path import isfile, join, exists
import argparse

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model, Model   # CHANGED
from tensorflow.keras.layers import Input

#CLI args
parser = argparse.ArgumentParser()
parser.add_argument("model", help="Ściezka to wyrenowanego .h5 modelu")
parser.add_argument("input_dir", help="Folder low-res")
parser.add_argument("output_dir", help="Folder wyjściowy")
parser.add_argument("--scale", type=int, default=2, help="Współczynnik skalowania (2)")

args = parser.parse_args()

if not exists(args.output_dir):
    makedirs(args.output_dir)


base = load_model(args.model, compile=False)


flex_input = Input(shape=(1, None, None))
x = flex_input
for layer in base.layers[1:]:
    x = layer(x)

m = Model(inputs=flex_input, outputs=x)

print("Using model:", args.model)
print("Input dir :", args.input_dir)
print("Output dir:", args.output_dir)
print("Scale     :", args.scale)

files = [f for f in listdir(args.input_dir) if isfile(join(args.input_dir, f)) and f.lower().endswith(".png")]
files.sort()

for name in files:
    in_path = join(args.input_dir, name)
    out_path = join(args.output_dir, name)

    #YCbCr conversion
    img = Image.open(in_path).convert("YCbCr")


    if args.scale != 1:
        w, h = img.size
        img = img.resize((w * args.scale, h * args.scale), Image.BICUBIC)


    y, cb, cr = img.split()


    y_np = np.array(y, dtype=np.float32)
    y_np = y_np[None, None, :, :]  # (1, 1, H, W)


    out = m.predict(y_np, verbose=0)[0, 0, :, :]  # (H_out, W_out)


    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    out_y = Image.fromarray(out)



    ow, oh = out_y.size
    cw, ch = cb.size

    if (ow, oh) != (cw, ch):

        left = (cw - ow) // 2
        top = (ch - oh) // 2
        right = left + ow
        bottom = top + oh
        cb_c = cb.crop((left, top, right, bottom))
        cr_c = cr.crop((left, top, right, bottom))
    else:
        cb_c, cr_c = cb, cr


    out_img_ycbcr = Image.merge("YCbCr", (out_y, cb_c, cr_c))
    out_img_rgb = out_img_ycbcr.convert("RGB")

    out_img_rgb.save(out_path)
    print("saved:", out_path)
