import sys
from os import listdir, makedirs
from os.path import join, isfile, exists
import argparse

import numpy as np
from PIL import Image

#CLI args
parser = argparse.ArgumentParser()
parser.add_argument("hr_dir", help="High-res folder")
parser.add_argument("lr_dir", help="Low-res folder")
parser.add_argument("out_dir", help="Folder na patche (powstaną subfoldery input/ i label/)")
parser.add_argument("--stride", type=int, default=28, help="Krok przesuwania okna patchy")
parser.add_argument("--max-patches-per-image",type=int,default=1000,help="Maksymalna liczba patchy na jeden obraz")

args = parser.parse_args()

input_dir = join(args.out_dir, "input")
label_dir = join(args.out_dir, "label")

if not (exists(args.hr_dir) and exists(args.lr_dir)):
    print("HR/LR directories not found")
    sys.exit(1)

if not exists(input_dir):
    makedirs(input_dir)
if not exists(label_dir):
    makedirs(label_dir)

#SRCNN patch settings
patch_size = 33
label_size = 21
stride = args.stride
border = (patch_size - label_size) // 2  # 6 px
max_patches_per_image = args.max_patches_per_image

print("Config:")
print("  patch_size:", patch_size)
print("  label_size:", label_size)
print("  stride:", stride)
print("  max_patches_per_image:", max_patches_per_image)
print()


hr_files = [f for f in listdir(args.hr_dir) if isfile(join(args.hr_dir, f))]
lr_files = [f for f in listdir(args.lr_dir) if isfile(join(args.lr_dir, f))]
common = sorted(set(hr_files) & set(lr_files))

if not common:
    print("No common files between HR and LR directories")
    sys.exit(1)

total_imgs = len(common)
print("Found", total_imgs, "matching HR/LR images")
print()

count = 1
total_patches = 0

for idx, name in enumerate(common, start=1):
    hr_path = join(args.hr_dir, name)
    lr_path = join(args.lr_dir, name)

    print(f"[{idx}/{total_imgs}] Processing image: {name} (total patches so far: {total_patches})")

    hr_img = Image.open(hr_path).convert("YCbCr")
    lr_img = Image.open(lr_path).convert("YCbCr")

    lr_up = lr_img.resize(hr_img.size, resample=Image.BICUBIC)

    hr_y = np.array(hr_img)[:, :, 0].astype(np.float32)
    lr_y = np.array(lr_up)[:, :, 0].astype(np.float32)

    h, w = hr_y.shape
    patches_for_this_image = 0

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            sub_input = lr_y[i:i + patch_size, j:j + patch_size]
            sub_label = hr_y[
                i + border:i + border + label_size,
                j + border:j + border + label_size
            ]

            sub_input_img = Image.fromarray(
                np.clip(sub_input, 0, 255).astype(np.uint8)
            )
            sub_label_img = Image.fromarray(
                np.clip(sub_label, 0, 255).astype(np.uint8)
            )

            fname = "%06d.png" % count
            sub_input_img.save(join(input_dir, fname))
            sub_label_img.save(join(label_dir, fname))

            count += 1
            patches_for_this_image += 1
            total_patches += 1

            if patches_for_this_image % 200 == 0:
                print(f"   {patches_for_this_image} patches from this image...")

            if patches_for_this_image >= max_patches_per_image:
                print(f"   Reached max_patches_per_image={max_patches_per_image} for this image, moving on.")
                break

        if patches_for_this_image >= max_patches_per_image:
            break

print()
print("Done.")
print("Total patches generated:", total_patches)
print("Input patches dir:", input_dir)
print("Label patches dir:", label_dir)
