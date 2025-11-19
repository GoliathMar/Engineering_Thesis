#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

#temporary hardcode, will change one day
BASE      = Path(__file__).parent.resolve()
HR_DIR    = BASE / "source"
BIC_DIR   = BASE / "output" / "bicubic"
SRCNN_DIR = BASE / "output" / "srcnn"
SRGAN_DIR = BASE / "output" / "srgan"

SHAVE = 6


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def crop_border(img: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return img
    return img[n:-n, n:-n, ...]


def center_crop_to_smallest(*imgs):

    h = min(i.shape[0] for i in imgs)
    w = min(i.shape[1] for i in imgs)

    def cc(x):
        top = (x.shape[0] - h) // 2
        left = (x.shape[1] - w) // 2
        return x[top: top + h, left: left + w, :]

    return [cc(i) for i in imgs]


def rgb_to_y(img_rgb: np.ndarray) -> np.ndarray:

    return rgb2ycbcr(img_rgb)[..., 0]


stems = sorted({p.stem for p in HR_DIR.glob("*.png")})

psnr_bicubic = []
psnr_srcnn   = []
psnr_srgan   = []

ssim_bicubic = []
ssim_srcnn   = []
ssim_srgan   = []

for stem in stems:
    hr_path    = HR_DIR    / f"{stem}.png"
    bic_path   = BIC_DIR   / f"{stem}.png"
    srcnn_path = SRCNN_DIR / f"{stem}.png"
    srgan_path = SRGAN_DIR / f"{stem}.png"

    if not (hr_path.is_file() and bic_path.is_file() and
            srcnn_path.is_file() and srgan_path.is_file()):
        print(f"[WARN] File missing for {stem}, skipping.")
        continue

    hr    = load_rgb(hr_path)
    bic   = load_rgb(bic_path)
    srcnn = load_rgb(srcnn_path)
    srgan = load_rgb(srgan_path)

    #shave 6 px
    hr    = crop_border(hr, SHAVE)
    bic   = crop_border(bic, SHAVE)
    srcnn = crop_border(srcnn, SHAVE)
    srgan = crop_border(srgan, SHAVE)


    hr, bic, srcnn, srgan = center_crop_to_smallest(hr, bic, srcnn, srgan)


    hr_y    = rgb_to_y(hr)
    bic_y   = rgb_to_y(bic)
    srcnn_y = rgb_to_y(srcnn)
    srgan_y = rgb_to_y(srgan)

    #PSNR
    psnr_b = peak_signal_noise_ratio(hr_y, bic_y,   data_range=255)
    psnr_s = peak_signal_noise_ratio(hr_y, srcnn_y, data_range=255)
    psnr_g = peak_signal_noise_ratio(hr_y, srgan_y, data_range=255)

    #SSIM
    ssim_b = structural_similarity(hr_y, bic_y,   data_range=255)
    ssim_s = structural_similarity(hr_y, srcnn_y, data_range=255)
    ssim_g = structural_similarity(hr_y, srgan_y, data_range=255)

    psnr_bicubic.append(psnr_b)
    psnr_srcnn.append(psnr_s)
    psnr_srgan.append(psnr_g)

    ssim_bicubic.append(ssim_b)
    ssim_srcnn.append(ssim_s)
    ssim_srgan.append(ssim_g)

    print(
        f"{stem}: "
        f"PSNR_bicubic={psnr_b:.3f} dB  |  PSNR_SRCNN={psnr_s:.3f} dB  |  PSNR_SRGAN={psnr_g:.3f} dB  ||  "
        f"SSIM_bicubic={ssim_b:.4f}  |  SSIM_SRCNN={ssim_s:.4f}  |  SSIM_SRGAN={ssim_g:.4f}"
    )

if psnr_bicubic:
    print("\n== Mean (channel Y, shave 6px) ==")
    print(f"bicubic: PSNR={np.mean(psnr_bicubic):.3f} dB, SSIM={np.mean(ssim_bicubic):.4f}")
    print(f"SRCNN  : PSNR={np.mean(psnr_srcnn):.3f} dB, SSIM={np.mean(ssim_srcnn):.4f}")
    print(f"SRGAN  : PSNR={np.mean(psnr_srgan):.3f} dB, SSIM={np.mean(ssim_srgan):.4f}")
else:
    print("Insufficient data to calculate mean.")
