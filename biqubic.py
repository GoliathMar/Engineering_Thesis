import argparse
from pathlib import Path
import tensorflow as tf

#CLI
parser = argparse.ArgumentParser()
parser.add_argument("hr_dir", help="Katalog z obrazami wejściowymi (np. lowres_2x)")
parser.add_argument("out_dir", help="Katalog wyjściowy na przeskalowane obrazy")
parser.add_argument("--scale", type=float, default=2.0, help="Współczynnik skalowania (np. 2.0 = 2x upsampling, 0.5 = 2x downsampling)")
args = parser.parse_args()

HR_DIR  = Path(args.hr_dir).resolve()
OUT_DIR = Path(args.out_dir).resolve()

SCALE = args.scale

OUT_DIR.mkdir(parents=True, exist_ok=True)


#PNG processing
for src in sorted(HR_DIR.rglob("*.png")):
    img = tf.io.decode_png(tf.io.read_file(str(src)), channels=0)
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]


    nh = tf.cast(tf.cast(h, tf.float32) * SCALE, tf.int32)
    nw = tf.cast(tf.cast(w, tf.float32) * SCALE, tf.int32)

    img_f = tf.image.convert_image_dtype(img, tf.float32)
    resized_f = tf.image.resize(img_f, (nh, nw), method="bicubic")
    resized_u8 = tf.image.convert_image_dtype(
        tf.clip_by_value(resized_f, 0.0, 1.0),
        tf.uint8
    )

    dst = OUT_DIR / src.relative_to(HR_DIR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tf.io.write_file(str(dst), tf.io.encode_png(resized_u8))

    print("saved:", dst)
