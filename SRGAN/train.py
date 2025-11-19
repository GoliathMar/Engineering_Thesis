#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses, Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras import mixed_precision


from model import build_generator, build_discriminator

mixed_precision.set_global_policy("mixed_float16")


#CLI Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--hr_dir", type=str, default="source", help="Katalog z HR")
parser.add_argument("--out_dir", type=str, default="SRGAN_checkpoints", help="Gdzie zapisać wagi")
parser.add_argument("--pretrain_epochs", type=int, default=5)
parser.add_argument("--gan_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--scale", type=int, default=2)
parser.add_argument("--patch_size", type=int, default=96)
args = parser.parse_args()

BASE_DIR = Path(__file__).parent.resolve()
HR_DIR = (BASE_DIR / args.hr_dir).resolve()
OUT_DIR = (BASE_DIR / args.out_dir).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE = args.scale
PATCH_SIZE = args.patch_size
LR_SIZE = PATCH_SIZE // SCALE

#Models
generator = build_generator(scale=SCALE)
discriminator = build_discriminator()

#VGG
vgg = VGG19(include_top=False, weights="imagenet")
vgg.trainable = False
content_model = Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)


def vgg_features(x):

    #Scaling from [-1, 1] to [0, 255]
    x = (x + 1.0) * 127.5
    x = vgg_preprocess(x)
    return content_model(x)


#Dataset Pipeline
hr_files = sorted([str(p) for p in HR_DIR.rglob("*.png")])
if not hr_files:
    raise SystemExit(f"Brak plików PNG w {HR_DIR}")


def load_and_process(path):

    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0, 1]


    hr = tf.image.random_crop(img, size=[PATCH_SIZE, PATCH_SIZE, 3])


    lr = tf.image.resize(hr, [LR_SIZE, LR_SIZE], method="bicubic")


    hr = (hr * 2.0) - 1.0
    lr = (lr * 2.0) - 1.0

    return lr, hr


ds = tf.data.Dataset.from_tensor_slices(hr_files)
ds = ds.shuffle(len(hr_files))
ds = ds.map(load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)


bce = losses.BinaryCrossentropy(from_logits=False)
mse = losses.MeanSquaredError()

g_opt = optimizers.Adam(learning_rate=1e-4, beta_1=0.9)
d_opt = optimizers.Adam(learning_rate=1e-4, beta_1=0.9)

LAMBDA_CONTENT = 1.0
LAMBDA_ADV = 1e-3


#Training Steps
@tf.function
def pretrain_step(lr, hr):
    with tf.GradientTape() as tape:
        sr = generator(lr, training=True)
        loss = mse(hr, sr)
    grads = tape.gradient(loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(grads, generator.trainable_variables))
    return loss


@tf.function
def gan_step(lr, hr):

    with tf.GradientTape() as tape_d:
        sr = generator(lr, training=True)

        real_pred = discriminator(hr, training=True)
        fake_pred = discriminator(sr, training=True)

        d_loss_real = bce(tf.ones_like(real_pred), real_pred)
        d_loss_fake = bce(tf.zeros_like(fake_pred), fake_pred)
        d_loss = d_loss_real + d_loss_fake

    grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(grads_d, discriminator.trainable_variables))


    with tf.GradientTape() as tape_g:
        sr = generator(lr, training=True)
        fake_pred = discriminator(sr, training=True)

        # Content loss (VGG)
        img_loss = mse(vgg_features(hr), vgg_features(sr))
        # Adversarial loss
        adv_loss = bce(tf.ones_like(fake_pred), fake_pred)

        g_loss = LAMBDA_CONTENT * img_loss + LAMBDA_ADV * adv_loss

    grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(grads_g, generator.trainable_variables))

    return d_loss, g_loss, img_loss, adv_loss


#Main Loop
if __name__ == "__main__":
    #Pretraining
    if args.pretrain_epochs > 0:
        print(f"\n=== Pre-training (MSE), {args.pretrain_epochs} epochs ===")
        for epoch in range(1, args.pretrain_epochs + 1):
            losses_pt = []
            for lr, hr in ds:
                l = pretrain_step(lr, hr)
                losses_pt.append(l)
            print(f"Epoch {epoch}/{args.pretrain_epochs} - MSE Loss: {np.mean(losses_pt):.4f}")

        generator.save_weights(str(OUT_DIR / "srresnet_pretrain.weights.h5"))

    #GAN Training
    if args.gan_epochs > 0:
        print(f"\n=== SRGAN Training, {args.gan_epochs} epochs ===")
        for epoch in range(1, args.gan_epochs + 1):
            d_stats, g_stats = [], []
            for lr, hr in ds:
                d_l, g_l, c_l, a_l = gan_step(lr, hr)
                d_stats.append(d_l)
                g_stats.append(g_l)

            print(f"Epoch {epoch}/{args.gan_epochs} - D_loss: {np.mean(d_stats):.4f} | G_loss: {np.mean(g_stats):.4f}")

            #Weights save
            generator.save_weights(str(OUT_DIR / f"srgan_epoch_{epoch}.weights.h5"))