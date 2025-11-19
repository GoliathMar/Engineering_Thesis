# model.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def residual_block(x, filters=64):

    skip = x
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([skip, x])

def upsample_block(x, filters, scale=2):

    x = layers.Conv2D(filters * (scale ** 2), 3, padding="same")(x)
    x = layers.Lambda(lambda t: tf.nn.depth_to_space(t, scale))(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    return x

def build_generator(scale=2, num_res_blocks=16):
    inp = layers.Input(shape=(None, None, 3))

    #Convolution
    x = layers.Conv2D(64, 9, padding="same")(inp)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    skip = x

    #Residual Blocks
    for _ in range(num_res_blocks):
        x = residual_block(x, 64)

    #Convolution + BN + Skip connection
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([skip, x])

    #Upsampling
    #For scale 2.0
    steps = int(np.log2(scale))
    for _ in range(steps):
        x = upsample_block(x, 64, scale=2)


    out = layers.Conv2D(3, 9, padding="same", activation="tanh")(x)

    return Model(inp, out, name="generator")

def disc_block(x, filters, strides):
    x = layers.Conv2D(filters, 3, strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def build_discriminator():
    inp = layers.Input(shape=(None, None, 3))

    x = layers.Conv2D(64, 3, strides=1, padding="same")(inp)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = disc_block(x, 64, 2)
    x = disc_block(x, 128, 1)
    x = disc_block(x, 128, 2)
    x = disc_block(x, 256, 1)
    x = disc_block(x, 256, 2)
    x = disc_block(x, 512, 1)
    x = disc_block(x, 512, 2)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return Model(inp, out, name="discriminator")