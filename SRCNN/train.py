import sys
from os import listdir
from os.path import isfile, join, exists

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save", help="Ścieżka do zapisania checkpointów")
parser.add_argument("data", help="Folder z danymi do trenowania")
args = parser.parse_args()

input_dir = join(args.data, "input")
label_dir = join(args.data, "label")

files_input = [f for f in listdir(input_dir) if f.endswith(".png")]
files_label = [f for f in listdir(label_dir) if f.endswith(".png")]

common = sorted(set(files_input) & set(files_label))
print("Input files   :", len(files_input))
print("Label files   :", len(files_label))
print("Common files  :", len(common))


if not (exists(input_dir) and exists(label_dir)):
    print("Input/label directories not found")
    sys.exit(1)

import numpy as np
import tensorflow as tf
import imageio.v2 as imageio

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

from sklearn.model_selection import train_test_split

inputs = Input(shape=(1, 33, 33))

x = Conv2D(64, (9, 9), activation='relu', kernel_initializer='he_normal', data_format='channels_first', padding='valid')(inputs)

x = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_normal', data_format='channels_first', padding='valid')(x)

x = Conv2D(1, (5, 5),kernel_initializer='he_normal', data_format='channels_first', padding='valid')(x)

m = Model(inputs=inputs, outputs=x)
m.compile(optimizer=Adam(learning_rate=0.001), loss='mse')



X_list = []
y_list = []

for i, f in enumerate(common, start=1):
    # INPUT
    img_in = imageio.imread(join(input_dir, f))
    if img_in.ndim == 2:
        img_in = img_in[:, :, None]
    X_list.append(img_in[None, :, :, 0])

    # LABEL
    img_lab = imageio.imread(join(label_dir, f))
    if img_lab.ndim == 2:
        img_lab = img_lab[:, :, None]
    y_list.append(img_lab[None, :, :, 0])

    if i % 10000 == 0:
        print(f"  loaded {i}/{len(common)} patch pairs")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print("Finished loading patches.")
print("  X shape:", X.shape)
print("  y shape:", y.shape)
print("  number of pairs:", X.shape[0])

epochs_per_cycle = 5
max_epochs = 100
max_cycles = max_epochs // epochs_per_cycle  # 100 / 5 = 20

count = 1
for _ in range(max_cycles):
    m.fit(X, y, batch_size=128, epochs=epochs_per_cycle)
    if args.save:
        current_epochs = count * epochs_per_cycle  # 5, 10, 15, ...
        print("Saving model", current_epochs)
        m.save(join(args.save, f"model_{current_epochs}.h5"))
    count += 1

