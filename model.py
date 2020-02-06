from __future__ import absolute_import, division, print_function, unicode_literals
import random
import tensorflow as tf
import pathlib
from pathlib import Path
import os

tf.compat.v1.enable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

data_root = Path("./data")
for item in data_root.iterdir():
    print("Item:", item)

all_image_paths = list(data_root.glob("*/*"))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print("Image count:", image_count)

label_names = sorted(
    item.name for item in data_root.glob("*/") if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("Label names:", label_names)
print("Label to index:", label_to_index)
print("All image labels:", all_image_labels)
print("All image paths:", all_image_paths)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.contrib.data.AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)

BATCH_SIZE = 2
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D

import numpy as np

model = Sequential()
model.add(Conv2D(64, (3, 3), activation=tf.nn.leaky_relu, input_shape=(256, 256, 3)))
model.add(Conv2D(64, (3, 3), activation=tf.nn.leaky_relu))
model.add(MaxPooling2D(3, 3))
model.add(Conv2D(128, (3, 3), activation=tf.nn.leaky_relu))
model.add(Conv2D(128, (3, 3), activation=tf.nn.leaky_relu))
model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.5))
model.add(Dense(len(label_names), activation="softmax"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

image_count = len(all_image_paths)

model.fit(ds, steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy() // BATCH_SIZE, epochs=5)

model.summary()

model.save("trash.h5")
