from __future__ import absolute_import, division, print_function, unicode_literals
from pathlib import Path
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pathlib
import random
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

train_data_root = Path("./data/train")
for item in train_data_root.iterdir():
    print("Train item:", item)

validation_data_root = Path("./data/validation")
for item in validation_data_root.iterdir():
    print("Validation item:", item)

all_train_image_paths = list(train_data_root.glob("*/*"))
all_train_image_paths = [str(path) for path in all_train_image_paths]

all_validation_image_paths = list(validation_data_root.glob("*/*"))
all_validation_image_paths = [str(path) for path in all_validation_image_paths]

train_image_count = len(all_train_image_paths)
print("Train image count:", train_image_count)

validation_image_count = len(all_validation_image_paths)
print("Validation image count:", validation_image_count)

label_names = sorted(
    item.name for item in train_data_root.glob("*/") if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))

print("Label names:", label_names)
print("Label to index:", label_to_index)
print("All train image paths:", all_train_image_paths)
print("All validation image paths:", all_validation_image_paths)

BATCH_SIZE = 2

print("Batch size:", BATCH_SIZE)

model = Sequential()

model.add(Conv2D(64, 3, strides=3, input_shape=(256, 256, 3)))
model.add(Activation(tf.nn.leaky_relu))

model.add(Conv2D(32, 3, strides=2))
model.add(Activation(tf.nn.leaky_relu))
model.add(BatchNormalization())

model.add(Conv2D(32, 3, strides=2))
model.add(Activation(tf.nn.leaky_relu))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(Conv2D(16, 3, strides=1))
model.add(Activation(tf.nn.leaky_relu))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(Conv2D(1, 3, strides=1))
model.add(Activation(tf.nn.leaky_relu))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation(tf.nn.leaky_relu))

model.add(Dense(len(label_names)))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_root, target_size=(
    256, 256), batch_size=BATCH_SIZE, class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_root, target_size=(256, 256), batch_size=BATCH_SIZE, class_mode='categorical')

model.fit_generator(train_generator, steps_per_epoch=train_image_count // BATCH_SIZE, epochs=600,
                    validation_data=validation_generator, validation_steps=validation_image_count // BATCH_SIZE)

model.summary()

model.save("trash.h5")
