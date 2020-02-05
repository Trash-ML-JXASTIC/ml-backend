from __future__ import absolute_import, division, print_function, unicode_literals
import random
import tensorflow as tf
import pathlib
from pathlib import Path
import os

tf.compat.v1.enable_eager_execution()

data_root = Path("./data")
for item in data_root.iterdir():
    print("Item: ", item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print("Image count: ", image_count)

label_names = sorted(
    item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image,
                       num_parallel_calls=tf.contrib.data.AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

print("All image paths: ", all_image_paths)

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


image_label_ds = ds.map(load_and_preprocess_from_path_label)

BATCH_SIZE = 32
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

mobile_net = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False)


def change_range(image, label):
    return 2*image-1, label


keras_ds = ds.map(change_range)

image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names), activation='softmax')])

logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
model.fit(ds, epochs=1, steps_per_epoch=tf.math.ceil(
    len(all_image_paths)/BATCH_SIZE).numpy())
model.save('trash.h5')
