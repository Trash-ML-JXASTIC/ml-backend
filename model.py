import os
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from pathlib import Path

from tensorflow.python.keras.callbacks import ModelCheckpoint

train_data_root = Path("./data/train")
for file in train_data_root.glob("*/._*.jpg"):
    os.remove(file)
for item in train_data_root.iterdir():
    print("Train item:", item)

validation_data_root = Path("./data/validation")
for file in validation_data_root.glob("*/._*.jpg"):
    os.remove(file)
for item in validation_data_root.iterdir():
    print("Validation item:", item)

all_train_image_paths = list(train_data_root.glob("*/*.jpg"))
all_train_image_paths = [str(path) for path in all_train_image_paths]

all_validation_image_paths = list(validation_data_root.glob("*/*.jpg"))
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

BATCH_SIZE = 16
if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, 256, 256)
else:
    input_shape = (256, 256, 3)
output_shape = (256, 256)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_data_root,
  image_size=(output_shape),
  batch_size=BATCH_SIZE)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  validation_data_root,
  image_size=(output_shape),
  batch_size=BATCH_SIZE)

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(input_shape)),
  layers.Conv2D(16, 3, padding='same', strides=2, activation=tf.nn.leaky_relu),
  layers.Conv2D(32, 3, padding='same', strides=2, activation=tf.nn.leaky_relu),
  layers.Conv2D(64, 3, padding='same', strides=2, activation=tf.nn.leaky_relu),
  layers.Conv2D(3, 3, padding='same'),
  layers.GlobalAvgPool2D(),
  layers.Activation(activation="softmax")
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
            optimizer="rmsprop", metrics=["accuracy"])

filepath = "model-improvement-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0,
                             save_best_only=False, save_weights_only=False, mode='auto', save_freq=1)
callbacks_list = [checkpoint]

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=30,
  callbacks=callbacks_list
)

model.summary()

model.save("trash.h5")
