from __future__ import absolute_import, division, print_function, unicode_literals
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dense, Dropout, Flatten, MaxPooling2D, ZeroPadding2D, Convolution2D, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
# config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 3.0
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

BATCH_SIZE = 16
if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, 256, 256)
else:
    input_shape = (256, 256, 3)

output_shape = (256, 256)

print("Batch size:", BATCH_SIZE)

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation(tf.nn.leaky_relu))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation(tf.nn.leaky_relu))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation(tf.nn.leaky_relu))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation(tf.nn.leaky_relu))
model.add(Dropout(0.2))
model.add(Dense(len(label_names)))
model.add(Activation('softmax'))


'''
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation(tf.nn.leaky_relu))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation(tf.nn.leaky_relu))
model.add(Dropout(0.2))
model.add(Dense(len(label_names)))
model.add(Activation('softmax'))
'''

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_root, target_size=(
    output_shape), batch_size=BATCH_SIZE, class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_root, target_size=(output_shape), batch_size=BATCH_SIZE, class_mode='categorical')

filepath = "model-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0,
                             save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]

model.fit_generator(train_generator, steps_per_epoch=train_image_count // BATCH_SIZE, epochs=30,
                    validation_data=validation_generator, validation_steps=validation_image_count // BATCH_SIZE, callbacks=callbacks_list)

model.summary()

model.save("trash.h5")
