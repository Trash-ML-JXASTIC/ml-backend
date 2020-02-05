from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib

data_root = pathlib.Path("data")
cardboard_dir = os.path.join(data_root, 'cardboard')
glass_dir = os.path.join(data_root, 'glass')

num_cardboard = len(os.listdir(cardboard_dir))
num_glass = len(os.listdir(glass_dir))
total = num_cardboard + num_glass

print('total training cardboard images:', num_cardboard)
print('total training glass images:', num_glass)
print("--")
print("Total training images:", total)

batch_size = 48
epochs = 5
IMG_HEIGHT = 512
IMG_WIDTH = 384

image_generator = ImageDataGenerator(rescale=1./255)
data_gen = image_generator.flow_from_directory(batch_size=batch_size,
                                               directory=data_root,
                                               shuffle=True,
                                               target_size=(
                                                   IMG_HEIGHT, IMG_WIDTH),
                                               class_mode='binary')

sample_training_images, _ = next(data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# plotImages(sample_training_images[:5])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    data_gen,
    steps_per_epoch=total // batch_size,
    epochs=epochs
)

model.save('trash.h5')
