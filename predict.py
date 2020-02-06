import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

tf.compat.v1.enable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(512, 384))
    # (height, width, channels)
    img_tensor = image.img_to_array(img)
    # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # imshow expects values in the range [0, 1]
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


# load model
model = load_model("trash.h5")

# image path
img_path = 'test/glass3.jpg'

# load a single image
new_image = load_image(img_path)

# check prediction
pred = model.predict(new_image)

print("Raw prediction data: ", pred)

pred_class = model.predict_classes(new_image)

print("Raw prediction class data: ", pred_class)

print("--")

labels_index = [
    "cardboard",
    "paper",
    "plastic",
    "glass",
    "metal",
    "trash"
]

print("Result:")

print("File: ", img_path)

print("Predicted class: ", labels_index[pred_class[0]])

print("Possibilities:")

i = 0
for label in labels_index:
    print("\t%s ==> %f" % (label, pred[0][i]))
    i = i + 1
