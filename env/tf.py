import tensorflow
from tensorflow.python.client import device_lib

print("Tensorflow version:", tensorflow.__version__)
device_lib.list_local_devices()
