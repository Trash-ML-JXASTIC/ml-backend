import tensorflow as tf
print(1, tf.config.list_physical_devices('GPU'))
print(2, tf.test.is_built_with_cuda())
