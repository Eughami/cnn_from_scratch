import tensorflow as tf
import numpy as np
import time
# Adjustable size for input data and filter kernel
input_height = 32
input_width = 32
input_channels = 1

filter_height = 3
filter_width = 3
input_channels = 3
output_channels = 32  # Number of filters

# Generate random input data and filter kernel
input_data = np.random.rand(10000, input_height, input_width, input_channels).astype(np.float32)
filter_kernel = np.random.rand(filter_height, filter_width, input_channels, output_channels).astype(np.float32)

# Create TensorFlow constants from the generated data
input_tensor = tf.constant(input_data)
filter_tensor = tf.constant(filter_kernel)

# Create a convolution layer using cuDNN
conv_layer = tf.keras.layers.Conv2D(
    filters=output_channels,
    kernel_size=(filter_height, filter_width),
    strides=(1, 1),
    padding='same',
    activation='relu',
    use_bias=True,
    input_shape=(input_height, input_width, input_channels)
)

start = time.time()
# Perform convolution using cuDNN
output_data = conv_layer(input_tensor)
print("Time taken: ", time.time() - start)
