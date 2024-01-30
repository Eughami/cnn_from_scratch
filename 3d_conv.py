from numba import cuda
import numpy as np
import tensorflow as tf
import time 
import keras
from scipy.signal import convolve2d

np.set_printoptions(precision=2, suppress=True, linewidth=np.inf)

def manual_convolution(input_array, kernel):
    input_shape = input_array.shape
    kernel_shape = kernel.shape
    output_shape = (input_shape[0] - kernel_shape[0] + 1, input_shape[1] - kernel_shape[1] + 1)
    output_array = np.zeros(output_shape)

    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            output_array[i, j] = np.sum(input_array[i:i+kernel_shape[0], j:j+kernel_shape[1]] * kernel)

    return output_array

# Manual convolution using CUDA with @cuda.jit
@cuda.jit
def manual_convolution_3d_cuda(larger_arrays, smaller_arrays, output_arrays):
    # larger and small arrays should have the same depth
    batch_size, larger_size_depth, larger_size_height, larger_size_width = larger_arrays.shape
    channels, smaller_size_depth, smaller_size_height, smaller_size_width = smaller_arrays.shape

    output_height = larger_size_height - smaller_size_height + 1
    output_width = larger_size_width - smaller_size_width + 1
    x, y, z = cuda.grid(3)

    if x < batch_size and y < output_height and z < output_width:
        for k in range(channels):
            value = 0.0
            for l in range(smaller_size_depth):
                for j in range(smaller_size_height):
                    for i in range(smaller_size_width):
                        print("(",x,y,z,")","convolving",larger_arrays[x, l, y + i, z + j], smaller_arrays[k, l, i, j])
                        re = larger_arrays[x, l, y + i, z + j] * smaller_arrays[k, l, i, j]
                        print("(",x,y,z,")","Adding",re, " to ", value)
                        value += re
            output_arrays[x, k, y, z] = value


@cuda.jit
def manual_maxpooling_cuda(input_arrays, output_arrays, window_size, stride):
    total_size, channels, _, _ = input_arrays.shape
    x, y, z = cuda.grid(3)
    
    if x < total_size and y < output_arrays.shape[2] and z < output_arrays.shape[3]:
        for k in range(channels):
            value = 0.0
            for i in range(window_size):
                for j in range(window_size):
                    value = max(value, input_arrays[x // channels, k, (stride*y + i), (stride*z + j)])
            output_arrays[x // channels, k, y, z] = value

# Number of larger arrays and smaller arrays
num_larger_arrays = 1
num_smaller_arrays = 1

# Size of each larger array and smaller array
larger_array_size = (num_larger_arrays, 1, 5, 5)
smaller_array_size = (num_smaller_arrays,1,  3, 3)

# Generate larger arrays and smaller arrays on CPU with random values limited to two decimal places
larger_arrays = np.round(np.random.rand(*larger_array_size) * 100) / 100
smaller_arrays = np.round(np.random.rand(*smaller_array_size) * 100) / 100

#output size
output_y = larger_array_size[2] - smaller_array_size[2] + 1
output_x = larger_array_size[3] - smaller_array_size[3] + 1

print("inputs\n",larger_arrays)
print("\nfilters\n",smaller_arrays)

# (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
# processed_input_array = [array[np.newaxis, :, :] if len(array.shape) == 2 else array for array in train_images]
# larger_arrays = np.array(processed_input_array)

# Define grid and block dimensions
threadsperblock = (8, 8, 8)
blockspergrid_x = (num_larger_arrays + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (output_y + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid_z = (output_x + threadsperblock[2] - 1) // threadsperblock[2]
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
print("blockspergrid",blockspergrid)

# Transfer arrays to GPU memory
start = time.time()
larger_arrays_gpu = cuda.to_device(larger_arrays)
smaller_arrays_gpu = cuda.to_device(smaller_arrays)

# Create output arrays on GPU
output_arrays_gpu = cuda.device_array((num_larger_arrays, num_smaller_arrays, output_y, output_x))
print("GPU transfer time",time.time()-start)
# Perform convolution on GPU for each combination of arrays
start = time.time()
manual_convolution_3d_cuda[blockspergrid, threadsperblock](larger_arrays_gpu, smaller_arrays_gpu, output_arrays_gpu)

# Transfer results from GPU memory to CPU for printing
results_gpu = output_arrays_gpu.copy_to_host()
print("GPU time",time.time()-start)
print("results_gpu",results_gpu.shape)
print("results_gpu",results_gpu)
# start = time.time()
# #Perform max pooling on GPU for each combination of arrays
# window_size = 2
# stride = 2
# output_y = output_y//window_size
# output_x = output_x//window_size
# output_arrays_gpu = cuda.device_array((num_larger_arrays, num_smaller_arrays, output_y, output_x))
# # blockspergrid_y = blockspergrid_y//window_size
# # blockspergrid_z = blockspergrid_z//window_size
# blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

# print("blockspergrid : ",blockspergrid)

# results_gpu = cuda.to_device(results_gpu)
# manual_maxpooling_cuda[blockspergrid, threadsperblock](results_gpu, output_arrays_gpu, window_size, stride)
# results_gpu = output_arrays_gpu.copy_to_host()
# print("results_gpu",results_gpu.shape)
# print("GPU time",time.time()-start)
# print("results_gpu",results_gpu[0])

# for i in range(results_gpu.shape[0]):
#     for j in range(results_gpu.shape[1]):
#         print(results_gpu[i, j, :, :])

def convolution(input_tensor, filter_tensor, stride=1):
    # Get dimensions
    N, C_in, H_in, W_in = input_tensor.shape
    _, C_out, H_filter, W_filter = filter_tensor.shape

    # Calculate output size
    H_out = ((H_in - H_filter) // stride) + 1
    W_out = ((W_in - W_filter) // stride) + 1

    # Initialize output tensor
    output_tensor = np.zeros((N, C_out, H_out, W_out))

    # Perform convolution
    for n in range(N):
        for c_out in range(C_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start, h_end = h_out * stride, h_out * stride + H_filter
                    w_start, w_end = w_out * stride, w_out * stride + W_filter

                    output_tensor[n, c_out, h_out, w_out] = np.sum(
                        input_tensor[n, :, h_start:h_end, w_start:w_end] * filter_tensor[:, c_out, :, :]
                    )

    return output_tensor



output_arrays_cpu = np.zeros_like(results_gpu)
# output_arrays_cpu = convolution(larger_arrays, smaller_arrays)
# Perform convolution on CPU
start = time.time()
for i, larger_array in enumerate(larger_arrays):
    for j, smaller_array in enumerate(smaller_arrays):
        result = np.zeros((output_y, output_x))
        for x in range(smaller_array_size[1]):
          result += manual_convolution(larger_array[x, :, :], smaller_array[x, :, :])
        output_arrays_cpu[i, j, :, :] = result
# for i in range(output_arrays_cpu.shape[0]):
    # for j in range(output_arrays_cpu.shape[1]):
        # output_arrays_cpu[i,j,:,:] = np.sum(larger_arrays[i,:,:,:] * smaller_arrays[j,:,:,:], axis=(1,2))

# output_arrays_cpu[0, 0, :, :] = convolve2d(larger_arrays[0,0, :, :], smaller_arrays[0,0, :, :], mode='valid')
# print("CPU time",time.time()-start)
print("results_cpu",output_arrays_cpu.shape)
print("results_cpu",output_arrays_cpu)
# for i in range(output_arrays_cpu.shape[0]):
    # for j in range(output_arrays_cpu.shape[1]):
        # print(output_arrays_cpu[i, j, :, :])

# Assuming multiple input and filter arrays
# start = time.time()
# input_data = tf.convert_to_tensor(larger_arrays, dtype=tf.float32)
# # print("input_data",input_data.shape)
# filter_data = tf.convert_to_tensor(smaller_arrays, dtype=tf.float32)
# # print("filter_data",filter_data.shape)

# with tf.profiler.experimental.Profile('./logs'):
# #     # Create a Conv2D layer with similar parameters as tf.nn.convolution
#     conv_layer = tf.keras.layers.Conv2D(filters=num_smaller_arrays, kernel_size=(smaller_array_size[2], smaller_array_size[3]), strides=(1, 1), padding='valid', data_format='channels_first')
#     # Apply the Conv2D layer to the input data
#     output_data = conv_layer(input_data)
#     print("tensorflow time",time.time()-start)
#     output_data = output_data.numpy()
#     print(output_data.shape)  # Print the shape of the output
# for i in range(output_data.shape[0]):
#     for j in range(output_data.shape[1]):
#         print(output_data[i, j, :, :])

# Perform 3D convolution with multiple inputs and filters
# output_tensor = tf.nn.convolution(input_data, filter_data, padding='VALID', data_format= 'NCHW')
# print("tensorflow",output_tensor.shape)  # Print the shape of the output
# output_data = output_tensor.numpy()
# print(output_data.shape)  # Print the shape of the output
# for i in range(output_data.shape[0]):
    # for j in range(output_data.shape[1]):
        # print(output_data[i, j, :, :])
if(np.allclose(results_gpu, output_arrays_cpu)):
    print("GPU and CPU results are same")
# if(np.allclose(results_gpu, output_data)):
#     print("GPU and tensorflow results are same")
#! memory
#! my implementation is better with higher resolution
#! tensorflow is better with number of inputs and filters
#! speed
#! mixed result 
#! do some comparison with different input and filter sizes i.e 32x32x32 and 3x3x3
#! after each layer filter depth must be same as number of filters in previous layer
#! do mine vs cpu vs cudnn (this can be done later to show supervisor my shit is actually working)