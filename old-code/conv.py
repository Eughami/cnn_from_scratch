from numba import cuda
import numpy as np

@cuda.jit
def manual_convolution_3d_cuda(larger_arrays, smaller_arrays, output_arrays):
    batch_size, larger_depth, larger_height, larger_width = larger_arrays.shape
    smaller_depth, smaller_height, smaller_width = smaller_arrays.shape[1:]

    x, y, z, w = cuda.grid(4)

    if x < batch_size and y < (larger_depth - smaller_depth + 1) and z < (larger_height - smaller_height + 1) and w < (larger_width - smaller_width + 1):
        for k in range(smaller_arrays.shape[0]):
            value = 0.0
            for i in range(smaller_depth):
                for j in range(smaller_height):
                    for l in range(smaller_width):
                        value += larger_arrays[x, y + i, z + j, w + l] * smaller_arrays[k, i, j, l]
            output_arrays[x, y, z, w, k] = value

# Number of larger arrays and smaller arrays
num_larger_arrays = 1
num_smaller_arrays = 1

# Size of each larger array and smaller array (3D)
larger_array_size = (num_larger_arrays, 3, 16, 16)
smaller_array_size = (num_smaller_arrays, 3, 3, 3)

# Generating larger arrays and smaller arrays on CPU with random values limited to two decimal places
larger_arrays_3d = np.round(np.random.rand(*larger_array_size) * 100) / 100
smaller_arrays_3d = np.round(np.random.rand(*smaller_array_size) * 100) / 100

# Transfer arrays to GPU memory
larger_arrays_3d_gpu = cuda.to_device(larger_arrays_3d)
smaller_arrays_3d_gpu = cuda.to_device(smaller_arrays_3d)

# Output size
output_depth = larger_array_size[1] - smaller_array_size[1] + 1
output_height = larger_array_size[2] - smaller_array_size[2] + 1
output_width = larger_array_size[3] - smaller_array_size[3] + 1

# Create output arrays on GPU
output_arrays_3d_gpu = cuda.device_array((num_larger_arrays, output_depth, output_height, output_width, num_smaller_arrays))

# Define grid and block dimensions
threadsperblock = (8, 8, 8)
blockspergrid_x = (num_larger_arrays + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (output_depth + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid_z = (output_height + threadsperblock[2] - 1) // threadsperblock[2]
blockspergrid_w = (output_width + threadsperblock[3] - 1) // threadsperblock[3]
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z, blockspergrid_w)

# Perform convolution on GPU for each combination of arrays
manual_convolution_3d_cuda[blockspergrid, threadsperblock](larger_arrays_3d_gpu, smaller_arrays_3d_gpu, output_arrays_3d_gpu)

# Transfer results from GPU memory to CPU for printing
results_3d_gpu = output_arrays_3d_gpu.copy_to_host()

print("larger_arrays_3d\n",larger_arrays_3d)
print("smaller_arrays_3d\n",smaller_arrays_3d)
print("results_3d_gpu\n",results_3d_gpu)