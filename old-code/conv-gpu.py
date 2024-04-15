from numba import cuda
import numpy as np
import time

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
def manual_convolution_cuda(larger_arrays, smaller_arrays, output_arrays):
    batch_size, larger_size, _ = larger_arrays.shape
    smaller_size = smaller_arrays.shape[1]
    x, y, z = cuda.grid(3)
    
    if x < batch_size and y < (larger_size - smaller_size + 1) and z < (larger_size - smaller_size + 1):
        for k in range(smaller_arrays.shape[0]):
            value = 0.0
            for i in range(smaller_size):
                for j in range(smaller_size):
                    value += larger_arrays[x, y + i, z + j] * smaller_arrays[k, i, j]
            output_arrays[x, y, z, k] = value

# Number of larger arrays and smaller arrays
num_larger_arrays = 128
num_smaller_arrays = 32

# Size of each larger array and smaller array
larger_array_size = (num_larger_arrays, 28, 28)
smaller_array_size = (num_smaller_arrays, 3, 3)

# Generate larger arrays and smaller arrays on CPU with random values limited to two decimal places
larger_arrays = np.round(np.random.rand(*larger_array_size) * 100) / 100
smaller_arrays = np.round(np.random.rand(*smaller_array_size) * 100) / 100

# Transfer arrays to GPU memory
larger_arrays_gpu = cuda.to_device(larger_arrays)
smaller_arrays_gpu = cuda.to_device(smaller_arrays)

# Create output arrays on GPU
output_arrays_gpu = cuda.device_array((num_larger_arrays, larger_array_size[1] - smaller_array_size[1] + 1, larger_array_size[2] - smaller_array_size[2] + 1, num_smaller_arrays))
# Define grid and block dimensions
threadsperblock = (8, 8, 1)
blockspergrid_x = (num_larger_arrays + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = ((larger_array_size[1] - smaller_array_size[1] + 1) + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid_z = ((larger_array_size[2] - smaller_array_size[2] + 1) + threadsperblock[2] - 1) // threadsperblock[2]
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
print("blockspergrid ", blockspergrid, " threadsperblock ", threadsperblock)

start = time.time()
# Perform convolution on GPU for each combination of arrays
manual_convolution_cuda[blockspergrid, threadsperblock](larger_arrays_gpu, smaller_arrays_gpu, output_arrays_gpu)

# Transfer results from GPU memory to CPU for printing
results_gpu = output_arrays_gpu.copy_to_host()
print("GPU execution time: ", (time.time() - start)*1000 , " ms")
# print("GPU:\n", results_gpu)

cpu_output_arrays = np.zeros((num_larger_arrays, larger_array_size[1] - smaller_array_size[1] + 1, larger_array_size[2] - smaller_array_size[2] + 1, num_smaller_arrays))
start_cpu = time.time()

for i, larger_array in enumerate(larger_arrays):
    for j, smaller_array in enumerate(smaller_arrays):
        result = manual_convolution(larger_array, smaller_array)
        cpu_output_arrays[i, :, :, j] = result

end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print("CPU Execution Time:", cpu_time*1000," ms")

# print("CPU:\n", cpu_output_arrays)

# check if arrays are the same with some tolerance
if np.allclose(cpu_output_arrays, results_gpu): 
    print("convolution success")
else:
    print("convolution failed")
