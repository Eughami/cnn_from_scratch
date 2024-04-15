from numba import cuda
import numpy as np

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


# Manual maxpooling using CUDA with @cuda.jit
@cuda.jit
def manual_maxpooling_cuda(input_arrays, output_arrays, window_size, stride):
    batch_size, input_size, _, _ = input_arrays.shape
    x, y, z = cuda.grid(3)
    
    if x < batch_size and y < output_arrays.shape[2] and z < output_arrays.shape[1]:
        for k in range(input_arrays.shape[3]):
            value = 0.0
            # print(x,y,z)
            for i in range(window_size):
                for j in range(window_size):
                    value = max(value, input_arrays[x, (stride*y + i ), (stride*z + j),k])
            output_arrays[x, y, z, k] = value


# Create events for measuring time
start_kernel = cuda.event()
end_kernel = cuda.event()

# Number of larger arrays and smaller arrays
num_larger_arrays = 2
num_smaller_arrays = 5

# Size of each larger array and smaller array
larger_array_size = (num_larger_arrays, 28, 28)
smaller_array_size = (num_smaller_arrays, 3, 3)

# Generate larger arrays and smaller arrays on CPU with random values limited to two decimal places
larger_arrays = np.round(np.random.rand(*larger_array_size) * 100) / 100
smaller_arrays = np.round(np.random.rand(*smaller_array_size) * 100) / 100

# Transfer arrays to GPU memory
larger_arrays_gpu = cuda.to_device(larger_arrays)
smaller_arrays_gpu = cuda.to_device(smaller_arrays)

#output size
output_y = larger_array_size[1] - smaller_array_size[1] + 1
output_x = larger_array_size[2] - smaller_array_size[2] + 1

# Create output arrays on GPU
output_arrays_gpu = cuda.device_array((num_larger_arrays, output_y, output_x, num_smaller_arrays))

# Define grid and block dimensions
threadsperblock = (8, 8, 1)
blockspergrid_x = (num_larger_arrays + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (output_y + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid_z = (output_x + threadsperblock[2] - 1) // threadsperblock[2]
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
print("blockspergrid : ",blockspergrid)


# Perform convolution on GPU for each combination of arrays
start_kernel.record()
manual_convolution_cuda[blockspergrid, threadsperblock](larger_arrays_gpu, smaller_arrays_gpu, output_arrays_gpu)
# Transfer results from GPU memory to CPU for printing
results_gpu = output_arrays_gpu.copy_to_host()
end_kernel.record()
end_kernel.synchronize()  # Wait for completion of kernel execution
kernel_time = cuda.event_elapsed_time(start_kernel, end_kernel)
print(f"Kernel execution time: {kernel_time} ms")


print("result conv")
print(results_gpu.shape)
# for i in range(results_gpu.shape[0]):
#     for j in range(results_gpu.shape[3]):
#         print(results_gpu[i, :, :, j])
#Perform max pooling on GPU for each combination of arrays
window_size = 2
stride = 2
output_y = output_y//window_size
output_x = output_x//window_size
output_arrays_gpu = cuda.device_array((num_larger_arrays, output_y, output_x, num_smaller_arrays))
blockspergrid_y = blockspergrid_y//window_size
blockspergrid_z = blockspergrid_z//window_size
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

print("blockspergrid : ",blockspergrid)

start_kernel.record()
results_gpu = cuda.to_device(results_gpu)
manual_maxpooling_cuda[blockspergrid, threadsperblock](results_gpu, output_arrays_gpu, window_size, stride)
results_gpu = output_arrays_gpu.copy_to_host()
end_kernel.record()
end_kernel.synchronize()  # Wait for completion of kernel execution
kernel_time = cuda.event_elapsed_time(start_kernel, end_kernel)
print(f"Kernel execution time: {kernel_time} ms")
print("result maxPooling")
print(results_gpu.shape)
# print("result maxPooling")
# for i in range(results_gpu.shape[0]):
#     for j in range(results_gpu.shape[3]):
#         print(results_gpu[i, :, :, j])
# ! this is not supported yet. it is for 3d convolution
# # Make another convolution layer 
# output_y = output_y - smaller_array_size[1] + 1
# output_x = output_x - smaller_array_size[2] + 1
# smaller_array_size = (num_smaller_arrays * 2, 3, 3)
# smaller_arrays = np.round(np.random.rand(*smaller_array_size) * 100) / 100
# output_arrays_gpu = cuda.device_array((num_larger_arrays, output_y, output_x, num_smaller_arrays))
# blockspergrid_y = (output_y + threadsperblock[1] - 1) // threadsperblock[1]
# blockspergrid_z = (output_x + threadsperblock[2] - 1) // threadsperblock[2]
# blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

# print("blockspergrid : ",blockspergrid)
# manual_convolution_cuda[blockspergrid, threadsperblock](results_gpu, smaller_arrays_gpu, output_arrays_gpu)
# results_gpu = output_arrays_gpu.copy_to_host()
# print("result conv")
# for i in range(results_gpu.shape[0]):
#     for j in range(results_gpu.shape[3]):
#         print(results_gpu[i, :, :, j])


