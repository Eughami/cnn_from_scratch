import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["OMP_NUM_THREADS"] = "1"  # Limit numpy to use only one CPU core
from convolution_layer import Convolution
from fully_connected import Fully_Connected
from utils import create_batches, cross_entropy_loss, cross_entropy_loss_gradient 
import tensorflow as tf
# tf.random.set_seed(42)
import sys
import numpy as np
# from scipy.signal import convolve2d
from tensorflow import keras
from keras.utils import to_categorical
from keras import datasets
from sklearn.metrics import accuracy_score
import time 
from numba import cuda
import numpy as np
# Set a seed for reproducibility
# np.random.seed(42)
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=np.inf,threshold=np.inf)

# Manual convolution using CUDA with @cuda.jit
@cuda.jit
def manual_convolution_3d_cuda(inputs, filters, output_arrays):
    _, input_channels, _, _ =inputs.shape
    batch_size, _, output_height, output_width = output_arrays.shape
    channels, _, smaller_size_height, smaller_size_width = filters.shape
  
    x, y, z = cuda.grid(3)

    if x < batch_size and y < output_height and z < output_width:
        for k in range(channels):
            value = 0.0
            for l in range(input_channels):
                for j in range(smaller_size_height):
                    for i in range(smaller_size_width):
                        value += inputs[x, l, y + i, z + j] * filters[k, l, i, j]
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
                    value = max(value, input_arrays[x, k, (stride*y + i), (stride*z + j)])
            output_arrays[x, k, y, z] = value
            
@cuda.jit
def manual_maxpooling_back_cuda(conv_input, dL_input, output_arrays, pool_size):
    total_size, channels, height, width = conv_input.shape
    
    x = cuda.grid(1)
    if x < total_size:
        for c in range(channels):
            for i in range(height // pool_size):
                for j in range(width // pool_size):
                    start_i = i * pool_size
                    start_j = j * pool_size
                    end_i = start_i + pool_size
                    end_j = start_j + pool_size

                    local_max = conv_input[x, c, start_i, start_j]
                    for m in range(start_i, end_i):
                        for n in range(start_j, end_j):
                            local_max = max(local_max, conv_input[x, c, m, n])

                    for m in range(start_i, end_i):
                        for n in range(start_j, end_j):
                            mask = (conv_input[x, c, m, n] == local_max)
                            output_arrays[x, c, m, n] = dL_input[x, c, i, j] * mask
    
@cuda.jit
def manual_convolution_3d_back_cuda(inputs, gradients, output_arrays):
    batch_size, _, output_height, output_width = output_arrays.shape
    _, smaller_size_depth, smaller_size_height, smaller_size_width = gradients.shape

    x, y, z = cuda.grid(3)

    if x < batch_size and y < output_height and z < output_width:
        for l in range(smaller_size_depth):
            value = 0.0
            for j in range(smaller_size_height):
                for i in range(smaller_size_width):
                    value += inputs[x, l, y + i, z + j] * gradients[x, l, i, j]
            output_arrays[x, l, y, z] = value
            
def cnn_kernel(input,filter):
    # Input/filter  should be in the format (N, C, H, W)
    # Process each array within the main array
    processed_input_array = [array[np.newaxis, :, :] if len(array.shape) == 2 else array for array in input]
    processed_filter_array = [array[np.newaxis, :, :] if len(array.shape) == 2 else array for array in filter]
    # Convert to numpy arrays
    processed_input_array = np.array(processed_input_array)
    processed_filter_array = np.array(processed_filter_array)
    
    num_larger_arrays, _, input_height, input_width= processed_input_array.shape
    num_smaller_arrays, _, filter_height, filter_width = processed_filter_array.shape

    output_y =input_height - filter_height + 1
    output_x = input_width - filter_width + 1
    
    threadsperblock = (8, 8, 8)
    blockspergrid_x = (num_larger_arrays + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (output_y + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (output_x + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Transfer arrays to GPU memory
    larger_arrays_gpu = cuda.to_device(processed_input_array)
    smaller_arrays_gpu = cuda.to_device(processed_filter_array)

    # Create output arrays on GPU
    output_arrays_gpu = cuda.device_array((num_larger_arrays, num_smaller_arrays, output_y, output_x))
    # Perform convolution on GPU for each combination of arrays
    manual_convolution_3d_cuda[blockspergrid, threadsperblock](larger_arrays_gpu, smaller_arrays_gpu, output_arrays_gpu)
    # Transfer results from GPU memory to CPU for printing
    results_gpu = output_arrays_gpu.copy_to_host()
    # Applying Relu Activtion function
    results_gpu = np.maximum(results_gpu, 0)
    return results_gpu

def pool_kernel(input,window_size, stride):
    # Input  should be in the format (N, C, H, W)
    processed_input_array = [array[np.newaxis, :, :] if len(array.shape) == 2 else array for array in input]
    processed_input_array = np.array(processed_input_array)
    
    num_larger_arrays, channels, input_height, input_width = processed_input_array.shape
    
    output_y = input_height//window_size
    output_x = input_width//window_size

    threadsperblock = (8, 8, 8)
    blockspergrid_x = (num_larger_arrays + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (output_y + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (output_x + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    larger_arrays_gpu = cuda.to_device(processed_input_array)    
    output_arrays_gpu = cuda.device_array((num_larger_arrays, channels, output_y, output_x))
    
    manual_maxpooling_cuda[blockspergrid, threadsperblock](larger_arrays_gpu, output_arrays_gpu, window_size, stride)
    results_gpu = output_arrays_gpu.copy_to_host()
    
    return results_gpu

def pool_back_kernel(conv_arr, dL_arr, pool_size):
    threadsperblock = 32
    blockspergrid = (conv_arr.shape[0] + threadsperblock - 1) // threadsperblock
    
    conv_arr_gpu = cuda.to_device(conv_arr)
    dl_arr_gpu = cuda.to_device(dL_arr)
    output_arrays_gpu = cuda.device_array_like(conv_arr)

    manual_maxpooling_back_cuda[blockspergrid, threadsperblock](conv_arr_gpu, dl_arr_gpu, output_arrays_gpu, pool_size)

    results_gpu = output_arrays_gpu.copy_to_host()
    return results_gpu
    
def cnn_back_kernel(inputs,gradients,mode='valid'):
    # Input/filter  should be in the format (N, C, H, W)
    # Process each array within the main array
    processed_input_array = [array[np.newaxis, :, :] if len(array.shape) == 2 else array for array in inputs]
    processed_gradients_array = [array[np.newaxis, :, :] if len(array.shape) == 2 else array for array in gradients]
    # Convert to numpy arrays
    processed_input_array = np.array(processed_input_array)
    processed_gradients_array = np.array(processed_gradients_array)
    _, channels, gradients_height, gradients_width = processed_gradients_array.shape
    if mode == 'full':
        processed_input_array = np.pad(processed_input_array, ((0, 0), (0, 0), (gradients_height - 1, gradients_height - 1), (gradients_width - 1, gradients_width - 1)), mode='constant')
    
    num_larger_arrays, _, input_height, input_width= processed_input_array.shape
    
    # print("processed_input_array",processed_input_array.shape)
    # print("processed_gradients_array",processed_gradients_array.shape)
    output_y =input_height - gradients_height + 1
    output_x = input_width - gradients_width + 1

    threadsperblock = (8, 8, 8)
    blockspergrid_x = (num_larger_arrays + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (output_y + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (output_x + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # print("blockspergrid",blockspergrid)
    # Transfer arrays to GPU memory
    larger_arrays_gpu = cuda.to_device(processed_input_array)
    smaller_arrays_gpu = cuda.to_device(processed_gradients_array)

    # Create output arrays on GPU
    output_arrays_gpu = cuda.device_array((num_larger_arrays, channels, output_y, output_x))
    # Perform convolution on GPU for each combination of arrays
    manual_convolution_3d_back_cuda[blockspergrid, threadsperblock](larger_arrays_gpu, smaller_arrays_gpu, output_arrays_gpu)
    results_gpu = output_arrays_gpu.copy_to_host()
    return results_gpu

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

X_train = train_images / 255.0
y_train = train_labels

X_test = test_images / 255.0
y_test = test_labels

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

epochs=20
lr=0.01
batch_size=32
pool_size=2


conv1 = Convolution(X_train[0].shape, 5, 2)
# print("conv1.filters",conv1.output_shape)
# print("conv1.filters",conv1.filter_shape)
# print("conv1.filters\n",conv1.filters)
h,w= conv1.output_shape[1:]
conv2 = Convolution((h//pool_size,w//pool_size), 3, 4)
# print("conv2.filters",conv2.output_shape)
# print("conv2.filters\n",conv2.filters)
# sys.exit()
full = Fully_Connected(25 * 4, 10)
# full = Fully_Connected(169, 10)

st = time.time()
at = []
tt = []
def train_network():
    num_samples = len(X_train)
    
    for epoch in range(epochs):
        t = time.time()
        total_loss = 0.0
        correct_predictions = 0

        # Create batches
        batches, num_batches = create_batches(X_train, y_train, batch_size)
        for batch_idx, (X_batch, y_batch) in enumerate(batches):
            full_conv_output = cnn_kernel(X_batch, conv1.filters)
            # print("full_conv_output\n",full_conv_output.shape)
            full_pool_out = pool_kernel(full_conv_output, pool_size, pool_size)
            # print("full_pool_out\n",full_pool_out.shape)
            full_conv_output2= cnn_kernel(full_pool_out, conv2.filters)
            # print("full_conv_output2\n",full_conv_output2.shape)
            full_pool_out2 = pool_kernel(full_conv_output2, pool_size, pool_size)
            # print("full_pool_out2\n",full_pool_out2.shape)
            num_images = full_conv_output.shape[0]
            all_full_back = np.empty_like(full_pool_out2)
            for i in range(num_images):
                full_out = full.forward(full_pool_out2[i])
                loss = cross_entropy_loss(full_out.flatten(), y_batch[i])
                total_loss += loss

                # Converting to One-Hot encoding
                one_hot_pred = np.zeros_like(full_out)
                one_hot_pred[np.argmax(full_out)] = 1
                one_hot_pred = one_hot_pred.flatten()

                num_pred = np.argmax(one_hot_pred)
                num_y = np.argmax(y_batch[i])

                if num_pred == num_y:
                    correct_predictions += 1

                # Backward pass
                gradient = cross_entropy_loss_gradient(y_batch[i], full_out.flatten()).reshape((-1, 1))
                all_full_back[i] = full.backward(gradient, lr)

            back_pool_out = pool_back_kernel(full_conv_output2,all_full_back,pool_size)
            # print("\n\n\nback_pool_out",back_pool_out.shape)
            back_conv_filters_out = cnn_back_kernel(full_pool_out, back_pool_out)
            # print("\n\n\nback_conv_filters_out",back_conv_filters_out.shape)
            conv2.filters -= lr * np.prod(back_conv_filters_out, axis=0)
            conv2.update_filters(conv2.filters)
            # print("\n\n\nconv2.filters",conv2.filters.shape)
            back_conv_input_out = cnn_back_kernel(back_pool_out, conv2.filters, mode='full')
            # print("\n\n\nback_conv_input_out",back_conv_input_out.shape)

            back_pool_out2 = pool_back_kernel(full_conv_output, back_conv_input_out, pool_size)
            # print("\n\n\nback_pool_out2",back_pool_out2.shape)
            back_conv_filters_out2 = cnn_back_kernel(X_batch, back_pool_out2)
            # print("\n\n\nback_conv_filters_out2",back_conv_filters_out2.shape)
            conv1.filters -= lr * np.prod(back_conv_filters_out2, axis=0)
            conv1.update_filters(conv1.filters)
            # print("\n\n\nconv1.filters",conv1.filters.shape)
            # sys.exit()


            
        average_loss = total_loss / num_samples
        accuracy = correct_predictions / num_samples * 100.0
        # print("correct_predictions",correct_predictions)
        # print("total_loss",total_loss)
        at.append(accuracy)
        tt.append(epoch+1)
        print(f"Epoch {epoch + 1}/{epochs} - Time: {time.time() - t:.2f} seconds - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")


train_network()

print("Time taken for training : ",time.time()-st , " seconds")
# Plotting the data
print("epoches",tt)
print("accuracy",at)
plt.close()

st = time.time()
predictions = []
# np.save("conv_filters.npy",conv.filters)
# np.save("conv_biases.npy",conv.biases)
# np.save("weights.npy",full.weights)
# np.save("biases.npy",full.biases)

conv_out = cnn_kernel(X_test, conv1.filters) 
pool_out = pool_kernel(conv_out, pool_size, pool_size)
conv_out = cnn_kernel(pool_out, conv2.filters)
pool_out = pool_kernel(conv_out, pool_size, pool_size)
for i in range(len(X_test)):
    flattened_output = pool_out[i].flatten()
    pred = full.forward(flattened_output)
    one_hot_pred = np.zeros_like(pred)
    one_hot_pred[np.argmax(pred)] = 1
    predictions.append(one_hot_pred.flatten())

# Convert one-hot encoded predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)
print("Time taken for testing : ",time.time()-st , " seconds")
# Calculate accuracy using sklearn's accuracy_score
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy :", accuracy)