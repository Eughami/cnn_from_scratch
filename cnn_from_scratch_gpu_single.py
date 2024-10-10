import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["OMP_NUM_THREADS"] = "1"  # Limit numpy to use only one CPU core
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras import datasets
from sklearn.metrics import accuracy_score
import time 
from numba import cuda
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=np.inf,threshold=np.inf)

# Manual convolution using CUDA with @cuda.jit
@cuda.jit
def manual_convolution_3d_cuda(inputs, filters, output_arrays):
    num_filters, output_height, output_width = output_arrays.shape
    channels, smaller_size_height, smaller_size_width = filters.shape

    x, y, z = cuda.grid(3)

    if x < num_filters and y < output_height and z < output_width:
        for k in range(channels):
            value = 0.0
            for j in range(smaller_size_height):
                for i in range(smaller_size_width):
                    value += inputs[y + i, z + j] * filters[k, i, j]
            output_arrays[k, y, z] = value

@cuda.jit
def manual_maxpooling_cuda(input_arrays, output_arrays, window_size, stride):
    channels, out_height, out_width = input_arrays.shape
    x, y, z = cuda.grid(3)
    
    if x < channels and y < out_height and z < out_width:
        for k in range(channels):
            value = 0.0
            for i in range(window_size):
                for j in range(window_size):
                    value = max(value, input_arrays[k, (stride*y + i), (stride*z + j)])
            output_arrays[k, y, z] = value
            
@cuda.jit
def manual_maxpooling_back_cuda(conv_input, dL_input, output_arrays, pool_size):
    channels, height, width = conv_input.shape
    
    x = cuda.grid(1)
    if x < channels:
        for c in range(channels):
            for i in range(height // pool_size):
                for j in range(width // pool_size):
                    start_i = i * pool_size
                    start_j = j * pool_size
                    end_i = start_i + pool_size
                    end_j = start_j + pool_size

                    local_max = conv_input[c, start_i, start_j]
                    for m in range(start_i, end_i):
                        for n in range(start_j, end_j):
                            local_max = max(local_max, conv_input[c, m, n])

                    for m in range(start_i, end_i):
                        for n in range(start_j, end_j):
                            mask = (conv_input[c, m, n] == local_max)
                            output_arrays[c, m, n] = dL_input[c, i, j] * mask
    
@cuda.jit
def manual_convolution_3d_back_cuda(inputs, gradients, output_arrays):
    channels, output_height, output_width = output_arrays.shape
    _, smaller_size_height, smaller_size_width = gradients.shape

    x, y, z = cuda.grid(3)

    if x < channels and y < output_height and z < output_width:
        value = 0.0
        for j in range(smaller_size_height):
            for i in range(smaller_size_width):
                value += inputs[y + i, z + j] * gradients[x, i, j]
        output_arrays[x, y, z] = value


def gaussian_init(shape, mean=0, stddev=0.1):
    """
    Perform Gaussian initialization for weights.
    
    Arguments:
    shape -- Shape of the weight tensor
    mean -- Mean of the Gaussian distribution (default 0)
    stddev -- Standard deviation of the Gaussian distribution (default 0.1)
    
    Returns:
    weights -- Initialized weights using Gaussian initialization
    """
    return np.random.normal(mean, stddev, shape)

def xavier_init(shape):
    """
    Perform Xavier initialization for weights.
    
    Arguments:
    shape -- Shape of the weight tensor
    
    Returns:
    weights -- Initialized weights using Xavier initialization
    """
    fan_in = np.prod(shape[1:])  # Compute the number of input units
    variance = 2.0 / (fan_in + shape[0])  # Xavier scaling factor
    stddev = np.sqrt(variance)
    return np.random.randn(*shape) * stddev
class Convolution:
    
    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape
        
        # Size of outputs and filters
        self.filter_shape = (num_filters, filter_size, filter_size) # (3,3)
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)
        
        # self.filters = gaussian_init((num_filters, filter_size, filter_size))
        self.filters = xavier_init((num_filters, filter_size, filter_size))
    
    def update_filters(self, new_val):
        self.filters = new_val

class Fully_Connected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size # Size of the inputs coming
        self.output_size = output_size # Size of the output producing

        self.weights = np.random.randn(output_size, self.input_size)
        self.biases = np.random.rand(output_size, 1)

    
    def softmax(self, z):
        # Shift the input values to avoid numerical instability
        shifted_z = z - np.max(z)
        
        # Exponentiate the shifted values
        exp_values = np.exp(shifted_z)
        
        # Calculate the sum of exponentiated values for normalization
        sum_exp_values = np.sum(exp_values, axis=0)
        
        # Calculate the softmax probabilities
        probabilities = exp_values / sum_exp_values
        
        return probabilities
    
    def softmax_derivative(self, s):
        diag_softmax = np.diagflat(s)
        outer_product = np.outer(s, s)
        return diag_softmax - outer_product
    
    def forward(self, input_data):
        self.input_data = input_data
        # Flattening the inputs from the previous layer into a vector
        flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases

        # Applying Softmax
        self.output = self.softmax(self.z)
        return self.output
    
    def backward(self, dL_dout, lr):
        # Calculate the gradient of the loss with respect to the pre-activation (z)
        dL_dy = np.dot(self.softmax_derivative(self.output), dL_dout)
        # Calculate the gradient of the loss with respect to the weights (dw)
        dL_dw = np.dot(dL_dy, self.input_data.flatten().reshape(1, -1))

        # Calculate the gradient of the loss with respect to the biases (db)
        dL_db = dL_dy

        # Calculate the gradient of the loss with respect to the input data (dL_dinput)
        dL_dinput = np.dot(self.weights.T, dL_dy)
        dL_dinput = dL_dinput.reshape(self.input_data.shape)

        # Update the weights and biases based on the learning rate and gradients
        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db

        # Return the gradient of the loss with respect to the input data
        return dL_dinput
    
def cross_entropy_loss(predictions, targets):

    num_samples = 10

    # Avoid numerical instability by adding a small epsilon value
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(targets * np.log(predictions)) / num_samples
    return loss

def cross_entropy_loss_gradient(actual_labels, predicted_probs):
    num_samples = actual_labels.shape[0]
    gradient = -actual_labels / (predicted_probs + 1e-7) / num_samples

    return gradient

import numpy as np
from numba import cuda

def cnn_kernel(input, filter):
    # Input should be in the format (H, W)
    # filter should be in the format (C, H, W)
    
    
    # Determine shapes after processing
    input_height, input_width = input.shape
    num_filter_channels, filter_height, filter_width = filter.shape

    # Calculate output dimensions after convolution
    output_y = input_height - filter_height + 1
    output_x = input_width - filter_width + 1

    # Define CUDA threads and blocks
    threadsperblock = (8, 8, 8)
    blockspergrid_x = (num_filter_channels + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (output_y + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (output_x + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Transfer arrays to GPU memory
    input_gpu = cuda.to_device(input)
    filter_gpu = cuda.to_device(filter)

    # Create output array on GPU
    output_gpu = cuda.device_array((num_filter_channels, output_y, output_x))

    # Perform convolution on GPU
    manual_convolution_3d_cuda[blockspergrid, threadsperblock](input_gpu, filter_gpu, output_gpu)

    # Transfer results from GPU to CPU
    results_gpu = output_gpu.copy_to_host()

    # Applying ReLU activation function
    results_gpu = np.maximum(results_gpu, 0)
    return results_gpu

def pool_kernel(input,window_size, stride):
    # Input  should be in the format (C, H, W)
    
    channels, input_height, input_width = input.shape
    
    output_y = input_height//window_size
    output_x = input_width//window_size

    threadsperblock = (8, 8, 8)
    blockspergrid_x = (channels + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (output_y + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (output_x + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    larger_arrays_gpu = cuda.to_device(input)    
    output_arrays_gpu = cuda.device_array((channels, output_y, output_x))
    
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
    
def cnn_back_kernel(inputs,gradients):
    input_height, input_width= inputs.shape
    channels, gradients_height, gradients_width = gradients.shape

    output_y =input_height - gradients_height + 1
    output_x = input_width - gradients_width + 1
    
    threadsperblock = (2, 2, 2)
    blockspergrid_x = (channels + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (output_y + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (output_x + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Transfer arrays to GPU memory
    larger_arrays_gpu = cuda.to_device(inputs)
    smaller_arrays_gpu = cuda.to_device(gradients)

    # Create output arrays on GPU
    output_arrays_gpu = cuda.device_array((channels, output_y, output_x))
    
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
lr=0.001
pool_size=2


f_size = 3
f_num = 8
conv = Convolution(X_train[0].shape, f_size, f_num)
out_size = (X_train[0].shape[0] - f_size + 1) // pool_size
full = Fully_Connected(out_size * out_size * f_num, 10)

st = time.time()

def train_network():
    num_samples = len(X_train)
    
    for epoch in range(epochs):
        t = time.time()
        total_loss = 0.0
        correct_predictions = 0

        # Loop through image
        for i in range(num_samples):
            
            full_conv_output = cnn_kernel(X_train[i], conv.filters)
            full_pool_out = pool_kernel(full_conv_output, pool_size, pool_size)
            
            full_out = full.forward(full_pool_out)
            loss = cross_entropy_loss(full_out.flatten(), y_train[i])
            total_loss += loss

            # Converting to One-Hot encoding
            one_hot_pred = np.zeros_like(full_out)
            one_hot_pred[np.argmax(full_out)] = 1
            one_hot_pred = one_hot_pred.flatten()

            num_pred = np.argmax(one_hot_pred)
            num_y = np.argmax(y_train[i])

            if num_pred == num_y:
                correct_predictions += 1

            # Backward pass
            gradient = cross_entropy_loss_gradient(y_train[i], full_out.flatten()).reshape((-1, 1))
            all_full_back = full.backward(gradient, lr)
            back_pool_out = pool_back_kernel(full_conv_output,all_full_back,pool_size)
            back_conv_out = cnn_back_kernel(X_train[i], back_pool_out)
            conv.filters -= lr * back_conv_out 
            conv.update_filters(conv.filters)
            
            
        average_loss = total_loss / num_samples
        accuracy = correct_predictions / num_samples * 100.0
        print(f"Epoch {epoch + 1}/{epochs} - Time: {time.time() - t:.2f} seconds - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")


train_network()

print("Time taken for training : ",time.time()-st , " seconds")

st = time.time()
predictions = []

conv_out = cnn_kernel(X_test, conv.filters) 
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