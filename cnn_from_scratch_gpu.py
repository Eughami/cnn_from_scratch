import numpy as np
from numba import cuda,prange
from scipy.signal import correlate2d
import tensorflow.keras as keras
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import time 

@cuda.jit
def corr2d_cuda(input_data, filters, output, num_filters):
    for i in prange(num_filters):
        output[i, :, :] = correlate2d(input_data, filters[i], mode="valid")

@cuda.jit
def max_pool_cuda(input_data, output, pool_size):
    num_channels, height, width = input_data.shape
    output_height = height // pool_size
    output_width = width // pool_size

    output.shape = (num_channels, output_height, output_width)
    for c in prange(num_channels):
        for i in prange(output_height):
            for j in prange(output_width):
                start_i = i * pool_size
                start_j = j * pool_size
                end_i = start_i + pool_size
                end_j = start_j + pool_size

                patch = input_data[c, start_i:end_i, start_j:end_j]
                output[c, i, j] = np.max(patch)

class Convolution:
    # ... (same as the original Convolution class but remove the forward and backward functions)
     def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape
        
        # Size of outputs and filters
        
        self.filter_shape = (num_filters, filter_size, filter_size) # (3,3)
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)
        
        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)

class MaxPool:
    # ... (same as the original MaxPool class but remove the forward and backward functions)
    def __init__(self, pool_size):
        self.pool_size = pool_size

@cuda.jit
def softmax(z):
    # Shift the input values to avoid numerical instability
    shifted_z = z - np.max(z)
    exp_values = np.exp(shifted_z)
    sum_exp_values = np.sum(exp_values, axis=0)
    log_sum_exp = np.log(sum_exp_values)

    probabilities = exp_values / sum_exp_values

    return probabilities

@cuda.jit
def softmax_derivative(s):
    return np.diagflat(s) - np.dot(s, s.T)

class Fully_Connected:
    # ... (same as the original Fully_Connected class but remove the forward and backward functions)
    def __init__(self, input_size, output_size):
        self.input_size = input_size # Size of the inputs coming
        self.output_size = output_size # Size of the output producing
        self.weights = np.random.randn(output_size, self.input_size)
        self.biases = np.random.rand(output_size, 1)

@cuda.jit
def cross_entropy_loss_cuda(predictions, targets):
    num_samples = 10

    # Avoid numerical instability by adding a small epsilon value
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(targets * np.log(predictions)) / num_samples
    return loss

@cuda.jit
def cross_entropy_loss_gradient_cuda(actual_labels, predicted_probs):
    num_samples = actual_labels.shape[0]
    gradient = -actual_labels / (predicted_probs + 1e-7) / num_samples

    return gradient

@cuda.jit
def train_network_cuda(X, y, conv, pool, full, lr=0.01, epochs=200):
    for epoch in prange(epochs):
        total_loss = 0.0
        correct_predictions = 0

        for i in prange(len(X)):
            # Forward pass
            input_data = X[i].astype(np.float32)
            conv_filters = conv.filters.astype(np.float32)
            corr2d_cuda[1, X_train[0].shape[0], X_train[0].shape[0], X_train[0].shape[0]](input_data, conv_filters, conv.output, conv.num_filters)
            pool_size = pool.pool_size
            max_pool_cuda[1, X_train[0].shape[0], X_train[0].shape[0], X_train[0].shape[0], 1, 1](conv.output, pool.output, pool_size)
            full_input = pool.output.reshape(-1, 1)
            full_output = full.output.astype(np.float32)
            full_weights = full.weights.astype(np.float32)
            full_biases = full.biases.astype(np.float32)

            loss = cross_entropy_loss_cuda[1, full_output.shape[0], full_output.shape[1]](full_output, y[i].astype(np.float32))
            total_loss += loss

            correct_predictions_temp = np.argmax(full_output, axis=-1)
            correct_predictions += np.sum(correct_predictions_temp == y[i])

            # Backward pass
            gradients = cross_entropy_loss_gradient_cuda[1, X_train[0].shape[0], full_output.shape[1]](y[i].astype(np.float32), full_output)
            gradients = gradients.reshape((-1, 1))
            full_back = np.zeros_like(full.output)
            full_back[:, np.newaxis, :] = gradients

            full_weights -= lr * np.sum(full_back * full.output, axis=(0, 1, 2))
            full_biases -= lr * np.sum(full_back, axis=(0, 1, 2))

        # Print epoch statistics
        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X_train) * 100.0
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")

@cuda.jit
def predict_cuda(input_sample, conv, pool, full):
    input_data = input_sample.astype(np.float32)
    conv_filters = conv.filters.astype(np.float32)
    corr2d_cuda[1, input_data.shape[0], input_data.shape[0], conv.filters.shape[0]](input_data, conv_filters, conv.output, conv.num_filters)
    pool_size = pool.pool_size
    max_pool_cuda[1, input_data.shape[0], input_data.shape[0], input_data.shape[0], 1, 1](conv.output, pool.output, pool_size)
    flattened_output = pool.output.reshape(-1, 1)
    predictions = full.forward(flattened_output)
    return predictions

def train_network(X, y, lr=0.01, epochs=20):
    cuda_X = np.ascontiguousarray(X, dtype=np.float32)
    cuda_y = np.ascontiguousarray(y, dtype=np.float32)

    conv_instance = Convolution(X[0].shape, 3, 1)
    pool_instance = MaxPool(2)
    full_instance = Fully_Connected(169, 10)

    print(conv_instance.filters)
    cuda_conv = cuda.device_array(np.random.randn(*conv_instance.filter_shape).shape, np.float32)
    cuda_pool = cuda.device_array(pool_instance.output_shape, np.float32)
    cuda_full = cuda.device_array(full_instance.output_shape, np.float32)

    train_network_cuda[1, len(X), 1](cuda_X, cuda_y, conv_instance, pool_instance, full_instance, lr, epochs)
    conv = cuda_conv.copy_to_host()
    pool = cuda_pool.copy_to_host()
    full = cuda_full.copy_to_host()

    return conv, pool, full

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
X_train = train_images[:8000] / 255.0
y_train = train_labels[:8000]

X_test = train_images[8000:10000] / 255.0
y_test = train_labels[8000:10000]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = np.ascontiguousarray(X_train, dtype=np.float32)
y_train = np.ascontiguousarray(y_train, dtype=np.float32)

st = time.time()
conv,pool,full =  train_network(X_train, y_train, lr=0.01, epochs=20)
print("Time taken for training : ",time.time()-st , " seconds")

predictions = []

for data in X_test:
    input_sample = np.ascontiguousarray(data, dtype=np.float32)
    pred = predict_cuda(input_sample, conv, pool, full)
    one_hot_pred = np.zeros_like(pred)
    one_hot_pred[np.argmax(pred)] = 1
    predictions.append(one_hot_pred.flatten())

predictions = np.array(predictions)
print("accuracy score : ",accuracy_score(predictions, y_test))