import numpy as np
import tensorflow.keras as keras
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import time 

import numpy as np

def manual_convolution(input_array, kernel, mode='valid'):
    input_shape = input_array.shape
    kernel_shape = kernel.shape
    
    if mode == 'valid':
        output_shape = (input_shape[0] - kernel_shape[0] + 1, input_shape[1] - kernel_shape[1] + 1)
    elif mode == 'same':
        output_shape = input_shape
    elif mode == 'full':
        output_shape = (input_shape[0] + kernel_shape[0] - 1, input_shape[1] + kernel_shape[1] - 1)
    else:
        raise ValueError("Invalid mode. Please use 'valid', 'same', or 'full'.")

    output_array = np.zeros(output_shape)

    # Pad the input array based on the mode
    if mode == 'valid':
        padded_input = input_array
    elif mode == 'same':
        pad_height = kernel_shape[0] // 2
        pad_width = kernel_shape[1] // 2
        padded_input = np.pad(input_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    elif mode == 'full':
        padded_input = np.pad(input_array, ((kernel_shape[0] - 1, kernel_shape[0] - 1),
                                            (kernel_shape[1] - 1, kernel_shape[1] - 1)),
                              mode='constant')
    else:
        raise ValueError("Invalid mode. Please use 'valid', 'same', or 'full'.")
    
    # Perform the convolution
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            total = 0
            for k in range(kernel_shape[0]):
                for l in range(kernel_shape[1]):
                    total += padded_input[i + k, j + l] * kernel[k, l]

            output_array[i, j] = total

    return output_array


class Convolution:
    
    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape
        
        # Size of outputs and filters
        
        self.filter_shape = (num_filters, filter_size, filter_size) # (3,3)
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)
        
        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)
    
    def forward(self, input_data):
        self.input_data = input_data
        # Initialized the input value
        output = np.zeros(self.output_shape)
        for i in range(self.num_filters):
            output[i] = manual_convolution(self.input_data, self.filters[i], mode="valid")
        #Applying Relu Activtion function
        output = np.maximum(output, 0)
        return output 
    
    def backward(self, dL_dout, lr):
        # Create a random dL_dout array to accommodate output gradients
        dL_dinput = np.zeros_like(self.input_data)
        dL_dfilters = np.zeros_like(self.filters)

        for i in range(self.num_filters):
                # Calculating the gradient of loss with respect to kernels
                dL_dfilters[i] = manual_convolution(self.input_data, dL_dout[i],mode="valid")

                # Calculating the gradient of loss with respect to inputs
                dL_dinput += manual_convolution(dL_dout[i],self.filters[i], mode="full")

        # Updating the parameters with learning rate
        self.filters -= lr * dL_dfilters
        self.biases -= lr * dL_dout

        # returning the gradient of inputs
        return dL_dinput
    
class MaxPool:

    def __init__(self, pool_size):
        self.pool_size = pool_size
    
    def forward(self, input_data):

        self.input_data = input_data
        self.num_channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        # Determining the output shape
        self.output = np.zeros((self.num_channels, self.output_height, self.output_width))

        # Iterating over different channels
        for c in range(self.num_channels):
            # Looping through the height
            for i in range(self.output_height):
                # looping through the width
                for j in range(self.output_width):

                    # Starting postition
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    # Ending Position
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    # Creating a patch from the input data
                    patch = input_data[c, start_i:end_i, start_j:end_j]

                    #Finding the maximum value from each patch/window
                    self.output[c, i, j] = np.max(patch)

        return self.output
    
    def backward(self, dL_dout, lr):
        dL_dinput = np.zeros_like(self.input_data)

        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = self.input_data[c, start_i:end_i, start_j:end_j]

                    mask = patch == np.max(patch)

                    dL_dinput[c,start_i:end_i, start_j:end_j] = dL_dout[c, i, j] * mask

        return dL_dinput
    
class Fully_Connected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size # Size of the inputs coming
        self.output_size = output_size # Size of the output producing
        self.weights = np.random.randn(output_size, self.input_size)
        self.biases = np.random.rand(output_size, 1)

    def softmax(self, z):
        # Shift the input values to avoid numerical instability
        shifted_z = z - np.max(z)
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0)
        log_sum_exp = np.log(sum_exp_values)

        # Compute the softmax probabilities
        probabilities = exp_values / sum_exp_values

        return probabilities
    
    def softmax_derivative(self, s):
        return np.diagflat(s) - np.dot(s, s.T)
    
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
    
def create_batches(data, labels, batch_size):
    num_samples = len(data)
    num_batches = num_samples // batch_size

    # Create batches of data and labels
    data_batches = np.array_split(data[:num_batches * batch_size], num_batches)
    label_batches = np.array_split(labels[:num_batches * batch_size], num_batches)

    return zip(data_batches, label_batches)

def train_network(X, y, conv, pool, full, lr=0.01, epochs=20, batch_size=32):
    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0

        # Create batches of data and labels
        batches = create_batches(X, y, batch_size)

        for batch_X, batch_y in batches:
            batch_loss = 0.0
            batch_correct = 0

            for i in range(len(batch_X)):
                # Forward pass
                conv_out = conv.forward(batch_X[i])
                pool_out = pool.forward(conv_out)
                full_out = full.forward(pool_out)
                loss = cross_entropy_loss(full_out.flatten(), batch_y[i])
                batch_loss += loss

                # Converting to One-Hot encoding
                one_hot_pred = np.zeros_like(full_out)
                one_hot_pred[np.argmax(full_out)] = 1
                one_hot_pred = one_hot_pred.flatten()

                num_pred = np.argmax(one_hot_pred)
                num_y = np.argmax(batch_y[i])

                if num_pred == num_y:
                    batch_correct += 1

                # Backward pass
                gradient = cross_entropy_loss_gradient(batch_y[i], full_out.flatten()).reshape((-1, 1))
                full_back = full.backward(gradient, lr)
                pool_back = pool.backward(full_back, lr)
                conv_back = conv.backward(pool_back, lr)

            total_loss += batch_loss / len(batch_X)
            correct_predictions += batch_correct

        # Print epoch statistics
        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X) * 100.0
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")

def predict(input_sample, conv, pool, full):
    # Forward pass through Convolution and pooling
    conv_out = conv.forward(input_sample)
    pool_out = pool.forward(conv_out)
    # Flattening
    flattened_output = pool_out.flatten()
    # Forward pass through fully connected layer
    predictions = full.forward(flattened_output)
    return predictions

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
X_train = train_images[:8000] / 255.0
y_train = train_labels[:8000]

X_test = train_images[8000:10000] / 255.0
y_test = train_labels[8000:10000]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

conv = Convolution(X_train[0].shape, 3, 1)
pool = MaxPool(2)
full = Fully_Connected(169, 10)

st = time.time()
train_network(X_train, y_train, conv, pool, full, lr=0.01, epochs=20)
print("Time taken for training : ",time.time()-st , " seconds")

predictions = []

for data in X_test:
    pred = predict(data, conv, pool, full)
    one_hot_pred = np.zeros_like(pred)
    one_hot_pred[np.argmax(pred)] = 1
    predictions.append(one_hot_pred.flatten())

# Convert one-hot encoded predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Calculate accuracy using sklearn's accuracy_score
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy score:", accuracy)