import os
os.environ["OMP_NUM_THREADS"] = "1"  # Limit numpy to use only one CPU core
from convolution_layer import Convolution
from max_pool import MaxPool
from utils import create_batches, cross_entropy_loss, cross_entropy_loss_gradient
import numpy as np
import tensorflow as tf
# tf.random.set_seed(42)
from keras import datasets
from scipy.signal import convolve2d
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import time 
import numpy as np
from fully_connected import Fully_Connected 
import sys
# Set a seed for reproducibility
# np.random.seed(42)
# np.set_printoptions(precision=4, suppress=True, linewidth=np.inf,threshold=np.inf)

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

X_train = train_images[0:1000] / 255.0
y_train = train_labels[0:1000]

X_test = test_images / 255.0
y_test = test_labels

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

filter_size = 3
filter_num = 8
pool_size = 2

conv = Convolution(X_train[0].shape, filter_size, filter_num)
pool = MaxPool(pool_size)
full = Fully_Connected(1352, 10)

def train_network(X, y, conv, pool, full, lr=0.01, epochs=2, batch_size=128):
    for epoch in range(epochs):
        t = time.time()
        total_loss = 0.0
        correct_predictions = 0

        batches, num_batches = create_batches(X, y, batch_size)
        for X_batch, y_batch in batches:
            batch_loss = 0.0
            batch_correct_predictions = 0
            dL_filters_temp = np.zeros((batch_size, filter_num, filter_size, filter_size))
                        
            for i in range(len(X_batch)):
                # Forward pass
                conv_out = conv.forward(X_batch[i])
                pool_out = pool.forward(conv_out)
                full_out = full.forward(pool_out)
                loss = cross_entropy_loss(full_out.flatten(), y_batch[i])
                batch_loss += loss

                # Converting to One-Hot encoding
                one_hot_pred = np.zeros_like(full_out)
                one_hot_pred[np.argmax(full_out)] = 1
                one_hot_pred = one_hot_pred.flatten()

                num_pred = np.argmax(one_hot_pred)
                num_y = np.argmax(y_batch[i])

                if num_pred == num_y:
                    batch_correct_predictions += 1

                # Backward pass
                gradient = cross_entropy_loss_gradient(y_batch[i], full_out.flatten()).reshape((-1, 1))
                full_back = full.backward(gradient, lr)
                pool_back = pool.backward(full_back, lr)
                
                conv_back, dL_filters = conv.backward(X_batch[i], pool_back, lr)
                dL_filters_temp[i] = dL_filters
            
            # Update filters after processing the entire batch
            dL_filters_temp = np.prod(dL_filters_temp, axis=0)
            conv.filters -= lr * dL_filters_temp
            conv.update_filters(conv.filters)
            
            total_loss += batch_loss
            correct_predictions += batch_correct_predictions

        # Print epoch statistics
        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X) * 100.0
        print(f"Epoch {epoch + 1}/{epochs} - Time: {time.time() - t:.2f} seconds - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%") 

def predict(input_sample, conv, pool, full):
    # Forward pass through Convolution and pooling
    conv_out = conv.forward(input_sample)
    pool_out = pool.forward(conv_out)
    # Flattening
    flattened_output = pool_out.flatten()
    # Forward pass through fully connected layer
    predictions = full.forward(flattened_output)
    return predictions

st = time.time()
train_network(X_train, y_train, conv, pool, full)
print("Time taken for training : ",time.time()-st , " seconds")

predictions = []
# np.save("conv_filters.npy",conv.filters)
# np.save("conv_biases.npy",conv.biases)
# np.save("weights.npy",full.weights)
# np.save("biases.npy",full.biases)
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
print("Accuracy :", accuracy)