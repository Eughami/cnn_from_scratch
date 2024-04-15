### 1. Data Preparation:

1. Load the MNIST dataset consisting of handwritten digits.
2. Normalize the pixel values to the range [0, 1].
3. Convert labels to one-hot encoded vectors.

### 2. Convolutional Neural Network (CNN) Architecture:

1. **Convolution Class**:

   - Initialize filters (kernels) randomly.
   - Perform forward pass convolution operation using manual_convolution function.
   - Apply ReLU activation function to the output.
   - Implement backward pass to accumulate gradients for filter weights.
   - Update filter weights using accumulated gradients after processing each batch.

2. **MaxPooling Class**:

   - Implement forward pass max-pooling operation.
   - During forward pass, store the indices of maximum values.
   - Implement backward pass to distribute gradients to the correct locations.

3. **Fully Connected (FC) Class**:
   - Initialize weights and biases randomly.
   - Implement forward pass by flattening the input, calculating the dot product with weights, and applying softmax activation.
   - Implement backward pass to update weights and biases using gradient descent.

### 3. Training Network:

1. Define a function to create batches of data.
2. Iterate over epochs:
   - Shuffle and split the data into batches.
   - Iterate over each batch:
     - Initialize temporary storage for filter weight updates.
     - Iterate over each image in the batch:
       - Perform forward pass through the network.
       - Calculate loss using cross-entropy.
       - Update temporary storage with gradient for each filter.
       - Update total loss and count correct predictions for the batch.
     - Update filter weights using the accumulated gradients in the temporary storage.
   - Print epoch-wise loss and accuracy.

### 4. Prediction:

1. Perform forward pass on test data to obtain predictions.
2. Convert predictions to class labels using argmax.
3. Calculate accuracy on the test dataset.

### 5. Results:

1. Measure the training time.
2. Print the final accuracy achieved on the test dataset.

### Overall Workflow:

1. Data is prepared and preprocessed.
2. CNN architecture is defined with convolutional, pooling, and fully connected layers.
3. The network is trained using backpropagation with gradient descent, with convolution weights updated after processing each batch.
4. Predictions are made on the test data to evaluate the model's performance.
5. Training time and accuracy are reported.
