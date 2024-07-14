1. **Input Layer**: Receive input data, typically images, of fixed size.

2. **Split Input into Batches of Size N**:
   a. Divide the input data into batches, each containing N images.

3. **For each Batch**:
   a. Initialize a temporary array to store the gradient of the filters

4. **For each Image in the Batch**:

   1. **Convolutional Layer (Convolution)**: Apply convolution operation to the image using filters to extract features.
   2. **Activation Function (ReLU)**: Apply ReLU to the feature maps obtained from the convolution operation.
   3. **Pooling Layer (Pooling)**: Downsample the feature maps obtained from the convolution step.
   4. **Flattening**: Flatten the output from pooling layer into a one-dimensional vector to prepare it for input into the output layer.
   5. **Output Layer**: Connect every neuron from the previous layer and produce the network's output using softmax activation function.
   6. **Loss**: Compute loss using cross_entropy_loss function
   7. **Loss Gradient**: Compute gradient for the output layer using cross_entropy_loss_gradient function.
   8. **FC Backward Pass**: Using the loss gradient compute the FC gradient.
   9. **FC weights/biases Update**: Update the weights and biases for the FC layer.
   10. **Pooling Backward Pass**: the gradient from the FC layer is passed back to only that neuron which achieved the max. All other neurons get zero gradient.
   11. **Convolution backward Pass**:Compute the filters gradients using pooling gradient and input image.
   12. **Gradients Accumulation**: Add the filters gradient to the temporary array.

5. **Update Filters**: After processing the entire batch:

   - Aggregate the gradients computed across the batch dimension.
   - Utilize these aggregated gradients to update the convolution filters using gradient descent.

6. **Repeat**: Iterate through steps 3-5 for each batch in the dataset.

7. **Repeat**: Repeat steps 2-6 for the specified number of epoch

8. **Evaluation**: After training, evaluate the model's performance on a separate validation dataset to assess its generalization ability.
