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
            # output_array[i, j] = np.sum(np.multiply(padded_input[i:i+kernel_shape[0], j:j+kernel_shape[1]], kernel))
            total = 0
            for k in range(kernel_shape[0]):
                for l in range(kernel_shape[1]):
                    total += padded_input[i + k, j + l] * kernel[k, l]

            output_array[i, j] = total

    return output_array


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

    return zip(data_batches, label_batches), num_batches