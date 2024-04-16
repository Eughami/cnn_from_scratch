import sys
from utils import manual_convolution
import numpy as np

class Convolution:
    
    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape
        
        # Size of outputs and filters    
        self.filter_shape = (num_filters, filter_size, filter_size) # (3,3)
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)
        
        self.filters = np.random.randn(*self.filter_shape)
        # self.filters = np.load("single_filters.npy")
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
    
    def backward(self, input, dL_dout, lr):
        # Create a random dL_dout array to accommodate output gradients
        dL_dinput = np.zeros_like(input)
        dL_dfilters = np.zeros_like(self.filters)
        print(dL_dout.shape)
        print(self.filters.shape)
        sys.exit()
        
        for i in range(self.num_filters):
                # Calculating the gradient of loss with respect to kernels
                dL_dfilters[i] = manual_convolution(input, dL_dout[i],mode="valid")

                # Calculating the gradient of loss with respect to inputs
                # dL_dinput += manual_convolution(dL_dout[i],self.filters[i], mode="full")

        # Updating the parameters with learning rate #!This is being done in the train_network function 
        # self.filters -= lr * dL_dfilters
        # self.biases -= lr * dL_dout

        # returning the gradient of inputs
        return dL_dinput, dL_dfilters
    
    def update_filters(self, new_val):
        self.filters = new_val
    