# https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
# https://www.youtube.com/watch?v=TEWy9vZcxW4
import numpy as np

class Layer():
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward(self, output_error, learning_rate):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias  # Z = XW + B
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)  # dE/dX -> see chain rule
        weights_error = np.dot(self.input.T, output_error)  # dE/dW

        self.weights = self.weights - learning_rate * weights_error 
        self.bias = self.bias - learning_rate * output_error 
        return input_error # Will act as output error for layer before this one
    
class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, output_error, learning_rate):
        return self.activation_derivative(self.input) * output_error    
