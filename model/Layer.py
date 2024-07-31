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
        self.bias = np.random.rand(output_size, 1) - 0.5
        

    def forward(self, input):
        # input.shape: (in_features or self.input_size, batch)
        self.input = input

        # Z = W^{T}X + b
        # W: (in_features, out_features or self.output_size)
        # b: (out_features, batch)
        # Z: (out_features, batch)
        self.output = np.dot(self.weights.T, self.input) + self.bias

        return self.output

    # computes dL/dW, dL/dB for a given output_error=dL/dY. Returns input_error=dL/dX.
    def backward(self, dLdy_dydZ, learning_rate):
        # dL/dX = dL/dy * dy/dZ * dZ/dX
        # dZ/dX = W^{T}
        input_error = np.dot(dLdy_dydZ, self.weights.T)

        # dL/dW = dL/dy * dy/dZ * dZ/dW
        # dZ/dW = X
        weights_error = dLdy_dydZ * self.input  # dL/dW

        # dL/db = dL/dy * dy/dZ * dZ/db
        # dZ/db = 1
        bias_error = dLdy_dydZ

        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * bias_error.reshape((bias_error.shape[1], 1))
        return input_error # Will act as output error for layer before this one
    
class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.z = input
        self.y = self.activation(input) # y = g(Z)
        return self.y

    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, dldy, learning_rate):
        dydZ = self.activation_derivative(self.z)
        dydZ = dydZ.reshape((1, dydZ.shape[0]))
        return dydZ * dldy    
