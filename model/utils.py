import numpy as np

################### Loss Functions ###################
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_derivative(y_true, y_pred):
    y_pred = y_pred.reshape((1, y_pred.shape[0]))
    return 2*(y_pred-y_true)/y_true.size # (batch, classes)

def categorical_cross_entropy(y_true, y_pred, n_classes=10):
    return -1/n_classes * np.sum(y_true * np.log(y_pred))

def categorical_cross_entropy_derivative(y_true, y_pred):
    y_pred = y_pred.reshape((1, y_pred.shape[0]))
    return y_pred - y_true # (batch, classes)


################### Activation Functions ###################
def tanh(x):
    return np.tanh(x);

def tanh_derivative(x):
    return 1-np.tanh(x)**2;

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def ReLU(z):
    return np.maximum(0, z)

# backward pass, receives input from next layer (so output of this layer)
def ReLU_derivative(output):
    # Apply the transformation to 1 if element > 0, else 0
    return np.where(output > 0, 1, 0)