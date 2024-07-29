from keras.datasets import mnist
import numpy as np
from model.network import Network
from model.layer import FCLayer, ActivationLayer
from keras.utils import to_categorical

def main():
    #loading the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Shapes
    # x_train: (60 000, 28, 28)
    # y_train: (60 000,)
    # x_train: (10 000, 28, 28)
    # y_train: (10 000,)

    # reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], 28*28, 1)   # shape: (28*28, 1)
    x_train = x_train.astype('float32')
    x_train /= 255

    x_test = x_test.reshape(x_test.shape[0], 28*28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255

    # encode output which is a number in range [0,9] into a vector of size 10
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    # network
    net = Network()
    net.add(FCLayer(28*28, 75))     # output_shape: (1, 75)
    net.add(ActivationLayer(tanh, tanh_derivative)) 
    net.add(FCLayer(75, 50))        # output_shape: (1, 50)
    net.add(ActivationLayer(tanh, tanh_derivative))
    net.add(FCLayer(50, 10))        #ouput_shape: (1, 10)
    net.add(ActivationLayer(tanh, tanh_derivative))

    # train
    net.compile(mse, mse_derivative)
    net.fit(x_train[:1000], y_train[:1000], n_epochs=30, learning_rate=0.1)

    # test on 3 samples
    out = net.predict(x_test[0:3])
    print("\n")
    print("predicted values : ")
    print([np.argmax(subarray) for subarray in out], end="\n")
    print("true values : ")
    print(np.argmax(y_test[:3], axis=1))
    return

# forward pass
def ReLU(z):
    return np.maximum(0, z)

# backward pass, receives input from next layer (so output of this layer)
def ReLU_derivative(output):
    # Apply the transformation to 1 if element > 0, else 0
    return np.where(output > 0, 1, 0)

# loss function 
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_derivative(y_true, y_pred):
    y_pred = y_pred.reshape((1, y_pred.shape[0]))
    return 2*(y_pred-y_true)/y_true.size # (batch, classes)

# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_derivative(x):
    return 1-np.tanh(x)**2;

if __name__ == "__main__":
    main()