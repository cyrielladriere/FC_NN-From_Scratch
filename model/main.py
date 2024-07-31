import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from model.network import Network
from model.layer import FCLayer, ActivationLayer
from model.utils import sigmoid, sigmoid_derivative, tanh, tanh_derivative, mse, mse_derivative


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
    net.add(ActivationLayer(sigmoid, sigmoid_derivative)) 
    net.add(FCLayer(75, 50))        # output_shape: (1, 50)
    net.add(ActivationLayer(sigmoid, sigmoid_derivative))
    net.add(FCLayer(50, 10))        #ouput_shape: (1, 10)
    net.add(ActivationLayer(sigmoid, sigmoid_derivative))

    # train
    net.compile(mse, mse_derivative)
    net.fit(x_train, y_train, n_epochs=200, learning_rate=0.001)

    # test
    out = net.predict(x_test)
    preds = [np.argmax(subarray) for subarray in out]
    actual = np.argmax(y_test, axis=1)

    # Calculate the boolean array of correct predictions
    correct_predictions = preds == actual

    # Calculate the accuracy as the mean of correct predictions
    accuracy = np.mean(correct_predictions)
    print("Accuracy on test set: ", accuracy)
    # print("true values : ")
    # print(np.argmax(y_test[:3], axis=1))
    return

if __name__ == "__main__":
    main()