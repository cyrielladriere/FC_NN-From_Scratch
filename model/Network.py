import numpy as np

class Network():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative
    
    def predict(self, input):
        result = []
        for sample in input:
            output = sample
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result
    
    def fit(self, x_train, y_train, n_epochs, learning_rate):
        samples = len(x_train)
        for epoch in range(n_epochs):
            loss = 0.0
            accuracy = 0.0
            for i, sample in enumerate(x_train):
                # forward prop
                output = sample
                for layer in self.layers:
                    output = layer.forward(output)

                # loss and accuracy are only used to show at end of each epoch, not used in computations of network 
                loss += self.loss(y_train[i], output)
                accuracy += 1 if np.argmax(output) == np.argmax(y_train[i]) else 0

                # backward prop 
                dldy = self.loss_derivative(y_train[i], output) # (1, classes)
                for layer in reversed(self.layers):
                    dldy = layer.backward(dldy, learning_rate)
            loss /= samples
            print(f"Epoch {epoch+1}/{n_epochs}: train_loss={loss: .6f}, train_accuracy={accuracy/len(x_train): .3f}")



