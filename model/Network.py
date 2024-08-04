import numpy as np
import time

class Network():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    def _evaluate(self, x_test, y_test):
        # test
        out = self.predict(x_test)
        preds = [np.argmax(subarray) for subarray in out]
        actual = np.argmax(y_test, axis=1)

        # Calculate the boolean array of correct predictions
        correct_predictions = preds == actual

        # Calculate the accuracy as the mean of correct predictions
        accuracy = np.mean(correct_predictions)
        return accuracy

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
    
    def fit(self, x_train, y_train, x_test, y_test, n_epochs, learning_rate):
        samples = len(x_train)
        for epoch in range(n_epochs):
            epoch_start = time.time()
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
            test_acc = self._evaluate(x_test, y_test)
            time_elapsed = time.time() - epoch_start
            print(f"[{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s] Epoch {epoch+1}/{n_epochs}: train_loss={loss: .6f}, train_accuracy={accuracy/len(x_train): .3f}, test_accuracy={test_acc: .3f}")
