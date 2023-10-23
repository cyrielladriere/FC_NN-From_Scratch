from keras.datasets import mnist
from model.model import model

def main():
    #loading the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #printing the shapes of the vectors 
    print('X_train: ' + str(x_train.shape))
    print('Y_train: ' + str(y_train.shape))
    print('X_test:  ' + str(x_test.shape))
    print('Y_test:  ' + str(y_test.shape))

    mdl = model(x_train, y_train)

    return

if __name__ == "__main__":
    main()