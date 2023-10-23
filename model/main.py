from keras.datasets import mnist
from model.model import model

def main():
    #loading the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Shapes
    # x_train: (60 000, 28, 28)
    # y_train: (60 000,)
    # x_train: (10 000, 28, 28)
    # y_train: (10 000,)

    # mdl = model(x_train, y_train)

    return

if __name__ == "__main__":
    main()