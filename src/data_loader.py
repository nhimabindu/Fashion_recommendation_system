import tensorflow as tf


def load_fashion_mnist():

    # Download and load Fashion-MNIST
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    # Normalize pixel values
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_fashion_mnist()

    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)

    print("Testing images:", x_test.shape)
    print("Testing labels:", y_test.shape)

    print("\nFashion-MNIST loaded successfully!")