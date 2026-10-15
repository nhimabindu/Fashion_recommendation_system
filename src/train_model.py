import os
import tensorflow as tf

from data_loader import load_fashion_mnist


# Reproducibility
tf.random.set_seed(42)

# Create models directory
os.makedirs("models", exist_ok=True)


# Load dataset
x_train, y_train, x_test, y_test = load_fashion_mnist()

# Add the channel dimension for CNN input
x_train = x_train[..., None]
x_test = x_test[..., None]


# Build CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, kernel_size=3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(
        64, kernel_size=3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128, activation="relu", name="image_features"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(10, activation="softmax")
])


# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Train the model
model.summary()

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)


# Evaluate on the test dataset
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print(f"\nTest accuracy: {test_accuracy:.4f}")


# Save the trained model
model.save("models/fashion_cnn.keras")

print("\nModel saved to models/fashion_cnn.keras")