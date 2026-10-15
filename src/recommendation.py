import numpy as np
import tensorflow as tf

from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import load_fashion_mnist


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def load_cnn_model():
    """Load the trained CNN."""

    return tf.keras.models.load_model(
        "models/fashion_cnn.keras"
    )


def extract_features(model, images):
    """Extract learned features from clothing images."""

    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("image_features").output
    )

    images = images[..., np.newaxis]

    features = feature_model.predict(
        images,
        batch_size=128,
        verbose=0
    )

    # Normalize feature vectors
    norms = np.linalg.norm(features, axis=1, keepdims=True)

    features = features / np.maximum(norms, 1e-10)

    return features


def build_recommendation_model(images):

    cnn = load_cnn_model()

    features = extract_features(cnn, images)

    similarity_matrix = cosine_similarity(features)

    return features, similarity_matrix


def recommend_products(
    images,
    labels,
    similarity_matrix,
    image_index,
    top_n=5
):
    """Find visually similar clothing images."""

    if not 0 <= image_index < len(images):
        raise IndexError("Image index is outside the dataset.")

    scores = similarity_matrix[image_index]

    ranked_indices = np.argsort(scores)[::-1]

    # Exclude the selected image
    ranked_indices = ranked_indices[
        ranked_indices != image_index
    ]

    recommended_indices = ranked_indices[:top_n]

    results = []

    for index in recommended_indices:

        results.append({
            "image_index": int(index),
            "class_name": CLASS_NAMES[int(labels[index])],
            "similarity_score": float(scores[index]),
            "image": images[index]
        })

    return results


if __name__ == "__main__":

    x_train, y_train, _, _ = load_fashion_mnist()

    # Use the same subset as the initial experiment
    sample_images = x_train[:1000]
    sample_labels = y_train[:1000]

    print("Extracting CNN features...")

    features, similarity_matrix = build_recommendation_model(
        sample_images
    )

    print("Feature shape:", features.shape)
    print("Similarity matrix shape:", similarity_matrix.shape)

    recommendations = recommend_products(
        sample_images,
        sample_labels,
        similarity_matrix,
        image_index=0,
        top_n=5
    )

    print(
        "\nSelected item:",
        CLASS_NAMES[int(sample_labels[0])]
    )

    print("\nRecommended items:")

    for item in recommendations:

        print(
            f"Index: {item['image_index']} | "
            f"Category: {item['class_name']} | "
            f"Similarity: {item['similarity_score']:.4f}"
        )