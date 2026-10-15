import numpy as np

from src.data_loader import load_fashion_mnist
from src.recommendation import (
    build_recommendation_model,
    recommend_products,
    CLASS_NAMES
)


def evaluate_recommendations():

    # Load the Fashion-MNIST dataset
    x_train, y_train, _, _ = load_fashion_mnist()

    # Use a subset for evaluation
    images = x_train[:1000]
    labels = y_train[:1000]

    print("Extracting CNN features...")

    _, similarity_matrix = build_recommendation_model(images)

    # Evaluate recommendations for 100 queries
    query_indices = np.linspace(
        0, len(images) - 1, 100, dtype=int
    )

    precision_scores = []
    category_match_scores = []

    for query_index in query_indices:

        recommendations = recommend_products(
            images,
            labels,
            similarity_matrix,
            image_index=int(query_index),
            top_n=5
        )

        recommended_labels = [
            CLASS_NAMES.index(item["class_name"])
            for item in recommendations
        ]

        true_label = int(labels[query_index])

        matches = sum(
            label == true_label
            for label in recommended_labels
        )

        precision_scores.append(matches / 5)

        category_match_scores.append(matches / 5)

    mean_precision = float(np.mean(precision_scores))

    print("\nEvaluation Results")
    print("------------------")
    print("Queries evaluated:", len(query_indices))
    print("Recommendations per query: 5")
    print(
        f"Mean same-category recommendation rate: "
        f"{mean_precision:.2%}"
    )


if __name__ == "__main__":
    evaluate_recommendations()