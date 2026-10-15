import matplotlib.pyplot as plt

from src.data_loader import load_fashion_mnist
from src.recommendation import (
    build_recommendation_model,
    recommend_products,
    CLASS_NAMES
)


# Load the Fashion-MNIST dataset
x_train, y_train, _, _ = load_fashion_mnist()

# Use 1,000 images for our initial experiment
sample_images = x_train[:1000]
sample_labels = y_train[:1000]

# Build the recommendation model
features, similarity_matrix = build_recommendation_model(
    sample_images
)

# Select an image
selected_index = 0

# Find similar clothing items
recommendations = recommend_products(
    sample_images,
    sample_labels,
    similarity_matrix,
    image_index=selected_index,
    top_n=5
)

# Create the visualization
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes = axes.flatten()

# Display the selected image
axes[0].imshow(sample_images[selected_index], cmap="gray")

axes[0].set_title(
    f"Selected Item\n{CLASS_NAMES[int(sample_labels[selected_index])]}"
)

axes[0].axis("off")

# Display the five recommendations
for position, item in enumerate(recommendations, start=1):

    axes[position].imshow(item["image"], cmap="gray")

    axes[position].set_title(
        f"{item['class_name']}\n"
        f"Similarity: {item['similarity_score']:.3f}"
    )

    axes[position].axis("off")

plt.suptitle(
    "Fashion Recommendation System",
    fontsize=16
)

plt.tight_layout()

plt.show()