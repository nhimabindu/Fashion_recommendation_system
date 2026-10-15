import pandas as pd

from src.recommendation import (
    build_recommendation_model,
    recommend_products
)


products = pd.DataFrame({
    "product_name": [
        "Black Casual Dress",
        "Black Summer Dress",
        "Red Party Dress",
        "White Sports Shoes",
        "Blue Casual Shirt",
        "Black Formal Dress"
    ],
    "category": [
        "dress", "dress", "dress",
        "shoes", "shirt", "dress"
    ],
    "color": [
        "black", "black", "red",
        "white", "blue", "black"
    ],
    "style": [
        "casual", "casual", "party",
        "sports", "casual", "formal"
    ],
    "description": [
        "comfortable everyday black dress",
        "lightweight black dress for summer",
        "elegant red dress for parties",
        "comfortable shoes for sports",
        "blue shirt for everyday wear",
        "black dress for formal occasions"
    ]
})


products, similarity_matrix = build_recommendation_model(
    products
)

recommendations = recommend_products(
    products,
    similarity_matrix,
    "Black Casual Dress",
    top_n=3
)

print("Recommended fashion products:")
print(
    recommendations[
        ["product_name", "similarity_score"]
    ].to_string(index=False)
)