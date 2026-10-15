# Fashion Recommendation System

An image-based fashion recommendation system built using
Python, TensorFlow, CNNs, and Streamlit.

## Overview

The application uses the Fashion-MNIST dataset to recommend
visually similar clothing images.

A convolutional neural network learns image representations.
Cosine similarity compares these representations to retrieve
similar items.

## Features

- Fashion-MNIST dataset loading
- CNN-based image feature extraction
- Content-based fashion recommendations
- Cosine similarity search
- Interactive Streamlit dashboard
- Recommendation evaluation

## Technology Stack

- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Pandas
- Matplotlib
- Streamlit

## Dataset

Fashion-MNIST contains 70,000 grayscale clothing images
across 10 categories.

The dataset is downloaded through TensorFlow.

## Installation

Clone the repository and enter the project directory.

Create a virtual environment and install dependencies:

    python -m venv .venv

On Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Install the required libraries:

    python -m pip install -r requirements.txt

## Train the Model

    python -m src.train_model

The trained model is saved to:

    models/fashion_cnn.keras

## Run the Application

    python -m streamlit run app.py

## Evaluate Recommendations

    python -m src.evaluate_model

## Limitations

Fashion-MNIST contains small grayscale images rather than
real-world product photographs, brand information, or
user purchase histories.

The current system focuses on image similarity rather
than personalized recommendations based on shopping
behaviour.