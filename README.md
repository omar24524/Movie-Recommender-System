# Movie Recommender System: SVD & NCF

## Overview
This project implements a multi-model movie recommendation engine. It compares traditional Matrix Factorization (SVD) with modern Neural Collaborative Filtering (NCF) to predict user movie ratings. The system is designed for modularity and is containerized with Docker to ensure consistency across development and production environments.

## Methodology
This project utilizes two primary recommendation techniques:
* **SVD (Singular Value Decomposition):** Implemented using the `scikit-surprise` library. This serves as our robust baseline model for collaborative filtering.
* **NCF (Neural Collaborative Filtering):** Implemented using `PyTorch`. This approach uses user/movie embeddings passed through a Multi-Layer Perceptron (MLP) to capture non-linear user-item interactions.

## Performance Metrics
The models were evaluated using Root Mean Squared Error (RMSE) on the MovieLens dataset:

| Model | RMSE |
| :--- | :--- |
| **SVD** | 0.8383 |
| **NCF** | 0.8880 |

## Project Structure
* `/data`: Contains raw `ratings.csv` and `movies.csv` files.
* `/models`: Contains serialized model weights (`svd_model.pkl`, `ncf_model.pkl`).
* `/src`: Source code for data preprocessing and NCF network architecture.
* `svd_model.ipynb`: Training and evaluation notebook for the SVD model.
* `NCF_model.ipynb`: Training and evaluation notebook for the NCF deep learning model.
* `Dockerfile` & `docker-compose.yml`: Configuration for containerization.

## Prerequisites
* Docker & Docker Compose
* Python 3.12+
* Dependencies: `torch`, `pandas`, `scikit-surprise`, `numpy`

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>