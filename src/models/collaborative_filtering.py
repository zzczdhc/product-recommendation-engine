"""Collaborative Filtering using Matrix Factorization (SVD/ALS)."""

import logging
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Note: requires `implicit` package
from implicit.als import AlternatingLeastSquares

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeFilteringModel:
    """Collaborative filtering recommendation model."""

    def __init__(
        self,
        method: str = "als",
        n_factors: int = 50,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 40.0,
    ):
        """Initialize CF model.

        Args:
            method: 'svd' or 'als'
            n_factors: Number of latent factors
            regularization: Regularization parameter
            iterations: Number of training iterations (for ALS)
            alpha: Confidence scaling (for ALS implicit feedback)
        """
        self.method = method
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

        self.model = None
        self.user_encoder = None
        self.product_encoder = None
        self.user_decoder = None
        self.product_decoder = None
        self.user_item_matrix = None

        # Populated for SVD
        self.user_factors = None
        self.item_factors = None

    def fit(self, implicit_ratings: pd.DataFrame) -> "CollaborativeFilteringModel":
        """Fit collaborative filtering model.

        Args:
            implicit_ratings: DataFrame with columns [user_id, product_id, final_rating]

        Returns:
            self
        """
        logger.info("Fitting Collaborative Filtering Model (%s)...", self.method.upper())

        # ==================== 1. Create User-Item Matrix ====================
        logger.info("Creating user-item sparse matrix...")

        unique_users = implicit_ratings["user_id"].unique()
        unique_products = implicit_ratings["product_id"].unique()

        self.user_encoder = {user: idx for idx, user in enumerate(unique_users)}
        self.product_encoder = {product: idx for idx, product in enumerate(unique_products)}

        self.user_decoder = {idx: user for user, idx in self.user_encoder.items()}
        self.product_decoder = {idx: product for product, idx in self.product_encoder.items()}

        user_indices = implicit_ratings["user_id"].map(self.user_encoder).values
        product_indices = implicit_ratings["product_id"].map(self.product_encoder).values
        ratings = implicit_ratings["final_rating"].values

        self.user_item_matrix = csr_matrix(
            (ratings, (user_indices, product_indices)),
            shape=(len(unique_users), len(unique_products)),
        )

        logger.info("Matrix shape: %s", self.user_item_matrix.shape)
        sparsity = 1 - self.user_item_matrix.nnz / (
            self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        )
        logger.info("Sparsity: %.4f%%", sparsity * 100)

        # ==================== 2. Train Model ====================
        if self.method == "svd":
            self._fit_svd()
        elif self.method == "als":
            self._fit_als()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def _fit_svd(self) -> None:
        """Fit SVD model."""
        logger.info("Training SVD model...")

        self.model = TruncatedSVD(n_components=self.n_factors, random_state=42)

        self.user_factors = self.model.fit_transform(self.user_item_matrix)
        self.item_factors = self.model.components_.T

        evr = float(self.model.explained_variance_ratio_.sum())
        logger.info("Explained variance ratio: %.4f", evr)

    def _fit_als(self) -> None:
        """Fit ALS model (implicit library)."""
        logger.info("Training ALS model...")

        self.model = AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
            random_state=42,
        )

        # ALS expects item-user matrix (transpose)
        item_user_matrix = self.user_item_matrix.T.tocsr()
        self.model.fit(item_user_matrix)

        logger.info("ALS training complete")

    def predict(self, user_id: int, product_id: int) -> float:
        """Predict rating for user-product pair."""
        if self.user_encoder is None or self.product_encoder is None:
            return 0.0

        if user_id not in self.user_encoder or product_id not in self.product_encoder:
            return 0.0

        user_idx = self.user_encoder[user_id]
        product_idx = self.product_encoder[product_id]

        if self.method == "svd":
            # user_factors: (n_users, k), item_factors: (n_items, k)
            prediction = float(np.dot(self.user_factors[user_idx], self.item_factors[product_idx]))
        else:
            # ALS latent factors live on the implicit model
            prediction = float(self.model.user_factors[user_idx].dot(self.model.item_factors[product_idx]))

        return prediction

    def get_recommendations(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_purchased: bool = True,
    ) -> List[Tuple[int, float]]:
        """Get top-N recommendations for a user."""
        if self.user_encoder is None:
            return []

        if user_id not in self.user_encoder:
            logger.warning("User %s not in training data", user_id)
            return []

        user_idx = self.user_encoder[user_id]

        if self.method == "svd":
            scores = np.dot(self.user_factors[user_idx], self.item_factors.T)

            if exclude_purchased:
                purchased_mask = self.user_item_matrix[user_idx].toarray().flatten() > 0
                scores[purchased_mask] = -np.inf

            top_indices = np.argsort(scores)[::-1][:top_n]
            return [(self.product_decoder[idx], float(scores[idx])) for idx in top_indices]

        # ALS
        item_user_matrix = self.user_item_matrix.T.tocsr()
        ids, scores = self.model.recommend(
            user_idx,
            item_user_matrix[user_idx],
            N=top_n,
            filter_already_liked_items=exclude_purchased,
        )
        return [(self.product_decoder[idx], float(score)) for idx, score in zip(ids, scores)]

    def get_similar_products(self, product_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """Get top-N similar products."""
        if self.product_encoder is None:
            return []

        if product_id not in self.product_encoder:
            logger.warning("Product %s not in training data", product_id)
            return []

        product_idx = self.product_encoder[product_id]

        if self.method == "svd":
            product_vector = self.item_factors[product_idx]
            denom = np.linalg.norm(self.item_factors, axis=1) * np.linalg.norm(product_vector)
            # Avoid divide-by-zero
            denom = np.where(denom == 0, 1e-12, denom)
            similarities = np.dot(self.item_factors, product_vector) / denom

            top_indices = np.argsort(similarities)[::-1][1 : top_n + 1]
            return [(self.product_decoder[idx], float(similarities[idx])) for idx in top_indices]

        # ALS
        ids, scores = self.model.similar_items(product_idx, N=top_n + 1)
        return [(self.product_decoder[idx], float(score)) for idx, score in zip(ids[1:], scores[1:])]

    def evaluate(self, test_ratings: pd.DataFrame) -> dict:
        """Evaluate model on test set."""
        logger.info("Evaluating model...")

        predictions: List[float] = []
        actuals: List[float] = []

        for _, row in test_ratings.iterrows():
            user_id = int(row["user_id"])
            product_id = int(row["product_id"])
            actual = float(row["final_rating"])

            pred = self.predict(user_id, product_id)

            # Only evaluate if we can make a prediction
            if pred > 0:
                predictions.append(pred)
                actuals.append(actual)

        if len(predictions) == 0:
            metrics = {"rmse": np.nan, "mae": np.nan, "correlation": 0.0, "coverage": 0.0}
            logger.info("RMSE: nan")
            logger.info("MAE: nan")
            logger.info("Correlation: 0.0000")
            logger.info("Coverage: 0.00%%")
            return metrics

        predictions_arr = np.asarray(predictions)
        actuals_arr = np.asarray(actuals)

        rmse = float(np.sqrt(np.mean((predictions_arr - actuals_arr) ** 2)))
        mae = float(np.mean(np.abs(predictions_arr - actuals_arr)))

        if len(predictions_arr) > 1:
            correlation = float(np.corrcoef(predictions_arr, actuals_arr)[0, 1])
        else:
            correlation = 0.0

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation,
            "coverage": len(predictions_arr) / len(test_ratings) if len(test_ratings) else 0.0,
        }

        logger.info("RMSE: %.4f", metrics["rmse"])
        logger.info("MAE: %.4f", metrics["mae"])
        logger.info("Correlation: %.4f", metrics["correlation"])
        logger.info("Coverage: %.2f%%", metrics["coverage"] * 100)

        return metrics

    def save(self, filepath: str) -> None:
        """Save model."""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "method": self.method,
                    "n_factors": self.n_factors,
                    "regularization": self.regularization,
                    "iterations": self.iterations,
                    "alpha": self.alpha,
                    "model": self.model,
                    "user_encoder": self.user_encoder,
                    "product_encoder": self.product_encoder,
                    "user_decoder": self.user_decoder,
                    "product_decoder": self.product_decoder,
                    "user_item_matrix": self.user_item_matrix,
                    "user_factors": self.user_factors,
                    "item_factors": self.item_factors,
                },
                f,
            )
        logger.info("Model saved to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "CollaborativeFilteringModel":
        """Load model."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        model = cls(
            method=data["method"],
            n_factors=data["n_factors"],
            regularization=data["regularization"],
            iterations=data["iterations"],
            alpha=data["alpha"],
        )

        model.model = data["model"]
        model.user_encoder = data["user_encoder"]
        model.product_encoder = data["product_encoder"]
        model.user_decoder = data["user_decoder"]
        model.product_decoder = data["product_decoder"]
        model.user_item_matrix = data["user_item_matrix"]
        model.user_factors = data.get("user_factors")
        model.item_factors = data.get("item_factors")

        logger.info("Model loaded from %s", filepath)
        return model