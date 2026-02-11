"""
Collaborative Filtering using Matrix Factorization (SVD/ALS)
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from implicit.als import AlternatingLeastSquares
from typing import List, Tuple, Optional
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeFilteringModel:
    """Collaborative filtering recommendation model"""

    def __init__(self,
                 method: str = 'als',
                 n_factors: int = 50,
                 regularization: float = 0.01,
                 iterations: int = 15,
                 alpha: float = 40.0):
        """
        Initialize CF model

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
        self.user_item_matrix = None

    def fit(self, implicit_ratings: pd.DataFrame) -> 'CollaborativeFilteringModel':
        """
        Fit collaborative filtering model

        Args:
            implicit_ratings: DataFrame with user_id, product_id, final_rating

        Returns:
            self
        """
        logger.info(f"Fitting Collaborative Filtering Model ({self.method.upper()})...")

        # ====================== 1. Create User-Item Matrix ======================
        logger.info("Creating user-item sparse matrix...")

        # Encode user and product IDs
        unique_users = implicit_ratings['user_id'].unique()
        unique_products = implicit_ratings['product_id'].unique()

        self.user_encoder = {user: idx for idx, user in enumerate(unique_users)}
        self.product_encoder = {product: idx for idx, product in enumerate(unique_products)}

        # Reverse mappings
        self.user_decoder = {idx: user for user, idx in self.user_encoder.items()}
        self.product_decoder = {idx: product for product, idx in self.product_encoder.items()}

        # Create sparse matrix
        user_indices = implicit_ratings['user_id'].map(self.user_encoder).values
        product_indices = implicit_ratings['product_id'].map(self.product_encoder).values
        ratings = implicit_ratings['final_rating'].values

        self.user_item_matrix = csr_matrix(
            (ratings, (user_indices, product_indices)),
            shape=(len(unique_users), len(unique_products))
        )

        logger.info(f"Matrix shape: {self.user_item_matrix.shape}")
        logger.info(f"Sparsity: {1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]):.4%}")

        # ====================== 1.5. ALS Safety: Remove Empty Item Columns ======================
        # The `implicit` ALS implementation can effectively drop / ignore items with zero interactions,
        # which can desync our `product_encoder` indices from `model.item_factors`.
        # To keep indices consistent, we proactively remove item columns with nnz == 0 when using ALS.
        if self.method == 'als':
            item_nnz = np.asarray(self.user_item_matrix.getnnz(axis=0)).ravel()  # nnz per item/column
            valid_item_mask = item_nnz > 0

            if not valid_item_mask.all():
                n_removed = int(np.sum(~valid_item_mask))
                logger.info(f"Removing {n_removed} empty items (nnz==0) before ALS training...")

                # Keep only non-empty item columns
                self.user_item_matrix = self.user_item_matrix[:, valid_item_mask]

                # Rebuild product encoder/decoder to match the new matrix column indices
                kept_old_item_indices = np.where(valid_item_mask)[0]
                old_product_decoder = self.product_decoder  # old: idx -> product_id

                self.product_decoder = {
                    new_idx: old_product_decoder[old_idx]
                    for new_idx, old_idx in enumerate(kept_old_item_indices)
                }
                self.product_encoder = {pid: idx for idx, pid in self.product_decoder.items()}

        # ====================== 2. Train Model ======================
        if self.method == 'svd':
            self._fit_svd()
        elif self.method == 'als':
            self._fit_als()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def _fit_svd(self):
        """Fit SVD model"""
        logger.info("Training SVD model...")

        self.model = TruncatedSVD(
            n_components=self.n_factors,
            random_state=42
        )

        # Fit and transform
        self.user_factors = self.model.fit_transform(self.user_item_matrix)
        self.item_factors = self.model.components_.T

        logger.info(f"Explained variance ratio: {self.model.explained_variance_ratio_.sum():.4f}")

    def _fit_als(self):
        """Fit ALS model (implicit library)"""
        logger.info("Training ALS model...")

        self.model = AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
            random_state=42
        )

        # ALS expects item-user matrix (transpose)
        item_user_matrix = self.user_item_matrix.T.tocsr()

        # Fit model
        self.model.fit(item_user_matrix)

        logger.info("ALS training complete")

    def predict(self, user_id: int, product_id: int) -> float:
        """
        Predict rating for user-product pair

        Args:
            user_id: User ID
            product_id: Product ID

        Returns:
            Predicted rating
        """
        if user_id not in self.user_encoder or product_id not in self.product_encoder:
            return 0.0

        user_idx = self.user_encoder[user_id]
        product_idx = self.product_encoder[product_id]

        if self.method == 'svd':
            prediction = np.dot(self.user_factors[user_idx], self.item_factors[product_idx])
        else:
            # ALS uses implicit's learned factors
            # Safety guard: avoid index errors if factors are smaller than our encoders
            if (user_idx >= self.model.user_factors.shape[0]) or (product_idx >= self.model.item_factors.shape[0]):
                return 0.0
            prediction = self.model.user_factors[user_idx].dot(self.model.item_factors[product_idx])

        return float(prediction)

    def get_recommendations(self,
                            user_id: int,
                            top_n: int = 10,
                            exclude_purchased: bool = True) -> List[Tuple[int, float]]:
        """
        Get top N recommendations for a user

        Args:
            user_id: User ID
            top_n: Number of recommendations
            exclude_purchased: Exclude already purchased products

        Returns:
            List of (product_id, score) tuples
        """
        if user_id not in self.user_encoder:
            logger.warning(f"User {user_id} not in training data")
            return []

        user_idx = self.user_encoder[user_id]

        if self.method == 'svd':
            # Compute scores for all products
            scores = np.dot(self.user_factors[user_idx], self.item_factors.T)

            # Get top N
            if exclude_purchased:
                purchased_mask = self.user_item_matrix[user_idx].toarray().flatten() > 0
                scores[purchased_mask] = -np.inf

            top_indices = np.argsort(scores)[::-1][:top_n]
            recommendations = [(self.product_decoder[idx], float(scores[idx])) for idx in top_indices]
        else:
            # ALS
            item_user_matrix = self.user_item_matrix.T.tocsr()

            if exclude_purchased:
                ids, scores = self.model.recommend(
                    user_idx,
                    item_user_matrix[user_idx],
                    N=top_n,
                    filter_already_liked_items=True
                )
            else:
                ids, scores = self.model.recommend(
                    user_idx,
                    item_user_matrix[user_idx],
                    N=top_n,
                    filter_already_liked_items=False
                )

            recommendations = [(self.product_decoder[idx], float(score)) for idx, score in zip(ids, scores)]

        return recommendations

    def get_similar_products(self,
                             product_id: int,
                             top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Get similar products

        Args:
            product_id: Product ID
            top_n: Number of similar products

        Returns:
            List of (product_id, similarity) tuples
        """
        if product_id not in self.product_encoder:
            logger.warning(f"Product {product_id} not in training data")
            return []

        product_idx = self.product_encoder[product_id]

        if self.method == 'svd':
            # Compute cosine similarity
            product_vector = self.item_factors[product_idx]
            similarities = np.dot(self.item_factors, product_vector) / (
                np.linalg.norm(self.item_factors, axis=1) * np.linalg.norm(product_vector)
            )

            # Get top N (excluding itself)
            top_indices = np.argsort(similarities)[::-1][1:top_n+1]
            similar_products = [(self.product_decoder[idx], float(similarities[idx])) for idx in top_indices]
        else:
            # ALS
            ids, scores = self.model.similar_items(product_idx, N=top_n+1)

            # Exclude the product itself (first result)
            similar_products = [(self.product_decoder[idx], float(score)) for idx, score in zip(ids[1:], scores[1:])]

        return similar_products

    def evaluate(self, test_ratings: pd.DataFrame) -> dict:
        """
        Evaluate model on test set

        Args:
            test_ratings: Test set with user_id, product_id, final_rating

        Returns:
            Dictionary with metrics
        """
        logger.info("Evaluating model...")

        predictions = []
        actuals = []

        for _, row in test_ratings.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            actual = row['final_rating']

            pred = self.predict(user_id, product_id)

            if pred > 0:  # Only evaluate if we can make a prediction
                predictions.append(pred)
                actuals.append(actual)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))

        # Correlation
        if len(predictions) > 1:
            correlation = np.corrcoef(predictions, actuals)[0, 1]
        else:
            correlation = 0.0

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'coverage': len(predictions) / len(test_ratings)
        }

        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"Correlation: {correlation:.4f}")
        logger.info(f"Coverage: {metrics['coverage']:.2%}")

        return metrics

    def save(self, filepath: str):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'n_factors': self.n_factors,
                'regularization': self.regularization,
                'iterations': self.iterations,
                'alpha': self.alpha,
                'model': self.model,
                'user_encoder': self.user_encoder,
                'product_encoder': self.product_encoder,
                'user_decoder': self.user_decoder,
                'product_decoder': self.product_decoder,
                'user_item_matrix': self.user_item_matrix,
                'user_factors': getattr(self, 'user_factors', None),
                'item_factors': getattr(self, 'item_factors', None)
            }, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CollaborativeFilteringModel':
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            method=data['method'],
            n_factors=data['n_factors'],
            regularization=data['regularization'],
            iterations=data['iterations'],
            alpha=data['alpha']
        )

        model.model = data['model']
        model.user_encoder = data['user_encoder']
        model.product_encoder = data['product_encoder']
        model.user_decoder = data['user_decoder']
        model.product_decoder = data['product_decoder']
        model.user_item_matrix = data['user_item_matrix']
        model.user_factors = data.get('user_factors')
        model.item_factors = data.get('item_factors')

        logger.info(f"Model loaded from {filepath}")
        return model