"""Collaborative Filtering using Matrix Factorization (SVD/ALS)."""

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

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
        fallback_on_invalid: bool = False,
    ):
        """Initialize CF model.

        Args:
            method: 'svd' or 'als'
            n_factors: Number of latent factors
            regularization: Regularization parameter
            iterations: Number of training iterations (for ALS)
            alpha: Confidence scaling (for ALS implicit feedback)
            fallback_on_invalid: For in-matrix pairs, fallback to global mean when
                ALS factors are invalid (NaN/Inf) or indices are desynced.
        """
        self.method = method
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.fallback_on_invalid = fallback_on_invalid

        self.model = None
        self.user_encoder = None
        self.product_encoder = None
        self.user_decoder = None
        self.product_decoder = None
        self.user_item_matrix = None

        # Populated for SVD
        self.user_factors = None
        self.item_factors = None

        # Diagnostics / robustness
        self.user_key_type_: Optional[type] = None
        self.product_key_type_: Optional[type] = None
        self.global_mean_: float = 0.0
        self.als_user_factor_finite_ratio_: float = 1.0
        self.als_item_factor_finite_ratio_: float = 1.0
        self.als_user_nonfinite_row_rate_: float = 0.0
        self.als_item_nonfinite_row_rate_: float = 0.0
        self.als_factors_sanitized_: bool = False

    def fit(self, implicit_ratings: pd.DataFrame) -> "CollaborativeFilteringModel":
        """Fit collaborative filtering model.

        Args:
            implicit_ratings: DataFrame with columns [user_id, product_id, final_rating]

        Returns:
            self
        """
        logger.info("Fitting Collaborative Filtering Model (%s)...", self.method.upper())

        required = {"user_id", "product_id", "final_rating"}
        missing = required - set(implicit_ratings.columns)
        if missing:
            raise ValueError(f"Missing required columns for fit(): {sorted(missing)}")

        # ==================== 1. Create User-Item Matrix ====================
        logger.info("Creating user-item sparse matrix...")

        unique_users = implicit_ratings["user_id"].unique()
        unique_products = implicit_ratings["product_id"].unique()

        if len(unique_users) == 0 or len(unique_products) == 0:
            raise ValueError("Empty training data after preprocessing/split.")

        self.user_key_type_ = type(unique_users[0])
        self.product_key_type_ = type(unique_products[0])

        self.user_encoder = {user: idx for idx, user in enumerate(unique_users)}
        self.product_encoder = {product: idx for idx, product in enumerate(unique_products)}

        self.user_decoder = {idx: user for user, idx in self.user_encoder.items()}
        self.product_decoder = {idx: product for product, idx in self.product_encoder.items()}

        user_indices = implicit_ratings["user_id"].map(self.user_encoder).values
        product_indices = implicit_ratings["product_id"].map(self.product_encoder).values
        ratings = implicit_ratings["final_rating"].astype(np.float32).values
        if not np.isfinite(ratings).all():
            raise ValueError("final_rating contains NaN/Inf; please clean preprocessing output.")
        self.global_mean_ = float(ratings.mean()) if len(ratings) else 0.0

        self.user_item_matrix = csr_matrix(
            (ratings, (user_indices, product_indices)),
            shape=(len(unique_users), len(unique_products)),
        )

        logger.info("Matrix shape: %s", self.user_item_matrix.shape)
        sparsity = 1 - self.user_item_matrix.nnz / (
            self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        )
        logger.info("Sparsity: %.4f%%", sparsity * 100)

        # ==================== ALS Safety: drop empty item columns ====================
        # Keep implicit ALS item_factors index aligned with our product_encoder indices.
        # Some pipelines may include products that end up as all-zeros in the matrix.
        self.dropped_empty_items_ = 0
        if self.method == "als":
            item_nnz = np.asarray(self.user_item_matrix.getnnz(axis=0)).ravel()
            valid_mask = item_nnz > 0

            if not valid_mask.all():
                self.dropped_empty_items_ = int((~valid_mask).sum())
                logger.info(
                    "ALS safety: dropping %d empty items (nnz==0) before training to keep indices aligned.",
                    self.dropped_empty_items_,
                )

                # Shrink matrix to valid item columns
                self.user_item_matrix = self.user_item_matrix[:, valid_mask]

                # Rebuild product encoder/decoder to match new column indices
                kept_old_cols = np.where(valid_mask)[0]
                old_decoder = self.product_decoder  # old idx -> product_id

                self.product_decoder = {
                    new_idx: old_decoder[old_idx] for new_idx, old_idx in enumerate(kept_old_cols)
                }
                self.product_encoder = {pid: idx for idx, pid in self.product_decoder.items()}

                logger.info("ALS safety: new matrix shape after drop: %s", self.user_item_matrix.shape)

        # ==================== 2. Train Model ====================
        if self.method == "svd":
            self._fit_svd()
        elif self.method == "als":
            self._fit_als()
            self._sanitize_als_factors()
            # Post-check: ensure implicit factors align with our matrix dimensions
            try:
                logger.info(
                    "ALS check: matrix n_users=%d, n_items=%d | model.user_factors=%d, model.item_factors=%d",
                    self.user_item_matrix.shape[0],
                    self.user_item_matrix.shape[1],
                    self.model.user_factors.shape[0],
                    self.model.item_factors.shape[0],
                )
                if (
                    self.model.user_factors.shape[0] != self.user_item_matrix.shape[0]
                    or self.model.item_factors.shape[0] != self.user_item_matrix.shape[1]
                ):
                    logger.warning(
                        (
                            "ALS factor shape mismatch detected. "
                            "Predictions may be invalid until orientation is fixed."
                        )
                    )
            except Exception:
                pass
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

        # Keep orientation as user-item matrix so:
        # - model.user_factors aligns to user indices
        # - model.item_factors aligns to product indices
        user_item_matrix = self.user_item_matrix.tocsr()
        self.model.fit(user_item_matrix)

        logger.info("ALS training complete")

    def _sanitize_als_factors(self) -> None:
        """Record ALS factor health and optionally sanitize non-finite values."""
        if self.method != "als" or self.model is None:
            return

        user_factors = getattr(self.model, "user_factors", None)
        item_factors = getattr(self.model, "item_factors", None)
        if user_factors is None or item_factors is None:
            return

        user_factors_arr = np.asarray(user_factors)
        item_factors_arr = np.asarray(item_factors)

        user_finite_mask = np.isfinite(user_factors_arr)
        item_finite_mask = np.isfinite(item_factors_arr)

        self.als_user_factor_finite_ratio_ = float(user_finite_mask.mean())
        self.als_item_factor_finite_ratio_ = float(item_finite_mask.mean())

        if user_factors_arr.ndim == 2 and len(user_factors_arr) > 0:
            self.als_user_nonfinite_row_rate_ = float((~user_finite_mask.all(axis=1)).mean())
        else:
            self.als_user_nonfinite_row_rate_ = 0.0

        if item_factors_arr.ndim == 2 and len(item_factors_arr) > 0:
            self.als_item_nonfinite_row_rate_ = float((~item_finite_mask.all(axis=1)).mean())
        else:
            self.als_item_nonfinite_row_rate_ = 0.0

        logger.info(
            (
                "ALS factor health: user_finite=%.4f%% item_finite=%.4f%% "
                "| user_nonfinite_rows=%.4f%% item_nonfinite_rows=%.4f%%"
            ),
            self.als_user_factor_finite_ratio_ * 100,
            self.als_item_factor_finite_ratio_ * 100,
            self.als_user_nonfinite_row_rate_ * 100,
            self.als_item_nonfinite_row_rate_ * 100,
        )

        self.als_factors_sanitized_ = False
        if user_finite_mask.all() and item_finite_mask.all():
            return

        logger.warning(
            "ALS produced non-finite factors. Replacing NaN/Inf with 0.0 to keep predictions scorable."
        )
        self.model.user_factors = np.nan_to_num(
            user_factors_arr, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.model.item_factors = np.nan_to_num(
            item_factors_arr, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.als_factors_sanitized_ = True

    @staticmethod
    def _cast_series_to_key_type(series: pd.Series, key_type: Optional[type]) -> pd.Series:
        """Cast test ids to encoder key type (best-effort)."""
        if key_type is None:
            return series
        try:
            return series.map(lambda x: key_type(x) if pd.notna(x) else x)
        except Exception:
            return series

    @staticmethod
    def _cast_scalar_to_key_type(value: Any, key_type: Optional[type]) -> Any:
        """Cast one id to encoder key type (best-effort)."""
        if key_type is None:
            return value
        if isinstance(value, np.generic):
            value = value.item()
        try:
            return key_type(value)
        except Exception:
            return value

    def _predict_with_status(self, user_id: Any, product_id: Any) -> Tuple[float, str]:
        """Predict with status code for diagnostics."""
        if self.user_encoder is None or self.product_encoder is None:
            return float("nan"), "missing_encoder"

        user_key = self._cast_scalar_to_key_type(user_id, self.user_key_type_)
        product_key = self._cast_scalar_to_key_type(product_id, self.product_key_type_)

        if user_key not in self.user_encoder:
            return float("nan"), "unseen_user"
        if product_key not in self.product_encoder:
            return float("nan"), "unseen_item"

        user_idx = int(self.user_encoder[user_key])
        product_idx = int(self.product_encoder[product_key])

        if self.user_item_matrix is None:
            return float("nan"), "missing_matrix"

        if user_idx < 0 or product_idx < 0:
            return float("nan"), "negative_index"
        if user_idx >= self.user_item_matrix.shape[0] or product_idx >= self.user_item_matrix.shape[1]:
            if self.fallback_on_invalid:
                return float(self.global_mean_), "fallback_matrix_oob"
            return float("nan"), "matrix_oob"

        if self.method == "svd":
            if self.user_factors is None or self.item_factors is None:
                return float("nan"), "missing_svd_factors"
            if user_idx >= self.user_factors.shape[0] or product_idx >= self.item_factors.shape[0]:
                if self.fallback_on_invalid:
                    return float(self.global_mean_), "fallback_svd_oob"
                return float("nan"), "svd_oob"
            pred = float(np.dot(self.user_factors[user_idx], self.item_factors[product_idx]))
            return (pred, "ok") if np.isfinite(pred) else (float("nan"), "nonfinite_pred")

        # ALS
        if (
            self.model is None
            or getattr(self.model, "user_factors", None) is None
            or getattr(self.model, "item_factors", None) is None
        ):
            return float("nan"), "missing_als_factors"

        if user_idx >= self.model.user_factors.shape[0] or product_idx >= self.model.item_factors.shape[0]:
            if self.fallback_on_invalid:
                return float(self.global_mean_), "fallback_als_oob"
            return float("nan"), "als_oob"

        user_vec = self.model.user_factors[user_idx]
        item_vec = self.model.item_factors[product_idx]

        if not np.isfinite(user_vec).all() or not np.isfinite(item_vec).all():
            if not self.fallback_on_invalid:
                return float("nan"), "nonfinite_factor_row"
            user_vec = np.nan_to_num(user_vec, nan=0.0, posinf=0.0, neginf=0.0)
            item_vec = np.nan_to_num(item_vec, nan=0.0, posinf=0.0, neginf=0.0)
            pred = float(np.dot(user_vec, item_vec))
            if np.isfinite(pred):
                return pred, "fallback_sanitized_factor"
            return float(self.global_mean_), "fallback_nonfinite_pred"

        pred = float(np.dot(user_vec, item_vec))
        if np.isfinite(pred):
            return pred, "ok"
        if self.fallback_on_invalid:
            return float(self.global_mean_), "fallback_nonfinite_pred"
        return float("nan"), "nonfinite_pred"

    def predict(self, user_id: int, product_id: int) -> float:
        """Predict score for a user-product pair.

        Returns np.nan when the pair is not scorable (unseen ids or any shape mismatch),
        so callers can filter with `np.isfinite`.
        """
        pred, _status = self._predict_with_status(user_id, product_id)
        return float(pred)

    def get_recommendations(
        self,
        user_id: Any,
        top_n: int = 10,
        exclude_purchased: bool = True,
    ) -> List[Tuple[int, float]]:
        """Get top-N recommendations for a user."""
        if self.user_encoder is None:
            return []

        user_key = self._cast_scalar_to_key_type(user_id, self.user_key_type_)
        if user_key not in self.user_encoder:
            logger.warning("User %s not in training data", user_id)
            return []

        user_idx = int(self.user_encoder[user_key])

        if self.method == "svd":
            scores = np.dot(self.user_factors[user_idx], self.item_factors.T)

            if exclude_purchased:
                purchased_mask = self.user_item_matrix[user_idx].toarray().flatten() > 0
                scores[purchased_mask] = -np.inf

            top_indices = np.argsort(scores)[::-1][:top_n]
            return [(self.product_decoder[idx], float(scores[idx])) for idx in top_indices]

        # ALS
        # recommend() expects a CSR row vector with shape (1, n_items)
        if self.model is None:
            return []

        user_item_csr = self.user_item_matrix.tocsr()
        user_items = user_item_csr[user_idx]
        if not hasattr(user_items, "shape") or len(user_items.shape) != 2:
            user_items = user_items.reshape(1, -1)

        ids, scores = self.model.recommend(
            user_idx,
            user_items,
            N=top_n,
            filter_already_liked_items=exclude_purchased,
        )
        return [(self.product_decoder[idx], float(score)) for idx, score in zip(ids, scores)]

    def get_similar_products(self, product_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """Get top-N similar products."""
        if self.product_encoder is None:
            return []

        product_key = self._cast_scalar_to_key_type(product_id, self.product_key_type_)
        if product_key not in self.product_encoder:
            logger.warning("Product %s not in training data", product_id)
            return []

        product_idx = int(self.product_encoder[product_key])

        if self.method == "svd":
            product_vector = self.item_factors[product_idx]
            denom = np.linalg.norm(self.item_factors, axis=1) * np.linalg.norm(product_vector)
            # Avoid divide-by-zero
            denom = np.where(denom == 0, 1e-12, denom)
            similarities = np.dot(self.item_factors, product_vector) / denom

            top_indices = np.argsort(similarities)[::-1][1 : top_n + 1]
            return [(self.product_decoder[idx], float(similarities[idx])) for idx in top_indices]

        # ALS
        if self.model is None or getattr(self.model, "item_factors", None) is None:
            return []
        if product_idx >= self.model.item_factors.shape[0]:
            return []
        ids, scores = self.model.similar_items(product_idx, N=top_n + 1)
        return [(self.product_decoder[idx], float(score)) for idx, score in zip(ids[1:], scores[1:])]

    def evaluate(self, test_ratings: pd.DataFrame) -> dict:
        """Evaluate model on test set.

        - Keep only finite predictions (not NaN/inf).
        - For ALS, scores can be negative; do NOT filter by pred > 0.
        - Report both overall coverage and in-matrix coverage.
        """
        logger.info("Evaluating model...")

        if test_ratings is None or len(test_ratings) == 0:
            return {
                "rmse": np.nan,
                "mae": np.nan,
                "correlation": 0.0,
                "spearman_correlation": 0.0,
                "coverage": 0.0,
                "coverage_in_matrix": 0.0,
                "cold_start_user_rate": np.nan,
                "cold_start_item_rate": np.nan,
                "fallback_rate_in_matrix": 0.0,
                "nonfinite_drop_rate_in_matrix": 0.0,
                "als_user_factor_finite_ratio": float(
                    getattr(self, "als_user_factor_finite_ratio_", 1.0)
                ),
                "als_item_factor_finite_ratio": float(
                    getattr(self, "als_item_factor_finite_ratio_", 1.0)
                ),
                "dropped_empty_items": int(getattr(self, "dropped_empty_items_", 0)),
            }

        # ----- cold-start diagnostics (w.r.t encoders built from TRAIN data) -----
        train_user_keys = set(self.user_encoder.keys()) if self.user_encoder is not None else set()
        train_item_keys = set(self.product_encoder.keys()) if self.product_encoder is not None else set()

        test_user_series = self._cast_series_to_key_type(test_ratings["user_id"], self.user_key_type_)
        test_item_series = self._cast_series_to_key_type(
            test_ratings["product_id"], self.product_key_type_
        )

        seen_user = test_user_series.isin(train_user_keys)
        seen_item = test_item_series.isin(train_item_keys)

        cold_user_rate = float((~seen_user).mean())
        cold_item_rate = float((~seen_item).mean())
        in_matrix_mask = seen_user & seen_item
        in_matrix_rate = float(in_matrix_mask.mean())

        logger.info(
            "Eval diagnostics: cold_start_user_rate=%.2f%%, cold_start_item_rate=%.2f%%, in_matrix_rate=%.2f%%",
            cold_user_rate * 100,
            cold_item_rate * 100,
            in_matrix_rate * 100,
        )
        logger.info(
            "Eval diagnostics: dropped_empty_items=%d",
            int(getattr(self, "dropped_empty_items_", 0)),
        )

        # ----- compute predictions only for in-matrix rows (others are unscorable anyway) -----
        predictions: List[float] = []
        actuals: List[float] = []
        status_counts: Dict[str, int] = {}

        eval_users = test_user_series.loc[in_matrix_mask].to_numpy()
        eval_items = test_item_series.loc[in_matrix_mask].to_numpy()
        eval_actuals = test_ratings.loc[in_matrix_mask, "final_rating"].astype(float).to_numpy()

        for user_id, product_id, actual in zip(eval_users, eval_items, eval_actuals):
            pred, status = self._predict_with_status(user_id, product_id)
            status_counts[status] = status_counts.get(status, 0) + 1
            if np.isfinite(pred):
                predictions.append(float(pred))
                actuals.append(float(actual))

        scorable = len(predictions)
        total = len(test_ratings)
        in_matrix_total = int(in_matrix_mask.sum())

        coverage_all = scorable / total if total else 0.0
        coverage_in_matrix = scorable / in_matrix_total if in_matrix_total else 0.0
        fallback_count = int(sum(v for k, v in status_counts.items() if k.startswith("fallback_")))
        fallback_rate_in_matrix = fallback_count / in_matrix_total if in_matrix_total else 0.0
        nonfinite_drop_count = int(status_counts.get("nonfinite_pred", 0))
        nonfinite_drop_rate = nonfinite_drop_count / in_matrix_total if in_matrix_total else 0.0

        if scorable == 0:
            metrics = {
                "rmse": np.nan,
                "mae": np.nan,
                "correlation": 0.0,
                "spearman_correlation": 0.0,
                "coverage": coverage_all,
                "coverage_in_matrix": coverage_in_matrix,
                "cold_start_user_rate": cold_user_rate,
                "cold_start_item_rate": cold_item_rate,
                "fallback_rate_in_matrix": fallback_rate_in_matrix,
                "nonfinite_drop_rate_in_matrix": nonfinite_drop_rate,
                "als_user_factor_finite_ratio": float(
                    getattr(self, "als_user_factor_finite_ratio_", 1.0)
                ),
                "als_item_factor_finite_ratio": float(
                    getattr(self, "als_item_factor_finite_ratio_", 1.0)
                ),
                "dropped_empty_items": int(getattr(self, "dropped_empty_items_", 0)),
            }
            logger.info(
                "RMSE: nan | MAE: nan | Corr: 0.0000 | coverage(all): %.2f%% | coverage(in-matrix): %.2f%%",
                coverage_all * 100,
                coverage_in_matrix * 100,
            )
            logger.info(
                "Predict status counts: %s | fallback_rate_in_matrix=%.2f%%",
                status_counts,
                fallback_rate_in_matrix * 100,
            )
            return metrics

        predictions_arr = np.asarray(predictions, dtype=float)
        actuals_arr = np.asarray(actuals, dtype=float)

        rmse = float(np.sqrt(np.mean((predictions_arr - actuals_arr) ** 2)))
        mae = float(np.mean(np.abs(predictions_arr - actuals_arr)))

        if len(predictions_arr) > 1:
            correlation = float(np.corrcoef(predictions_arr, actuals_arr)[0, 1])
            spearman_correlation = float(
                pd.Series(predictions_arr).corr(pd.Series(actuals_arr), method="spearman")
            )
        else:
            correlation = 0.0
            spearman_correlation = 0.0

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation,
            "spearman_correlation": spearman_correlation,
            "coverage": coverage_all,
            "coverage_in_matrix": coverage_in_matrix,
            "cold_start_user_rate": cold_user_rate,
            "cold_start_item_rate": cold_item_rate,
            "fallback_rate_in_matrix": fallback_rate_in_matrix,
            "nonfinite_drop_rate_in_matrix": nonfinite_drop_rate,
            "als_user_factor_finite_ratio": float(getattr(self, "als_user_factor_finite_ratio_", 1.0)),
            "als_item_factor_finite_ratio": float(getattr(self, "als_item_factor_finite_ratio_", 1.0)),
            "dropped_empty_items": int(getattr(self, "dropped_empty_items_", 0)),
        }

        logger.info("RMSE: %.4f", metrics["rmse"])
        logger.info("MAE: %.4f", metrics["mae"])
        logger.info("Correlation: %.4f", metrics["correlation"])
        logger.info("Spearman Correlation: %.4f", metrics["spearman_correlation"])
        logger.info("Coverage (all test rows): %.2f%%", metrics["coverage"] * 100)
        logger.info("Coverage (in-matrix rows only): %.2f%%", metrics["coverage_in_matrix"] * 100)
        logger.info(
            "Fallback rate (in-matrix): %.2f%% | Non-finite drop rate (in-matrix): %.2f%%",
            fallback_rate_in_matrix * 100,
            nonfinite_drop_rate * 100,
        )
        logger.info("Predict status counts: %s", status_counts)

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
                    "fallback_on_invalid": self.fallback_on_invalid,
                    "user_key_type_": self.user_key_type_,
                    "product_key_type_": self.product_key_type_,
                    "global_mean_": self.global_mean_,
                    "als_user_factor_finite_ratio_": self.als_user_factor_finite_ratio_,
                    "als_item_factor_finite_ratio_": self.als_item_factor_finite_ratio_,
                    "als_user_nonfinite_row_rate_": self.als_user_nonfinite_row_rate_,
                    "als_item_nonfinite_row_rate_": self.als_item_nonfinite_row_rate_,
                    "als_factors_sanitized_": self.als_factors_sanitized_,
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
            fallback_on_invalid=data.get("fallback_on_invalid", False),
        )

        model.model = data["model"]
        model.user_encoder = data["user_encoder"]
        model.product_encoder = data["product_encoder"]
        model.user_decoder = data["user_decoder"]
        model.product_decoder = data["product_decoder"]
        model.user_item_matrix = data["user_item_matrix"]
        model.user_factors = data.get("user_factors")
        model.item_factors = data.get("item_factors")
        model.fallback_on_invalid = data.get("fallback_on_invalid", False)
        model.user_key_type_ = data.get("user_key_type_")
        model.product_key_type_ = data.get("product_key_type_")
        model.global_mean_ = float(data.get("global_mean_", 0.0))
        model.als_user_factor_finite_ratio_ = float(data.get("als_user_factor_finite_ratio_", 1.0))
        model.als_item_factor_finite_ratio_ = float(data.get("als_item_factor_finite_ratio_", 1.0))
        model.als_user_nonfinite_row_rate_ = float(data.get("als_user_nonfinite_row_rate_", 0.0))
        model.als_item_nonfinite_row_rate_ = float(data.get("als_item_nonfinite_row_rate_", 0.0))
        model.als_factors_sanitized_ = bool(data.get("als_factors_sanitized_", False))

        logger.info("Model loaded from %s", filepath)
        return model
