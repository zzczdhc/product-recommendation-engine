"""
Data preprocessing utilities for product recommendation engine
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess Instacart data for recommendation models"""

    def __init__(self,
                 min_user_orders: int = 5,
                 min_product_orders: int = 20,
                 max_basket_size: int = 100):
        """
        Initialize preprocessor

        Args:
            min_user_orders: Minimum orders per user to keep
            min_product_orders: Minimum orders per product to keep
            max_basket_size: Maximum basket size (filter outliers)
        """
        self.min_user_orders = min_user_orders
        self.min_product_orders = min_product_orders
        self.max_basket_size = max_basket_size

        self.active_users = None
        self.popular_products = None
        self.stats = {}

    def fit(self, orders: pd.DataFrame, prior: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessor to data (identify users and products to keep)

        Args:
            orders: Orders dataframe
            prior: Prior order products dataframe

        Returns:
            self
        """
        logger.info("Fitting preprocessor...")

        # Get prior orders only
        prior_orders = orders[orders['eval_set'] == 'prior'][['order_id', 'user_id']]

        # Count orders per user
        user_order_counts = prior_orders.groupby('user_id')['order_id'].nunique()
        self.active_users = set(user_order_counts[user_order_counts >= self.min_user_orders].index)

        # Count orders per product
        product_order_counts = prior.groupby('product_id')['order_id'].nunique()
        self.popular_products = set(product_order_counts[product_order_counts >= self.min_product_orders].index)

        # Store statistics
        self.stats = {
            'total_users': len(user_order_counts),
            'active_users': len(self.active_users),
            'user_retention_rate': len(self.active_users) / len(user_order_counts) if len(user_order_counts) > 0 else 0.0,
            'total_products': len(product_order_counts),
            'popular_products': len(self.popular_products),
            'product_retention_rate': len(self.popular_products) / len(product_order_counts) if len(product_order_counts) > 0 else 0.0
        }

        logger.info(
            f"Active users: {self.stats['active_users']:,} / {self.stats['total_users']:,} "
            f"({self.stats['user_retention_rate']:.1%})"
        )
        logger.info(
            f"Popular products: {self.stats['popular_products']:,} / {self.stats['total_products']:,} "
            f"({self.stats['product_retention_rate']:.1%})"
        )

        return self

    def transform(self,
                  orders: pd.DataFrame,
                  prior: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform data (filter and clean)

        Args:
            orders: Orders dataframe
            prior: Prior order products dataframe

        Returns:
            Tuple of (filtered_orders, filtered_prior)
        """
        if self.active_users is None or self.popular_products is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        logger.info("Transforming data...")

        # Filter orders
        orders_filtered = orders[orders['user_id'].isin(self.active_users)].copy()

        # Filter prior products to orders we keep + popular products
        prior_filtered = prior[
            (prior['order_id'].isin(orders_filtered['order_id'])) &
            (prior['product_id'].isin(self.popular_products))
        ].copy()

        # Filter baskets by size
        basket_sizes = prior_filtered.groupby('order_id').size()
        valid_orders = basket_sizes[
            (basket_sizes >= 2) & (basket_sizes <= self.max_basket_size)
        ].index

        orders_filtered = orders_filtered[orders_filtered['order_id'].isin(valid_orders)].copy()
        prior_filtered = prior_filtered[prior_filtered['order_id'].isin(valid_orders)].copy()

        logger.info(f"Filtered orders: {len(orders_filtered):,}")
        logger.info(f"Filtered transactions: {len(prior_filtered):,}")

        return orders_filtered, prior_filtered

    def fit_transform(self,
                      orders: pd.DataFrame,
                      prior: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit and transform in one step"""
        return self.fit(orders, prior).transform(orders, prior)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in dataframe

    Strategy:
    - days_since_prior_order: Fill with median (or 30 for first order)
    - Other numerical: Fill with median
    - Categorical: Fill with 'Unknown'
    """
    df = df.copy()

    # Handle days_since_prior_order specially (NaN means first order)
    if 'days_since_prior_order' in df.columns:
        # Fill NaN with 30 (assume monthly for first order)
        df['days_since_prior_order'].fillna(30, inplace=True)

    # Fill other numerical columns with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Filled {col} missing values with median: {median_val}")

    # Fill categorical columns with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna('Unknown', inplace=True)
            logger.info(f"Filled {col} missing values with 'Unknown'")

    return df


def create_implicit_ratings(orders: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """
    Create implicit rating matrix from reorder behavior

    Rating = (# times user reordered product) / (# total orders by user-product)

    Returns:
        DataFrame with columns: user_id, product_id, final_rating
    """
    logger.info("Creating implicit ratings from reorder behavior...")

    # Get prior orders with user_id
    prior_orders = orders[orders['eval_set'] == 'prior'][['order_id', 'user_id']]
    prior_with_user = prior.merge(prior_orders, on='order_id')

    # Count reorders per (user, product)
    user_product_reorders = (
        prior_with_user.groupby(['user_id', 'product_id'])['reordered']
        .sum()
        .reset_index(name='reorder_count')
    )

    # Count total orders per (user, product)
    user_product_orders = (
        prior_with_user.groupby(['user_id', 'product_id'])['order_id']
        .nunique()
        .reset_index(name='order_count')
    )

    # Merge
    ratings = user_product_reorders.merge(user_product_orders, on=['user_id', 'product_id'])

    # Calculate implicit rating
    # Rating = reorder_count / order_count (ranges from 0 to 1)
    ratings['implicit_rating'] = ratings['reorder_count'] / ratings['order_count']

    # Alternative: Add purchase frequency as well
    user_total_orders = (
        prior_orders.groupby('user_id')['order_id']
        .nunique()
        .reset_index(name='user_total_orders')
    )

    ratings = ratings.merge(user_total_orders, on='user_id')
    ratings['purchase_frequency'] = ratings['order_count'] / ratings['user_total_orders']

    # Final rating: weighted combination
    # 70% reorder rate + 30% purchase frequency
    ratings['final_rating'] = (
        0.7 * ratings['implicit_rating'] +
        0.3 * ratings['purchase_frequency']
    )

    logger.info(f"Created {len(ratings):,} implicit ratings")
    logger.info(f"Rating distribution:\n{ratings['final_rating'].describe()}")

    return ratings[['user_id', 'product_id', 'final_rating']]