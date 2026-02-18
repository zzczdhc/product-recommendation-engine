"""
Association Rule Mining using FP-Growth/Apriori
"""

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from typing import List, Tuple, Optional
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssociationRuleModel:
    """Association rule mining for product recommendations"""

    def __init__(self,
                 min_support: float = 0.003,
                 min_confidence: float = 0.1,
                 min_lift: float = 1.5,
                 max_len: int = 2):
        """
        Initialize association rule model

        Args:
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            min_lift: Minimum lift threshold
            max_len: Maximum itemset length
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_len = max_len

        self.frequent_itemsets = None
        self.rules = None

    def fit(self,
            prior: pd.DataFrame,
            top_k_products: Optional[int] = 2000) -> "AssociationRuleModel":
        """
        Fit association rule model

        Args:
            prior: DataFrame with order_id and product_id
            top_k_products: Only use top K most popular products (memory optimization)

        Returns:
            self
        """
        logger.info("Fitting Association Rule Model...")

        # ===================== 1. Build Baskets =====================
        logger.info("Building baskets...")
        baskets = prior.groupby('order_id')['product_id'].apply(list).values

        # Filter baskets with at least 2 items
        baskets = [b for b in baskets if len(b) >= 2]
        logger.info(f"Total baskets: {len(baskets):,}")
        logger.info(f"Average basket size: {np.mean([len(b) for b in baskets]):.2f}")

        # ===================== 2. Limit to Top K Products =====================
        if top_k_products:
            logger.info(f"Limiting to top {top_k_products} products...")
            product_counts = prior['product_id'].value_counts()
            top_products = set(product_counts.head(top_k_products).index)

            # Filter baskets
            baskets = [[p for p in basket if p in top_products] for basket in baskets]
            baskets = [b for b in baskets if len(b) >= 2]
            logger.info(f"Filtered baskets: {len(baskets):,}")

        # ===================== 3. Transaction Encoding =====================
        logger.info("Encoding transactions...")
        te = TransactionEncoder()
        te_array = te.fit(baskets).transform(baskets)
        basket_df = pd.DataFrame(te_array, columns=te.columns_)

        # Convert to string columns (required by mlxtend sometimes when IDs are ints)
        basket_df.columns = basket_df.columns.astype(str)

        logger.info(f"Encoded shape: {basket_df.shape}")
        sparsity = 1 - (basket_df.sum().sum() / (basket_df.shape[0] * basket_df.shape[1]))
        logger.info(f"Sparsity: {sparsity:.2%}")

        # ===================== 4. Mine Frequent Itemsets =====================
        logger.info(f"Mining frequent itemsets (min_support={self.min_support})...")
        self.frequent_itemsets = fpgrowth(
            basket_df,
            min_support=self.min_support,
            use_colnames=True,
            max_len=self.max_len,
            verbose=0
        )

        logger.info(f"Found {len(self.frequent_itemsets):,} frequent itemsets")

        if len(self.frequent_itemsets) > 0:
            self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(len)
            logger.info("Itemset distribution:")
            logger.info(self.frequent_itemsets['length'].value_counts().sort_index())

        # ===================== 5. Generate Association Rules =====================
        if self.frequent_itemsets is not None and len(self.frequent_itemsets) > 0:
            logger.info(
                f"Generating rules (min_confidence={self.min_confidence}, min_lift={self.min_lift})..."
            )

            self.rules = association_rules(
                self.frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence
            )

            # Filter by lift
            self.rules = self.rules[self.rules['lift'] >= self.min_lift].copy()

            # Add lengths
            self.rules['ante_len'] = self.rules['antecedents'].apply(len)
            self.rules['cons_len'] = self.rules['consequents'].apply(len)

            logger.info(f"Generated {len(self.rules):,} rules")

            rule_dist = self.rules.groupby(['ante_len', 'cons_len']).size()
            logger.info(f"Rule distribution:\n{rule_dist}")
        else:
            logger.warning("No frequent itemsets found. Try lowering min_support.")
            self.rules = pd.DataFrame()

        return self

    def get_recommendations(self,
                            product_id: int,
                            top_n: int = 5,
                            only_pairs: bool = True) -> List[Tuple[int, float]]:
        """
        Get recommendations for a product

        Args:
            product_id: Product ID
            top_n: Number of recommendations
            only_pairs: Only return 1-to-1 rules

        Returns:
            List of (product_id, lift) tuples
        """
        if self.rules is None or len(self.rules) == 0:
            return []

        rules_subset = self.rules.copy()

        if only_pairs:
            rules_subset = rules_subset[
                (rules_subset['ante_len'] == 1) & (rules_subset['cons_len'] == 1)
            ]

        # Find rules where product_id is in antecedent
        product_str = str(product_id)
        mask = rules_subset['antecedents'].apply(lambda x: product_str in [str(i) for i in x])
        matched_rules = rules_subset[mask]

        if len(matched_rules) == 0:
            return []

        # Extract recommendations
        recommendations = []
        for _, row in matched_rules.iterrows():
            consequents = [int(x) for x in row['consequents']]
            lift = row['lift']

            for cons_product in consequents:
                if cons_product != product_id:
                    recommendations.append((cons_product, lift))

        # Sort by lift and return top N
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
        return recommendations

    def get_basket_recommendations(self,
                                   product_ids: List[int],
                                   top_n: int = 5,
                                   require_full_match: bool = False,
                                   min_overlap: int = 1,
                                   overlap_weighted: bool = True) -> List[Tuple[int, float]]:
        """
        Get recommendations for a basket of products

        Args:
            product_ids: List of product IDs in basket
            top_n: Number of recommendations
            require_full_match: If True, antecedent must be a subset of basket
            min_overlap: Minimum overlap size between basket and antecedent
            overlap_weighted: If True, down-weight partial matches by overlap ratio

        Returns:
            List of (product_id, lift) tuples
        """
        if self.rules is None or len(self.rules) == 0:
            return []

        product_strs = set(str(p) for p in product_ids)
        min_overlap = max(1, int(min_overlap))

        def overlap_size(antecedents) -> int:
            ante_strs = set(str(i) for i in antecedents)
            overlap = len(ante_strs & product_strs)
            if require_full_match:
                return overlap if ante_strs.issubset(product_strs) else 0
            return overlap if overlap >= min_overlap else 0

        overlap_series = self.rules['antecedents'].apply(overlap_size)
        matched_rules = self.rules[overlap_series > 0].copy()
        if len(matched_rules) > 0:
            matched_rules['overlap'] = overlap_series[overlap_series > 0].astype(float).values

        if len(matched_rules) == 0:
            return []

        # Aggregate recommendations: keep max score for each consequent
        recommendations = {}
        for _, row in matched_rules.iterrows():
            consequents = [int(x) for x in row['consequents']]
            lift = float(row['lift'])
            overlap = float(row.get('overlap', 1.0))
            ante_len = max(1.0, float(row.get('ante_len', len(row['antecedents']))))
            score = lift * (overlap / ante_len) if overlap_weighted else lift

            for cons_product in consequents:
                if cons_product not in product_ids:
                    if cons_product in recommendations:
                        recommendations[cons_product] = max(recommendations[cons_product], score)
                    else:
                        recommendations[cons_product] = score

        # Sort and return top N
        recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return recommendations

    def save(self, filepath: str):
        """Save model to disk"""
        with open(filepath, "wb") as f:
            pickle.dump({
                'min_support': self.min_support,
                'min_confidence': self.min_confidence,
                'min_lift': self.min_lift,
                'max_len': self.max_len,
                'frequent_itemsets': self.frequent_itemsets,
                'rules': self.rules
            }, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AssociationRuleModel":
        """Load model from disk"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        model = cls(
            min_support=data['min_support'],
            min_confidence=data['min_confidence'],
            min_lift=data['min_lift'],
            max_len=data['max_len']
        )
        model.frequent_itemsets = data['frequent_itemsets']
        model.rules = data['rules']

        logger.info(f"Model loaded from {filepath}")
        return model
