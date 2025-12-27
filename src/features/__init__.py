"""
Features Module
===============
Feature engineering utilities for CLV modeling.
"""

from .rfm_features import calculate_rfm, calculate_rfm_scores, segment_rfm
from .behavioral_features import (
    calculate_purchase_patterns,
    calculate_inter_purchase_time,
    calculate_product_diversity,
    calculate_time_based_features
)
from .feature_engineering import (
    create_feature_matrix,
    scale_features,
    create_lag_features,
    create_rolling_features,
    handle_categorical_features
)

__all__ = [
    'calculate_rfm',
    'calculate_rfm_scores',
    'segment_rfm',
    'calculate_purchase_patterns',
    'calculate_inter_purchase_time',
    'calculate_product_diversity',
    'calculate_time_based_features',
    'create_feature_matrix',
    'scale_features',
    'create_lag_features',
    'create_rolling_features',
    'handle_categorical_features'
]
