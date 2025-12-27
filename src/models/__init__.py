"""
Models Module
=============
CLV prediction and customer segmentation models.
"""

from .clv_probabilistic import BGNBDModel, GammaGammaModel, calculate_clv
from .clv_ml import CLVPredictor, train_test_clv_model, compare_models
from .segmentation import (
    CustomerSegmentation,
    find_optimal_clusters,
    create_segment_labels
)

__all__ = [
    'BGNBDModel',
    'GammaGammaModel',
    'calculate_clv',
    'CLVPredictor',
    'train_test_clv_model',
    'compare_models',
    'CustomerSegmentation',
    'find_optimal_clusters',
    'create_segment_labels'
]
