"""
Evaluation Metrics Module
=========================
Metrics for evaluating CLV models.
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def calculate_decile_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Perform decile analysis."""
    df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
    df['decile'] = pd.qcut(df['predicted'], q=10, labels=range(1, 11), duplicates='drop')
    return df.groupby('decile').agg({'actual': ['mean', 'sum', 'count'], 'predicted': ['mean', 'sum']})


if __name__ == "__main__":
    print("Evaluation Metrics Module")
