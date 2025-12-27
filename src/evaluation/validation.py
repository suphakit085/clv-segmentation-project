"""
Model Validation Module
=======================
Cross-validation and model validation utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold, TimeSeriesSplit


def time_series_cv(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
    """Perform time series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    
    return {'mean_rmse': np.mean(scores), 'std_rmse': np.std(scores)}


def k_fold_cv(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
    """Perform K-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    
    return {'mean_rmse': np.mean(scores), 'std_rmse': np.std(scores)}


if __name__ == "__main__":
    print("Model Validation Module")
