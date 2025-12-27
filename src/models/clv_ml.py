"""
CLV Machine Learning Models Module
==================================
Machine learning models for Customer Lifetime Value prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings


class CLVPredictor:
    """
    Machine learning-based CLV predictor.
    Supports multiple regression algorithms.
    """
    
    MODELS = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor
    }
    
    def __init__(self, model_type: str = 'random_forest', **model_params):
        """
        Initialize the CLV predictor.
        
        Args:
            model_type: Type of model to use
            **model_params: Additional parameters for the model
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(self.MODELS.keys())}")
        
        self.model_type = model_type
        self.model = self.MODELS[model_type](**model_params)
        self.feature_names = None
        self.fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CLVPredictor':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix
            y: Target variable (CLV)
            
        Returns:
            Self
        """
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict CLV for new customers.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted CLV values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True CLV values
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        return {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions)
        }
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                       cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error')
        
        return {
            'mean_cv_rmse': np.sqrt(-scores.mean()),
            'std_cv_rmse': np.sqrt(-scores).std()
        }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (if available).
        
        Returns:
            DataFrame with feature importance or None
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        elif hasattr(self.model, 'coef_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': self.model.coef_
            }).sort_values('coefficient', key=abs, ascending=False)
            return importance
        
        return None


def train_test_clv_model(X: pd.DataFrame, 
                         y: pd.Series,
                         model_type: str = 'random_forest',
                         test_size: float = 0.2,
                         random_state: int = 42,
                         **model_params) -> Tuple[CLVPredictor, Dict[str, float]]:
    """
    Train and test a CLV model with train/test split.
    
    Args:
        X: Feature matrix
        y: Target variable
        model_type: Type of model to use
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        **model_params: Additional model parameters
        
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = CLVPredictor(model_type=model_type, **model_params)
    model.fit(X_train, y_train)
    
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics


def compare_models(X: pd.DataFrame, 
                   y: pd.Series,
                   models: List[str] = None,
                   cv: int = 5) -> pd.DataFrame:
    """
    Compare multiple models using cross-validation.
    
    Args:
        X: Feature matrix
        y: Target variable
        models: List of model types to compare
        cv: Number of cross-validation folds
        
    Returns:
        DataFrame with comparison results
    """
    if models is None:
        models = ['linear', 'ridge', 'random_forest', 'gradient_boosting']
    
    results = []
    
    for model_type in models:
        try:
            predictor = CLVPredictor(model_type=model_type)
            cv_scores = predictor.cross_validate(X, y, cv=cv)
            results.append({
                'model': model_type,
                'mean_rmse': cv_scores['mean_cv_rmse'],
                'std_rmse': cv_scores['std_cv_rmse']
            })
        except Exception as e:
            warnings.warn(f"Failed to evaluate {model_type}: {str(e)}")
    
    return pd.DataFrame(results).sort_values('mean_rmse')


if __name__ == "__main__":
    print("CLV Machine Learning Models Module")
