"""
Feature Engineering Module
==========================
Functions for creating and transforming features for CLV modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Optional, Tuple


def create_feature_matrix(rfm_features: pd.DataFrame,
                          behavioral_features: pd.DataFrame,
                          customer_id_col: str = 'customer_id') -> pd.DataFrame:
    """
    Combine multiple feature DataFrames into a single feature matrix.
    
    Args:
        rfm_features: DataFrame with RFM features
        behavioral_features: DataFrame with behavioral features
        customer_id_col: Name of customer ID column
        
    Returns:
        Combined feature matrix
    """
    feature_matrix = rfm_features.merge(
        behavioral_features, 
        on=customer_id_col, 
        how='outer'
    )
    
    return feature_matrix


def scale_features(df: pd.DataFrame,
                   columns: List[str],
                   method: str = 'standard') -> Tuple[pd.DataFrame, object]:
    """
    Scale numerical features.
    
    Args:
        df: Input DataFrame
        columns: Columns to scale
        method: Scaling method ('standard', 'minmax')
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    df_scaled = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    return df_scaled, scaler


def create_lag_features(df: pd.DataFrame,
                        customer_id_col: str = 'customer_id',
                        date_col: str = 'date',
                        value_cols: List[str] = None,
                        lag_periods: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """
    Create lag features for time series analysis.
    
    Args:
        df: Input DataFrame sorted by date
        customer_id_col: Name of customer ID column
        date_col: Name of date column
        value_cols: Columns to create lag features for
        lag_periods: List of lag periods
        
    Returns:
        DataFrame with lag features added
    """
    df_lagged = df.copy()
    df_lagged = df_lagged.sort_values([customer_id_col, date_col])
    
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in value_cols:
        for lag in lag_periods:
            df_lagged[f'{col}_lag_{lag}'] = df_lagged.groupby(customer_id_col)[col].shift(lag)
    
    return df_lagged


def create_rolling_features(df: pd.DataFrame,
                            customer_id_col: str = 'customer_id',
                            date_col: str = 'date',
                            value_col: str = 'amount',
                            windows: List[int] = [3, 7, 30]) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        df: Input DataFrame sorted by date
        customer_id_col: Name of customer ID column
        date_col: Name of date column
        value_col: Column to compute rolling features for
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling features added
    """
    df_rolling = df.copy()
    df_rolling = df_rolling.sort_values([customer_id_col, date_col])
    
    for window in windows:
        df_rolling[f'{value_col}_rolling_mean_{window}'] = df_rolling.groupby(customer_id_col)[value_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df_rolling[f'{value_col}_rolling_sum_{window}'] = df_rolling.groupby(customer_id_col)[value_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum()
        )
    
    return df_rolling


def handle_categorical_features(df: pd.DataFrame,
                                categorical_cols: List[str],
                                method: str = 'onehot') -> pd.DataFrame:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical columns
        method: Encoding method ('onehot', 'label')
        
    Returns:
        DataFrame with encoded categorical features
    """
    df_encoded = df.copy()
    
    if method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    elif method == 'label':
        for col in categorical_cols:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return df_encoded


if __name__ == "__main__":
    print("Feature Engineering Module")
