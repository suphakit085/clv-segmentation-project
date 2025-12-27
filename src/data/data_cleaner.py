"""
Data Cleaner Module
===================
Functions for cleaning and preprocessing raw data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataframe.
    
    Args:
        df: Input DataFrame
        subset: Column names to consider for identifying duplicates
        
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset)


def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'drop',
                          fill_value: Optional[any] = None) -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Args:
        df: Input DataFrame
        strategy: Strategy to handle missing values ('drop', 'fill', 'mean', 'median', 'mode')
        fill_value: Value to use when strategy is 'fill'
        
    Returns:
        DataFrame with missing values handled
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    elif strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def remove_outliers(df: pd.DataFrame, 
                    columns: List[str],
                    method: str = 'iqr',
                    threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        method: Method to detect outliers ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < threshold]
            
    return df_clean


def clean_transaction_data(df: pd.DataFrame,
                           amount_column: str = 'amount',
                           date_column: str = 'date') -> pd.DataFrame:
    """
    Clean transaction data by removing invalid transactions.
    
    Args:
        df: Input DataFrame with transaction data
        amount_column: Name of the amount column
        date_column: Name of the date column
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove negative amounts
    df_clean = df_clean[df_clean[amount_column] > 0]
    
    # Remove future dates
    df_clean = df_clean[df_clean[date_column] <= pd.Timestamp.now()]
    
    # Remove duplicates
    df_clean = remove_duplicates(df_clean)
    
    return df_clean


if __name__ == "__main__":
    print("Data Cleaner Module")
