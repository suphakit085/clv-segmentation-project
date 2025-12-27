"""
Behavioral Features Module
==========================
Functions for calculating customer behavioral features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def calculate_purchase_patterns(df: pd.DataFrame,
                                customer_id_col: str = 'customer_id',
                                date_col: str = 'date',
                                amount_col: str = 'amount') -> pd.DataFrame:
    """
    Calculate purchase pattern features for each customer.
    
    Args:
        df: Transaction DataFrame
        customer_id_col: Name of customer ID column
        date_col: Name of date column
        amount_col: Name of amount column
        
    Returns:
        DataFrame with purchase pattern features
    """
    df_sorted = df.sort_values([customer_id_col, date_col])
    
    features = df.groupby(customer_id_col).agg({
        amount_col: ['mean', 'std', 'min', 'max', 'sum'],
        date_col: ['min', 'max', 'count']
    })
    
    features.columns = [
        'avg_transaction_amount',
        'std_transaction_amount',
        'min_transaction_amount',
        'max_transaction_amount',
        'total_revenue',
        'first_purchase_date',
        'last_purchase_date',
        'transaction_count'
    ]
    
    # Calculate customer lifetime in days
    features['customer_lifetime_days'] = (
        features['last_purchase_date'] - features['first_purchase_date']
    ).dt.days
    
    # Calculate average purchase frequency (transactions per month)
    features['avg_purchase_frequency'] = features['transaction_count'] / (
        features['customer_lifetime_days'] / 30 + 1
    )
    
    return features.reset_index()


def calculate_inter_purchase_time(df: pd.DataFrame,
                                   customer_id_col: str = 'customer_id',
                                   date_col: str = 'date') -> pd.DataFrame:
    """
    Calculate inter-purchase time statistics for each customer.
    
    Args:
        df: Transaction DataFrame
        customer_id_col: Name of customer ID column
        date_col: Name of date column
        
    Returns:
        DataFrame with inter-purchase time features
    """
    df_sorted = df.sort_values([customer_id_col, date_col])
    
    # Calculate days between purchases
    df_sorted['days_since_last_purchase'] = df_sorted.groupby(customer_id_col)[date_col].diff().dt.days
    
    ipt_features = df_sorted.groupby(customer_id_col)['days_since_last_purchase'].agg([
        'mean', 'std', 'min', 'max'
    ])
    
    ipt_features.columns = [
        'avg_days_between_purchases',
        'std_days_between_purchases',
        'min_days_between_purchases',
        'max_days_between_purchases'
    ]
    
    return ipt_features.reset_index()


def calculate_product_diversity(df: pd.DataFrame,
                                customer_id_col: str = 'customer_id',
                                product_col: str = 'product_id',
                                category_col: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate product diversity metrics for each customer.
    
    Args:
        df: Transaction DataFrame
        customer_id_col: Name of customer ID column
        product_col: Name of product ID column
        category_col: Name of product category column (optional)
        
    Returns:
        DataFrame with product diversity features
    """
    features = df.groupby(customer_id_col).agg({
        product_col: 'nunique'
    })
    
    features.columns = ['unique_products_purchased']
    
    if category_col and category_col in df.columns:
        features['unique_categories'] = df.groupby(customer_id_col)[category_col].nunique()
    
    return features.reset_index()


def calculate_time_based_features(df: pd.DataFrame,
                                   customer_id_col: str = 'customer_id',
                                   date_col: str = 'date',
                                   amount_col: str = 'amount') -> pd.DataFrame:
    """
    Calculate time-based behavioral features.
    
    Args:
        df: Transaction DataFrame
        customer_id_col: Name of customer ID column
        date_col: Name of date column
        amount_col: Name of amount column
        
    Returns:
        DataFrame with time-based features
    """
    df_copy = df.copy()
    df_copy['month'] = df_copy[date_col].dt.month
    df_copy['day_of_week'] = df_copy[date_col].dt.dayofweek
    df_copy['hour'] = df_copy[date_col].dt.hour
    
    # Preferred shopping day
    preferred_day = df_copy.groupby(customer_id_col)['day_of_week'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
    )
    
    # Weekend vs weekday preference
    df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6])
    weekend_ratio = df_copy.groupby(customer_id_col)['is_weekend'].mean()
    
    features = pd.DataFrame({
        'preferred_shopping_day': preferred_day,
        'weekend_purchase_ratio': weekend_ratio
    }).reset_index()
    
    return features


if __name__ == "__main__":
    print("Behavioral Features Module")
