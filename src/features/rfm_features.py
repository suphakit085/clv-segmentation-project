"""
RFM Features Module
===================
Functions for calculating Recency, Frequency, and Monetary features.
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime


def calculate_rfm(df: pd.DataFrame,
                  customer_id_col: str = 'customer_id',
                  date_col: str = 'date',
                  amount_col: str = 'amount',
                  reference_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
    
    Args:
        df: Transaction DataFrame
        customer_id_col: Name of customer ID column
        date_col: Name of date column
        amount_col: Name of amount column
        reference_date: Reference date for recency calculation (defaults to max date in data)
        
    Returns:
        DataFrame with RFM metrics per customer
    """
    if reference_date is None:
        reference_date = df[date_col].max()
    
    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,  # Recency
        customer_id_col: 'count',  # Frequency (using customer_id as proxy for transaction count)
        amount_col: 'sum'  # Monetary
    })
    
    # Rename columns properly
    rfm.columns = ['recency', 'frequency', 'monetary']
    
    # Fix frequency calculation
    rfm['frequency'] = df.groupby(customer_id_col)[date_col].nunique()
    
    return rfm.reset_index()


def calculate_rfm_scores(rfm_df: pd.DataFrame,
                         r_col: str = 'recency',
                         f_col: str = 'frequency',
                         m_col: str = 'monetary',
                         n_bins: int = 5) -> pd.DataFrame:
    """
    Calculate RFM scores using quantile-based binning.
    
    Args:
        rfm_df: DataFrame with RFM metrics
        r_col: Name of recency column
        f_col: Name of frequency column
        m_col: Name of monetary column
        n_bins: Number of bins for scoring (default 5)
        
    Returns:
        DataFrame with RFM scores added
    """
    df = rfm_df.copy()
    
    # Recency: lower is better, so we reverse the scoring
    df['R_score'] = pd.qcut(df[r_col], q=n_bins, labels=range(n_bins, 0, -1), duplicates='drop')
    
    # Frequency: higher is better
    df['F_score'] = pd.qcut(df[f_col].rank(method='first'), q=n_bins, labels=range(1, n_bins + 1), duplicates='drop')
    
    # Monetary: higher is better
    df['M_score'] = pd.qcut(df[m_col].rank(method='first'), q=n_bins, labels=range(1, n_bins + 1), duplicates='drop')
    
    # Convert to int
    for col in ['R_score', 'F_score', 'M_score']:
        df[col] = df[col].astype(int)
    
    # Calculate combined RFM score
    df['RFM_score'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)
    df['RFM_total'] = df['R_score'] + df['F_score'] + df['M_score']
    
    return df


def segment_rfm(rfm_scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign customer segments based on RFM scores.
    
    Args:
        rfm_scored_df: DataFrame with RFM scores
        
    Returns:
        DataFrame with segment labels added
    """
    df = rfm_scored_df.copy()
    
    def assign_segment(row):
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        
        if r >= 4 and f >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r <= 2 and f >= 4:
            return 'At Risk'
        elif r <= 2 and f <= 2:
            return 'Lost'
        elif r >= 3 and f == 1:
            return 'Promising'
        elif r == 3 and f >= 2:
            return 'Need Attention'
        else:
            return 'Others'
    
    df['segment'] = df.apply(assign_segment, axis=1)
    
    return df


if __name__ == "__main__":
    print("RFM Features Module")
