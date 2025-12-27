"""
Data Loader Module
==================
Functions for loading raw data from various sources.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union


def load_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame containing the loaded data
    """
    return pd.read_csv(filepath, **kwargs)


def load_excel(filepath: Union[str, Path], sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Load data from an Excel file.
    
    Args:
        filepath: Path to the Excel file
        sheet_name: Name of the sheet to load
        **kwargs: Additional arguments passed to pd.read_excel
        
    Returns:
        DataFrame containing the loaded data
    """
    return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)


def load_transaction_data(filepath: Union[str, Path], 
                          date_column: str = 'date',
                          customer_id_column: str = 'customer_id',
                          **kwargs) -> pd.DataFrame:
    """
    Load transaction data with proper date parsing.
    
    Args:
        filepath: Path to the data file
        date_column: Name of the date column
        customer_id_column: Name of the customer ID column
        **kwargs: Additional arguments passed to the loader
        
    Returns:
        DataFrame with parsed dates and proper data types
    """
    df = load_csv(filepath, parse_dates=[date_column], **kwargs)
    df[customer_id_column] = df[customer_id_column].astype(str)
    return df


if __name__ == "__main__":
    print("Data Loader Module")
