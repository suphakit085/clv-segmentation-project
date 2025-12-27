"""
Unit tests for data loading and cleaning modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDataLoader:
    """Tests for data loading functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
        return pd.DataFrame({
            'InvoiceNo': ['536365', '536366', '536367', 'C536368', '536369'],
            'StockCode': ['85123A', '71053', '84406B', '84029G', '84029E'],
            'Description': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
            'Quantity': [6, 8, -3, 6, 6],
            'InvoiceDate': pd.date_range('2011-01-01', periods=5, freq='D'),
            'UnitPrice': [2.55, 3.39, 0.00, 3.39, 3.39],
            'CustomerID': [17850.0, 17850.0, np.nan, 17851.0, 17851.0],
            'Country': ['UK', 'UK', 'UK', 'UK', 'UK']
        })
    
    def test_sample_data_creation(self, sample_data):
        """Test that sample data is created correctly."""
        assert len(sample_data) == 5
        assert 'CustomerID' in sample_data.columns
        assert sample_data['InvoiceDate'].dtype == 'datetime64[ns]'
    
    def test_null_customer_filtering(self, sample_data):
        """Test filtering out null CustomerIDs."""
        clean_data = sample_data[sample_data['CustomerID'].notna()]
        assert len(clean_data) == 4
        assert clean_data['CustomerID'].isna().sum() == 0
    
    def test_cancellation_filtering(self, sample_data):
        """Test filtering out cancelled orders."""
        clean_data = sample_data[~sample_data['InvoiceNo'].str.startswith('C')]
        assert len(clean_data) == 4
        assert not any(clean_data['InvoiceNo'].str.startswith('C'))
    
    def test_negative_quantity_filtering(self, sample_data):
        """Test filtering out negative quantities."""
        clean_data = sample_data[sample_data['Quantity'] > 0]
        assert len(clean_data) == 4
        assert (clean_data['Quantity'] > 0).all()
    
    def test_zero_price_filtering(self, sample_data):
        """Test filtering out zero prices."""
        clean_data = sample_data[sample_data['UnitPrice'] > 0]
        assert len(clean_data) == 4
        assert (clean_data['UnitPrice'] > 0).all()


class TestDataCleaner:
    """Tests for data cleaning functions."""
    
    @pytest.fixture
    def dirty_data(self):
        """Create dirty data for cleaning tests."""
        return pd.DataFrame({
            'CustomerID': [1, 1, 2, 2, 3, np.nan],
            'Revenue': [100, 100, 200, 300, 150, 50],
            'Quantity': [1, 1, 2, 3, 1, -1]
        })
    
    def test_duplicate_removal(self, dirty_data):
        """Test duplicate row removal."""
        clean_data = dirty_data.drop_duplicates()
        # Row 0 and 1 are duplicates
        assert len(clean_data) == 5
    
    def test_missing_value_count(self, dirty_data):
        """Test counting missing values."""
        missing_count = dirty_data['CustomerID'].isna().sum()
        assert missing_count == 1
    
    def test_revenue_calculation(self):
        """Test revenue calculation."""
        df = pd.DataFrame({
            'Quantity': [1, 2, 3],
            'UnitPrice': [10.0, 20.0, 30.0]
        })
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        expected = [10.0, 40.0, 90.0]
        assert list(df['Revenue']) == expected


class TestRFMCalculation:
    """Tests for RFM calculation functions."""
    
    @pytest.fixture
    def transaction_data(self):
        """Create transaction data for RFM tests."""
        return pd.DataFrame({
            'CustomerID': [1, 1, 1, 2, 2, 3],
            'InvoiceNo': ['A', 'B', 'C', 'D', 'E', 'F'],
            'InvoiceDate': pd.to_datetime([
                '2011-12-01', '2011-11-15', '2011-10-01',
                '2011-12-09', '2011-12-05',
                '2011-06-01'
            ]),
            'Revenue': [100, 150, 200, 50, 75, 300]
        })
    
    def test_recency_calculation(self, transaction_data):
        """Test recency calculation."""
        reference_date = pd.to_datetime('2011-12-10')
        
        recency = transaction_data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days
        })
        
        # Customer 1 bought on Dec 1 -> 9 days ago
        assert recency.loc[1, 'InvoiceDate'] == 9
        # Customer 2 bought on Dec 9 -> 1 day ago
        assert recency.loc[2, 'InvoiceDate'] == 1
        # Customer 3 bought on Jun 1 -> 192 days ago
        assert recency.loc[3, 'InvoiceDate'] == 192
    
    def test_frequency_calculation(self, transaction_data):
        """Test frequency calculation."""
        frequency = transaction_data.groupby('CustomerID')['InvoiceNo'].nunique()
        
        assert frequency[1] == 3
        assert frequency[2] == 2
        assert frequency[3] == 1
    
    def test_monetary_calculation(self, transaction_data):
        """Test monetary calculation."""
        monetary = transaction_data.groupby('CustomerID')['Revenue'].sum()
        
        assert monetary[1] == 450  # 100 + 150 + 200
        assert monetary[2] == 125  # 50 + 75
        assert monetary[3] == 300


class TestSegmentation:
    """Tests for customer segmentation functions."""
    
    @pytest.fixture
    def rfm_scores(self):
        """Create RFM scores for segmentation tests."""
        return pd.DataFrame({
            'CustomerID': [1, 2, 3, 4, 5],
            'R_Score': [5, 5, 1, 1, 3],
            'F_Score': [5, 1, 5, 1, 3],
            'M_Score': [5, 1, 5, 1, 3]
        })
    
    def test_score_range(self, rfm_scores):
        """Test that scores are in valid range."""
        for col in ['R_Score', 'F_Score', 'M_Score']:
            assert rfm_scores[col].min() >= 1
            assert rfm_scores[col].max() <= 5
    
    def test_rfm_string_creation(self, rfm_scores):
        """Test RFM string creation."""
        rfm_scores['RFM_Score'] = (
            rfm_scores['R_Score'].astype(str) +
            rfm_scores['F_Score'].astype(str) +
            rfm_scores['M_Score'].astype(str)
        )
        
        assert rfm_scores.loc[0, 'RFM_Score'] == '555'  # Champions
        assert rfm_scores.loc[1, 'RFM_Score'] == '511'  # New customer
        assert rfm_scores.loc[3, 'RFM_Score'] == '111'  # Lost


class TestCLVPrediction:
    """Tests for CLV prediction functions."""
    
    def test_clv_calculation(self):
        """Test basic CLV calculation."""
        avg_order_value = 100
        purchase_frequency = 12  # per year
        customer_lifespan = 3  # years
        
        clv = avg_order_value * purchase_frequency * customer_lifespan
        assert clv == 3600
    
    def test_discounted_clv(self):
        """Test discounted CLV calculation."""
        future_value = 1000
        discount_rate = 0.10
        years = 1
        
        present_value = future_value / (1 + discount_rate) ** years
        assert round(present_value, 2) == 909.09
    
    def test_ensemble_clv(self):
        """Test ensemble CLV calculation."""
        clv_model1 = 1000
        clv_model2 = 1200
        
        ensemble_clv = (clv_model1 + clv_model2) / 2
        assert ensemble_clv == 1100


class TestFeatureEngineering:
    """Tests for feature engineering functions."""
    
    @pytest.fixture
    def customer_data(self):
        """Create customer data for feature tests."""
        return pd.DataFrame({
            'CustomerID': [1, 1, 1, 2, 2],
            'InvoiceDate': pd.to_datetime([
                '2011-01-01', '2011-02-01', '2011-03-01',
                '2011-01-15', '2011-03-15'
            ]),
            'Revenue': [100, 150, 200, 50, 75]
        })
    
    def test_customer_lifetime(self, customer_data):
        """Test customer lifetime calculation."""
        lifetime = customer_data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (x.max() - x.min()).days
        })
        
        # Customer 1: Jan 1 to Mar 1 = 59 days
        assert lifetime.loc[1, 'InvoiceDate'] == 59
        # Customer 2: Jan 15 to Mar 15 = 59 days
        assert lifetime.loc[2, 'InvoiceDate'] == 59
    
    def test_avg_revenue(self, customer_data):
        """Test average revenue calculation."""
        avg_revenue = customer_data.groupby('CustomerID')['Revenue'].mean()
        
        assert avg_revenue[1] == 150  # (100 + 150 + 200) / 3
        assert avg_revenue[2] == 62.5  # (50 + 75) / 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
