"""
Unit tests for feature engineering and RFM modules.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestRFMFeatures:
    """Tests for RFM feature calculation."""
    
    @pytest.fixture
    def transactions(self):
        """Create sample transactions."""
        np.random.seed(42)
        n_customers = 100
        n_transactions = 500
        
        customer_ids = np.random.choice(range(1, n_customers + 1), n_transactions)
        dates = pd.date_range('2011-01-01', periods=365, freq='D')
        invoice_dates = np.random.choice(dates, n_transactions)
        revenues = np.random.exponential(50, n_transactions)
        
        return pd.DataFrame({
            'CustomerID': customer_ids,
            'InvoiceNo': [f'INV{i}' for i in range(n_transactions)],
            'InvoiceDate': invoice_dates,
            'Revenue': revenues
        })
    
    def test_rfm_dataframe_creation(self, transactions):
        """Test RFM dataframe creation."""
        reference_date = transactions['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        rfm = transactions.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'Revenue': 'sum'
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        assert len(rfm) == transactions['CustomerID'].nunique()
        assert 'Recency' in rfm.columns
        assert 'Frequency' in rfm.columns
        assert 'Monetary' in rfm.columns
    
    def test_recency_is_positive(self, transactions):
        """Test that recency is always positive."""
        reference_date = transactions['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        rfm = transactions.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days
        })
        
        assert (rfm['InvoiceDate'] >= 0).all()
    
    def test_frequency_is_positive(self, transactions):
        """Test that frequency is always positive."""
        frequency = transactions.groupby('CustomerID')['InvoiceNo'].nunique()
        assert (frequency >= 1).all()
    
    def test_monetary_is_positive(self, transactions):
        """Test that monetary is always positive."""
        monetary = transactions.groupby('CustomerID')['Revenue'].sum()
        assert (monetary > 0).all()


class TestRFMScoring:
    """Tests for RFM scoring functions."""
    
    @pytest.fixture
    def rfm_data(self):
        """Create RFM data for scoring."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'CustomerID': range(1, n + 1),
            'Recency': np.random.randint(1, 365, n),
            'Frequency': np.random.randint(1, 20, n),
            'Monetary': np.random.exponential(500, n)
        })
    
    def test_quintile_scoring(self, rfm_data):
        """Test quintile-based scoring."""
        rfm_data['R_Score'] = pd.qcut(
            rfm_data['Recency'], 
            q=5, 
            labels=[5, 4, 3, 2, 1],
            duplicates='drop'
        )
        
        # Check score distribution
        score_counts = rfm_data['R_Score'].value_counts()
        assert len(score_counts) <= 5
    
    def test_score_range_validation(self, rfm_data):
        """Test that scores are in valid range."""
        rfm_data['R_Score'] = pd.qcut(
            rfm_data['Recency'].rank(method='first'), 
            q=5, 
            labels=[5, 4, 3, 2, 1]
        ).astype(int)
        
        assert rfm_data['R_Score'].min() >= 1
        assert rfm_data['R_Score'].max() <= 5


class TestBehavioralFeatures:
    """Tests for behavioral feature engineering."""
    
    @pytest.fixture
    def order_data(self):
        """Create order data."""
        return pd.DataFrame({
            'CustomerID': [1, 1, 1, 2, 2, 2, 3],
            'OrderID': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            'OrderDate': pd.to_datetime([
                '2011-01-01', '2011-02-01', '2011-03-01',
                '2011-01-15', '2011-02-15', '2011-03-15',
                '2011-06-01'
            ]),
            'Quantity': [5, 3, 7, 2, 4, 6, 10],
            'Revenue': [50, 30, 70, 20, 40, 60, 100]
        })
    
    def test_items_per_order(self, order_data):
        """Test items per order calculation."""
        items_per_order = order_data.groupby('CustomerID')['Quantity'].mean()
        
        assert items_per_order[1] == 5  # (5 + 3 + 7) / 3
        assert items_per_order[2] == 4  # (2 + 4 + 6) / 3
        assert items_per_order[3] == 10
    
    def test_avg_order_value(self, order_data):
        """Test average order value calculation."""
        aov = order_data.groupby('CustomerID')['Revenue'].mean()
        
        assert aov[1] == 50  # (50 + 30 + 70) / 3
        assert aov[2] == 40  # (20 + 40 + 60) / 3
        assert aov[3] == 100
    
    def test_order_std(self, order_data):
        """Test order value standard deviation."""
        order_std = order_data.groupby('CustomerID')['Revenue'].std(ddof=0)
        
        # Customer 1: [50, 30, 70] -> std = 16.33
        assert round(order_std[1], 2) == 16.33


class TestInterPurchaseTime:
    """Tests for inter-purchase time calculations."""
    
    @pytest.fixture
    def purchase_data(self):
        """Create purchase data with dates."""
        return pd.DataFrame({
            'CustomerID': [1, 1, 1, 2, 2],
            'PurchaseDate': pd.to_datetime([
                '2011-01-01', '2011-01-15', '2011-02-15',
                '2011-01-01', '2011-03-01'
            ])
        }).sort_values(['CustomerID', 'PurchaseDate'])
    
    def test_days_between_purchases(self, purchase_data):
        """Test days between purchases calculation."""
        purchase_data['PrevPurchase'] = purchase_data.groupby('CustomerID')['PurchaseDate'].shift(1)
        purchase_data['DaysSinceLast'] = (
            purchase_data['PurchaseDate'] - purchase_data['PrevPurchase']
        ).dt.days
        
        # Customer 1: Jan 1 -> Jan 15 = 14 days, Jan 15 -> Feb 15 = 31 days
        customer1 = purchase_data[purchase_data['CustomerID'] == 1]['DaysSinceLast'].dropna()
        assert list(customer1) == [14, 31]
    
    def test_avg_inter_purchase_time(self, purchase_data):
        """Test average inter-purchase time."""
        purchase_data['PrevPurchase'] = purchase_data.groupby('CustomerID')['PurchaseDate'].shift(1)
        purchase_data['DaysSinceLast'] = (
            purchase_data['PurchaseDate'] - purchase_data['PrevPurchase']
        ).dt.days
        
        avg_ipt = purchase_data.groupby('CustomerID')['DaysSinceLast'].mean()
        
        # Customer 1: (14 + 31) / 2 = 22.5
        assert avg_ipt[1] == 22.5
        # Customer 2: only one gap of 59 days
        assert avg_ipt[2] == 59


class TestFeatureScaling:
    """Tests for feature scaling functions."""
    
    @pytest.fixture
    def feature_data(self):
        """Create feature data for scaling tests."""
        return pd.DataFrame({
            'Feature1': [10, 20, 30, 40, 50],
            'Feature2': [100, 200, 300, 400, 500]
        })
    
    def test_standard_scaling(self, feature_data):
        """Test standard scaling."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_data)
        
        # Scaled features should have mean ~0 and std ~1
        assert abs(scaled[:, 0].mean()) < 1e-10
        assert abs(scaled[:, 0].std() - 1) < 0.1
    
    def test_scaling_preserves_shape(self, feature_data):
        """Test that scaling preserves data shape."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_data)
        
        assert scaled.shape == feature_data.shape


class TestProductDiversity:
    """Tests for product diversity metrics."""
    
    @pytest.fixture
    def product_data(self):
        """Create product data."""
        return pd.DataFrame({
            'CustomerID': [1, 1, 1, 2, 2, 3, 3, 3, 3],
            'ProductID': ['A', 'B', 'C', 'A', 'A', 'A', 'B', 'C', 'D']
        })
    
    def test_unique_products(self, product_data):
        """Test unique products per customer."""
        unique_products = product_data.groupby('CustomerID')['ProductID'].nunique()
        
        assert unique_products[1] == 3  # A, B, C
        assert unique_products[2] == 1  # Only A
        assert unique_products[3] == 4  # A, B, C, D
    
    def test_product_diversity_ratio(self, product_data):
        """Test product diversity ratio."""
        diversity = product_data.groupby('CustomerID').agg({
            'ProductID': ['nunique', 'count']
        })
        diversity.columns = ['unique', 'total']
        diversity['ratio'] = diversity['unique'] / diversity['total']
        
        # Customer 1: 3 unique / 3 total = 1.0
        assert diversity.loc[1, 'ratio'] == 1.0
        # Customer 2: 1 unique / 2 total = 0.5
        assert diversity.loc[2, 'ratio'] == 0.5


class TestTimeBasedFeatures:
    """Tests for time-based feature engineering."""
    
    @pytest.fixture
    def time_data(self):
        """Create time-based data."""
        dates = pd.to_datetime([
            '2011-01-03',  # Monday
            '2011-01-04',  # Tuesday
            '2011-01-08',  # Saturday
            '2011-01-09',  # Sunday
            '2011-01-10',  # Monday
        ])
        
        return pd.DataFrame({
            'CustomerID': [1, 1, 1, 1, 1],
            'Date': dates
        })
    
    def test_day_of_week(self, time_data):
        """Test day of week extraction."""
        time_data['DayOfWeek'] = time_data['Date'].dt.dayofweek
        
        assert time_data['DayOfWeek'].iloc[0] == 0  # Monday
        assert time_data['DayOfWeek'].iloc[2] == 5  # Saturday
    
    def test_weekend_flag(self, time_data):
        """Test weekend flag creation."""
        time_data['DayOfWeek'] = time_data['Date'].dt.dayofweek
        time_data['IsWeekend'] = time_data['DayOfWeek'].isin([5, 6]).astype(int)
        
        # 2 weekend days out of 5
        assert time_data['IsWeekend'].sum() == 2
    
    def test_weekend_ratio(self, time_data):
        """Test weekend purchase ratio."""
        time_data['DayOfWeek'] = time_data['Date'].dt.dayofweek
        time_data['IsWeekend'] = time_data['DayOfWeek'].isin([5, 6]).astype(int)
        
        weekend_ratio = time_data['IsWeekend'].mean()
        assert weekend_ratio == 0.4  # 2/5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
