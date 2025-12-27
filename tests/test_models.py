"""
Unit tests for CLV models and segmentation modules.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCLVModels:
    """Tests for CLV prediction models."""
    
    @pytest.fixture
    def training_data(self):
        """Create training data for CLV models."""
        np.random.seed(42)
        n = 100
        
        X = pd.DataFrame({
            'Recency': np.random.randint(1, 365, n),
            'Frequency': np.random.randint(1, 20, n),
            'Monetary': np.random.exponential(500, n),
            'Lifetime': np.random.randint(30, 365, n)
        })
        
        # CLV is a function of the features
        y = (X['Monetary'] * 0.5 + 
             X['Frequency'] * 20 + 
             np.random.normal(0, 50, n))
        
        return X, y
    
    def test_random_forest_training(self, training_data):
        """Test Random Forest training."""
        X, y = training_data
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_linear_regression_training(self, training_data):
        """Test Linear Regression training."""
        X, y = training_data
        
        model = LinearRegression()
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_model_r2_score(self, training_data):
        """Test that R² score is calculated correctly."""
        X, y = training_data
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        
        r2 = r2_score(y, predictions)
        assert 0 <= r2 <= 1  # R² should be between 0 and 1 for good models
    
    def test_feature_importance(self, training_data):
        """Test feature importance extraction."""
        X, y = training_data
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importance = model.feature_importances_
        assert len(importance) == X.shape[1]
        assert abs(sum(importance) - 1.0) < 1e-10  # Should sum to 1


class TestProbabilisticModels:
    """Tests for probabilistic CLV models."""
    
    @pytest.fixture
    def rfm_summary(self):
        """Create RFM summary for probabilistic models."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'CustomerID': range(1, n + 1),
            'frequency': np.random.poisson(5, n),
            'recency': np.random.randint(1, 365, n),
            'T': np.random.randint(30, 400, n),
            'monetary': np.random.exponential(100, n)
        })
    
    def test_expected_purchases(self, rfm_summary):
        """Test expected purchases calculation."""
        # Simple heuristic model
        rfm_summary['expected_purchases'] = (
            rfm_summary['frequency'] / rfm_summary['T'] * 365
        )
        
        assert (rfm_summary['expected_purchases'] >= 0).all()
    
    def test_probability_alive(self, rfm_summary):
        """Test probability alive calculation."""
        rfm_summary['days_since_purchase'] = rfm_summary['T'] - rfm_summary['recency']
        rfm_summary['avg_gap'] = rfm_summary['T'] / (rfm_summary['frequency'] + 1)
        
        # Simple decay model
        rfm_summary['prob_alive'] = np.exp(
            -rfm_summary['days_since_purchase'] / (2 * rfm_summary['avg_gap'])
        )
        rfm_summary['prob_alive'] = rfm_summary['prob_alive'].clip(0, 1)
        
        assert (rfm_summary['prob_alive'] >= 0).all()
        assert (rfm_summary['prob_alive'] <= 1).all()
    
    def test_clv_calculation(self, rfm_summary):
        """Test CLV calculation from components."""
        expected_purchases = 5
        expected_monetary = 100
        prob_alive = 0.8
        discount_rate = 0.1
        
        clv = expected_purchases * expected_monetary * prob_alive / (1 + discount_rate)
        
        assert clv == pytest.approx(363.64, rel=0.01)


class TestKMeansClustering:
    """Tests for K-Means clustering."""
    
    @pytest.fixture
    def cluster_data(self):
        """Create data for clustering tests."""
        np.random.seed(42)
        
        # Create 3 distinct clusters
        cluster1 = np.random.randn(30, 3) + [0, 0, 0]
        cluster2 = np.random.randn(30, 3) + [5, 5, 5]
        cluster3 = np.random.randn(30, 3) + [10, 0, 5]
        
        data = np.vstack([cluster1, cluster2, cluster3])
        return pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3'])
    
    def test_kmeans_fitting(self, cluster_data):
        """Test K-Means fitting."""
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(cluster_data)
        
        assert len(labels) == len(cluster_data)
        assert len(np.unique(labels)) == 3
    
    def test_silhouette_score(self, cluster_data):
        """Test silhouette score calculation."""
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(cluster_data)
        
        score = silhouette_score(cluster_data, labels)
        assert -1 <= score <= 1
        assert score > 0.3  # Should be reasonably good for well-separated clusters
    
    def test_cluster_centers(self, cluster_data):
        """Test cluster centers extraction."""
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(cluster_data)
        
        centers = kmeans.cluster_centers_
        assert centers.shape == (3, 3)  # 3 clusters, 3 features


class TestOptimalClusters:
    """Tests for finding optimal number of clusters."""
    
    @pytest.fixture
    def elbow_data(self):
        """Create data for elbow method tests."""
        np.random.seed(42)
        return np.random.randn(100, 4)
    
    def test_inertia_decreases(self, elbow_data):
        """Test that inertia decreases with more clusters."""
        inertias = []
        
        for k in range(2, 6):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(elbow_data)
            inertias.append(kmeans.inertia_)
        
        # Inertia should generally decrease
        assert inertias[0] > inertias[-1]
    
    def test_silhouette_variation(self, elbow_data):
        """Test silhouette score for different k values."""
        silhouettes = []
        
        for k in range(2, 6):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(elbow_data)
            score = silhouette_score(elbow_data, labels)
            silhouettes.append(score)
        
        # All silhouette scores should be valid
        assert all(-1 <= s <= 1 for s in silhouettes)


class TestHierarchicalClustering:
    """Tests for hierarchical clustering."""
    
    @pytest.fixture
    def hierarchical_data(self):
        """Create data for hierarchical clustering."""
        np.random.seed(42)
        return np.random.randn(50, 3)
    
    def test_agglomerative_clustering(self, hierarchical_data):
        """Test agglomerative clustering."""
        from sklearn.cluster import AgglomerativeClustering
        
        hc = AgglomerativeClustering(n_clusters=3)
        labels = hc.fit_predict(hierarchical_data)
        
        assert len(labels) == len(hierarchical_data)
        assert len(np.unique(labels)) == 3


class TestClusterProfiles:
    """Tests for cluster profile generation."""
    
    @pytest.fixture
    def clustered_data(self):
        """Create clustered data with labels."""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'CustomerID': range(1, 101),
            'Recency': np.random.randint(1, 365, 100),
            'Frequency': np.random.randint(1, 20, 100),
            'Monetary': np.random.exponential(500, 100),
            'Cluster': np.random.choice([0, 1, 2], 100)
        })
        
        return data
    
    def test_profile_calculation(self, clustered_data):
        """Test cluster profile calculation."""
        profiles = clustered_data.groupby('Cluster').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum']
        })
        
        assert len(profiles) == 3  # 3 clusters
    
    def test_cluster_sizes(self, clustered_data):
        """Test cluster size calculation."""
        sizes = clustered_data['Cluster'].value_counts()
        
        assert sum(sizes) == len(clustered_data)


class TestSegmentLabeling:
    """Tests for segment labeling functions."""
    
    @pytest.fixture
    def segment_data(self):
        """Create segment data for labeling tests."""
        return pd.DataFrame({
            'Cluster': [0, 1, 2],
            'Avg_Recency': [30, 100, 250],
            'Avg_Frequency': [10, 5, 2],
            'Avg_Monetary': [2500, 800, 200]
        })
    
    def test_label_assignment(self, segment_data):
        """Test segment label assignment."""
        def assign_label(row):
            if row['Avg_Recency'] < 50 and row['Avg_Monetary'] > 2000:
                return 'VIP'
            elif row['Avg_Recency'] < 150:
                return 'Active'
            else:
                return 'Dormant'
        
        segment_data['Label'] = segment_data.apply(assign_label, axis=1)
        
        assert segment_data.loc[0, 'Label'] == 'VIP'
        assert segment_data.loc[1, 'Label'] == 'Active'
        assert segment_data.loc[2, 'Label'] == 'Dormant'


class TestModelEvaluation:
    """Tests for model evaluation metrics."""
    
    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        assert rmse == pytest.approx(10.0, rel=0.01)
    
    def test_mae_calculation(self):
        """Test MAE calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        mae = np.mean(np.abs(y_true - y_pred))
        assert mae == 10.0
    
    def test_mape_calculation(self):
        """Test MAPE calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 220, 330, 440, 550])
        
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        assert mape == 10.0  # 10% error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
