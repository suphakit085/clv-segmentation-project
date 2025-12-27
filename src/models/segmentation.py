"""
Customer Segmentation Module
============================
Advanced customer segmentation using clustering algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class CustomerSegmentation:
    """
    Customer segmentation using various clustering algorithms.
    """
    
    ALGORITHMS = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'hierarchical': AgglomerativeClustering,
        'gmm': GaussianMixture
    }
    
    def __init__(self, algorithm: str = 'kmeans', n_clusters: int = 5, **kwargs):
        """
        Initialize the segmentation model.
        
        Args:
            algorithm: Clustering algorithm to use
            n_clusters: Number of clusters (not applicable for DBSCAN)
            **kwargs: Additional algorithm parameters
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(self.ALGORITHMS.keys())}")
        
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False
        self.labels_ = None
    
    def fit(self, X: pd.DataFrame) -> 'CustomerSegmentation':
        """
        Fit the segmentation model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit model
        if self.algorithm == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42, **self.kwargs)
        elif self.algorithm == 'dbscan':
            self.model = DBSCAN(**self.kwargs)
        elif self.algorithm == 'hierarchical':
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters, **self.kwargs)
        elif self.algorithm == 'gmm':
            self.model = GaussianMixture(n_components=self.n_clusters, random_state=42, **self.kwargs)
        
        if self.algorithm == 'gmm':
            self.model.fit(X_scaled)
            self.labels_ = self.model.predict(X_scaled)
        else:
            self.labels_ = self.model.fit_predict(X_scaled)
        
        self.fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X_scaled)
        else:
            # For algorithms without predict (e.g., hierarchical clustering)
            raise NotImplementedError(f"{self.algorithm} does not support prediction on new data")
    
    def evaluate(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate clustering quality.
        
        Args:
            X: Feature matrix used for fitting
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        X_scaled = self.scaler.transform(X)
        
        # Check if we have valid clusters (more than 1 unique label)
        n_labels = len(np.unique(self.labels_))
        if n_labels < 2:
            return {'silhouette': np.nan, 'calinski_harabasz': np.nan, 'davies_bouldin': np.nan}
        
        return {
            'silhouette': silhouette_score(X_scaled, self.labels_),
            'calinski_harabasz': calinski_harabasz_score(X_scaled, self.labels_),
            'davies_bouldin': davies_bouldin_score(X_scaled, self.labels_)
        }
    
    def get_cluster_profiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get profiles for each cluster.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with cluster statistics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        X_with_labels = X.copy()
        X_with_labels['cluster'] = self.labels_
        
        profiles = X_with_labels.groupby('cluster').agg(['mean', 'std', 'count'])
        
        return profiles


def find_optimal_clusters(X: pd.DataFrame, 
                          max_clusters: int = 10,
                          algorithm: str = 'kmeans') -> Tuple[int, pd.DataFrame]:
    """
    Find optimal number of clusters using multiple metrics.
    
    Args:
        X: Feature matrix
        max_clusters: Maximum number of clusters to try
        algorithm: Clustering algorithm to use
        
    Returns:
        Tuple of (optimal number of clusters, evaluation DataFrame)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    
    for k in range(2, max_clusters + 1):
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X_scaled)
            inertia = model.inertia_
        elif algorithm == 'gmm':
            model = GaussianMixture(n_components=k, random_state=42)
            model.fit(X_scaled)
            labels = model.predict(X_scaled)
            inertia = -model.score(X_scaled)  # Negative log-likelihood
        else:
            raise ValueError(f"Optimal cluster finding not supported for {algorithm}")
        
        results.append({
            'n_clusters': k,
            'silhouette': silhouette_score(X_scaled, labels),
            'calinski_harabasz': calinski_harabasz_score(X_scaled, labels),
            'davies_bouldin': davies_bouldin_score(X_scaled, labels),
            'inertia': inertia
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal based on silhouette score
    optimal_k = results_df.loc[results_df['silhouette'].idxmax(), 'n_clusters']
    
    return int(optimal_k), results_df


def create_segment_labels(cluster_profiles: pd.DataFrame,
                          custom_labels: Optional[Dict[int, str]] = None) -> Dict[int, str]:
    """
    Create meaningful labels for segments based on their profiles.
    
    Args:
        cluster_profiles: DataFrame with cluster statistics
        custom_labels: Optional custom labels for clusters
        
    Returns:
        Dictionary mapping cluster IDs to labels
    """
    if custom_labels:
        return custom_labels
    
    # Default labeling based on common naming conventions
    n_clusters = len(cluster_profiles.index.get_level_values(0).unique())
    
    default_labels = {
        0: 'Segment A',
        1: 'Segment B', 
        2: 'Segment C',
        3: 'Segment D',
        4: 'Segment E',
        5: 'Segment F',
        6: 'Segment G',
        7: 'Segment H',
        8: 'Segment I',
        9: 'Segment J'
    }
    
    return {i: default_labels.get(i, f'Segment {i}') for i in range(n_clusters)}


if __name__ == "__main__":
    print("Customer Segmentation Module")
