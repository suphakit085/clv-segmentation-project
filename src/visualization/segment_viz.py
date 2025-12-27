"""
Segment Visualization Module
============================
Visualizations for customer segments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


def plot_segment_distribution(segment_labels: pd.Series,
                              figsize: tuple = (10, 6),
                              save_path: Optional[str] = None):
    """Plot distribution of customers across segments."""
    fig, ax = plt.subplots(figsize=figsize)
    segment_labels.value_counts().plot(kind='bar', ax=ax, color=sns.color_palette('viridis', len(segment_labels.unique())))
    ax.set_title('Customer Distribution by Segment')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_segment_profiles(profiles: pd.DataFrame,
                          features: List[str],
                          figsize: tuple = (12, 8),
                          save_path: Optional[str] = None):
    """Plot radar chart of segment profiles."""
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    
    for segment in profiles.index:
        values = profiles.loc[segment, features].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', label=str(segment))
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title('Segment Profiles')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_rfm_segments(rfm_data: pd.DataFrame,
                      figsize: tuple = (10, 8),
                      save_path: Optional[str] = None):
    """Plot RFM segment scatter plot."""
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(rfm_data['recency'], rfm_data['frequency'],
                        c=rfm_data['monetary'], s=50, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_title('RFM Segmentation')
    plt.colorbar(scatter, label='Monetary')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


if __name__ == "__main__":
    print("Segment Visualization Module")
