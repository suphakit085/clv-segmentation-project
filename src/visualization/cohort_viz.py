"""
Cohort Visualization Module
===========================
Visualizations for cohort analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_cohort_heatmap(cohort_data: pd.DataFrame,
                        title: str = 'Cohort Analysis',
                        figsize: tuple = (12, 8),
                        cmap: str = 'Blues',
                        save_path: Optional[str] = None):
    """Plot cohort retention heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cohort_data, annot=True, fmt='.1%', cmap=cmap, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Cohort Period')
    ax.set_ylabel('Cohort')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_retention_curve(cohort_data: pd.DataFrame,
                         figsize: tuple = (10, 6),
                         save_path: Optional[str] = None):
    """Plot retention curves for cohorts."""
    fig, ax = plt.subplots(figsize=figsize)
    
    for cohort in cohort_data.index[:5]:
        ax.plot(cohort_data.columns, cohort_data.loc[cohort], marker='o', label=str(cohort))
    
    ax.set_title('Cohort Retention Curves')
    ax.set_xlabel('Period')
    ax.set_ylabel('Retention Rate')
    ax.legend(title='Cohort')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_cohort_revenue(cohort_revenue: pd.DataFrame,
                        figsize: tuple = (12, 6),
                        save_path: Optional[str] = None):
    """Plot cumulative revenue by cohort."""
    fig, ax = plt.subplots(figsize=figsize)
    cohort_revenue.cumsum(axis=1).plot(kind='area', stacked=True, ax=ax, alpha=0.7)
    ax.set_title('Cumulative Revenue by Cohort')
    ax.set_xlabel('Period')
    ax.set_ylabel('Revenue')
    ax.legend(title='Cohort', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


if __name__ == "__main__":
    print("Cohort Visualization Module")
