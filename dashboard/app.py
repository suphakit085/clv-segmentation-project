"""
CLV Segmentation Dashboard
==========================
Interactive Streamlit dashboard for CLV analysis and customer segmentation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os

# Page config
st.set_page_config(
    page_title="CLV Segmentation Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .segment-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> Customer Lifetime Value Dashboard</h1>', unsafe_allow_html=True)

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"


@st.cache_data
def load_data():
    """Load all data files"""
    data = {}
    
    try:
        # Load cleaned transactions
        if (PROCESSED_DIR / "online_retail_clean.csv").exists():
            data['transactions'] = pd.read_csv(
                PROCESSED_DIR / "online_retail_clean.csv",
                parse_dates=['InvoiceDate']
            )
        
        # Load RFM segments
        if (FEATURES_DIR / "rfm_segments.csv").exists():
            data['rfm'] = pd.read_csv(FEATURES_DIR / "rfm_segments.csv")
        
        # Load CLV predictions
        if (FEATURES_DIR / "clv_predictions.csv").exists():
            data['clv'] = pd.read_csv(FEATURES_DIR / "clv_predictions.csv")
        
        # Load segment profiles
        if (FEATURES_DIR / "segment_profiles.csv").exists():
            data['profiles'] = pd.read_csv(FEATURES_DIR / "segment_profiles.csv", index_col=0)
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
    
    return data


def create_metric_cards(data):
    """Display key metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    if 'rfm' in data:
        rfm = data['rfm']
        col1.metric("Total Customers", f"{len(rfm):,}")
        col2.metric("Total Revenue", f"£{rfm['Monetary'].sum():,.0f}")
        col3.metric("Avg. CLV", f"£{data['clv']['CLV_ensemble'].mean():,.2f}" if 'clv' in data else "N/A")
        col4.metric("Segments", rfm['Segment'].nunique())


def create_segment_distribution(data):
    """Create segment distribution chart"""
    if 'rfm' not in data:
        return None
    
    segment_counts = data['rfm']['Segment'].value_counts()
    
    fig = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Segments Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=400)
    
    return fig


def create_rfm_scatter(data):
    """Create RFM scatter plot"""
    if 'rfm' not in data:
        return None
    
    sample = data['rfm'].sample(min(2000, len(data['rfm'])))
    
    fig = px.scatter(
        sample,
        x='Recency',
        y='Frequency',
        color='Segment',
        size='Monetary',
        hover_data=['CustomerID', 'Monetary'],
        title="RFM Customer Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=500)
    
    return fig


def create_clv_distribution(data):
    """Create CLV distribution chart"""
    if 'clv' not in data:
        return None
    
    fig = px.histogram(
        data['clv'],
        x='CLV_ensemble',
        nbins=50,
        title="CLV Distribution",
        labels={'CLV_ensemble': 'Predicted CLV (£)'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.add_vline(
        x=data['clv']['CLV_ensemble'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: £{data['clv']['CLV_ensemble'].mean():,.0f}"
    )
    fig.update_layout(height=400)
    
    return fig


def create_segment_profile_heatmap(data):
    """Create segment profile heatmap"""
    if 'profiles' not in data:
        return None
    
    profiles = data['profiles']
    
    # Normalize for heatmap
    cols = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']
    cols = [c for c in cols if c in profiles.columns]
    
    if not cols:
        return None
    
    normalized = profiles[cols].copy()
    for col in cols:
        normalized[col] = (normalized[col] - normalized[col].min()) / \
                          (normalized[col].max() - normalized[col].min())
    
    fig = px.imshow(
        normalized,
        labels=dict(x="Metric", y="Segment", color="Normalized Value"),
        title="Segment Profiles Heatmap",
        color_continuous_scale="RdYlGn",
        aspect='auto'
    )
    fig.update_layout(height=400)
    
    return fig


def create_revenue_by_segment(data):
    """Create revenue by segment chart"""
    if 'rfm' not in data:
        return None
    
    revenue_by_segment = data['rfm'].groupby('Segment')['Monetary'].sum().sort_values(ascending=True)
    
    fig = px.bar(
        x=revenue_by_segment.values,
        y=revenue_by_segment.index,
        orientation='h',
        title="Total Revenue by Segment",
        labels={'x': 'Revenue (£)', 'y': 'Segment'},
        color=revenue_by_segment.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    
    return fig


# Main app
def main():
    # Load data
    data = load_data()
    
    if not data:
        st.warning("⚠️ Please run the notebooks first to generate the data files.")
        st.info("""
        **To get started:**
        1. Run notebooks 01-07 in order
        2. This will generate the required data files
        3. Refresh this dashboard
        """)
        
        # Show file upload option
        st.subheader("Or upload your own data:")
        uploaded_file = st.file_uploader("Upload Transaction Data (CSV)", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df):,} rows")
            st.dataframe(df.head())
        return
    
    # Sidebar
    st.sidebar.header(" Dashboard Controls")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select View",
        ["Overview", "RFM Analysis", "CLV Predictions", "Segment Deep Dive"]
    )
    
    # Filters
    if 'rfm' in data:
        segments = ['All'] + list(data['rfm']['Segment'].unique())
        selected_segment = st.sidebar.selectbox("Filter by Segment", segments)
        
        if selected_segment != 'All':
            data['rfm'] = data['rfm'][data['rfm']['Segment'] == selected_segment]
            if 'clv' in data:
                data['clv'] = data['clv'][data['clv']['CustomerID'].isin(data['rfm']['CustomerID'])]
    
    # Main content based on page
    if page == "Overview":
        st.header(" Business Overview")
        
        # Metrics
        create_metric_cards(data)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_segment_distribution(data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_revenue_by_segment(data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # CLV Distribution
        fig = create_clv_distribution(data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "RFM Analysis":
        st.header(" RFM Segmentation Analysis")
        
        # RFM Scatter
        fig = create_rfm_scatter(data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # RFM Statistics
        if 'rfm' in data:
            st.subheader("RFM Statistics by Segment")
            
            rfm_stats = data['rfm'].groupby('Segment').agg({
                'CustomerID': 'count',
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': ['mean', 'sum']
            }).round(2)
            
            rfm_stats.columns = ['Customers', 'Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Total Revenue']
            st.dataframe(rfm_stats.style.format({
                'Avg Monetary': '£{:,.2f}',
                'Total Revenue': '£{:,.0f}'
            }))
    
    elif page == "CLV Predictions":
        st.header("💰 Customer Lifetime Value Predictions")
        
        if 'clv' in data:
            # CLV metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predicted CLV", f"£{data['clv']['CLV_ensemble'].sum():,.0f}")
            col2.metric("Average CLV", f"£{data['clv']['CLV_ensemble'].mean():,.2f}")
            col3.metric("Median CLV", f"£{data['clv']['CLV_ensemble'].median():,.2f}")
            
            # Distribution
            fig = create_clv_distribution(data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Top customers
            st.subheader("Top 10 Customers by CLV")
            top_customers = data['clv'].nlargest(10, 'CLV_ensemble')[
                ['CustomerID', 'Monetary', 'CLV_ensemble', 'Segment']
            ]
            st.dataframe(top_customers.style.format({
                'Monetary': '£{:,.2f}',
                'CLV_ensemble': '£{:,.2f}'
            }))
    
    elif page == "Segment Deep Dive":
        st.header(" Segment Deep Dive")
        
        # Segment heatmap
        fig = create_segment_profile_heatmap(data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment details
        if 'profiles' in data:
            st.subheader("Segment Profiles")
            st.dataframe(data['profiles'].style.format({
                'Avg_Monetary': '£{:,.2f}' if 'Avg_Monetary' in data['profiles'].columns else None,
                'Total_Revenue': '£{:,.0f}' if 'Total_Revenue' in data['profiles'].columns else None
            }))
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with using Streamlit")
    st.sidebar.markdown("CLV Segmentation Project")


if __name__ == "__main__":
    main()
