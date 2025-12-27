"""
CLV Segmentation Dashboard
==========================
Streamlit dashboard for CLV analysis and customer segmentation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="CLV Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Lifetime Value & Segmentation Dashboard")

# Sidebar
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Transaction Data", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.sidebar.success(f"✅ Loaded {len(df):,} records")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎯 RFM Analysis", "💰 CLV Prediction", "👥 Segmentation"])
    
    with tab1:
        st.header("Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        col4.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
        st.subheader("Data Statistics")
        st.dataframe(df.describe())
    
    with tab2:
        st.header("RFM Analysis")
        st.info("Configure RFM analysis parameters in the sidebar")
        
        col1, col2 = st.columns(2)
        with col1:
            customer_id_col = st.selectbox("Customer ID Column", df.columns)
        with col2:
            date_col = st.selectbox("Date Column", df.columns)
        
        amount_col = st.selectbox("Amount Column", df.columns)
        
        if st.button("Calculate RFM"):
            st.success("RFM calculation completed!")
            st.info("Implement RFM calculation logic here")
    
    with tab3:
        st.header("CLV Prediction")
        st.info("Configure CLV prediction parameters")
        
        model_type = st.selectbox("Model Type", ["BG/NBD", "Machine Learning", "Simple Regression"])
        prediction_period = st.slider("Prediction Period (months)", 1, 24, 12)
        
        if st.button("Predict CLV"):
            st.success("CLV prediction completed!")
            st.info("Implement CLV prediction logic here")
    
    with tab4:
        st.header("Customer Segmentation")
        st.info("Configure segmentation parameters")
        
        n_segments = st.slider("Number of Segments", 2, 10, 5)
        algorithm = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Hierarchical", "GMM"])
        
        if st.button("Run Segmentation"):
            st.success("Segmentation completed!")
            st.info("Implement segmentation logic here")

else:
    st.info("👈 Please upload your transaction data to get started")
    
    st.markdown("""
    ## Getting Started
    
    1. **Upload your data** - CSV or Excel file with transaction data
    2. **Configure columns** - Map your data columns to required fields
    3. **Analyze RFM** - Calculate Recency, Frequency, Monetary metrics
    4. **Predict CLV** - Use probabilistic or ML models
    5. **Segment Customers** - Group customers by behavior
    
    ### Expected Data Format
    
    Your data should contain:
    - **Customer ID**: Unique identifier for each customer
    - **Transaction Date**: Date of each transaction
    - **Transaction Amount**: Value of each transaction
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
