# CLV Segmentation Project

📊 **Customer Lifetime Value (CLV) Prediction and Customer Segmentation Toolkit**

## Overview

This project provides a comprehensive toolkit for:
- **Customer Lifetime Value (CLV) Prediction** using probabilistic and machine learning models
- **Customer Segmentation** using RFM analysis and clustering algorithms
- **Cohort Analysis** for understanding customer behavior over time
- **Interactive Dashboard** for visualizing insights

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clv-segmentation.git
cd clv-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

## 📁 Project Structure

```
clv-segmentation-project/
├── data/
│   ├── raw/                      # Original data
│   ├── processed/                # Cleaned data
│   └── features/                 # Feature store
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cohort_analysis.ipynb
│   ├── 03_rfm_segmentation.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_clv_modeling.ipynb
│   ├── 06_advanced_segmentation.ipynb
│   └── 07_business_recommendations.ipynb
├── src/
│   ├── data/                     # Data loading and cleaning
│   ├── features/                 # Feature engineering
│   ├── models/                   # CLV and segmentation models
│   ├── evaluation/               # Model evaluation metrics
│   └── visualization/            # Plotting utilities
├── tests/                        # Unit tests
├── dashboard/                    # Streamlit dashboard
├── requirements.txt
├── README.md
└── setup.py
```

## 📚 Notebooks

1. **01_data_exploration.ipynb** - Exploratory data analysis
2. **02_cohort_analysis.ipynb** - Customer cohort analysis
3. **03_rfm_segmentation.ipynb** - RFM-based segmentation
4. **04_feature_engineering.ipynb** - Feature creation
5. **05_clv_modeling.ipynb** - CLV prediction models
6. **06_advanced_segmentation.ipynb** - Advanced clustering
7. **07_business_recommendations.ipynb** - Business insights

## 🔧 Key Features

### RFM Analysis
- Calculate Recency, Frequency, Monetary metrics
- Automatic RFM scoring and segmentation
- Customizable segment labels

### CLV Models
- **BG/NBD Model** - Probabilistic customer lifetime prediction
- **Gamma-Gamma Model** - Average transaction value estimation
- **Machine Learning Models** - Random Forest, Gradient Boosting

### Segmentation
- K-Means clustering
- Hierarchical clustering
- DBSCAN
- Gaussian Mixture Models

## 📊 Usage Examples

### RFM Analysis

```python
from src.features import calculate_rfm, calculate_rfm_scores, segment_rfm

# Calculate RFM metrics
rfm = calculate_rfm(transactions_df, 
                    customer_id_col='customer_id',
                    date_col='date',
                    amount_col='amount')

# Score customers
rfm_scored = calculate_rfm_scores(rfm)

# Assign segments
rfm_segmented = segment_rfm(rfm_scored)
```

### CLV Prediction

```python
from src.models import BGNBDModel, GammaGammaModel

# Fit BG/NBD model
bgnbd = BGNBDModel()
bgnbd.fit(frequency, recency, T)

# Predict future transactions
expected_purchases = bgnbd.predict_transactions(frequency, recency, T, t=12)
```

### Customer Segmentation

```python
from src.models import CustomerSegmentation

# Create and fit segmentation model
segmenter = CustomerSegmentation(algorithm='kmeans', n_clusters=5)
segmenter.fit(feature_matrix)

# Get cluster labels
labels = segmenter.labels_
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Dashboard

The Streamlit dashboard provides:
- Data upload and preview
- Interactive RFM analysis
- CLV prediction interface
- Segment visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
