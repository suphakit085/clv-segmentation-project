# CLV Segmentation Project

**Customer Lifetime Value (CLV) Prediction and Customer Segmentation Toolkit**

## Overview

This project provides a comprehensive toolkit for:
- **Customer Lifetime Value (CLV) Prediction** using probabilistic and machine learning models
- **Customer Segmentation** using RFM analysis and clustering algorithms
- **Cohort Analysis** for understanding customer behavior over time
- **Interactive Dashboard** for visualizing insights

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/suphakit085/clv-segmentation-project.git
cd clv-segmentation-project

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
python -m streamlit run dashboard/app.py
```

## Project Structure

```
clv-segmentation-project/
├── data/
│   ├── raw/                      # Original data
│   ├── processed/                # Cleaned data + images
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
├── tests/                        # Unit tests (55 tests)
├── dashboard/                    # Streamlit dashboard
├── docs/                         # Documentation
│   ├── executive_summary.md
│   ├── technical_report.md
│   ├── blog_post.md
│   ├── presentation.md
│   └── video_script.md
├── .github/workflows/            # CI/CD pipeline
├── requirements.txt
├── environment.yml
├── README.md
└── setup.py
```

## Notebooks

1. **01_data_exploration.ipynb** - Exploratory data analysis
2. **02_cohort_analysis.ipynb** - Customer cohort analysis
3. **03_rfm_segmentation.ipynb** - RFM-based segmentation
4. **04_feature_engineering.ipynb** - Feature creation
5. **05_clv_modeling.ipynb** - CLV prediction models
6. **06_advanced_segmentation.ipynb** - Advanced clustering
7. **07_business_recommendations.ipynb** - Business insights & ROI

## Key Features

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

## Dashboard

The Streamlit dashboard provides:
- Customer overview with KPIs
- Interactive RFM analysis
- CLV prediction interface
- Segment visualization

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## Documentation

- **Executive Summary** - One-page business summary
- **Technical Report** - Detailed methodology and findings
- **Blog Post** - 2000+ word article for Medium
- **Presentation** - 15-20 slides for stakeholders
- **Video Script** - 5-minute walkthrough

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact
suphakit.kae@kkumail.com

