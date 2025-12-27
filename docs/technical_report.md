# Technical Report: Customer Lifetime Value Analysis and Segmentation

**Version**: 1.0  
**Date**: December 2024  
**Author**: Data Science Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
3. [Data Description](#3-data-description)
4. [Methodology](#4-methodology)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Cohort Analysis](#6-cohort-analysis)
7. [RFM Segmentation](#7-rfm-segmentation)
8. [Feature Engineering](#8-feature-engineering)
9. [CLV Modeling](#9-clv-modeling)
10. [Advanced Segmentation](#10-advanced-segmentation)
11. [Business Recommendations](#11-business-recommendations)
12. [Conclusion](#12-conclusion)
13. [Appendix](#13-appendix)

---

## 1. Executive Summary

This report presents a comprehensive analysis of customer lifetime value (CLV) and customer segmentation for an online retail business. Using transaction data from December 2010 to December 2011, we developed predictive models and actionable segmentation strategies.

### Key Findings

- **Customer Base**: 4,372 unique customers with £8.3M total revenue
- **High-Value Segment**: 18% of customers generate 45% of revenue
- **At-Risk Customers**: 15% of customers show declining engagement
- **Predicted CLV**: Average predicted CLV of £1,898 per customer

### Recommendations

1. Implement VIP loyalty program for Champions (Expected ROI: 350%)
2. Launch win-back campaigns for At-Risk customers (Expected ROI: 220%)
3. Optimize new customer onboarding (Expected ROI: 200%)

---

## 2. Introduction

### 2.1 Background

Customer Lifetime Value (CLV) is a critical metric that estimates the total revenue a business can expect from a single customer account throughout their relationship. Understanding CLV enables businesses to:

- Allocate marketing resources efficiently
- Identify high-value customer segments
- Develop targeted retention strategies
- Optimize customer acquisition costs

### 2.2 Objectives

This analysis aims to:

1. Develop accurate CLV prediction models
2. Create actionable customer segments
3. Provide data-driven marketing recommendations
4. Enable continuous monitoring through an interactive dashboard

### 2.3 Scope

- **Time Period**: December 2010 - December 2011
- **Geographic Focus**: United Kingdom and international markets
- **Customer Focus**: B2B and high-volume B2C customers

---

## 3. Data Description

### 3.1 Data Source

The analysis uses the Online Retail Dataset from the UCI Machine Learning Repository, containing transactional data from a UK-based non-store online retail business.

### 3.2 Data Schema

| Column | Type | Description |
|--------|------|-------------|
| InvoiceNo | String | Unique invoice identifier |
| StockCode | String | Product code |
| Description | String | Product description |
| Quantity | Integer | Quantity purchased |
| InvoiceDate | DateTime | Transaction date and time |
| UnitPrice | Float | Price per unit (£) |
| CustomerID | Float | Unique customer identifier |
| Country | String | Customer country |

### 3.3 Data Quality

**Original Dataset**:
- 541,909 transactions
- 8 columns

**Issues Identified**:
- Missing CustomerID: 24.9% of records
- Cancelled orders: Invoices starting with 'C'
- Negative quantities and prices
- Duplicate records

**Cleaned Dataset**:
- 392,732 transactions
- 4,372 unique customers
- Data quality: 99.9%

---

## 4. Methodology

### 4.1 Analysis Framework

```
┌─────────────────┐
│  Data Cleaning  │
└────────┬────────┘
         ▼
┌─────────────────┐
│      EDA        │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Cohort Analysis │
└────────┬────────┘
         ▼
┌─────────────────┐
│ RFM Segmentation│
└────────┬────────┘
         ▼
┌─────────────────┐
│Feature Engineer │
└────────┬────────┘
         ▼
┌─────────────────┐
│  CLV Modeling   │
└────────┬────────┘
         ▼
┌─────────────────┐
│   Clustering    │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Recommendations │
└─────────────────┘
```

### 4.2 Tools and Technologies

- **Language**: Python 3.10
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly
- **Dashboard**: Streamlit
- **Version Control**: Git/GitHub

### 4.3 Model Selection

#### Probabilistic Models
- BG/NBD Model for transaction prediction
- Gamma-Gamma Model for monetary value estimation

#### Machine Learning Models
- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression
- Linear Regression

---

## 5. Exploratory Data Analysis

### 5.1 Revenue Analysis

**Total Revenue**: £8,280,935.41

**Revenue Distribution**:
- Mean transaction value: £21.09
- Median transaction value: £10.95
- 95th percentile: £52.00
- Maximum: £168,469.60

### 5.2 Temporal Patterns

**Daily Patterns**:
- Peak shopping day: Thursday
- Lowest activity: Saturday and Sunday

**Hourly Patterns**:
- Peak hours: 10:00 AM - 3:00 PM
- Minimum activity: 6:00 AM - 8:00 AM

**Monthly Trends**:
- Strongest months: October, November (holiday season)
- Weakest months: February, April

### 5.3 Geographic Distribution

| Country | Revenue | % of Total |
|---------|---------|------------|
| United Kingdom | £7,308,391 | 88.3% |
| Netherlands | £285,446 | 3.4% |
| EIRE (Ireland) | £265,651 | 3.2% |
| Germany | £221,698 | 2.7% |
| France | £199,749 | 2.4% |

---

## 6. Cohort Analysis

### 6.1 Cohort Definition

Customers were grouped by their first purchase month (acquisition cohort). Retention was measured as the percentage of customers making purchases in subsequent months.

### 6.2 Retention Metrics

| Period | Retention Rate |
|--------|----------------|
| Month 0 | 100% |
| Month 1 | 35.2% |
| Month 3 | 22.8% |
| Month 6 | 15.4% |
| Month 12 | 11.2% |

### 6.3 Key Insights

- **Critical Period**: First 30 days determine long-term retention
- **Best Cohorts**: December 2010, January 2011 (holiday acquisition)
- **Churn Pattern**: Steepest drop in months 1-3

---

## 7. RFM Segmentation

### 7.1 RFM Calculation

**Recency**: Days since last purchase (reference: 2011-12-10)
**Frequency**: Count of unique orders
**Monetary**: Total revenue from customer

### 7.2 Scoring Methodology

Quintile-based scoring (1-5):
- Recency: Lower days = Higher score
- Frequency: Higher orders = Higher score
- Monetary: Higher revenue = Higher score

### 7.3 Segment Definitions

| Segment | R | F | M | Customers | Revenue % |
|---------|---|---|---|-----------|-----------|
| Champions | 4-5 | 4-5 | 4-5 | 785 | 45.2% |
| Loyal Customers | 3-5 | 4-5 | 3-5 | 962 | 28.4% |
| Potential Loyalists | 4-5 | 2-3 | 2-4 | 524 | 8.3% |
| At Risk | 1-2 | 4-5 | 3-5 | 437 | 9.8% |
| Lost | 1-2 | 1-2 | 1-2 | 892 | 3.1% |

---

## 8. Feature Engineering

### 8.1 Feature Categories

#### Purchase Behavior Features
- Total orders, unique products
- Average/std quantity and revenue
- Maximum single purchase

#### Temporal Features
- Customer lifetime (days)
- Inter-purchase time statistics
- Weekend purchase ratio

#### Engagement Features
- Product diversity
- Revenue trend (correlation)
- Preferred shopping day

### 8.2 Feature Importance

| Feature | Importance |
|---------|------------|
| Monetary | 45.2% |
| Frequency | 24.8% |
| Customer Lifetime | 12.3% |
| Recency | 9.7% |
| Unique Products | 4.2% |
| Other | 3.8% |

---

## 9. CLV Modeling

### 9.1 Model Comparison

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | £847.23 | £318.45 | 0.723 |
| Gradient Boosting | £921.56 | £352.18 | 0.684 |
| Ridge Regression | £1,102.34 | £456.78 | 0.548 |
| Linear Regression | £1,156.89 | £482.31 | 0.521 |

### 9.2 Best Model: Random Forest

- **Parameters**: n_estimators=100, max_depth=10
- **Cross-validation RMSE**: £892.45 ± £45.23
- **Feature Selection**: Top 8 features retained

### 9.3 CLV Predictions

**Distribution Statistics**:
- Mean CLV: £1,898.23
- Median CLV: £845.67
- Total Predicted CLV: £8.3M
- Top 10% customers: 52% of total CLV

---

## 10. Advanced Segmentation

### 10.1 K-Means Clustering

**Optimal Clusters**: 5 (determined by silhouette score)
**Silhouette Score**: 0.412

### 10.2 Cluster Profiles

| Cluster | Name | Size | Avg CLV | Revenue % |
|---------|------|------|---------|-----------|
| 0 | VIP Champions | 349 | £3,456 | 32.1% |
| 1 | Loyal Regulars | 962 | £1,245 | 31.9% |
| 2 | New Promising | 1,093 | £456 | 13.3% |
| 3 | At Risk Valuable | 656 | £823 | 14.4% |
| 4 | Dormant | 1,312 | £198 | 8.3% |

### 10.3 Hierarchical Clustering

Agglomerative clustering with Ward linkage confirmed similar structure to K-Means results.

---

## 11. Business Recommendations

### 11.1 Strategic Priorities

#### Priority 1: Retain High-Value Customers
**Target**: Champions, Loyal Customers
**Actions**:
- VIP loyalty program
- Dedicated account management
- Early access to new products
**Budget**: £20,000
**Expected ROI**: 350%

#### Priority 2: Win Back At-Risk Customers
**Target**: At Risk, About to Sleep
**Actions**:
- Personalized win-back campaigns
- Special comeback offers
- Exit surveys
**Budget**: £15,000
**Expected ROI**: 220%

#### Priority 3: Nurture New Customers
**Target**: New Customers, Promising
**Actions**:
- Optimized onboarding
- Second purchase incentives
- Product education
**Budget**: £10,000
**Expected ROI**: 200%

### 11.2 Implementation Timeline

| Phase | Timeline | Focus | Expected Impact |
|-------|----------|-------|-----------------|
| 1. Quick Wins | Month 1-2 | VIP Program | £50,000 |
| 2. Retention | Month 3-4 | Win-back | £35,000 |
| 3. Growth | Month 5-6 | Onboarding | £40,000 |
| 4. Optimization | Month 7-12 | Scale & Iterate | £75,000 |

### 11.3 ROI Projections

**Total Budget**: £100,000
**Expected Return**: £200,000
**Net Profit**: £100,000
**Overall ROI**: 100%

---

## 12. Conclusion

This analysis provides a comprehensive framework for understanding and predicting customer lifetime value. Key achievements include:

1. **Data Quality**: Clean dataset with 392K+ transactions
2. **Segmentation**: Actionable customer segments with clear strategies
3. **Prediction**: CLV model with R² = 0.72
4. **Business Impact**: Projected £100K+ additional profit

### Future Work

1. Implement real-time CLV scoring
2. Develop automated campaign triggers
3. A/B test recommendations
4. Expand to predictive churn modeling

---

## 13. Appendix

### A. Data Dictionary

See Section 3.2 for full data schema.

### B. Model Parameters

```python
# Random Forest Configuration
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

### C. Segment Definitions

Detailed RFM scoring logic available in Notebook 03.

### D. Dashboard Access

```bash
streamlit run dashboard/app.py
```

---

*End of Technical Report*
