# 📈 Sales Forecasting & Demand Intelligence Dashboard

## Overview

Businesses generate large amounts of sales data every day, but making meaningful decisions from that data can be challenging. This project presents an end-to-end Sales Forecasting and Demand Intelligence Dashboard that transforms historical sales records into practical business insights.

The system analyses historical sales trends, forecasts future demand, detects unusual sales behaviour, and groups products based on their demand characteristics. It also provides an interactive dashboard that enables business users to explore sales performance without requiring any technical knowledge.

The project was developed as part of a Data Science internship to demonstrate the complete workflow of business data analysis, forecasting, anomaly detection, and dashboard development.

---

# Project Objectives

- Analyse historical retail sales data.
- Identify important sales trends and seasonal patterns.
- Forecast future sales using time-series models.
- Detect unusual sales behaviour automatically.
- Segment products according to demand.
- Build an interactive dashboard for business decision-making.

---

# Key Features

- 📊 Interactive Sales Dashboard
- 📈 Monthly and Yearly Sales Analysis
- 🌍 Region, Category, Segment and Year Filters
- 🔮 Sales Forecasting
- 🚨 Sales Anomaly Detection
- 📦 Product Demand Segmentation
- 📥 Download Filtered Dataset
- 📉 Business Intelligence Visualizations

---

# Machine Learning Models Used

### Forecasting Models
- SARIMA
- Facebook Prophet
- XGBoost

After comparing all forecasting models, **XGBoost achieved the best predictive performance** with the lowest forecasting errors and was identified as the most suitable model for production use.

### Anomaly Detection
- Isolation Forest

### Product Segmentation
- K-Means Clustering

### Dimensionality Reduction
- Principal Component Analysis (PCA)

---

# Exploratory Data Analysis Highlights

The exploratory analysis revealed several valuable business insights:

- Technology was the highest revenue-generating product category.
- The East region showed the strongest and most consistent sales performance.
- Sales followed a clear seasonal pattern.
- Customer demand increased significantly during September, November and December.
- Historical sales data showed predictable trends suitable for demand forecasting.

---

# Dashboard Pages

## 📊 Sales Overview

The Sales Overview page provides a complete summary of business performance using interactive filters and key performance indicators.

Features include:

- Total Sales
- Total Orders
- Average Sales
- Monthly Sales Trend
- Yearly Sales Analysis
- Sales by Region
- Category Distribution
- Top Selling Products
- Download Filtered Dataset

---

## 🔮 Forecast Explorer

The Forecast Explorer predicts future sales based on historical data.

Users can generate forecasts by:

- Product Category
- Region

The dashboard compares historical sales with future predicted sales through an interactive visualization.

---

## 🚨 Anomaly Report

The Anomaly Report automatically detects unusual sales behaviour using the Isolation Forest algorithm.

This helps identify unexpected spikes or drops in sales that may require further business investigation.

---

## 📦 Demand Segments

Products are grouped into demand-based clusters using K-Means Clustering.

The visualization helps identify products with:

- High Demand
- Medium Demand
- Low Demand
- Variable Demand

This supports better inventory planning and stock management.

---

# Business Insights

The analysis produced several practical business findings:

- Technology products contributed the highest overall sales.
- Sales demand showed strong seasonal behaviour.
- XGBoost produced the most accurate sales forecasts among all evaluated models.
- High-demand products should receive inventory priority.
- Anomaly detection helps identify unusual business events that require management attention.

---

# Project Structure

```
Sales-Forecasting-Demand-Intelligence-System
│
├── app.py
├── analysis.ipynb
├── train.csv
├── requirements.txt
├── README.md
├── summary.pdf
└── charts/
```

---

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- XGBoost

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Sales-Forecasting-Demand-Intelligence-System.git
```

Move into the project folder:

```bash
cd Sales-Forecasting-Demand-Intelligence-System
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# Business Benefits

This system helps businesses:

- Improve inventory planning
- Forecast future sales demand
- Detect unusual sales behaviour
- Understand seasonal sales trends
- Reduce inventory-related costs
- Support data-driven decision making

---

# Future Scope

Possible future improvements include:

- Integration with live business databases
- Real-time sales monitoring
- Inventory optimization models
- Customer demand prediction
- Cloud deployment with automated forecasting
- Advanced deep learning forecasting models

---

# Author

**Mandeep Kumar Roshan**

B.Tech Computer Science Engineering

Lovely Professional University

---

# License

This project was developed for educational and internship purposes.
