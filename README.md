# SaaS Customer Churn Prediction Pipeline

This repository implements a production-grade Machine Learning pipeline using Python to predict subscription churn for a hypothetical SaaS enterprise. The system leverages LightGBM optimized via cross-validation to identify high-risk customer accounts proactively.

## 🛠️ Project Structure
```text
├── main.py              # End-to-end execution script (Data Gen -> Eval)
├── requirements.txt     # Python dependency configuration
└── README.md            # Technical documentation
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python main.py
```

## 📊 Core Architecture Features
* **Preprocessing:** Automated handling of skewed outliers via the Interquartile Range (IQR) method and numerical normalization via `StandardScaler`.
* **Imbalance Treatment:** Implements synthetic oversampling to handle severe class imbalances.
* **Algorithm:** LightGBM Classifier chosen for high-velocity execution and tabular efficiency.
* **Validation:** Stratified 5-Fold Cross-Validation monitoring Recall as the primary operational metric.
