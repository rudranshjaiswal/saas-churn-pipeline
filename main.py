"""
SaaS Customer Churn Prediction Pipeline
Designed for Week 3 ML Development and Evaluation Plan.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==========================================
# STEP 1: HYPOTHETICAL DATA GENERATION
# ==========================================
def generate_synthetic_data(num_samples=2000):
    np.random.seed(42)
    print("[INFO] Generating synthetic SaaS customer metrics...")
    
    data = {
        'customer_id': [f"CUST_{i}" for i in range(num_samples)],
        'usage_velocity': np.random.exponential(scale=2.0, size=num_samples), # Skewed usage metric
        'api_error_rate': np.random.uniform(0.0, 0.05, size=num_samples),
        'support_tickets': np.random.poisson(lam=1.5, size=num_samples),
        'contract_type': np.random.choice(['Monthly', 'Annual', 'Bi-Annual'], size=num_samples, p=[0.6, 0.3, 0.1]),
        'region': np.random.choice(['US', 'EU', 'APAC'], size=num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce outliers into usage velocity to simulate dirty real-world data
    outlier_indices = np.random.choice(num_samples, size=40, replace=False)
    df.loc[outlier_indices, 'usage_velocity'] = df.loc[outlier_indices, 'usage_velocity'] * 10
    
    # Generate imbalanced target label (approx 15% Churn)
    churn_probability = (
        0.4 * (df['contract_type'] == 'Monthly') +
        0.3 * (df['api_error_rate'] > 0.03) +
        0.2 * (df['support_tickets'] > 3) -
        0.3 * (df['usage_velocity'])
    )
    # Map probabilities to binary targets safely
    prob_normalized = 1 / (1 + np.exp(-churn_probability))
    df['churn'] = (prob_normalized > 0.62).astype(int)
    
    print(f"[INFO] Dataset generated successfully. Total Shape: {df.shape} | Churn Rate: {df['churn'].mean():.2%}")
    return df

# ==========================================
# STEP 2: CUSTOM OUTLIER HANDLING (IQR METHOD)
# ==========================================
def handle_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Cap outliers to upper/lower bounds to prevent model distortion
        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    return df_clean

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    raw_data = generate_synthetic_data(num_samples=2500)
    
    # 2. Preprocessing & Cleaning Outliers
    numerical_cols = ['usage_velocity', 'api_error_rate', 'support_tickets']
    categorical_cols = ['contract_type', 'region']
    
    cleaned_data = handle_outliers_iqr(raw_data, numerical_cols)
    
    # Separate Features and Target
    X = cleaned_data[numerical_cols + categorical_cols]
    y = cleaned_data['churn']
    
    # 3. Structural Column Transformers (Encoding & Scaling)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ]
    )
    
    # 4. Define Imbalance Mitigation (SMOTE) & Model Training Pipeline
    # LightGBM hyperparameter initialization
    lgb_params = {
        'objective': 'binary',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'verbosity': -1,
        'random_state': 42
    }
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)), # Synthetically balances target labels
        ('classifier', lgb.LGBMClassifier(**lgb_params))
    ])
    
    # 5. Stratified 5-Fold Cross-Validation Matrix Setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_recalls = []
    cv_precisions = []
    cv_aucs = []
    
    print("\n[INFO] Starting Stratified 5-Fold Cross-Validation Loop...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train Pipeline (Fits transformers, applies SMOTE, trains LightGBM)
        pipeline.fit(X_train, y_train)
        
        # Inference Evaluation
        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)[:, 1]
        
        # Score Computations
        fold_recall = recall_score(y_val, preds)
        fold_precision = precision_score(y_val, preds)
        fold_auc = roc_auc_score(y_val, probs)
        
        cv_recalls.append(fold_recall)
        cv_precisions.append(fold_precision)
        cv_aucs.append(fold_auc)
        
        print(f"--- Fold {fold} Metrics -> Recall: {fold_recall:.4f} | Precision: {fold_precision:.4f} | ROC AUC: {fold_auc:.4f}")
        
    # 6. Print Final Aggregated Pipeline Validation Performance Report
    print("\n" + "="*50)
    print("FINAL MACHINE LEARNING CROSS-VALIDATION SUMMARY")
    print("="*50)
    print(f"Mean Pipeline Recall (Primary Metric) : {np.mean(cv_recalls):.4f}")
    print(f"Mean Pipeline Precision              : {np.mean(cv_precisions):.4f}")
    print(f"Mean Pipeline ROC AUC                : {np.mean(cv_aucs):.4f}")
    print("="*50)
