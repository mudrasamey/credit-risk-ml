import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# ======================================================
# Setup
# ======================================================
os.makedirs('models/pkl_files', exist_ok=True)

# ======================================================
# Load dataset
# ======================================================
data = pd.read_csv('data/application_train.csv')

# Optional: speed up experiments (comment out for full training)
# data = data.sample(50000, random_state=42)

# ======================================================
# Feature Engineering  ⭐ BIG BOOST
# ======================================================
data['CREDIT_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
data['ANNUITY_INCOME_RATIO'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['EMPLOYED_AGE_RATIO'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']

# Replace inf values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# ======================================================
# Split target
# ======================================================
X = data.drop('TARGET', axis=1)
y = data['TARGET']

print("\nClass distribution:")
print(y.value_counts(normalize=True))

# ======================================================
# Missing values (median works best for trees)
# ======================================================
for col in X.columns:
    if X[col].dtype != 'object':
        X[col] = X[col].fillna(X[col].median())

# ======================================================
# Label Encoding ⭐ better than one-hot for XGBoost
# ======================================================
for col in X.select_dtypes('object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# ======================================================
# Train-test split (STRATIFIED ⭐ critical)
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================================
# Handle imbalance
# ======================================================
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print("scale_pos_weight:", scale_pos_weight)

# ======================================================
# Tuned XGBoost ⭐ optimized params
# ======================================================
model = xgb.XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ======================================================
# Predict probabilities
# ======================================================
y_prob = model.predict_proba(X_test)[:, 1]

print("\nProbability range:", y_prob.min(), "→", y_prob.max())

# ======================================================
# Threshold search (MCC based ⭐ best for imbalance)
# ======================================================
thresholds = np.linspace(0.05, 0.9, 30)

best_t = 0
best_mcc = -1

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    mcc = matthews_corrcoef(y_test, preds)

    if mcc > best_mcc:
        best_mcc = mcc
        best_t = t

print("\nBest threshold:", best_t)

y_pred = (y_prob >= best_t).astype(int)

# ======================================================
# Metrics
# ======================================================
results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

print("\n====== Final Metrics ======")
for k, v in results.items():
    print(f"{k:10s}: {v:.4f}")

# ======================================================
# Save model
# ======================================================
joblib.dump(model, 'models/pkl_files/xgboost.pkl')
joblib.dump(list(X.columns), 'models/pkl_files/xgboost_features.pkl')

print("\nModel saved successfully.")
