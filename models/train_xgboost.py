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

RANDOM_STATE = 42

# ======================================================
# Load dataset
# ======================================================
data = pd.read_csv('data/application_train.csv')

# Optional speed up
# data = data.sample(60000, random_state=RANDOM_STATE)

# ======================================================
# Feature Engineering
# ======================================================
data['CREDIT_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
data['ANNUITY_INCOME_RATIO'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['EMPLOYED_AGE_RATIO'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']

data.replace([np.inf, -np.inf], np.nan, inplace=True)

# ======================================================
# Split target
# ======================================================
X = data.drop('TARGET', axis=1)
y = data['TARGET']

print("\nClass distribution:")
print(y.value_counts(normalize=True))

# ======================================================
# Missing value handling (median)
# ======================================================
for col in X.columns:
    if X[col].dtype != 'object':
        X[col] = X[col].fillna(X[col].median())

# ======================================================
# Label Encoding + SAVE encoders ⭐ CRITICAL
# ======================================================
encoders = {}

for col in X.select_dtypes('object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# ======================================================
# Train/Test split (STRATIFIED)
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# ======================================================
# Handle imbalance
# ======================================================
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print("scale_pos_weight:", scale_pos_weight)

# ======================================================
# Tuned XGBoost
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
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ======================================================
# Probability predictions
# ======================================================
y_prob = model.predict_proba(X_test)[:, 1]

print("\nProbability range:", y_prob.min(), "→", y_prob.max())

# ======================================================
# Threshold tuning (MCC best for imbalance)
# ======================================================
thresholds = np.linspace(0.3, 0.7, 40)

best_t = 0.67
best_mcc = -1

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    mcc = matthews_corrcoef(y_test, preds)

    if mcc > best_mcc:
        best_mcc = mcc
        best_t = t

print("\nBest threshold:", best_t)

# ======================================================
# Final metrics
# ======================================================
y_pred = (y_prob >= best_t).astype(int)

results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

print("\n====== XGBoost Results ======")
for k, v in results.items():
    print(f"{k:10s}: {v:.4f}")

# ======================================================
# Save EVERYTHING ⭐ REQUIRED FOR STREAMLIT
# ======================================================
joblib.dump(model, 'models/pkl_files/xgboost.pkl')
joblib.dump(list(X.columns), 'models/pkl_files/xgboost_features.pkl')
joblib.dump(encoders, 'models/pkl_files/xgboost_encoders.pkl')
joblib.dump(best_t, 'models/pkl_files/xgboost_threshold.pkl')

print("\nModel + features + encoders + threshold saved successfully.")
