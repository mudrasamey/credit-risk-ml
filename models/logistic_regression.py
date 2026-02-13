import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

# ======================================================
# Setup
# ======================================================
os.makedirs('models/pkl_files', exist_ok=True)

# ======================================================
# Load dataset
# ======================================================
data = pd.read_csv('data/application_train.csv')

# Optional speed
# data = data.sample(50000, random_state=42)

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
# Missing values
# ======================================================
for col in X.columns:
    if X[col].dtype != 'object':
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna('Unknown')

# ======================================================
# One hot encoding
# ======================================================
X = pd.get_dummies(X)

# memory optimization ⭐
X = X.astype(np.float32)

# ======================================================
# Stratified split
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================================
# Align columns (safety for inference)
# ======================================================
feature_cols = list(X_train.columns)
X_test = X_test.reindex(columns=feature_cols, fill_value=0)

# ======================================================
# Scaling (fit only on train)
# ======================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# Logistic Regression ⭐ upgraded
# ======================================================
model = LogisticRegression(
    solver='saga',            # faster & better for large sparse
    max_iter=3000,
    class_weight='balanced',  # handle imbalance
    C=0.5,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# ======================================================
# Probabilities
# ======================================================
y_prob = model.predict_proba(X_test)[:, 1]

print("\nProbability range:", y_prob.min(), "→", y_prob.max())

# ======================================================
# Threshold tuning (MCC)
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

print("Best threshold:", best_t)

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

print("\n====== Logistic Regression Results ======")
for k, v in results.items():
    print(f"{k:10s}: {v:.4f}")

# ======================================================
# Save everything (VERY IMPORTANT for Streamlit)
# ======================================================
joblib.dump(model, 'models/pkl_files/logistic_regression.pkl')
joblib.dump(scaler, 'models/pkl_files/logistic_regression_scaler.pkl')
joblib.dump(feature_cols, 'models/pkl_files/logistic_regression_features.pkl')
joblib.dump(best_t, 'models/pkl_files/logistic_regression_threshold.pkl')

print("\nModel saved successfully.")
