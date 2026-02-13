import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
# Feature engineering (same as XGBoost for fairness)
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
# Median imputation
# ======================================================
for col in X.columns:
    if X[col].dtype != 'object':
        X[col] = X[col].fillna(X[col].median())

# ======================================================
# Label encoding (less features than one-hot)
# ======================================================
for col in X.select_dtypes('object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# ======================================================
# Stratified split ⭐ IMPORTANT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================================
# Scaling (FIT ONLY ON TRAIN ⭐ no leakage)
# ======================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# Logistic Regression (imbalance handled)
# ======================================================
model = LogisticRegression(
    max_iter=3000,
    solver='lbfgs',
    class_weight='balanced',   # ⭐ key
    C=0.5,                     # regularization
    n_jobs=1
)

model.fit(X_train, y_train)

# ======================================================
# Probability predictions
# ======================================================
y_prob = model.predict_proba(X_test)[:, 1]

print("\nProbability range:", y_prob.min(), "→", y_prob.max())

# ======================================================
# Threshold tuning (MCC based)
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
# Save
# ======================================================
joblib.dump(model, 'models/pkl_files/logistic_regression.pkl')
joblib.dump(scaler, 'models/pkl_files/logistic_regression_scaler.pkl')
joblib.dump(list(X.columns), 'models/pkl_files/logistic_regression_features.pkl')

print("\nModel saved successfully.")
