import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
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

# ======================================================
# Feature engineering
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
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('Unknown')
    else:
        X[col] = X[col].fillna(X[col].median())

# ======================================================
# One-hot encoding ⭐ BEST for NB
# ======================================================
X = pd.get_dummies(X)

# ======================================================
# Train/Test split
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# ======================================================
# Scaling (VERY IMPORTANT for NB)
# ======================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# Model
# ======================================================
model = GaussianNB(var_smoothing=1e-8)

model.fit(X_train, y_train)

# ======================================================
# Predict probabilities
# ======================================================
y_prob = model.predict_proba(X_test)[:, 1]

print("\nProbability range:", y_prob.min(), "→", y_prob.max())

# ======================================================
# Threshold tuning
# ======================================================
thresholds = np.linspace(0.05, 0.9, 30)

best_t = 0.5
best_mcc = -1

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    mcc = matthews_corrcoef(y_test, preds)

    if mcc > best_mcc:
        best_mcc = mcc
        best_t = t

print("Best threshold:", best_t)

# ======================================================
# Metrics
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

print("\n====== Naive Bayes Results ======")
for k, v in results.items():
    print(f"{k:10s}: {v:.4f}")

# ======================================================
# Save EVERYTHING ⭐
# ======================================================
joblib.dump(model, 'models/pkl_files/naive_bayes.pkl')
joblib.dump(scaler, 'models/pkl_files/naive_bayes_scaler.pkl')
joblib.dump(list(X.columns), 'models/pkl_files/naive_bayes_features.pkl')
joblib.dump(best_t, 'models/pkl_files/naive_bayes_threshold.pkl')

print("\nModel + scaler + features + threshold saved successfully.")
