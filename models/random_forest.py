import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
# data = data.sample(60000, random_state=42)

# ======================================================
# Feature engineering (same as other models)
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
# Label encoding ⭐ trees prefer this
# ======================================================
for col in X.select_dtypes('object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# ======================================================
# Stratified split ⭐ critical
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================================
# Tuned Random Forest ⭐
# ======================================================
model = RandomForestClassifier(
    n_estimators=300,          # ⭐ more trees = better
    max_depth=10,             # prevents overfit
    min_samples_leaf=50,
    min_samples_split=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=1                  # avoids resource tracker noise
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

print("\n====== Random Forest Results ======")
for k, v in results.items():
    print(f"{k:10s}: {v:.4f}")

# ======================================================
# Save
# ======================================================
joblib.dump(model, 'models/pkl_files/random_forest.pkl')
joblib.dump(list(X.columns), 'models/pkl_files/random_forest_features.pkl')

print("\nModel saved successfully.")
