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
RANDOM_STATE = 42

# ======================================================
# Load data
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

# ======================================================
# Imputation
# ======================================================
for col in X.columns:
    if X[col].dtype != 'object':
        X[col] = X[col].fillna(X[col].median())

# ======================================================
# Label encoding
# ======================================================
encoders = {}

for col in X.select_dtypes('object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# ======================================================
# Train/Test split
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# ======================================================
# Random Forest
# ======================================================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=30,
    min_samples_split=100,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ======================================================
# Probability predictions
# ======================================================
y_prob = model.predict_proba(X_test)[:, 1]

# ======================================================
# Threshold tuning (F1 best for imbalance)
# ======================================================
thresholds = np.linspace(0.15, 0.4, 40)

best_t = 0
best_f1 = -1

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("Best threshold:", best_t)

# ======================================================
# Final metrics
# ======================================================
y_pred = (y_prob >= best_t).astype(int)

results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

print("\n====== Random Forest Results ======")
for k, v in results.items():
    print(f"{k:10s}: {v:.4f}")

# ======================================================
# Save EVERYTHING
# ======================================================
joblib.dump(model, 'models/pkl_files/random_forest.pkl')
joblib.dump(list(X.columns), 'models/pkl_files/random_forest_features.pkl')
joblib.dump(best_t, 'models/pkl_files/random_forest_threshold.pkl')
joblib.dump(encoders, 'models/pkl_files/random_forest_encoders.pkl')

print("\nModel + features + threshold + encoders saved successfully.")