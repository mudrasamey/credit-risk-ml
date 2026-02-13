import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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

# ======================================================
# Median imputation
# ======================================================
for col in X.columns:
    if X[col].dtype != 'object':
        X[col] = X[col].fillna(X[col].median())

# ======================================================
# Label encoding (trees prefer this)
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
# Decision Tree (regularized)
# ======================================================
model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=100,
    min_samples_split=500,
    class_weight='balanced',
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

# ======================================================
# Probabilities
# ======================================================
y_prob = model.predict_proba(X_test)[:, 1]

# ======================================================
# Threshold tuning (MCC best for imbalance)
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

print("\n====== Decision Tree Results ======")
for k, v in results.items():
    print(f"{k:10s}: {v:.4f}")

# ======================================================
# Save EVERYTHING (important for Streamlit)
# ======================================================
joblib.dump(model, 'models/pkl_files/decision_tree.pkl')
joblib.dump(list(X.columns), 'models/pkl_files/decision_tree_features.pkl')
joblib.dump(encoders, 'models/pkl_files/decision_tree_encoders.pkl')
joblib.dump(best_t, 'models/pkl_files/decision_tree_threshold.pkl')

print("\nModel + features + threshold saved successfully.")
