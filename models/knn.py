# ======================================================
# KNN - Credit Risk Classification (PIPELINE VERSION)
# ======================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline   # ⭐ important


# ======================================================
# Setup
# ======================================================
os.makedirs('models/pkl_files', exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 40000


# ======================================================
# Load dataset
# ======================================================
data = pd.read_csv('data/application_train.csv')

if len(data) > SAMPLE_SIZE:
    data = data.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)


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
# Missing values + One-hot
# ======================================================
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('Unknown')
    else:
        X[col] = X[col].fillna(X[col].median())

X = pd.get_dummies(X)


# ======================================================
# Train-test split
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)


# ======================================================
# Save feature list for Streamlit alignment
# ======================================================
train_cols = list(X_train.columns)
X_test = X_test.reindex(columns=train_cols, fill_value=0)


# ======================================================
# ⭐ KNN PIPELINE
# ======================================================
model = Pipeline(steps=[
    ('scaler', StandardScaler()),      # scaling
    ('smote', SMOTE(random_state=RANDOM_STATE)),  # imbalance fix
    ('pca', PCA(n_components=50)),     # dimensionality reduction
    ('knn', KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='euclidean',
        n_jobs=-1
    ))
])


# ======================================================
# Train
# ======================================================
model.fit(X_train, y_train)


# ======================================================
# Probabilities
# ======================================================
y_prob = model.predict_proba(X_test)[:, 1]


# ======================================================
# Threshold tuning (MCC)
# ======================================================
thresholds = np.linspace(0.05, 0.8, 40)

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
# Final predictions
# ======================================================
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

print("\n====== KNN Results ======")
for k, v in results.items():
    print(f"{k:10s}: {v:.4f}")


# ======================================================
# Save model + features
# ======================================================
joblib.dump(model, 'models/pkl_files/knn.pkl')
joblib.dump(train_cols, 'models/pkl_files/knn_features.pkl')

print("\nPipeline model saved successfully.")
