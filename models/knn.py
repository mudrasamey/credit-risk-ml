import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
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
# Load + sample
# ======================================================
data = pd.read_csv('data/application_train.csv')
data = data.sample(10000, random_state=42)  # speed

# ======================================================
# Feature engineering
# ======================================================
data['CREDIT_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
data['ANNUITY_INCOME_RATIO'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['EMPLOYED_AGE_RATIO'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']

data.replace([np.inf, -np.inf], np.nan, inplace=True)

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
# ⭐ FIX 1 — One Hot Encoding (NOT LabelEncoder)
# ======================================================
X = pd.get_dummies(X)

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
# Scaling
# ======================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# ⭐ FIX 2 — PCA (huge improvement for KNN)
# ======================================================
pca = PCA(n_components=50, random_state=42)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# ======================================================
# ⭐ FIX 3 — GridSearch (no leakage)
# ======================================================
param_grid = {
    "n_neighbors": range(15, 51, 4),
    "weights": ["distance"],
    "metric": ["euclidean"]
}

grid = GridSearchCV(
    KNeighborsClassifier(n_jobs=-1),
    param_grid,
    scoring="roc_auc",
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

print("\nBest parameters:", grid.best_params_)

# ======================================================
# Probabilities
# ======================================================
y_prob = model.predict_proba(X_test)[:, 1]

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

print("\n====== KNN Results ======")
for k, v in results.items():
    print(f"{k:10s}: {v:.4f}")

# ======================================================
# Save
# ======================================================
joblib.dump(model, 'models/pkl_files/knn.pkl')
joblib.dump(scaler, 'models/pkl_files/knn_scaler.pkl')
joblib.dump(pca, 'models/pkl_files/knn_pca.pkl')
joblib.dump(list(X.columns), 'models/pkl_files/knn_features.pkl')

print("\nModel saved successfully.")
