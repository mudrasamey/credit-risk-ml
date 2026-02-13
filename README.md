# 💳 Credit Risk / Loan Default Prediction

## 📌 Problem Statement
The goal of this project is to implement and evaluate multiple Machine Learning classification models to predict whether a customer will default on a loan (Target 1) or not (Target 0). This is a critical task for financial institutions to minimize credit risk and improve the accuracy of loan approval decisions. The project follows a full end-to-end workflow: data preprocessing, model implementation, evaluation using multiple metrics, and deployment via a Streamlit web application.

---

## 📊 Dataset Description

### Overview
- **Home Credit Default Risk (Kaggle)**
- **Dataset Source:** Home Credit Default Risk (Kaggle)
- **Dataset URL:** https://www.kaggle.com/competitions/home-credit-default-risk/data
- **Instance Size:** 307,511
- **Feature Size:**  122 attributes
- **Target Variable:** Binary (Categorical)
  - 0: No Default
  - 1: Default

### Dataset Challenges
**1. Heavily Imbalanced Classes**
The dataset is heavily imbalanced (~92% non-default vs ~8% default).. Accuracy can be misleading.

**2. High Dimensionality**
The dataset contains 122 features, which may lead to overfitting and increased model complexity.

**3. Missing Values**
Several features contain missing or incomplete information that require preprocessing.

**4. Feature Engineering Complexity**
Many variables (especially credit history and bureau data) require transformation or aggregation.

### Feature Description
A few key original and engineered features include:

**Original Features:**
- **`SK_ID_CURR`**: ID of the loan application.
- **`TARGET`**: The target variable (1 - client with payment difficulties, 0 - all other cases).
- **`AMT_INCOME_TOTAL`**: Total income of the client.
- **`AMT_CREDIT`**: Credit amount of the loan.
- **`AMT_ANNUITY`**: Loan annuity.
- **`DAYS_BIRTH`**: Client's age in days at the time of application.
- **`DAYS_EMPLOYED`**: How many days before the application the person started current employment.

**Engineered Features:**
- **`CREDIT_INCOME_RATIO`**: The ratio of the credit amount to the client's total income (`AMT_CREDIT` / `AMT_INCOME_TOTAL`).
- **`ANNUITY_INCOME_RATIO`**: The ratio of the loan annuity to the client's total income (`AMT_ANNUITY` / `AMT_INCOME_TOTAL`).
- **`EMPLOYED_AGE_RATIO`**: The ratio of days employed to the client's age (`DAYS_EMPLOYED` / `DAYS_BIRTH`).

### Data Preprocessing
- Missing value imputation (median/mode)
- One-hot / Label encoding
- Feature engineering:
  - CREDIT_INCOME_RATIO
  - ANNUITY_INCOME_RATIO
  - EMPLOYED_AGE_RATIO
- StandardScaler (for LR, KNN, NB)
- Stratified train/test split

---

## 🤖 Models Implemented

### Evaluation Matrix

| Model                  | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|------------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression    | 0.8755   | 0.7327  | 0.2613    | 0.2807  | 0.2707   | 0.2028  |
| Decision Tree          | 0.7817   | 0.7267  | 0.1912    | 0.5115  | 0.2783   | 0.2105  |
| K-Nearest Neighbors    | 0.7509   | 0.5288  | 0.1036    | 0.2649  | 0.1490   | 0.0400  |
| Naive Bayes            | 0.1094   | 0.5132  | 0.0838    | 0.9891  | 0.1545   | 0.0321  |
| Random Forest          | 0.5756   | 0.8093  | 0.1496    | 0.8870  | 0.2560   | 0.2390  |
| XGBoost                | 0.8626   | 0.7918  | 0.2769    | 0.4156  | 0.3324   | 0.2658  |

---

## 📊 Metric Interpretation

### Why Accuracy is misleading?
Because 92% of customers don’t default, predicting "No Default" always gives high accuracy.

### Important Metrics
- AUC → ranking quality
- Recall → catching defaulters
- Precision → fewer false alarms
- MCC → balanced metric for imbalanced datasets
- F1 → precision/recall balance

---

## 🧠 Model Insights

| ML Model Name                 | Observation about Model Performance                                                                                                                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Logistic Regression**       | Strong baseline model. Handles imbalance well using `class_weight='balanced'`. Produces stable AUC and good recall. Interpretable and fast. One of the most reliable models for this dataset.                                                          |
| **Decision Tree**             | Captures non-linear patterns but tends to overfit. High recall but low precision, resulting in many false positives. Less stable than ensemble methods.                                                                                                |
| **k-Nearest Neighbors (kNN)** | Performs poorly due to high dimensional sparse features and class imbalance. Distance-based methods struggle when minority class is small. Accuracy appears high but predicts mostly majority class → near zero recall. Not suitable for this dataset. |
| **Naive Bayes**               | Assumption of feature independence is violated. Predicts many positives causing extremely high recall but very low precision and accuracy. Weak overall performance.                                                                                   |
| **Random Forest (Ensemble)**  | Much better than single tree. Reduces overfitting and improves stability. Good balance between precision and recall. Strong MCC. Reliable ensemble baseline.                                                                                           |
| **XGBoost (Ensemble)**        | Best overall performer. Highest AUC and strong generalization. Handles imbalance well with boosting and class weighting. Captures complex patterns effectively. Final recommended model.                                                               |


---
## 🏆 Key Findings

### ✅ Best Overall → XGBoost
- Highest MCC
- High AUC
- Balanced precision & recall
- Strong generalization

### ✅ Strong Baseline → Logistic Regression
- Stable 
- Interpretable
- Fast

## ⚠ Random Forest
- Very high recall
- Slightly lower precision

## ❌ Poor performers
- KNN → struggles in high dimensions
- Naive Bayes → independence assumption violated

---

## 🖥️ Streamlit Dashboard

Features:
- Dataset overview
- Class distribution charts
- Model comparison table
- ROC curves
- Precision-Recall curves
- Confusion matrices
- Feature importance
- Adjustable threshold
- Live predictions

---

## ▶️ How to Run

### Install dependencies
pip install -r requirements.txt

### Train models
python models/logistic_regression.py  
python models/decision_tree.py  
python models/knn.py  
python models/naive_bayes.py  
python models/random_forest.py  
python models/train_xgboost.py  

### Launch dashboard
streamlit run app.py

---

## 📂 Project Structure

```
credit-risk-ml/
│
├── data/
│
├── models/
│   ├── pkl_files/
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   └── train_xgboost.py
│
├── .streamlit/
│
├── app.py
├── requirements.txt
├── retrain_all_models.py
└── README.md
```

---

## 🚀 Future Improvements

- SMOTE / Oversampling
- Hyperparameter tuning
- Feature selection
- SHAP explainability
- Probability calibration
- Cost-sensitive learning
