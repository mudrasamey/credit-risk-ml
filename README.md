# 💳 Credit Risk / Loan Default Prediction

## 📌 Overview
This project builds multiple **Machine Learning models** to predict whether a customer will **default on a loan**.

Target:
- 1 → Default
- 0 → No Default

The goal is to help financial institutions **minimize risk** and **improve loan approval decisions**.

---

## 📊 Dataset

**Home Credit Default Risk (Kaggle)**

Features include:
- Income
- Employment details
- Credit history
- Demographics
- Loan information

### Class Distribution
- No Default → ~92%
- Default → ~8%

This creates a **highly imbalanced classification problem**, so:
- Accuracy is misleading
- AUC, Recall, MCC are more meaningful

---

## ⚙️ Machine Learning Pipeline

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

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest
- XGBoost

---

## 🎯 Threshold Optimization

Instead of using default probability = 0.5:
- Threshold tuned using **MCC (Matthews Correlation Coefficient)**
- Improves minority class detection

---

## 📈 Final Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-------|-----------|-----------|-----------|-----------|
| Logistic Regression | 0.8544 | 0.7311 | 0.2441 | 0.3669 | 0.2932 | 0.2213 |
| Decision Tree | 0.7262 | 0.7172 | 0.1679 | 0.5881 | 0.2612 | 0.1982 |
| KNN | 0.9177 | 0.8136 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Naive Bayes | 0.1453 | 0.5780 | 0.0846 | 0.9563 | 0.1555 | 0.0310 |
| Random Forest | 0.8792 | 0.7465 | 0.2706 | 0.2758 | 0.2732 | 0.2073 |
| XGBoost | 0.9118 | 0.7587 | 0.4007 | 0.1446 | 0.2125 | 0.2027 |

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

---

## 👤 Author

Amey Mudras  
Senior Full Stack Developer | Machine Learning Practitioner  
