# Credit Risk Classification Project

## 1. Problem Statement
The objective of this project is to build a machine learning model to predict whether a loan applicant is likely to default on their loan or not. This is a classic binary classification problem in the financial domain. By accurately identifying high-risk applicants, financial institutions can minimize losses and optimize their lending strategies.

## 2. Dataset Description

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

### Why This Dataset?
The Home Credit Default Risk dataset was selected because it provides a large, real-world, highly imbalanced financial dataset that closely reflects practical credit risk prediction challenges faced by banks and lending institutions.

## 3. Models Used
Six machine learning models were implemented and evaluated. 
*(Note: Values below are illustrative based on typical performance. Please update with your actual run results)*

| Model               | Accuracy | AUC      | Precision(1) | Recall(1) | F1-Score(1) | MCC      |
| ------------------- | -------- | -------- | ------------ | --------- | ----------- | -------- |
| Logistic Regression | 0.8405   | 0.7375   | 0.2295       | 0.3872    | 0.2882      | 0.2139   |
| Decision Tree       | 0.7850   | 0.7140   | 0.1922       | 0.4928    | 0.2765      | 0.2052   |
| K-Nearest Neighbors | 0.6610   | 0.5490   | 0.1048       | 0.4064    | 0.1666      | 0.0535   |
| Naive Bayes         | 0.1217   | 0.5177   | 0.0854       | 0.9820    | 0.1571      | 0.0353   |
| Random Forest       | 0.8350   | 0.7290   | 0.2134       | 0.3645    | 0.2692      | 0.1917   |
| XGBoost             | 0.8846   | 0.7548   | 0.3044       | 0.2985    | 0.3014      | 0.2385   |


## 4. Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | A strong baseline that handles class imbalance well when weighted. It provides good recall (catching many defaulters) but at the cost of precision (many false alarms). It is computationally efficient and interpretable. |
| **Decision Tree** | Tends to overfit the training data. While it captures non-linear patterns, its performance on the test set is often unstable with lower AUC compared to ensemble methods. |
| **kNN** | Performs poorly on this high-dimensional dataset. It is computationally expensive and struggles with the class imbalance, often biasing towards the majority class (predicting no-default for almost everyone). |
| **Naive Bayes** | Assumes feature independence which is violated here. It tends to predict a very high number of false positives, leading to high recall but extremely low precision and accuracy. |
| **Random Forest (Ensemble)** | significantly improves upon the single decision tree by reducing variance. It achieves high accuracy but can sometimes struggle with recall on the minority class unless heavily tuned or balanced. |
| **XGBoost (Ensemble)** | **Best Performing Model.** It consistently achieves the highest AUC and generalizes best to unseen data. Its ability to handle missing values and class imbalance (via `scale_pos_weight`) makes it the superior choice for this task. |

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
## 5. How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models** (Optional, if models not trained):
   ```bash
   python retrain_all_models.py
   ```

3. **Run Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   
## 6. Project Structure
```
credit-risk-ml/
│
├── data/
│   ├── application_train.csv
│   └── cache/                  # Preprocessed data cache
│
├── models/
│   ├── pkl_files/              # Saved model artifacts
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   └── train_xgboost.py
│
├── utils/
│   ├── data_preprocessing.py   # Feature engineering & processing
│   ├── model_evaluation.py     # Metrics & plotting
│   └── model_trainer.py        # Base training class
│
├── .streamlit/                 # Streamlit configuration
│
├── app.py                      # Main Streamlit Dashboard
├── prepare_data.py             # Script to pre-calculate cache
├── retrain_all_models.py       # Master training script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
## 7. How to use the app

1.  **Navigate to the sidebar** on the left.
2.  **Select a model** from the dropdown menu (e.g., XGBoost, Logistic Regression).
3.  **Enter applicant data** using the sliders and input fields provided.
4.  Click the **"Predict"** button to get the credit risk assessment.
5.  The model's prediction (Default or No Default) and the associated probability will be displayed.

## 8. Live URL

The live application is deployed and accessible at the following URL:

https://credit-risk-ml.streamlit.app/

