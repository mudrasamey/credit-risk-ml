import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath, sample_size=None, random_state=42):
    """Loads dataset and optionally samples it."""
    data = pd.read_csv(filepath)
    if sample_size and len(data) > sample_size:
        data = data.sample(sample_size, random_state=random_state)
    return data

def engineer_features(data):
    """Performs standard feature engineering on the dataset."""
    # Create derived features
    data['CREDIT_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    data['ANNUITY_INCOME_RATIO'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    data['EMPLOYED_AGE_RATIO'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    
    # Handle infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data

def split_features_target(data, target_col='TARGET'):
    """Splits dataframe into X (features) and y (target)."""
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    return X, y

def impute_missing_values(X):
    """Imputes missing values: median for numeric, 'Unknown' for categorical."""
    # We operate on a copy to avoid SettingWithCopy warnings if applicable
    X = X.copy()
    for col in X.columns:
        if X[col].dtype != 'object':
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna('Unknown')
    return X

def encode_features(X, method='onehot', encoders=None):
    """
    Encodes categorical features.
    
    Args:
        X (pd.DataFrame): Feature matrix
        method (str): 'onehot' (pd.get_dummies) or 'label' (LabelEncoder)
        encoders (dict): Dictionary of existing encoders for 'label' method (optional)
        
    Returns:
        X_encoded (pd.DataFrame): Encoded features
        encoders (dict/None): Dictionary of fitted encoders (if method='label')
    """
    if method == 'onehot':
        # One-hot encoding
        X_encoded = pd.get_dummies(X)
        # Memory optimization common in original code
        X_encoded = X_encoded.astype(np.float32)
        return X_encoded, None
        
    elif method == 'label':
        X_encoded = X.copy()
        new_encoders = {}
        
        categorical_cols = X_encoded.select_dtypes('object').columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            # If encoders provided (e.g., during inference/test), usage would be tricky 
            # with LabelEncoder as it needs to handle unseen labels. 
            # For this simple refactor, we fit new ones or use provided ones if we were strict. 
            # But the original code fits on the whole X before split in some cases 
            # or relies on simple transformation.
            # 
            # Original code logic:
            # - RF/XGBoost: Fits LabelEncoder on X (before split) in some files, 
            #   or handles commonly.
            
            # Implementation matching original logic: fit on current data
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            new_encoders[col] = le
            
        return X_encoded, new_encoders
    
    else:
        raise ValueError(f"Unknown encoding method: {method}")

def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train-test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

def align_test_features(X_train, X_test, fill_value=0):
    """Aligns X_test columns to match X_train columns (useful for OneHot)."""
    feature_cols = list(X_train.columns)
    X_test_aligned = X_test.reindex(columns=feature_cols, fill_value=fill_value)
    return X_test_aligned
