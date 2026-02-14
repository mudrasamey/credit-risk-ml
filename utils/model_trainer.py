import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.data_preprocessing import (
    load_data, engineer_features, impute_missing_values,
    encode_features, split_data, align_test_features,
    split_features_target
)
from utils.model_evaluation import (
    tune_threshold, calculate_metrics, print_metrics
)

class BaseModelTrainer:
    def __init__(self, model_name, encoding_method='onehot', use_scaling=False, balancing_method='none'):
        """
        Base Trainer Class for all models.
        
        Args:
            model_name (str): Name of the model
            encoding_method (str): 'onehot' or 'label'
            use_scaling (bool): Whether to apply StandardScaler
            balancing_method (str): 'none', 'undersample', or 'smote'
        """
        self.model_name = model_name
        self.encoding_method = encoding_method
        self.use_scaling = use_scaling
        self.balancing_method = balancing_method
        
        # State
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_cols = None
        self.encoders = None
        self.scaler = None
        self.best_threshold = 0.5
        
        # Setup directories
        os.makedirs('models/pkl_files', exist_ok=True)

    def load_and_preprocess(self, filepath, sample_size=None, random_state=42):
        """Loads data, feature engineers, imputes, encodes, scales, and balances."""
        
        # Check cache first
        cache_dir = 'data/cache'
        X_path = os.path.join(cache_dir, 'X_cleaned.pkl')
        y_path = os.path.join(cache_dir, 'y.pkl')
        
        if os.path.exists(X_path) and os.path.exists(y_path) and sample_size is None:
            print(f"Loading cached data from {cache_dir}...")
            X = joblib.load(X_path)
            y = joblib.load(y_path)
        else:
            print(f"Loading raw data from {filepath}...")
            df = load_data(filepath, sample_size, random_state)
            
            # Feature Engineering
            print("Feature engineering...")
            df = engineer_features(df)
            
            # Split X, y
            X, y = split_features_target(df)
            
            # Impute
            print("Imputing missing values...")
            X = impute_missing_values(X)
        
        # Encode
        print(f"Encoding features ({self.encoding_method})...")
        X_encoded, self.encoders = encode_features(X, method=self.encoding_method)
        
        # Split (Stratified)
        print("Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X_encoded, y, random_state=random_state
        )
        
        # Balancing (Undersample or SMOTE) - Applied ONLY to Train
        if self.balancing_method == 'undersample':
            print("Applying Random Undersampling...")
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=random_state)
            self.X_train, self.y_train = rus.fit_resample(self.X_train, self.y_train)
            print(f"Resampled Train Shape: {self.X_train.shape}")
            
        elif self.balancing_method == 'smote':
            print("Applying SMOTE...")
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"Resampled Train Shape: {self.X_train.shape}")
            
            # Print Class Distribution
            unique, counts = np.unique(self.y_train, return_counts=True)
            total = sum(counts)
            print(f"Class Distribution after SMOTE:")
            for val, count in zip(unique, counts):
                print(f"  Target={val}: {count} ({count/total:.2%})")
        
        # Save feature columns for alignment
        self.feature_cols = list(self.X_train.columns)
        
        # Align test set (important for one-hot)
        if self.encoding_method == 'onehot':
            self.X_test = align_test_features(self.X_train, self.X_test)
            
        # Scaling (fit on BALANCED train, transform both)
        if self.use_scaling:
            print("Scaling features...")
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
        return self

    def train(self, model):
        """Trains the model."""
        print(f"Training {self.model_name}...")
        self.model = model
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate(self, threshold_metric='mcc'):
        """evaluates the model and tunes threshold."""
        print("Evaluating model...")
        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(self.X_test)[:, 1]
        else:
            print("Model does not support predict_proba, using predict.")
            y_prob = self.model.predict(self.X_test) # Fallback
            
        # Tune threshold
        print(f"Tuning threshold (metric: {threshold_metric})...")
        self.best_threshold, best_score = tune_threshold(
            self.y_test, y_prob, metric=threshold_metric
        )
        print(f"Best threshold: {self.best_threshold:.4f} (Score: {best_score:.4f})")
        
        # Predictions at best threshold
        y_pred = (y_prob >= self.best_threshold).astype(int)
        
        # Metrics
        results = calculate_metrics(self.y_test, y_pred, y_prob)
        print_metrics(self.model_name, results)
        return results

    def save_artifacts(self):
        """Saves model, features, scaler, encoders, and threshold."""
        print("Saving artifacts...")
        base_path = 'models/pkl_files'
        
        # Save Model
        joblib.dump(self.model, f'{base_path}/{self.model_name}.pkl')
        
        # Save Feature Cols
        joblib.dump(self.feature_cols, f'{base_path}/{self.model_name}_features.pkl')
        
        # Save Threshold
        joblib.dump(self.best_threshold, f'{base_path}/{self.model_name}_threshold.pkl')
        
        # Save Scaler (if used)
        if self.scaler:
            joblib.dump(self.scaler, f'{base_path}/{self.model_name}_scaler.pkl')
            
        # Save Encoders (if label encoding used)
        if self.encoders:
            joblib.dump(self.encoders, f'{base_path}/{self.model_name}_encoders.pkl')
            
        print(f"Artifacts saved to {base_path}/")

    def get_scale_pos_weight(self):
        """Calculates scale_pos_weight for XGBoost."""
        if self.y_train is None:
            raise ValueError("Data not loaded yet.")
        
        neg = (self.y_train == 0).sum()
        pos = (self.y_train == 1).sum()
        return neg / pos
