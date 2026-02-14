import os
import joblib
import pandas as pd
from utils.data_preprocessing import (
    load_data, engineer_features, impute_missing_values, split_features_target
)

def main():
    print("Loading raw data...")
    # Load raw data
    df = load_data('data/application_train.csv')
    
    print("Engineering features...")
    # Add new features
    df = engineer_features(df)
    
    print("Splitting features and target...")
    X, y = split_features_target(df)
    
    print("Imputing missing values...")
    # Fill missing values
    X = impute_missing_values(X)
    
    # Recombine for saving (so we can still split X/y later if needed, 
    # but since trainers expect them separate mostly, let's save X and y separate 
    # OR save a combined cleaning dataframe. 
    # 
    # Actually, ModelTrainer expects data to start from load_and_preprocess logic.
    # To support the current flow, let's save the PRE-ENCODED dataframe (X + y).
    
    print("Saving processed data to cache...")
    os.makedirs('data/cache', exist_ok=True)
    
    # Save X (features) and y (target) separately for efficiency
    joblib.dump(X, 'data/cache/X_cleaned.pkl')
    joblib.dump(y, 'data/cache/y.pkl')
    
    print("✅ Data preparation complete. Cached files saved to data/cache/")

if __name__ == "__main__":
    main()
