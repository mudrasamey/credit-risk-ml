import subprocess
import sys
import time

def run_model(script_name):
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    # Run the script using the current python executable
    try:
        subprocess.run([sys.executable, f'models/{script_name}'], check=True)
        print(f"\n✅ {script_name} completed in {time.time() - start_time:.2f} seconds")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_name} failed with error code {e.returncode}")

def main():
    models = [
        'logistic_regression.py',
        'random_forest.py',
        'train_xgboost.py',
        'decision_tree.py',
        'naive_bayes.py',
        'knn.py'
    ]
    
    print("Starting retraining of all models...")
    total_start = time.time()
    
    for model in models:
        run_model(model)
        
    print(f"\nAll models retrained in {time.time() - total_start:.2f} seconds.")

if __name__ == "__main__":
    main()
