import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

def tune_threshold(y_true, y_prob, metric='mcc', threshold_range=(0.05, 0.9), num_steps=30):
    """
    Finds the best classification threshold based on the specified metric.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('mcc' or 'f1')
        threshold_range: Tuple of (min, max) threshold
        num_steps: Number of thresholds to try
    
    Returns:
        best_threshold: The threshold that maximizes the metric
        best_score: The best score achievied
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_steps)
    best_threshold = 0.5
    best_score = -1
    
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        
        if metric == 'mcc':
            score = matthews_corrcoef(y_true, preds)
        elif metric == 'f1':
            score = f1_score(y_true, preds)
        else:
            # Default to MCC if unknown
            score = matthews_corrcoef(y_true, preds)
            
        if score > best_score:
            best_score = score
            best_threshold = t
            
    return best_threshold, best_score

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates standard binary classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC)
        
    Returns:
        results: Dictionary of metrics
    """
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    
    if y_prob is not None:
        try:
            results["AUC"] = roc_auc_score(y_true, y_prob)
        except:
            results["AUC"] = 0.0
            
    return results

def print_metrics(model_name, results):
    """Prints metrics in a formatted way."""
    print(f"\n====== {model_name} Results ======")
    for k, v in results.items():
        print(f"{k:10s}: {v:.4f}")
