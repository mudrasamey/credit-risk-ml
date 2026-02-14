import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
from utils.model_trainer import BaseModelTrainer

# ======================================================
# Initialize Trainer
# ======================================================
trainer = BaseModelTrainer(
    model_name='xgboost',
    encoding_method='label',
    use_scaling=False
)

# ======================================================
# Load Data & Preprocess
# ======================================================
trainer.load_and_preprocess(
    filepath='data/application_train.csv',
    # sample_size=60000  # Optional: Uncomment for faster testing
)

# ======================================================
# Configure Model (with Boosted Recall)
# ======================================================
scale_pos_weight = trainer.get_scale_pos_weight()
# Boosting the weight by 1.5x to aggressively target Recall
recall_boost_factor = 1.5 
final_weight = scale_pos_weight * recall_boost_factor

print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")
print(f"Applied final_weight (with {recall_boost_factor}x boost): {final_weight:.4f}")

model = xgb.XGBClassifier(
    n_estimators=1000,            # Increased slightly
    learning_rate=0.02,           # Lowered slightly for precision
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,           # Lowered to capture minority samples
    max_delta_step=1,             # Crucial for imbalanced data convergence
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    scale_pos_weight=final_weight,
    eval_metric='auc',
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)


# ======================================================
# Train, Evaluate, Save
# ======================================================
trainer.train(model)
trainer.evaluate(threshold_metric='mcc')
trainer.save_artifacts()
