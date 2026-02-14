import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from utils.model_trainer import BaseModelTrainer

# ======================================================
# Initialize Trainer
# ======================================================
trainer = BaseModelTrainer(
    model_name='random_forest',
    encoding_method='label',
    use_scaling=False
)

# ======================================================
# Load Data & Preprocess
# ======================================================
trainer.load_and_preprocess(
    filepath='data/application_train.csv'
)

# ======================================================
# Define Model
# ======================================================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=30,
    min_samples_split=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# ======================================================
# Train, Evaluate, Save
# ======================================================
trainer.train(model)
trainer.evaluate(threshold_metric='f1')
trainer.save_artifacts()