import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.tree import DecisionTreeClassifier
from utils.model_trainer import BaseModelTrainer

# ======================================================
# Initialize Trainer
# ======================================================
trainer = BaseModelTrainer(
    model_name='decision_tree',
    encoding_method='label',
    use_scaling=False,
    balancing_method='none'
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
model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=100,
    min_samples_split=500,
    class_weight='balanced',
    random_state=42
)

# ======================================================
# Train, Evaluate, Save
# ======================================================
trainer.train(model)
trainer.evaluate(threshold_metric='mcc')
trainer.save_artifacts()
