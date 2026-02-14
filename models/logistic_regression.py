import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from utils.model_trainer import BaseModelTrainer

# ======================================================
# Initialize Trainer
# ======================================================
trainer = BaseModelTrainer(
    model_name='logistic_regression',
    encoding_method='onehot',
    use_scaling=True,
    balancing_method='none'
)

# ======================================================
# Load Data & Preprocess
# ======================================================
trainer.load_and_preprocess(
    filepath='data/application_train.csv',
    # sample_size=50000 
)

# ======================================================
# Define Model
# ======================================================
model = LogisticRegression(
    solver='saga',
    max_iter=3000,
    class_weight='balanced',
    C=0.5,
    n_jobs=-1,
    random_state=42
)

# ======================================================
# Train, Evaluate, Save
# ======================================================
trainer.train(model)
trainer.evaluate(threshold_metric='mcc')
trainer.save_artifacts()
