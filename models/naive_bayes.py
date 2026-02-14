import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.naive_bayes import GaussianNB
from utils.model_trainer import BaseModelTrainer

# ======================================================
# Initialize Trainer
# ======================================================
trainer = BaseModelTrainer(
    model_name='naive_bayes',
    encoding_method='onehot',
    use_scaling=True
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
model = GaussianNB(var_smoothing=1e-8)

# ======================================================
# Train, Evaluate, Save
# ======================================================
trainer.train(model)
trainer.evaluate(threshold_metric='mcc')
trainer.save_artifacts()
