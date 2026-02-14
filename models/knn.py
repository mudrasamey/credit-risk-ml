import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from utils.model_trainer import BaseModelTrainer

# ======================================================
# Initialize Trainer
# ======================================================
# Note: use_scaling=True because we removed it from Pipeline
trainer = BaseModelTrainer(
    model_name='knn',
    encoding_method='onehot',
    use_scaling=True,
    balancing_method='none'
)

# ======================================================
# Load Data & Preprocess
# ======================================================
trainer.load_and_preprocess(
    filepath='data/application_train.csv',
    # sample_size=40000 # Commented out to match other models (fair comparison)
)

# ======================================================
# Define Model (Pipeline)
# ======================================================
model = Pipeline(steps=[
    # Scaler and SMOTE handled by Trainer now
    ('pca', PCA(n_components=50)),                # dimensionality reduction
    ('knn', KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='euclidean',
        n_jobs=-1
    ))
])

# ======================================================
# Train, Evaluate, Save
# ======================================================
trainer.train(model)
trainer.evaluate(threshold_metric='mcc')
trainer.save_artifacts()
