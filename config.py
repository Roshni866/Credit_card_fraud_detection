# config.py - Configuration for fraud detection project
"""
Simple configuration file for easy parameter modification.
"""

# Dataset configuration
DATA_PATH = 'creditcard.csv'
TEST_SIZE = 0.3
RANDOM_STATE = 42
 

# Model training
N_ESTIMATORS = 100  # For tree-based models
MAX_ITERATIONS = 1000  # For Logistic Regression

# Cross validation
CV_FOLDS = 5

# SMOTE (class imbalance handling)
SMOTE_RANDOM_STATE = 42


# Visualization
PLOT_DPI = 300
PLOT_FORMAT = 'png'

# Model names
MODELS_TO_TRAIN = [
    'Logistic Regression',
    'Random Forest',
    'Gradient Boosting',
    'XGBoost'
]

# Best model to save (set during training)
BEST_MODEL_NAME = 'XGBoost'
