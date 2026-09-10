"""
config.py — Single source of truth for Pratibimb.

Every module that needs PREDICTION_FEATURES, AVAILABLE_MODELS, or upload
limits must import from here. This prevents the 8-vs-12 drift that existed
between the training scripts and app.py.
"""

# ── Model features ────────────────────────────────────────────────────────────
# These 8 features match exactly what every pipeline .pkl was trained on.
# Do NOT add features here without retraining ALL five models.
PREDICTION_FEATURES = [
    'Age',
    'MonthlyIncome',
    'JobSatisfaction',
    'OverTime',
    'YearsAtCompany',
    'WorkLifeBalance',
    'JobLevel',
    'DistanceFromHome',
]

# ── Model registry ────────────────────────────────────────────────────────────
AVAILABLE_MODELS = ['rf', 'knn', 'lr', 'svm', 'xgb']

MODEL_DISPLAY_NAMES = {
    'rf':  'Random Forest',
    'knn': 'K-Nearest Neighbors',
    'lr':  'Logistic Regression',
    'svm': 'Support Vector Machine',
    'xgb': 'XGBoost',
}

# ── Upload limits ─────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB hard cap on any uploaded file
MAX_DISPLAY_ROWS = 100                  # Max rows shown in batch results table
MAX_VIZ_ROWS     = 5_000               # Sample cap for visualisation charts
MAX_STORED_FILES = 10                   # Oldest files pruned beyond this count

ALLOWED_EXTENSIONS = {'csv'}

# Columns that MUST be present for a dataset to be usable by the prediction
# pipeline. The preprocess route additionally requires 'Attrition'.
REQUIRED_PREDICTION_COLS = list(PREDICTION_FEATURES)
REQUIRED_PREPROCESS_COLS = list(PREDICTION_FEATURES) + ['Attrition']
