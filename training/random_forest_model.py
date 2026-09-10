"""
random_forest_model.py — Train and save the Random Forest attrition pipeline.

Features are imported from config.py so they can never drift from what app.py
expects. The dataset path is relative to the project root.
"""
import os
import sys

# Allow importing config from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PREDICTION_FEATURES  # noqa: E402

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET = os.path.join(
    os.path.dirname(__file__), '..', 'datasets',
    'IBM-HR-Analytics-Employee-Attrition-and-Performance.csv'
)
data = pd.read_csv(DATASET)

# ── Features & target ─────────────────────────────────────────────────────────
X = data[PREDICTION_FEATURES].copy()
y = data['Attrition']

le = LabelEncoder()
X['OverTime'] = le.fit_transform(X['OverTime'])
y = le.fit_transform(y)

imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Model ─────────────────────────────────────────────────────────────────────
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

pipeline = {'model': rf_model, 'scaler': scaler, 'imputer': imputer}

out = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_pipeline.pkl')
joblib.dump(pipeline, out)
print(f"Random Forest pipeline saved → {out}")
print(f"n_features_in_: {rf_model.n_features_in_}")