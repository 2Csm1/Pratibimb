"""
testing.py — Quick smoke-test for a saved pipeline.
Run from the project root: python training/testing.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PREDICTION_FEATURES  # noqa: E402

import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_pipeline.pkl')
pipeline = joblib.load(MODEL_PATH)

print(f"\nModel expects: {pipeline['model'].n_features_in_} features")
print(f"Feature names: {PREDICTION_FEATURES}")

input_data = {
    'Age': 35,
    'MonthlyIncome': 6500,
    'JobSatisfaction': 4,
    'OverTime': 'No',
    'YearsAtCompany': 5,
    'WorkLifeBalance': 3,
    'JobLevel': 3,
    'DistanceFromHome': 7,
}

input_df = pd.DataFrame([input_data])
input_df['OverTime'] = input_df['OverTime'].map({'No': 0, 'Yes': 1})

input_processed = pipeline['imputer'].transform(input_df)
input_processed = pipeline['scaler'].transform(input_processed)
prediction = pipeline['model'].predict(input_processed)

if prediction[0] == 1:
    print("\n⚠️ Employee is expected to leave the company.")
else:
    print("\n✅ Employee is expected to stay in the company.")