import joblib
import pandas as pd
import numpy as np

# Load the trained pipeline
model_path = "C:/PHYTON/my projects/project/models/rf_pipeline.pkl"  # Update with your actual path
pipeline = joblib.load(model_path)  # Load the pipeline (contains model, scaler, imputer)

# Define the features the model expects (based on training)
expected_features = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'OverTime', 'YearsAtCompany',
                    'WorkLifeBalance', 'JobLevel', 'DistanceFromHome']

# Static input data (only the 8 required features)
input_data = {
    'Age': 35,
    'MonthlyIncome': 65,
    'JobSatisfaction': 4,
    'OverTime': 'No',
    'YearsAtCompany': 5,
    'WorkLifeBalance': 3,
    'JobLevel': 3,
    'DistanceFromHome': 7
}

# Debugging: Check expected features
print("\nModel expects:", pipeline['model'].n_features_in_, "features")
print("Expected feature names:", expected_features)
print("Features in input data:", list(input_data.keys()))

# Verify all required features are present
missing_features = set(expected_features) - set(input_data.keys())
if missing_features:
    print(f"Error: Missing features {missing_features}")
    exit()

# Convert input_data to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical variable 'OverTime'
input_df['OverTime'] = input_df['OverTime'].map({'No': 0, 'Yes': 1})

# Preprocess the input data using the pipeline's imputer and scaler
input_processed = pipeline['imputer'].transform(input_df)  # Handle missing values (if any)
input_processed = pipeline['scaler'].transform(input_processed)  # Scale features

# Make prediction
prediction = pipeline['model'].predict(input_processed)

# Display the result
if prediction[0] == 1:
    print("\n⚠️ Employee is expected to leave the company.")
else:
    print("\n✅ Employee is expected to stay in the company.")