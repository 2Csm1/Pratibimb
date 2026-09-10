"""
knn_model.py — Train and save the K-Nearest Neighbors attrition pipeline.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PREDICTION_FEATURES  # noqa: E402

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

DATASET = os.path.join(
    os.path.dirname(__file__), '..', 'datasets',
    'IBM-HR-Analytics-Employee-Attrition-and-Performance.csv'
)
data = pd.read_csv(DATASET)

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

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

pipeline = {'model': knn_model, 'scaler': scaler, 'imputer': imputer}

out = os.path.join(os.path.dirname(__file__), '..', 'models', 'knn_pipeline.pkl')
joblib.dump(pipeline, out)
print(f"KNN pipeline saved → {out}")
print(f"n_features_in_: {knn_model.n_features_in_}")