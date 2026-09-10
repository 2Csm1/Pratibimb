import sys, os
import json
import hashlib
import random
import datetime
import numpy as np
import pandas as pd
import joblib
import sklearn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PREDICTION_FEATURES, AVAILABLE_MODELS, MODEL_DISPLAY_NAMES
from evaluation import build_pipeline, evaluate_all

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    random.seed(42)
    np.random.seed(42)

    data_path = os.path.join("datasets", "IBM-HR-Analytics-Employee-Attrition-and-Performance.csv")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    dataset_sha256 = compute_sha256(data_path)
    
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    X = df[PREDICTION_FEATURES]
    y = df['Attrition']
    
    prevalence = float(y.mean())
    n_rows = len(df)
    n_features = len(PREDICTION_FEATURES)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    print("Training and saving production pipelines (no class weights)...")
    manifest_models = {}
    metrics_summary = {}
    
    for name in AVAILABLE_MODELS:
        random.seed(42)
        np.random.seed(42)
        pipeline = build_pipeline(name)
        imputer = pipeline['imputer']
        scaler = pipeline['scaler']
        model = pipeline['model']
        
        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)
        
        model.fit(X_scaled, y)
        
        model_path = os.path.join("models", f"{name}_pipeline.pkl")
        joblib.dump(pipeline, model_path, protocol=4)
        print(f"Saved {name} pipeline to {model_path}")
        
        manifest_models[name] = {
            'path': f"models/{name}_pipeline.pkl",
            'n_features_in_': n_features
        }
        
    print("Evaluating models with cross-validation...")
    eval_results = evaluate_all(X, y, seed=42)
    
    best_model = None
    for name in AVAILABLE_MODELS:
        if not eval_results[name]['is_baseline']:
            metrics_summary[name] = {
                'average_precision_mean': eval_results[name]['metrics']['average_precision']['mean'],
                'roc_auc_mean': eval_results[name]['metrics']['roc_auc']['mean']
            }
            if eval_results[name]['rank'] == 1:
                best_model = name

    metrics_data = {
        'models': eval_results,
        'best_model': best_model,
        'prevalence': prevalence,
        'n_rows': n_rows,
        'n_features': n_features,
        'features': list(PREDICTION_FEATURES),
        'seed': 42,
        'sklearn_version': sklearn.__version__,
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'dataset_sha256': dataset_sha256
    }
    
    metrics_path = os.path.join("artifacts", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
    
    manifest_data = {
        'training_date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'dataset_sha256': dataset_sha256,
        'sklearn_version': sklearn.__version__,
        'python_version': sys.version,
        'models': manifest_models,
        'metrics_summary': metrics_summary
    }
    
    manifest_path = os.path.join("models", "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=4)
    print(f"Saved manifest to {manifest_path}")
    
if __name__ == "__main__":
    main()
