import sys, os
import time
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
    accuracy_score
)

sys.path.insert(0, os.path.dirname(__file__))
from config import PREDICTION_FEATURES, AVAILABLE_MODELS, MODEL_DISPLAY_NAMES

def build_estimator(name: str, class_weight=None, scale_pos_weight=None):
    if name == 'rf':
        rf_kwargs = {'n_estimators': 100, 'random_state': 42}
        if class_weight == 'balanced':
            rf_kwargs['class_weight'] = 'balanced'
        return RandomForestClassifier(**rf_kwargs)
    elif name == 'knn':
        return KNeighborsClassifier(n_neighbors=5)
    elif name == 'lr':
        lr_kwargs = {'max_iter': 1000, 'random_state': 42, 'solver': 'lbfgs'}
        if class_weight == 'balanced':
            lr_kwargs['class_weight'] = 'balanced'
        return LogisticRegression(**lr_kwargs)
    elif name == 'svm':
        svm_kwargs = {'kernel': 'rbf', 'probability': True, 'random_state': 42}
        if class_weight == 'balanced':
            svm_kwargs['class_weight'] = 'balanced'
        return SVC(**svm_kwargs)
    elif name == 'xgb':
        xgb_kwargs = {'n_estimators': 100, 'random_state': 42, 'eval_metric': 'logloss'}
        if class_weight == 'balanced' and scale_pos_weight is not None:
            xgb_kwargs['scale_pos_weight'] = scale_pos_weight
        return XGBClassifier(**xgb_kwargs)
    elif name == 'baseline_prior':
        return DummyClassifier(strategy='prior', random_state=42)
    elif name == 'baseline_stratified':
        return DummyClassifier(strategy='stratified', random_state=42)
    else:
        raise ValueError(f"Unknown model name: {name}")

def build_pipeline(name: str, class_weight=None, scale_pos_weight=None):
    return {
        'imputer': SimpleImputer(strategy='mean'),
        'scaler': StandardScaler(),
        'model': build_estimator(name, class_weight=class_weight, scale_pos_weight=scale_pos_weight)
    }

def evaluate_all(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    models_to_eval = AVAILABLE_MODELS + ['baseline_prior', 'baseline_stratified']
    
    results = {}
    for name in models_to_eval:
        display_name = MODEL_DISPLAY_NAMES.get(name, name)
        is_baseline = False
        if name == 'baseline_prior':
            display_name = 'Baseline (Always Majority)'
            is_baseline = True
        elif name == 'baseline_stratified':
            display_name = 'Baseline (Stratified Random)'
            is_baseline = True
            
        results[name] = {
            'display_name': display_name,
            'is_baseline': is_baseline,
            'metrics_raw': {
                'average_precision': [],
                'roc_auc': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'balanced_accuracy': [],
                'matthews_corrcoef': [],
                'brier_score': [],
                'accuracy': [],
                'fit_time': []
            }
        }
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    n_pos = np.sum(y_arr == 1)
    n_neg = np.sum(y_arr == 0)
    scale_pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        
        for name in models_to_eval:
            pipeline = build_pipeline(name)
            imputer = pipeline['imputer']
            scaler = pipeline['scaler']
            model = pipeline['model']
            
            start_time = time.time()
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)
            
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)
            
            model.fit(X_train_scaled, y_train)
            fit_time = time.time() - start_time
            
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = model.decision_function(X_test_scaled)
                
            metrics = results[name]['metrics_raw']
            metrics['fit_time'].append(fit_time)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics['average_precision'].append(average_precision_score(y_test, y_prob))
                metrics['roc_auc'].append(roc_auc_score(y_test, y_prob))
                metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
                metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
                metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
                metrics['balanced_accuracy'].append(balanced_accuracy_score(y_test, y_pred))
                metrics['matthews_corrcoef'].append(matthews_corrcoef(y_test, y_pred))
                metrics['brier_score'].append(brier_score_loss(y_test, y_prob))
                metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                
    for name, data in results.items():
        data['metrics'] = {}
        for metric_name, values in data['metrics_raw'].items():
            data['metrics'][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        del data['metrics_raw']
        
    model_names = [n for n in models_to_eval if not results[n]['is_baseline']]
    model_names.sort(key=lambda n: results[n]['metrics']['average_precision']['mean'], reverse=True)
    
    for i, name in enumerate(model_names):
        results[name]['rank'] = i + 1
        
    for name in models_to_eval:
        if results[name]['is_baseline']:
            results[name]['rank'] = None
            
    return results
