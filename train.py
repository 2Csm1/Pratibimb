"""
train.py — Train production pipelines and generate evaluation artifacts.

Usage:  python train.py

Outputs:
    models/<name>_pipeline.pkl   — fitted dict pipelines (imputer, scaler, model)
    models/manifest.json         — training metadata
    artifacts/metrics.json       — cross-validated metrics, calibration data,
                                   threshold sensitivity analysis
"""

import sys
import os
import json
import hashlib
import random
import datetime
import warnings

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PREDICTION_FEATURES, AVAILABLE_MODELS, MODEL_DISPLAY_NAMES
from evaluation import build_pipeline, evaluate_all, assess_calibration
from costs import (
    optimal_threshold,
    sweep_thresholds,
    REPLACEMENT_COST_FRACTION,
    INTERVENTION_COST,
    INTERVENTION_SUCCESS_RATE,
)


def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _calibration_comparison(pipeline_dict, X_scaled, y, seed=42):
    """Compare raw vs isotonic vs sigmoid calibration via 5-fold CV.

    With only ~237 positive cases (16% of 1470), isotonic regression
    frequently overfits — so we expect sigmoid to win on smaller models.

    Returns (best_method, results_dict) where best_method is one of
    'raw', 'isotonic', 'sigmoid'.
    """
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    base_model = pipeline_dict['model']
    methods = {
        'raw': None,
        'isotonic': 'isotonic',
        'sigmoid': 'sigmoid',
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = {}

    for label, method in methods.items():
        if method is None:
            # Raw: just get out-of-fold probabilities
            oof_probs = np.zeros(len(y))
            for train_idx, test_idx in skf.split(X_scaled, y):
                mdl = clone(base_model)
                mdl.fit(X_scaled[train_idx], y[train_idx])
                if hasattr(mdl, 'predict_proba'):
                    oof_probs[test_idx] = mdl.predict_proba(
                        X_scaled[test_idx]
                    )[:, 1]
                else:
                    oof_probs[test_idx] = mdl.decision_function(
                        X_scaled[test_idx]
                    )
        else:
            # Calibrated: wrap in CalibratedClassifierCV with cv=5
            cal = CalibratedClassifierCV(
                clone(base_model), method=method, cv=5
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cal.fit(X_scaled, y)
            # Get out-of-fold probs via cross_val_predict
            oof_probs = np.zeros(len(y))
            for train_idx, test_idx in skf.split(X_scaled, y):
                cal_fold = CalibratedClassifierCV(
                    clone(base_model), method=method, cv=3
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cal_fold.fit(X_scaled[train_idx], y[train_idx])
                oof_probs[test_idx] = cal_fold.predict_proba(
                    X_scaled[test_idx]
                )[:, 1]

        brier = float(brier_score_loss(y, oof_probs))
        results[label] = {'brier_score': brier}

    best_method = min(results, key=lambda m: results[m]['brier_score'])
    return best_method, results


def _fit_calibrated_model(base_model, X_scaled, y, method):
    """Fit a CalibratedClassifierCV and return it (or the raw model)."""
    if method == 'raw':
        mdl = clone(base_model)
        mdl.fit(X_scaled, y)
        return mdl
    else:
        cal = CalibratedClassifierCV(
            clone(base_model), method=method, cv=5
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal.fit(X_scaled, y)
        return cal


def _run_sensitivity_analysis(y, oof_probs, monthly_incomes):
    """Sweep INTERVENTION_SUCCESS_RATE over [0.1..0.5] and record impacts."""
    from costs import replacement_cost as _rc

    rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    median_income = float(np.median(monthly_incomes))
    R = REPLACEMENT_COST_FRACTION * 12 * median_income
    n = len(y)

    rows = []
    for s in rates:
        thresh = min(1.0, INTERVENTION_COST / (R * s)) if R * s > 0 else 1.0
        # compute expected cost at this threshold for the median employee
        sweep = sweep_thresholds(y, oof_probs, monthly_incomes,
                                 grid=np.array([thresh]))
        cost_per_employee = sweep['per_employee_costs'][0]
        rows.append({
            'success_rate': s,
            'threshold': round(float(thresh), 4),
            'expected_cost_per_employee': round(float(cost_per_employee), 2),
            'median_monthly_income': round(median_income, 2),
        })

    return rows


def main():
    random.seed(42)
    np.random.seed(42)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir, "datasets",
        "IBM-HR-Analytics-Employee-Attrition-and-Performance.csv",
    )
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    dataset_sha256 = compute_sha256(data_path)

    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    X = df[PREDICTION_FEATURES].copy()
    y = df['Attrition'].values

    prevalence = float(y.mean())
    n_rows = len(df)
    n_features = len(PREDICTION_FEATURES)

    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "artifacts"), exist_ok=True)

    # ── Phase 1: Train production pipelines with calibration ──────────────
    print("Training production pipelines with calibration selection...")
    manifest_models = {}
    calibration_results = {}

    for name in AVAILABLE_MODELS:
        random.seed(42)
        np.random.seed(42)

        pipeline = build_pipeline(name)
        imputer = pipeline['imputer']
        scaler = pipeline['scaler']
        base_model = pipeline['model']

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        # Compare calibration methods
        print(f"  [{name}] Comparing calibration: raw vs isotonic vs sigmoid...")
        best_method, cal_comparison = _calibration_comparison(
            pipeline, X_scaled, y, seed=42
        )
        print(f"  [{name}] Winner: {best_method} "
              f"(Brier: {cal_comparison[best_method]['brier_score']:.4f})")

        calibration_results[name] = {
            'best_method': best_method,
            'comparison': cal_comparison,
        }

        # Fit the winning model on the FULL dataset
        random.seed(42)
        np.random.seed(42)
        shipped_model = _fit_calibrated_model(
            base_model, X_scaled, y, best_method
        )

        # Save as the standard pipeline dict format
        pipeline['model'] = shipped_model
        model_path = os.path.join(base_dir, "models", f"{name}_pipeline.pkl")
        joblib.dump(pipeline, model_path, protocol=4)
        print(f"  [{name}] Saved to {model_path}")

        manifest_models[name] = {
            'path': f"models/{name}_pipeline.pkl",
            'n_features_in_': n_features,
            'calibration_method': best_method,
        }

    # ── Phase 2: Cross-validated evaluation ───────────────────────────────
    print("Evaluating models with cross-validation...")
    eval_results = evaluate_all(X, y, seed=42)

    best_model = None
    metrics_summary = {}
    for name in AVAILABLE_MODELS:
        if not eval_results[name]['is_baseline']:
            metrics_summary[name] = {
                'average_precision_mean': eval_results[name]['metrics'][
                    'average_precision']['mean'],
                'roc_auc_mean': eval_results[name]['metrics'][
                    'roc_auc']['mean'],
            }
            if eval_results[name]['rank'] == 1:
                best_model = name

    # ── Phase 3: Calibration assessment for each model ────────────────────
    print("Assessing calibration curves...")
    calibration_data = {}
    for name in AVAILABLE_MODELS:
        pipeline = build_pipeline(name)
        cal_info = assess_calibration(pipeline, X, y, seed=42)
        calibration_data[name] = cal_info
        print(f"  [{name}] Brier={cal_info['brier_score']:.4f}, "
              f"ECE={cal_info['ece']:.4f}")

    # ── Phase 4: Sensitivity analysis ─────────────────────────────────────
    print("Running threshold sensitivity analysis...")
    # Use the best model's out-of-fold probabilities for sensitivity
    best_pipeline = build_pipeline(best_model)
    imp = best_pipeline['imputer']
    scl = best_pipeline['scaler']

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_arr = np.asarray(X, dtype=float)
    oof_probs = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X_arr, y):
        p = build_pipeline(best_model)
        Xtr = p['scaler'].fit_transform(
            p['imputer'].fit_transform(X_arr[train_idx])
        )
        Xte = p['scaler'].transform(p['imputer'].transform(X_arr[test_idx]))
        p['model'].fit(Xtr, y[train_idx])
        oof_probs[test_idx] = p['model'].predict_proba(Xte)[:, 1]

    monthly_incomes = df['MonthlyIncome'].values
    sensitivity = _run_sensitivity_analysis(y, oof_probs, monthly_incomes)
    print(f"  Sensitivity analysis: {len(sensitivity)} scenarios computed")

    # Also run a full threshold sweep for the best model
    threshold_sweep = sweep_thresholds(y, oof_probs, monthly_incomes)
    print(f"  Optimal threshold (full sweep): "
          f"{threshold_sweep['optimal_threshold']:.3f}")

    # Sanity check: median income threshold
    median_income = float(df['MonthlyIncome'].median())
    sanity_threshold = optimal_threshold(median_income)
    print(f"\n  SANITY CHECK: median MonthlyIncome = {median_income:.0f}")
    print(f"  optimal_threshold({median_income:.0f}) = {sanity_threshold:.4f}")
    assert abs(sanity_threshold - 0.113) < 0.01, (
        f"Sanity check FAILED: expected ~0.113, got {sanity_threshold:.4f}"
    )
    print("  [OK] Sanity check passed (~0.113)")

    # ── Phase 5: Write artifacts ──────────────────────────────────────────
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    metrics_data = {
        'models': eval_results,
        'best_model': best_model,
        'prevalence': prevalence,
        'n_rows': n_rows,
        'n_features': n_features,
        'features': list(PREDICTION_FEATURES),
        'seed': 42,
        'sklearn_version': sklearn.__version__,
        'timestamp': timestamp,
        'dataset_sha256': dataset_sha256,
        'calibration': {
            'per_model': calibration_data,
            'calibration_method_selected': {
                name: calibration_results[name]
                for name in AVAILABLE_MODELS
            },
        },
        'threshold_sensitivity': sensitivity,
        'threshold_sweep': {
            'model': best_model,
            'thresholds': threshold_sweep['thresholds'],
            'per_employee_costs': threshold_sweep['per_employee_costs'],
            'optimal_threshold': threshold_sweep['optimal_threshold'],
            'optimal_cost': threshold_sweep['optimal_cost'],
        },
        'cost_assumptions': {
            'replacement_cost_fraction': REPLACEMENT_COST_FRACTION,
            'intervention_cost_usd': INTERVENTION_COST,
            'intervention_success_rate': INTERVENTION_SUCCESS_RATE,
            'source': 'SHRM/Gallup 50-200% of annual salary; low end used',
        },
    }

    metrics_path = os.path.join(base_dir, "artifacts", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\nSaved metrics to {metrics_path}")

    manifest_data = {
        'training_date': timestamp,
        'dataset_sha256': dataset_sha256,
        'sklearn_version': sklearn.__version__,
        'python_version': sys.version,
        'models': manifest_models,
        'metrics_summary': metrics_summary,
    }

    manifest_path = os.path.join(base_dir, "models", "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=4)
    print(f"Saved manifest to {manifest_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PR-AUC RESULTS (5-fold stratified CV, ranked)")
    print("=" * 60)
    ranked = sorted(
        [(n, eval_results[n]) for n in AVAILABLE_MODELS],
        key=lambda x: x[1]['metrics']['average_precision']['mean'],
        reverse=True,
    )
    for name, data in ranked:
        m = data['metrics']['average_precision']
        cal = calibration_results[name]
        print(f"  {data['rank']}. {data['display_name']:25s}  "
              f"PR-AUC={m['mean']:.4f}±{m['std']:.4f}  "
              f"calibration={cal['best_method']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
