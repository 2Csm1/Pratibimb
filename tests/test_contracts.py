"""
tests/test_contracts.py — Phase 0 contract tests for Pratibimb.

These tests lock three critical invariants:
  1. Every pipeline .pkl was trained on exactly len(PREDICTION_FEATURES) features.
  2. The /download route rejects path-traversal attempts with HTTP 400.
  3. The CSV validation helper rejects bad files before they reach pandas.

Run with: pytest tests/test_contracts.py -v
"""
import io
import os
import sys
import pytest

# Make sure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import joblib
from config import PREDICTION_FEATURES, AVAILABLE_MODELS


# ── Helper paths ──────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATHS = {
    'rf':  os.path.join(ROOT, 'models', 'rf_pipeline.pkl'),
    'knn': os.path.join(ROOT, 'models', 'knn_pipeline.pkl'),
    'lr':  os.path.join(ROOT, 'models', 'lr_pipeline.pkl'),
    'svm': os.path.join(ROOT, 'models', 'svm_pipeline.pkl'),
    'xgb': os.path.join(ROOT, 'models', 'xgb_pipeline.pkl'),
}


# ═════════════════════════════════════════════════════════════════════════════
# Contract 1 — Feature count
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
def test_pipeline_feature_count(model_name):
    """
    Every saved pipeline must have been trained on exactly the same number of
    features as defined in config.PREDICTION_FEATURES.

    If this test fails it means a model was retrained with a different feature
    list than what config.py declares. Fix: either retrain or update config.py.
    """
    pkl_path = MODEL_PATHS[model_name]
    assert os.path.isfile(pkl_path), f"Pipeline file not found: {pkl_path}"

    pipeline = joblib.load(pkl_path)
    n_actual   = pipeline['model'].n_features_in_
    n_expected = len(PREDICTION_FEATURES)

    assert n_actual == n_expected, (
        f"Pipeline '{model_name}' has n_features_in_={n_actual} "
        f"but config.PREDICTION_FEATURES has {n_expected} entries. "
        f"Retrain the model or fix config.py."
    )


@pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
def test_pipeline_imputer_feature_names(model_name):
    """
    The imputer stored in each pipeline should have been fit on exactly the
    features listed in PREDICTION_FEATURES (same order).
    """
    pkl_path = MODEL_PATHS[model_name]
    pipeline = joblib.load(pkl_path)
    imputer  = pipeline['imputer']

    if hasattr(imputer, 'feature_names_in_'):
        actual_names = list(imputer.feature_names_in_)
        assert actual_names == PREDICTION_FEATURES, (
            f"Pipeline '{model_name}' imputer feature names {actual_names} "
            f"do not match config.PREDICTION_FEATURES {PREDICTION_FEATURES}."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Contract 2 — Path traversal in /download
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Flask test client with a temp upload folder."""
    # Set SECRET_KEY so app doesn't raise at startup
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("FLASK_DEBUG", "1")

    # Point uploads to a temp dir
    import app as app_module
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    app_module.UPLOAD_FOLDER = str(tmp_path)

    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.mark.parametrize("malicious_filename", [
    "../../etc/passwd",
    "../app.py",
    "%2F%2Fetc%2Fpasswd",
    "....//....//etc/passwd",
])
def test_download_path_traversal_blocked(client, malicious_filename):
    """
    Path-traversal payloads must be rejected with HTTP 400, not served.
    """
    response = client.get(f"/download/{malicious_filename}")
    assert response.status_code in (400, 404), (
        f"Expected 400 or 404 for path traversal attempt '{malicious_filename}', "
        f"got {response.status_code}."
    )


def test_download_valid_file(client, tmp_path):
    """A legitimate file in the uploads folder must be served normally."""
    legit = tmp_path / "test.csv"
    legit.write_text("col1,col2\n1,2\n")

    import app as app_module
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)

    response = client.get("/download/test.csv")
    assert response.status_code == 200


# ═════════════════════════════════════════════════════════════════════════════
# Contract 3 — CSV upload validation
# ═════════════════════════════════════════════════════════════════════════════

def _make_file(content: bytes, filename: str):
    """Minimal file-like object mimicking werkzeug FileStorage."""
    class FakeFile:
        def __init__(self, data, name):
            self.filename = name
            self._buf = io.BytesIO(data)
        def read(self):
            return self._buf.read()
        def seek(self, pos):
            self._buf.seek(pos)
    return FakeFile(content, filename)


def test_csv_validation_rejects_wrong_extension():
    from app import validate_csv_upload
    f = _make_file(b"a,b\n1,2", "data.xlsx")
    ok, msg = validate_csv_upload(f)
    assert not ok
    assert "xlsx" in msg.lower() or "invalid" in msg.lower()


def test_csv_validation_rejects_oversized_file():
    from app import validate_csv_upload
    big = b"a,b\n" + b"1,2\n" * (3 * 1024 * 1024)   # ~24 MB of rows
    f = _make_file(big, "big.csv")
    ok, msg = validate_csv_upload(f)
    assert not ok
    assert "large" in msg.lower() or "size" in msg.lower()


def test_csv_validation_rejects_missing_columns():
    from app import validate_csv_upload
    content = b"Name,Department\nAlice,HR\n"
    f = _make_file(content, "employees.csv")
    ok, msg = validate_csv_upload(f, required_cols=['Age', 'MonthlyIncome'])
    assert not ok
    assert "Age" in msg or "missing" in msg.lower()


def test_csv_validation_accepts_valid_file():
    from app import validate_csv_upload
    header = ",".join(PREDICTION_FEATURES) + "\n"
    row    = ",".join(["35", "6500", "3", "No", "5", "3", "2", "10"]) + "\n"
    content = (header + row).encode()
    f = _make_file(content, "valid.csv")
    ok, msg = validate_csv_upload(f, required_cols=PREDICTION_FEATURES)
    assert ok, f"Expected valid CSV to pass, got error: {msg}"
