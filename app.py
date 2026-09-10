import os
import io
import pandas as pd
import numpy as np
import joblib
import time
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, abort, send_from_directory
from werkzeug.utils import secure_filename
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import glob

from config import (
    PREDICTION_FEATURES,
    AVAILABLE_MODELS,
    MODEL_DISPLAY_NAMES,
    MAX_UPLOAD_BYTES,
    MAX_DISPLAY_ROWS,
    MAX_VIZ_ROWS,
    MAX_STORED_FILES,
    ALLOWED_EXTENSIONS,
    REQUIRED_PREDICTION_COLS,
    REQUIRED_PREPROCESS_COLS,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Fix: require SECRET_KEY in production; allow a dev-only fallback only when
# FLASK_DEBUG is explicitly set. Never ship "supersecretkey" to production.
_secret = os.getenv("SECRET_KEY")
if not _secret:
    if os.getenv("FLASK_DEBUG", "0") in ("1", "true", "True"):
        _secret = "dev-only-insecure-key"
        logging.warning(
            "[SECURITY] SECRET_KEY is not set. Using an insecure dev key. "
            "Set SECRET_KEY before deploying."
        )
    else:
        raise RuntimeError(
            "SECRET_KEY environment variable is not set. "
            "Refusing to start in non-debug mode without a secret key. "
            "Set SECRET_KEY=<random-string> in your environment."
        )
app.secret_key = _secret

if os.getenv("RENDER", "false").lower() == "true":
    UPLOAD_FOLDER = '/opt/render/project/src/uploads'
else:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prune oldest uploads beyond MAX_STORED_FILES
_existing = sorted(glob.glob(os.path.join(UPLOAD_FOLDER, "*.csv")), key=os.path.getmtime)
for _f in _existing[:-MAX_STORED_FILES]:
    try:
        os.remove(_f)
    except OSError:
        pass

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model cache ───────────────────────────────────────────────────────────────
model_cache = {}

_BASE_DIR = os.path.dirname(__file__)
MODEL_PATHS = {
    'rf':  os.path.join(_BASE_DIR, 'models', 'rf_pipeline.pkl'),
    'knn': os.path.join(_BASE_DIR, 'models', 'knn_pipeline.pkl'),
    'lr':  os.path.join(_BASE_DIR, 'models', 'lr_pipeline.pkl'),
    'svm': os.path.join(_BASE_DIR, 'models', 'svm_pipeline.pkl'),
    'xgb': os.path.join(_BASE_DIR, 'models', 'xgb_pipeline.pkl'),
}


def get_model(model_name):
    """Load and cache a model pipeline, asserting the feature contract."""
    if model_name not in model_cache:
        pipeline = joblib.load(MODEL_PATHS[model_name])
        # Fix: assert feature count matches config — fail loudly on mismatch.
        n = pipeline['model'].n_features_in_
        expected = len(PREDICTION_FEATURES)
        if n != expected:
            raise RuntimeError(
                f"Pipeline '{model_name}' was trained on {n} features but "
                f"config.PREDICTION_FEATURES has {expected}. "
                f"Retrain the model or update config.py."
            )
        model_cache[model_name] = pipeline
    return model_cache[model_name]


# ── Preload IBM HR dataset ────────────────────────────────────────────────────
DATASET_PATH = os.path.join(_BASE_DIR, 'datasets', 'IBM-HR-Analytics-Employee-Attrition-and-Performance.csv')
try:
    df = pd.read_csv(DATASET_PATH)
    df['AttritionNumeric'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    logger.info(f"Loaded IBM HR dataset with {len(df)} rows. Median MonthlyIncome: {df['MonthlyIncome'].median()}")
except FileNotFoundError:
    logger.error(f"IBM HR dataset not found at {DATASET_PATH}. Using fallback median income.")
    df = pd.DataFrame()


# ── CSV upload validation ─────────────────────────────────────────────────────
def validate_csv_upload(file, required_cols=None):
    """
    Validate an uploaded file object before processing.

    Returns (True, None) on success or (False, error_message) on failure.
    Reads the file into memory to check size; rewinds the stream afterwards.
    """
    # Extension check
    filename = file.filename or ""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type '.{ext}'. Only CSV files are accepted."

    # Size check — read into buffer so we never trust Content-Length
    buf = file.read()
    if len(buf) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return False, f"File is too large. Maximum allowed size is {mb} MB."
    file.seek(0)  # rewind for downstream read

    # Schema check — parse only first rows for speed
    try:
        sample = pd.read_csv(io.BytesIO(buf), nrows=5)
    except Exception as exc:
        return False, f"Could not parse CSV: {exc}"

    if required_cols:
        missing = [c for c in required_cols if c not in sample.columns]
        if missing:
            return False, (
                f"CSV is missing required column(s): {', '.join(missing)}. "
                f"Required: {', '.join(required_cols)}"
            )

    return True, None


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_dataset(file_path, output_file=None):
    if output_file is None:
        output_file = os.path.join(UPLOAD_FOLDER, "preprocessed_data.csv")
    try:
        logger.info(f"Starting preprocessing for file: {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Loaded CSV with columns: {list(data.columns)}")

        missing_cols = [c for c in REQUIRED_PREPROCESS_COLS if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        X = data[PREDICTION_FEATURES].copy()
        y = data['Attrition']

        le = LabelEncoder()
        X['OverTime'] = le.fit_transform(X['OverTime'].astype(str))
        y = le.fit_transform(y.astype(str))

        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        preprocessed_data = pd.concat([X, pd.Series(y, name='Attrition')], axis=1)
        preprocessed_data.to_csv(output_file, index=False)
        logger.info(f"Preprocessed dataset saved as {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


# ── Time window analysis ──────────────────────────────────────────────────────
def predict_time_window(employee_data):
    try:
        monthly_income = float(employee_data.get('MonthlyIncome', 0))
        job_satisfaction = int(employee_data.get('JobSatisfaction', 0))
        years_at_company = int(employee_data.get('YearsAtCompany', 0))
        overtime = employee_data.get('OverTime', 'No')

        base_risk = 0.0
        risk_factors = []
        median_income = df['MonthlyIncome'].median() if not df.empty else 6500.0

        if monthly_income < median_income:
            base_risk += 0.3
            risk_factors.append(f"Below median salary (Current: ${monthly_income:,.0f}, Median: ${median_income:,.2f})")
        if job_satisfaction < 3:
            base_risk += 0.25
            risk_factors.append(f"Low job satisfaction (Score: {job_satisfaction}/4)")
        if overtime == 'Yes':
            base_risk += 0.2
            risk_factors.append("Regular overtime work")
        if years_at_company < 2:
            base_risk += 0.25
            risk_factors.append(f"Short tenure ({years_at_company} years)")

        if not risk_factors:
            risk_factors.append("No significant risk factors identified")

        predictions = {
            '30 days': {
                'risk_score': min(base_risk * 1.2, 1.0),
                'contributing_factors': risk_factors,
                'impact': 'Immediate intervention needed' if base_risk * 1.2 > 0.7 else 'Monitor situation',
            },
            '60 days': {
                'risk_score': min(base_risk * 1.1, 1.0),
                'contributing_factors': risk_factors,
                'impact': 'Plan intervention' if base_risk * 1.1 > 0.7 else 'Regular check-ins',
            },
            '90 days': {
                'risk_score': min(base_risk, 1.0),
                'contributing_factors': risk_factors,
                'impact': 'Strategic planning needed' if base_risk > 0.7 else 'Standard monitoring',
            },
        }
        logger.info(f"Time window analysis: base_risk={base_risk:.2f}, factors={risk_factors}")
        return predictions
    except Exception as e:
        logger.error(f"Error in time window prediction: {e}")
        _err = {'risk_score': 0.0, 'contributing_factors': ['Error calculating risk'], 'impact': 'Unable to assess'}
        return {'30 days': _err, '60 days': _err, '90 days': _err}


# ── Retention strategy ────────────────────────────────────────────────────────
def generate_retention_strategy(employee_data):
    try:
        strategies = {
            'immediate_actions': [],
            'medium_term_actions': [],
            'long_term_actions': [],
            'priority_level': 'medium',
        }
        monthly_income = float(employee_data.get('MonthlyIncome', 0))
        median_income = df['MonthlyIncome'].median() if not df.empty else 6500.0
        if monthly_income < median_income:
            strategies['immediate_actions'].append({
                'action': 'Review compensation package',
                'reason': 'Below median salary for role',
                'impact': 'High',
                'timeframe': '30 days',
            })
        job_satisfaction = int(employee_data.get('JobSatisfaction', 0))
        if job_satisfaction < 3:
            strategies['immediate_actions'].append({
                'action': 'Schedule one-on-one meeting',
                'reason': 'Low job satisfaction score',
                'impact': 'High',
                'timeframe': '7 days',
            })
            strategies['medium_term_actions'].append({
                'action': 'Create development plan',
                'reason': 'Career growth opportunity',
                'impact': 'Medium',
                'timeframe': '60 days',
            })
        if employee_data.get('OverTime') == 'Yes':
            strategies['immediate_actions'].append({
                'action': 'Review workload distribution',
                'reason': 'Regular overtime indicates potential burnout risk',
                'impact': 'High',
                'timeframe': '14 days',
            })
        years_at_company = int(employee_data.get('YearsAtCompany', 0))
        if years_at_company < 2:
            strategies['medium_term_actions'].append({
                'action': 'Assign mentor',
                'reason': 'New employee retention',
                'impact': 'Medium',
                'timeframe': '30 days',
            })
        elif years_at_company > 5:
            strategies['long_term_actions'].append({
                'action': 'Consider for leadership development',
                'reason': 'Experienced employee growth',
                'impact': 'Medium',
                'timeframe': '90 days',
            })
        if len(strategies['immediate_actions']) >= 2:
            strategies['priority_level'] = 'high'
        elif len(strategies['immediate_actions']) == 0:
            strategies['priority_level'] = 'low'
        return strategies
    except Exception as e:
        logger.error(f"Error generating retention strategy: {e}")
        return None


# ── Historical trends ─────────────────────────────────────────────────────────
def analyze_historical_trends():
    try:
        if df.empty:
            raise ValueError("IBM HR dataset is not available.")
        trends = {
            'department_trends': {},
            'overall_attrition_rate': float(df['AttritionNumeric'].mean()),
            'total_employees': len(df),
            'department_statistics': {},
        }
        dept_stats = df.groupby('Department').agg(
            AttritionCount=('AttritionNumeric', 'count'),
            AttritionMean=('AttritionNumeric', 'mean'),
            AvgIncome=('MonthlyIncome', 'mean'),
            AvgTenure=('YearsAtCompany', 'mean'),
        ).round(2)

        for dept in dept_stats.index:
            trends['department_statistics'][dept] = {
                'employee_count': int(dept_stats.loc[dept, 'AttritionCount']),
                'attrition_rate': float(dept_stats.loc[dept, 'AttritionMean']),
                'avg_salary': float(dept_stats.loc[dept, 'AvgIncome']),
                'avg_tenure': float(dept_stats.loc[dept, 'AvgTenure']),
            }
        avg_attrition = df['AttritionNumeric'].mean()
        trends['high_risk_departments'] = [
            dept for dept in trends['department_statistics']
            if trends['department_statistics'][dept]['attrition_rate'] > avg_attrition
        ]
        role_stats = (
            df.groupby('JobRole')['AttritionNumeric']
            .mean()
            .sort_values(ascending=False)
        )
        trends['role_trends'] = {role: float(rate) for role, rate in role_stats.items()}
        return trends
    except Exception as e:
        logger.error(f"Error analyzing historical trends: {e}")
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def main():
    return render_template('main.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            selected_model = request.form.get('model', 'svm')
            if selected_model not in AVAILABLE_MODELS:
                raise ValueError(f"Invalid model selected: {selected_model}")

            input_data = {
                'Age':              int(request.form['age']),
                'MonthlyIncome':    float(request.form['monthly_income']),
                'JobSatisfaction':  int(request.form['job_satisfaction']),
                'OverTime':         request.form['overtime'],
                'YearsAtCompany':   int(request.form['years_at_company']),
                'WorkLifeBalance':  int(request.form['work_life_balance']),
                'JobLevel':         int(request.form['job_level']),
                'DistanceFromHome': int(request.form['distance_from_home']),
            }

            input_df = pd.DataFrame([input_data])
            input_df['OverTime'] = input_df['OverTime'].map({'No': 0, 'Yes': 1})

            pipeline = get_model(selected_model)
            input_processed = pd.DataFrame(
                pipeline['imputer'].transform(input_df), columns=PREDICTION_FEATURES
            )
            input_processed = pd.DataFrame(
                pipeline['scaler'].transform(input_processed), columns=PREDICTION_FEATURES
            )
            prediction = pipeline['model'].predict(input_processed)[0]

            if hasattr(pipeline['model'], 'predict_proba'):
                try:
                    prob = pipeline['model'].predict_proba(input_processed)[0][1]
                    risk_percentage = f"{prob * 100:.1f}%"
                except Exception as e:
                    logger.error(f"Error calculating probability: {e}")
                    risk_percentage = "Not available"
            else:
                risk_percentage = "Not supported by this model"

            if prediction == 1:
                result = f"⚠️ Employee is expected to leave. Risk: {risk_percentage} (Model: {selected_model.upper()})"
            else:
                result = f"✅ Employee is expected to stay. Risk: {risk_percentage} (Model: {selected_model.upper()})"

            time_window_result = predict_time_window(input_data)
            return render_template(
                'home.html',
                prediction=result,
                time_window=time_window_result,
                models=AVAILABLE_MODELS,
                model_names=MODEL_DISPLAY_NAMES,
                selected_model=selected_model,
            )
        except ValueError as e:
            flash(f"Invalid input: {e}", "danger")
        except Exception as e:
            logger.error(f"Error in home route: {e}")
            flash(f"An error occurred: {e}", "danger")
        return render_template('home.html', models=AVAILABLE_MODELS, model_names=MODEL_DISPLAY_NAMES, selected_model='svm')

    return render_template(
        'home.html',
        prediction=None,
        time_window=None,
        models=AVAILABLE_MODELS,
        model_names=MODEL_DISPLAY_NAMES,
        selected_model='svm',
    )


@app.route('/retention_strategy', methods=['GET', 'POST'])
def retention_strategy():
    if request.method == 'POST':
        try:
            input_data = {
                'MonthlyIncome':   float(request.form['monthly_income']),
                'JobSatisfaction': int(request.form['job_satisfaction']),
                'OverTime':        request.form['overtime'],
                'YearsAtCompany':  int(request.form['years_at_company']),
            }
            strategy = generate_retention_strategy(input_data)
            if strategy:
                return render_template('retention_strategy.html', strategy=strategy)
            else:
                flash("Error generating retention strategy.", "danger")
        except ValueError as e:
            flash(f"Invalid input: {e}", "danger")
    return render_template('retention_strategy.html', strategy=None)


@app.route('/historical_trends', methods=['GET', 'POST'])
def historical_trends():
    if request.method == 'POST':
        trends = analyze_historical_trends()
        if trends:
            return render_template('historical_trends.html', trends=trends)
        else:
            flash("Error analyzing historical trends.", "danger")
    return render_template('historical_trends.html', trends=None)


@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    graphs = []
    annotations = []
    preprocessed_file = None

    if request.method == 'POST':
        # ── Preprocessing upload ───────────────────────────────────────────
        if 'preprocess_file' in request.files and request.files['preprocess_file'].filename:
            file = request.files['preprocess_file']
            ok, err = validate_csv_upload(file, required_cols=REQUIRED_PREPROCESS_COLS)
            if not ok:
                flash(f"Upload error: {err}", "danger")
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(filepath)
                    preprocessed_file = preprocess_dataset(filepath)
                    flash("Dataset preprocessed successfully. Download available below.", "success")
                except Exception as e:
                    flash(f"Error preprocessing dataset: {e}", "danger")

        # ── Visualisation upload ───────────────────────────────────────────
        if 'dataset' in request.files and request.files['dataset'].filename:
            file = request.files['dataset']
            ok, err = validate_csv_upload(file, required_cols=REQUIRED_PREDICTION_COLS)
            if not ok:
                flash(f"Upload error: {err}", "danger")
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(filepath)
                    df_uploaded = pd.read_csv(filepath)
                    if len(df_uploaded) > MAX_VIZ_ROWS:
                        df_uploaded = df_uploaded.sample(n=MAX_VIZ_ROWS, random_state=42)

                    start_time = time.time()
                    fig_config = {'displayModeBar': False, 'staticPlot': True}
                    colors = px.colors.sequential.Blues_r

                    # Chart 1: Attrition distribution pie
                    fig1 = px.pie(
                        df_uploaded, names='Attrition',
                        title='Attrition Distribution',
                        color_discrete_sequence=colors,
                    )
                    fig1.update_layout(width=500, height=400)
                    graphs.append(pio.to_html(fig1, full_html=False, config=fig_config))
                    annotations.append("Percentage of employees who stayed vs. left.")

                    # Chart 2: Monthly income vs attrition (box)
                    fig2 = px.box(
                        df_uploaded, x='Attrition', y='MonthlyIncome',
                        title='Monthly Income vs Attrition',
                        color='Attrition', color_discrete_sequence=colors,
                    )
                    fig2.update_layout(width=500, height=400)
                    graphs.append(pio.to_html(fig2, full_html=False, config=fig_config))
                    annotations.append("Income distribution for employees who stayed vs. left.")

                    # Chart 3: Job satisfaction histogram
                    fig3 = px.histogram(
                        df_uploaded, x='JobSatisfaction', color='Attrition',
                        barmode='group', title='Job Satisfaction by Attrition',
                        color_discrete_sequence=colors,
                    )
                    fig3.update_layout(width=500, height=400)
                    graphs.append(pio.to_html(fig3, full_html=False, config=fig_config))
                    annotations.append("Job satisfaction levels and their relation to attrition.")

                    # Chart 4: Attrition rate by Job Role (replaces old duplicate box plot)
                    if 'JobRole' in df_uploaded.columns and 'Attrition' in df_uploaded.columns:
                        role_attr = (
                            df_uploaded.assign(AttritionNum=df_uploaded['Attrition'].map({'Yes': 1, 'No': 0}))
                            .groupby('JobRole')['AttritionNum']
                            .mean()
                            .mul(100)
                            .round(1)
                            .sort_values(ascending=False)
                            .reset_index()
                        )
                        role_attr.columns = ['JobRole', 'AttritionRate']
                        fig4 = px.bar(
                            role_attr, x='AttritionRate', y='JobRole',
                            orientation='h',
                            title='Attrition Rate by Job Role (%)',
                            color='AttritionRate',
                            color_continuous_scale='Blues',
                        )
                        fig4.update_layout(width=500, height=400, yaxis={'categoryorder': 'total ascending'})
                        graphs.append(pio.to_html(fig4, full_html=False, config=fig_config))
                        annotations.append("Which job roles have the highest attrition risk.")

                    # Chart 5: Age vs attrition (violin)
                    fig5 = px.violin(
                        df_uploaded, x='Attrition', y='Age',
                        title='Age vs Attrition',
                        color='Attrition', color_discrete_sequence=colors,
                    )
                    fig5.update_layout(width=500, height=400)
                    graphs.append(pio.to_html(fig5, full_html=False, config=fig_config))
                    annotations.append("Age distribution by attrition status.")

                    # Chart 6: Years at company vs monthly income scatter
                    fig6 = px.scatter(
                        df_uploaded, x='YearsAtCompany', y='MonthlyIncome',
                        color='Attrition',
                        title='Years at Company vs Monthly Income by Attrition',
                        color_discrete_sequence=colors,
                    )
                    fig6.update_layout(width=500, height=400)
                    graphs.append(pio.to_html(fig6, full_html=False, config=fig_config))
                    annotations.append("Relationship between tenure and income, coloured by attrition.")

                    logger.info(f"Graph generation took {time.time() - start_time:.2f}s")
                except Exception as e:
                    flash(f"Error generating graphs: {e}", "danger")

    return render_template('analysis.html', graphs=graphs, annotations=annotations, preprocessed_file=preprocessed_file)


@app.route('/download/<path:filename>')
def download_file(filename):
    """
    Serve a file from the uploads folder.

    Fix: run secure_filename, resolve the absolute path, and verify it is
    inside UPLOAD_FOLDER before serving. Rejects path-traversal attempts
    (e.g. ../../etc/passwd) with HTTP 400.
    """
    safe_name = secure_filename(filename)
    if not safe_name:
        abort(400, description="Invalid filename.")

    upload_dir = os.path.realpath(app.config['UPLOAD_FOLDER'])
    target     = os.path.realpath(os.path.join(upload_dir, safe_name))

    if not target.startswith(upload_dir + os.sep) and target != upload_dir:
        logger.warning(f"Path traversal attempt blocked: '{filename}' resolved to '{target}'")
        abort(400, description="Invalid filename.")

    if not os.path.isfile(target):
        abort(404, description="File not found.")

    try:
        return send_from_directory(upload_dir, safe_name, as_attachment=True)
    except Exception as e:
        flash(f"Error downloading file: {e}", "danger")
        return redirect(url_for('analysis'))


@app.route('/batch', methods=['GET', 'POST'])
def batch():
    """
    Batch attrition prediction from an uploaded CSV.
    Fix: respects user's model selection instead of hardcoding Random Forest.
    """
    if request.method == 'POST':
        file = request.files.get('batch_file')
        if not file or not file.filename:
            flash("Please select a CSV file.", "danger")
            return render_template('batch.html', table=None, models=AVAILABLE_MODELS, model_names=MODEL_DISPLAY_NAMES, selected_model='rf')

        selected_model = request.form.get('model', 'rf')
        if selected_model not in AVAILABLE_MODELS:
            flash(f"Invalid model '{selected_model}'. Defaulting to Random Forest.", "warning")
            selected_model = 'rf'

        ok, err = validate_csv_upload(file, required_cols=REQUIRED_PREDICTION_COLS)
        if not ok:
            flash(f"Upload error: {err}", "danger")
            return render_template('batch.html', table=None, models=AVAILABLE_MODELS, model_names=MODEL_DISPLAY_NAMES, selected_model=selected_model)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            data = pd.read_csv(filepath)
            truncated = len(data) > MAX_DISPLAY_ROWS
            display_df = data.head(MAX_DISPLAY_ROWS).copy() if truncated else data.copy()

            input_df = data[PREDICTION_FEATURES].copy()
            input_df['OverTime'] = input_df['OverTime'].map({'No': 0, 'Yes': 1, 'no': 0, 'yes': 1})

            pipeline = get_model(selected_model)
            input_processed = pd.DataFrame(
                pipeline['imputer'].transform(input_df), columns=PREDICTION_FEATURES
            )
            input_processed = pd.DataFrame(
                pipeline['scaler'].transform(input_processed), columns=PREDICTION_FEATURES
            )
            predictions = pipeline['model'].predict(input_processed)
            probs = pipeline['model'].predict_proba(input_processed)[:, 1]

            risk_levels = ['High' if p >= 0.7 else 'Medium' if p >= 0.3 else 'Low' for p in probs]
            n = len(display_df)
            display_df['Risk']       = risk_levels[:n]
            display_df['Risk %']     = (probs[:n] * 100).round(1)
            display_df['Prediction'] = ['Leave' if p == 1 else 'Stay' for p in predictions[:n]]

            def style_row(row):
                if row['Prediction'] == 'Leave':
                    colors_map = {'High': '#ff9999', 'Medium': '#ffcc99', 'Low': '#ffeecc'}
                    c = colors_map.get(row['Risk'], '#ffffff')
                else:
                    c = '#99ff99'
                return [f'background-color: {c}' for _ in row]

            cols = PREDICTION_FEATURES + ['Risk', 'Risk %', 'Prediction']
            styled_df = display_df[cols].style.apply(style_row, axis=1).to_html()
            message = (
                f"Showing first {MAX_DISPLAY_ROWS} of {len(data)} employees."
                if truncated else f"Analyzed {len(data)} employees."
            )
            model_label = MODEL_DISPLAY_NAMES.get(selected_model, selected_model.upper())
            return render_template(
                'batch.html',
                table=styled_df,
                message=message,
                model_label=model_label,
                models=AVAILABLE_MODELS,
                model_names=MODEL_DISPLAY_NAMES,
                selected_model=selected_model,
            )
        except Exception as e:
            flash(f'Error processing file: {e}', "danger")

    return render_template('batch.html', table=None, models=AVAILABLE_MODELS, model_names=MODEL_DISPLAY_NAMES, selected_model='rf')


@app.route('/comparison')
def comparison():
    """
    Display pre-computed evaluation metrics.

    Reads artifacts/metrics.json (generated by train.py) and renders the
    comparison template.  This route NEVER calls .fit() and NEVER touches
    get_model(), so it cannot mutate the cached production pipelines used
    by /home and /batch.
    """
    import json

    metrics_path = os.path.join(_BASE_DIR, 'artifacts', 'metrics.json')

    if not os.path.isfile(metrics_path):
        flash(
            "Evaluation results not found. "
            "Please run  python train.py  to generate model metrics.",
            "warning",
        )
        return render_template('comparison.html', metrics=None, best_model=None)

    try:
        with open(metrics_path, 'r', encoding='utf-8') as fh:
            metrics = json.load(fh)

        best_model = metrics.get('best_model')
        return render_template(
            'comparison.html',
            metrics=metrics,
            best_model=best_model,
        )
    except Exception as e:
        logger.error(f"Error loading evaluation metrics: {e}")
        flash(f"Error loading evaluation metrics: {e}", "danger")
        return redirect(url_for('main'))


@app.route('/healthz')
def health_check():
    return "OK", 200


# ── Jinja2 custom filters ─────────────────────────────────────────────────────
def _basename(path):
    return os.path.basename(path)


app.jinja_env.filters['basename'] = _basename
app.jinja_env.filters['zip'] = zip


if __name__ == '__main__':
    debug_mode = os.getenv("FLASK_DEBUG", "0") in ("1", "true", "True")
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=debug_mode)