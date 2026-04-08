# Pratibimb

Pratibimb is a comprehensive Employee Attrition Prediction and HR Analytics platform. Built with Python, Flask, and Scikit-Learn, it leverages machine learning to assess the risk of employees leaving an organization and provides actionable insights and retention strategies.

## Key Features

- **Employee Attrition Prediction:** Evaluate the likelihood of an employee leaving using various predictive models, including Random Forest, K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machines (SVM), and XGBoost.
- **Time Window Analysis:** Predict risk levels and identify contributing factors over different periods (30, 60, and 90 days), allowing HR to intervene effectively.
- **Retention Strategy Generation:** Automatically generates tailored strategies suggesting immediate, medium-term, and long-term actions to retain at-risk employees based on their salary, job satisfaction, tenure, and overtime habits.
- **Historical Trends & Analytics:** Visually analyze historical HR data, exploring attrition rates across departments and roles, utilizing Plotly for dynamic, interactive visualizations.
- **Batch Processing:** Upload a CSV to process and analyze multiple employees simultaneously, outputting risk probabilities and predictions in a tabular format.
- **Model Comparison:** Compare the performance of the integrated ML models on the test set, evaluating their accuracy, precision, recall, F1 scores, and training time.

## Technologies Used

- **Backend / Web Framework:** Python, Flask, Werkzeug, Gunicorn
- **Machine Learning & Data Processing:** Scikit-Learn, XGBoost, Pandas, Numpy, Joblib, SciPy
- **Data Visualization:** Plotly, Matplotlib, Seaborn
- **Frontend Engine:** Jinja2

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/CSM4416/Pratibimb.git
   cd Pratibimb
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the application:
   Open your browser and navigate to `http://localhost:5000`.

## Pre-trained Models

The application relies on pre-trained Scikit-Learn pipelines stored in the `models` directory:
- `rf_pipeline.pkl`
- `knn_pipeline.pkl`
- `lr_pipeline.pkl`
- `svm_pipeline.pkl`
- `xgb_pipeline.pkl`

*(Note: Ensure you have placed or trained your models successfully into the `models` folder for predictions to work.)*
