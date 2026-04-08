import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
data = pd.read_csv(r'C:\python12\my projects\EmployeeAttritionProject\datasets\IBM-HR-Analytics-Employee-Attrition-and-Performance.csv')


# Select features and target
features = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'OverTime', 'YearsAtCompany',
            'WorkLifeBalance', 'JobLevel', 'DistanceFromHome']
X = data[features]
y = data['Attrition']

# Handle categorical variables
le = LabelEncoder()
X.loc[:, 'OverTime'] = le.fit_transform(X['OverTime'])
y = le.fit_transform(y)  # Convert 'Yes'/'No' to 1/0

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
knn_model = KNeighborsClassifier(n_neighbors=5)  # Default k=5, adjust as needed
knn_model.fit(X_train, y_train)

# Bundle model, scaler, and imputer into a dictionary
pipeline = {
    'model': knn_model,
    'scaler': scaler,
    'imputer': imputer
}

# Save the entire pipeline as a single .pkl file
joblib.dump(pipeline, 'knn_pipeline.pkl')

print("KNN pipeline saved as 'knn_pipeline.pkl'.")