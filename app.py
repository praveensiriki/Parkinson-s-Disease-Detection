import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib
from flask import Flask, render_template, request

# Load the dataset
data = pd.read_csv("PDD.csv")

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['Patient ID', "Parkinson's Diagnosis"])
y = data["Parkinson's Diagnosis"]

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Impute missing values for numerical columns using mean strategy
X_imputed_numeric = X.drop(columns=categorical_columns)
imputer_numeric = SimpleImputer(strategy='mean')
X_imputed_numeric = imputer_numeric.fit_transform(X_imputed_numeric)

# Impute missing values for categorical columns using mode strategy
X_imputed_categorical = X[categorical_columns]
imputer_categorical = SimpleImputer(strategy='most_frequent')
X_imputed_categorical = imputer_categorical.fit_transform(X_imputed_categorical)

# One-hot encode categorical variables
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_imputed_categorical)

# Concatenate the imputed numerical and encoded categorical data
X_imputed = np.concatenate((X_imputed_numeric, X_encoded.toarray()), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the trained model and preprocessing objects
joblib.dump(model, 'model.pkl')
joblib.dump(imputer_numeric, 'imputer_numeric.pkl')
joblib.dump(imputer_categorical, 'imputer_categorical.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load the trained model and preprocessing objects
model = joblib.load('model.pkl')
imputer_numeric = joblib.load('imputer_numeric.pkl')
imputer_categorical = joblib.load('imputer_categorical.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Create the Flask application
app = Flask(__name__)

# Define routes and functions for the Flask application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = float(request.form['age'])
    gender = request.form['gender']
    tremor = request.form['tremor']
    muscular_symptoms = request.form['muscular-symptoms']
    sleep_disturbances = request.form['sleep-disturbances']
    fatigue = request.form['fatigue']
    cognitive_symptoms = request.form['cognitive-symptoms']
    speech_difficulties = request.form['speech-difficulties']
    mood_changes = request.form['mood-changes']
    sense_of_smell = request.form['sense-of-smell']
    urinary_symptoms = request.form['urinary-symptoms']
    facial_symptoms = request.form['facial-symptoms']
    other_symptoms = request.form['other-symptoms']

    # Prepare user input for prediction
    user_input = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Tremor': [tremor],
        'Muscular Symptoms': [muscular_symptoms],
        'Sleep Disturbances': [sleep_disturbances],
        'Fatigue': [fatigue],
        'Cognitive Symptoms': [cognitive_symptoms],
        'Speech Difficulties': [speech_difficulties],
        'Mood Changes': [mood_changes],
        'Sense of Smell': [sense_of_smell],
        'Urinary Symptoms': [urinary_symptoms],
        'Facial Symptoms': [facial_symptoms],
        'Other Common Symptoms': [other_symptoms]
    })

    # Impute missing values for numerical columns
    user_input_numeric = user_input.drop(columns=categorical_columns)
    user_input_numeric_imputed = imputer_numeric.transform(user_input_numeric)

    # One-hot encode categorical variables
    user_input_categorical = user_input[categorical_columns]
    user_input_categorical_encoded = encoder.transform(user_input_categorical)

    # Concatenate the imputed numerical and encoded categorical data
    user_input_imputed = np.concatenate((user_input_numeric_imputed, user_input_categorical_encoded.toarray()), axis=1)

    # Standardize the user input
    user_input_scaled = scaler.transform(user_input_imputed)

    # Make prediction
    prediction = model.predict(user_input_scaled)

    # Return the prediction
    return render_template('result.html', prediction=prediction)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
