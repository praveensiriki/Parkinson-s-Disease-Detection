import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv("PDD.csv")

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode categorical variables
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Tremor'] = label_encoder.fit_transform(data['Tremor'])
data['Muscular Symptoms'] = label_encoder.fit_transform(data['Muscular Symptoms'])
data['Sleep Disturbances'] = label_encoder.fit_transform(data['Sleep Disturbances'])
data['Fatigue'] = label_encoder.fit_transform(data['Fatigue'])
data['Cognitive Symptoms'] = label_encoder.fit_transform(data['Cognitive Symptoms'])
data['Speech Difficulties'] = label_encoder.fit_transform(data['Speech Difficulties'])
data['Mood Changes'] = label_encoder.fit_transform(data['Mood Changes'])
data['Sense of Smell'] = label_encoder.fit_transform(data['Sense of Smell'])
data['Urinary Symptoms'] = label_encoder.fit_transform(data['Urinary Symptoms'])
data['Facial Symptoms'] = label_encoder.fit_transform(data['Facial Symptoms'])
data['Other Common Symptoms'] = label_encoder.fit_transform(data['Other Common Symptoms'])
data["Parkinson's Diagnosis"] = label_encoder.fit_transform(data["Parkinson's Diagnosis"])

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['Patient ID', "Parkinson's Diagnosis"])
y = data["Parkinson's Diagnosis"]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Function to collect new patient data from user inputs
def collect_patient_data():
    new_patient_data = {}
    for column in X.columns:
        new_patient_data[column] = input(f"Enter patient's {column}: ")
    return pd.DataFrame(new_patient_data, index=[0])

# Collect new patient data
new_patient_data = collect_patient_data()

# Encode categorical variables for the new patient data
for column in new_patient_data.select_dtypes(include='object'):
    if new_patient_data[column].iloc[0] not in label_encoder.classes_:
        new_patient_data[column] = -1
    else:
        new_patient_data[column] = label_encoder.transform([new_patient_data[column].iloc[0]])[0]

# Impute missing values for the new patient data
new_patient_data_imputed = imputer.transform(new_patient_data)

# Standardize the new patient data
new_patient_data_scaled = scaler.transform(new_patient_data_imputed)

# Make prediction for the new patient
prediction = model.predict(new_patient_data_scaled)
print("Predicted Parkinson's Diagnosis:", label_encoder.inverse_transform(prediction)[0])
