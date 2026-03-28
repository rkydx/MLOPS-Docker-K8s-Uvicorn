# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')

# Get list of columns in the dataset and print it
columns = df.columns
print("Columns in the dataset:", columns)

# Prepare the data for training
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age']]
y = df['Outcome']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'diabetes_model.pkl')
print("Model trained and saved as 'diabetes_model.pkl'")