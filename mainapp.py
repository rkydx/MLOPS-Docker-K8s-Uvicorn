# Import required libraries
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# Define a Pydantic model for input data
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API!"}

@app.post("/predict")
def predict_diabetes(input_data: DiabetesInput):
    # Prepare the input data for prediction
    input_data = np.array([[input_data.Pregnancies, input_data.Glucose, input_data.BloodPressure, input_data.BMI, input_data.Age]])
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Return the prediction result
    return {"Diabetic or not? ": bool(prediction[0])}