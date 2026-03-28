# Diabetes Prediction MLOps Project

End-to-end MLOps project for diabetes prediction using FastAPI, Docker, and Kubernetes deployment.

---

## Project Overview

This project demonstrates a complete machine learning lifecycle:
- Train a model using real-world dataset
- Serve predictions via FastAPI REST API
- Containerize using Docker
- Deploy on Kubernetes cluster

---

## Model Training

- Dataset: Diabetes dataset 
- Algorithm: Random Forest Classifier
- Features used:
  - Pregnancies
  - Glucose
  - BloodPressure
  - BMI
  - Age

---

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- Docker
- Kubernetes (K8s)
- Uvicorn

---

## Project Structure
```markdown
.
├── mainapp.py # FastAPI application
├── data_preparation_train.py # Model training script
├── diabetes_model.pkl # Trained model
├── requirements.txt # Dependencies
├── Dockerfile # Containerization
├── deployment.yaml # Kubernetes Deployment & Service
```

---

## Run Locally

```bash
pip install -r requirements.txt

uvicorn mainapp:app --reload
```

Application UI : http://localhost:8000/docs

## API Endpoint

### Request Body
```markdown
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "BMI": 25.5,
  "Age": 30
}
```
### Response
```
{
  "Diabetic or not?": true
}
```

