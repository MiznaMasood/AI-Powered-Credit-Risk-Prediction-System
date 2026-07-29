import joblib
import numpy as np

# Load model
model = joblib.load("saved_models/best_model.pkl")

def predict_risk(input_data):
    data = np.array([input_data])
    prediction = model.predict(data)

    if prediction[0] == 1:
        return "High Risk"
    else:
        return "Low Risk"