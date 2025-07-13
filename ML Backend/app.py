import os
from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)


model_path1 = r'C:\Users\hs298\Desktop\Projects\Life Scoop\ML Backend\Models\heart_attack_modelV1.joblib'
model = load(model_path1)
model_path2 = r"C:\Users\hs298\Desktop\Projects\Life Scoop\ML Backend\Models\Mental_health.joblib"
mental_model = load(model_path2)
model_path3 = r"C:\Users\hs298\Desktop\Projects\Life Scoop\ML Backend\Models\Diabetes.joblib"
diabetes_model = load(model_path3)
model_path4 = r"C:\Users\hs298\Desktop\Projects\Life Scoop\ML Backend\Models\Hypertension.joblib"
hypertension_model = load(model_path4)

@app.route('/predict-mental-health', methods=['POST'])
def predict_mental_health():
    try:
        # Expected input features
        expected_features = ['age', 'gender', 'employment_status', 'work_environment',
                                'mental_health_history', 'seeks_treatment', 'stress_level',
                                'sleep_hours', 'physical_activity_days',
                                'depression_score', 'anxiety_score',
                                'social_support_score', 'productivity_score']

        input_data = request.get_json()

        # Check if all required features are present
        if not all(feature in input_data for feature in expected_features):
            return jsonify({'error': 'Missing one or more required features'}), 400

        # Convert to DataFrame
        data_df = pd.DataFrame([input_data])

        # Model prediction
        proba = mental_model.predict_proba(data_df)[0]  # Assuming 3-class softmax output
        risk_class = np.argmax(proba) + 1  # Maps to: High=1, Medium=2, Low=3

        # Optional: assign readable labels
        risk_labels = {1: "High", 2: "Medium", 3: "Low"}
        risk_label_str = risk_labels[risk_class]

        return jsonify({
            'mental_risk_class': int(risk_class),  # cast from np.int64 to Python int
            'mental_risk_level': risk_label_str,
            'confidence_scores': {
            'High': float(round(proba[0], 3)),
            'Medium': float(round(proba[1], 3)),
            'Low': float(round(proba[2], 3))

            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-diabetes', methods=['POST'])
def predict_diabetes():
    try:
        expected_keys = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
        
        input_data = request.get_json()

        # Validate keys
        if not all(k in input_data for k in expected_keys):
            return jsonify({'error': 'Missing one or more required features'}), 400

        # Ensure proper column names match model
        data_df = pd.DataFrame([{
            'Pregnancies': input_data['Pregnancies'],
            'Glucose': input_data['Glucose'],
            'BloodPressure': input_data['BloodPressure'],
            'Insulin': input_data['Insulin'],
            'BMI': input_data['BMI'],
            'Age': input_data['Age']
        }])

        # Predict
        proba = diabetes_model.predict_proba(data_df)[0][1]
        risk = int(proba > 0.5)

        return jsonify({
            'diabetes_probability': round(float(proba), 3),
            'diabetes_risk': 'Positive' if risk else 'Negative',
            'risk_score_percent': round(float(proba * 100), 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def risk_label(prob, low_thresh=0.30, high_thresh=0.50):
    if prob < low_thresh:
        return 'Low'
    elif prob < high_thresh:
        return 'Moderate'
    else:
        return 'High'

@app.route('/predict/heart', methods=['POST'])
def predictheart():
    try:
        # Parse input JSON data
        input_data = request.get_json()
        expected_features = ['Sex', 'Cholesterol', 'Diabetes', 'Smoking', 'Alcohol Consumption']
        # Ensure all required features are present
        if not all(feature in input_data for feature in expected_features):
            return jsonify({'error': 'Missing one or more required features'}), 400

        # Convert input to DataFrame
        data_df = pd.DataFrame([input_data])
        data_df["Diabetes"] = 1 - data_df["Diabetes"]
        data_df["Alcohol Consumption"] = 1 - data_df["Alcohol Consumption"]

        # Make prediction
        proba = model.predict_proba(data_df)[0][1]  
        print(proba)
        threshold = 0.25  
        risk = 1 if proba > threshold else 0
        risk_level = risk_label(proba)


        return jsonify({
        'risk_score': round((proba*100)%100, 3),
        'risk_level': risk_level
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-hypertension', methods=['POST'])
def predict_hypertension():
    try:
        expected_keys = ['male', 'age', 'currentSmoker', 'diabetes', 'BMI', 'heartRate']
        input_data = request.get_json()

        # Validate input
        if not all(k in input_data for k in expected_keys):
            return jsonify({'error': 'Missing one or more required features'}), 400

        # Convert to DataFrame
        data_df = pd.DataFrame([{
            'male': input_data['male'],
            'age': input_data['age'],
            'currentSmoker': input_data['currentSmoker'],
            'diabetes': input_data['diabetes'],
            'BMI': input_data['BMI'],
            'heartRate': input_data['heartRate']
        }])

        # Predict
        proba = hypertension_model.predict_proba(data_df)[0][1]  
        threshold = 0.4
        risk = int(proba > threshold)

        return jsonify({
            'hypertension_probability': round(float(proba), 3),
            'hypertension_risk': 'Positive' if risk else 'Negative',
            'risk_score_percent': round(float(proba * 100), 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)