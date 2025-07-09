import os
from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)
# Load your saved model
model_path = r'C:\Users\hs298\Desktop\Projects\Life Scoop\ML Backend\Models\heart_attack_modelV1.joblib'
model = load(model_path)
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

if __name__ == '__main__':
    app.run(debug=True)