from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import InconsistentVersionWarning



# Load models


with ignore_warnings(category=InconsistentVersionWarning):
    mental_health_model = joblib.load(r'C:\Users\hs298\Desktop\Life-Scope\ML Models\menV1.joblib')
    heart_attack_model = joblib.load(r'C:\Users\hs298\Desktop\Life-Scope\ML Models\XGBV4.joblib')
app = Flask(__name__)

@app.route('/predict/mental-health', methods=['POST'])
def predict_mental_health():
    try:
        data = request.get_json()
        
        # Create DataFrame with expected features
        input_data = pd.DataFrame([{
            'age': data['age'],
            'gender': data['gender'],
            'employment_status': data['employment_status'],
            'work_environment': data['work_environment'],
            'mental_health_history': data['mental_health_history'],
            'seeks_treatment': data['seeks_treatment'],
            'stress_level': data['stress_level'],
            'sleep_hours': data['sleep_hours'],
            'physical_activity_days': data['physical_activity_days'],
            'depression_score': data['depression_score'],
            'anxiety_score': data['anxiety_score'],
            'social_support_score': data['social_support_score'],
            'productivity_score': data['productivity_score']
        }])

        prediction = mental_health_model.predict(input_data)[0]
        return jsonify({'mental_health_risk': str(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/heart-attack', methods=['POST'])
def predict_heart_attack():
    try:
        data = request.get_json()
        
        # Process categorical features
        processed = {
            'Age': data['Age'],
            'RestingBP': data['RestingBP'],
            'Cholesterol': data['Cholesterol'],
            'FastingBS': data['FastingBS'],
            'MaxHR': data['MaxHR'],
            'Oldpeak': data['Oldpeak'],
            'Sex_F': data['Sex'] == 'F',
            'Sex_M': data['Sex'] == 'M',
            'ChestPainType_ASY': data['ChestPainType'] == 'ASY',
            'ChestPainType_ATA': data['ChestPainType'] == 'ATA',
            'ChestPainType_NAP': data['ChestPainType'] == 'NAP',
            'ChestPainType_TA': data['ChestPainType'] == 'TA',
            'RestingECG_LVH': data['RestingECG'] == 'LVH',
            'RestingECG_Normal': data['RestingECG'] == 'Normal',
            'RestingECG_ST': data['RestingECG'] == 'ST',
            'ExerciseAngina_N': data['ExerciseAngina'] == 'N',
            'ExerciseAngina_Y': data['ExerciseAngina'] == 'Y',
            'ST_Slope_Down': data['ST_Slope'] == 'Down',
            'ST_Slope_Flat': data['ST_Slope'] == 'Flat',
            'ST_Slope_Up': data['ST_Slope'] == 'Up'
        }

        # Ensure correct column order
        columns = [
            'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
            'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA',
            'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH',
            'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N',
            'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
        ]

        input_df = pd.DataFrame([processed], columns=columns)
        prediction = heart_attack_model.predict(input_df)[0]
        
        y_proba = heart_attack_model.predict_proba(input_df)[:, 1]
        threshold = 0.25  
        prediction = (y_proba >= threshold).astype(int)
        
        return jsonify({'heart_attack_risk': bool(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
