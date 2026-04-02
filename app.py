"""
CREDIT RISK API - Flask REST Endpoint
Run with: python app.py
Endpoints:
  GET  /          - API information
  GET  /health    - Health check
  POST /predict   - Get credit risk prediction
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

print("Loading model and preprocessing objects...")
model = joblib.load('credit_model.joblib')
scaler = joblib.load('scaler.joblib')
encoders = joblib.load('label_encoders.joblib')

numerical_cols = ['age', 'income', 'loan_amount', 'credit_score', 
                  'debt_to_income', 'employment_years', 'late_payments']
categorical_cols = ['gender', 'education', 'marital_status']

@app.route('/')
def home():
    return jsonify({
        'service': 'Credit Risk Prediction API',
        'version': '1.0',
        'endpoints': {
            '/': 'This information',
            '/health': 'Health check',
            '/predict': 'POST - Submit customer data for risk prediction'
        },
        'example_request': {
            'age': 45,
            'gender': 'Male',
            'income': 60000,
            'loan_amount': 20000,
            'credit_score': 720,
            'debt_to_income': 0.3,
            'employment_years': 10,
            'late_payments': 0,
            'education': 'Bachelor',
            'marital_status': 'Married'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': str(pd.Timestamp.now())
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        required_fields = numerical_cols + categorical_cols
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing fields: {missing_fields}',
                'message': 'Prediction failed'
            }), 400
        
        input_df = pd.DataFrame([data])
        
        for col in categorical_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = encoders[col].transform(input_df[col])
                except ValueError as e:
                    return jsonify({
                        'error': f'Invalid value for {col}: {data[col]}',
                        'message': 'Prediction failed'
                    }), 400
        
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        if probability > 0.7:
            risk_level = "Low Risk"
            recommendation = "Approve"
        elif probability > 0.3:
            risk_level = "Medium Risk"
            recommendation = "Review Required"
        else:
            risk_level = "High Risk"
            recommendation = "Decline"
        
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_category': 'Good' if prediction == 1 else 'Bad',
            'risk_level': risk_level,
            'recommendation': recommendation,
            'message': 'Success'
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
