"""
================================================================================
PART 8: DEPLOYMENT & ETHICS - From Prototype to Value
================================================================================
Complete Python script covering:
1. Model training and serialization (pickle, joblib)
2. Flask REST API creation for model deployment
3. Fairness auditing and bias detection
4. Model card generation and monitoring plan
5. Business impact analysis

Author: Data Science Course
Date: 2024
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import pickle
import joblib
import json
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

print("=" * 80)
print("PART 8: DEPLOYMENT & ETHICS - From Prototype to Value")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)


# ============================================================================
# SECTION 2: CREATE SYNTHETIC DATASET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: DATA GENERATION - Credit Risk Dataset")
print("=" * 80)

np.random.seed(42)
n_samples = 5000

print(f"\nGenerating {n_samples} synthetic customer records...")

data = {
    'age': np.random.randint(18, 70, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
    'income': np.random.normal(50000, 20000, n_samples).astype(int),
    'loan_amount': np.random.normal(15000, 10000, n_samples).astype(int),
    'credit_score': np.random.randint(300, 850, n_samples),
    'debt_to_income': np.random.uniform(0, 0.5, n_samples),
    'employment_years': np.random.randint(0, 40, n_samples),
    'late_payments': np.random.poisson(1, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples)
}

df = pd.DataFrame(data)

risk_score = (
    -df['credit_score'] / 1000 +
    df['debt_to_income'] * 3 +
    df['late_payments'] * 0.5 +
    np.random.normal(0, 0.3, n_samples)
)
df['risk'] = (risk_score > np.median(risk_score)).astype(int)

print(f"\n[OK] Dataset created successfully!")
print(f"   Shape: {df.shape}")
print(f"   Features: {list(df.columns)}")
print(f"\n   Target distribution:")
print(f"   Good credit (1): {df['risk'].sum():,} ({df['risk'].mean()*100:.1f}%)")
print(f"   Bad credit (0):  {(len(df)-df['risk'].sum()):,} ({(1-df['risk'].mean())*100:.1f}%)")


# ============================================================================
# SECTION 3: PREPROCESSING AND MODEL TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: PREPROCESSING AND MODEL TRAINING")
print("=" * 80)

X = df.drop('risk', axis=1)
y = df['risk']

categorical_cols = ['gender', 'education', 'marital_status']
numerical_cols = ['age', 'income', 'loan_amount', 'credit_score', 
                  'debt_to_income', 'employment_years', 'late_payments']

print(f"\nCategorical features: {categorical_cols}")
print(f"Numerical features: {numerical_cols}")

print("\nEncoding categorical features...")
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
    print(f"   {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\nScaling numerical features...")
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print(f"   Scaled {len(numerical_cols)} features to zero mean, unit variance")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split:")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"\n[OK] Model trained successfully!")
print(f"   Training accuracy: {train_score:.4f}")
print(f"   Test accuracy: {test_score:.4f}")
print(f"   ROC-AUC: {auc_score:.4f}")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))


# ============================================================================
# SECTION 4: MODEL SERIALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: MODEL SERIALIZATION - Saving for Deployment")
print("=" * 80)

print("\nSaving with pickle...")
with open('credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   [OK] Saved as 'credit_model.pkl'")

print("\nSaving with joblib...")
joblib.dump(model, 'credit_model.joblib')
print("   [OK] Saved as 'credit_model.joblib'")

print("\nSaving preprocessing objects...")
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(encoders, 'label_encoders.joblib')
print("   [OK] Saved scaler.joblib and label_encoders.joblib")

print("\nVerifying model loading...")
with open('credit_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
sample = X_test.iloc[[0]]
pred_original = model.predict(sample)[0]
pred_loaded = loaded_model.predict(sample)[0]

if pred_original == pred_loaded:
    print("   [OK] Model loads correctly! Ready for deployment.")
else:
    print("   [ERROR] Model loading failed!")


# ============================================================================
# SECTION 5: FLASK API TEMPLATE GENERATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: FLASK API TEMPLATE GENERATION")
print("=" * 80)

api_code = '''"""
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
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(api_code)

print("\n[OK] Flask API template saved as 'app.py'")
print("\nAPI Usage Instructions:")
print("   1. Install Flask: pip install flask")
print("   2. Run API: python app.py")
print("   3. Test with curl")


# ============================================================================
# SECTION 6: FAIRNESS AUDIT - BIAS DETECTION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: FAIRNESS AUDIT - Bias Detection")
print("=" * 80)

df['prediction'] = model.predict(X)
df['probability'] = model.predict_proba(X)[:, 1]

df['age_group'] = pd.cut(
    df['age'], 
    bins=[0, 30, 50, 100], 
    labels=['Young (<30)', 'Middle (30-50)', 'Senior (>50)']
)

print("\n" + "-" * 80)
print("FAIRNESS METRICS BY DEMOGRAPHIC GROUP")
print("-" * 80)

print("\n1. GENDER ANALYSIS")
print("-" * 50)

gender_stats = {}
for gender in ['Male', 'Female']:
    group = df[df['gender'] == gender]
    approval_rate = group['prediction'].mean()
    avg_prob = group['probability'].mean()
    count = len(group)
    gender_stats[gender] = {'approval_rate': approval_rate, 'count': count}
    
    print(f"\n{gender}:")
    print(f"   Customers: {count}")
    print(f"   Approval rate: {approval_rate:.1%}")
    print(f"   Avg risk probability: {avg_prob:.3f}")

approval_male = gender_stats['Male']['approval_rate']
approval_female = gender_stats['Female']['approval_rate']
disparate_impact_gender = min(approval_female, approval_male) / max(approval_female, approval_male)

print(f"\nDisparate Impact Ratio (Female/Male): {disparate_impact_gender:.3f}")
print(f"   {'[PASS]' if disparate_impact_gender >= 0.8 else '[FLAG] Possible discrimination'}")

print("\n2. AGE GROUP ANALYSIS")
print("-" * 50)

age_stats = {}
for age_group in ['Young (<30)', 'Middle (30-50)', 'Senior (>50)']:
    group = df[df['age_group'] == age_group]
    approval_rate = group['prediction'].mean()
    count = len(group)
    age_stats[age_group] = {'approval_rate': approval_rate, 'count': count}
    
    print(f"\n{age_group}:")
    print(f"   Customers: {count}")
    print(f"   Approval rate: {approval_rate:.1%}")

approval_senior = age_stats['Senior (>50)']['approval_rate']
approval_middle = age_stats['Middle (30-50)']['approval_rate']
disparate_impact_age = approval_senior / approval_middle

print(f"\nDisparate Impact Ratio (Senior/Middle): {disparate_impact_age:.3f}")
print(f"   {'[PASS]' if disparate_impact_age >= 0.8 else '[FLAG] Possible age discrimination'}")

print("\n3. ERROR RATE ANALYSIS")
print("-" * 50)
print("False Positive: Bad loan approved")
print("False Negative: Good loan rejected")

for gender in ['Male', 'Female']:
    group = df[df['gender'] == gender]
    
    tp = ((group['prediction'] == 1) & (group['risk'] == 1)).sum()
    tn = ((group['prediction'] == 0) & (group['risk'] == 0)).sum()
    fp = ((group['prediction'] == 1) & (group['risk'] == 0)).sum()
    fn = ((group['prediction'] == 0) & (group['risk'] == 1)).sum()
    
    total_bad = (group['risk'] == 0).sum()
    total_good = (group['risk'] == 1).sum()
    
    fp_rate = fp / total_bad if total_bad > 0 else 0
    fn_rate = fn / total_good if total_good > 0 else 0
    
    print(f"\n{gender}:")
    print(f"   False Positive Rate: {fp_rate:.3f} ({fp}/{total_bad})")
    print(f"   False Negative Rate: {fn_rate:.3f} ({fn}/{total_good})")

print("\n" + "=" * 80)
print("BIAS SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

bias_flags = []
if disparate_impact_gender < 0.8:
    bias_flags.append("Gender disparity detected")
if disparate_impact_age < 0.8:
    bias_flags.append("Age disparity detected")

if bias_flags:
    print("\n[WARNING] BIAS ALERT: The following issues were detected:")
    for flag in bias_flags:
        print(f"   * {flag}")
    print("\nRECOMMENDED ACTIONS:")
    print("   1. Collect more diverse training data")
    print("   2. Consider fairness-aware algorithms")
    print("   3. Implement human oversight for borderline cases")
else:
    print("\n[OK] FAIRNESS CHECK PASSED: No severe disparate impact detected")


# ============================================================================
# SECTION 7: MODEL CARD GENERATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: MODEL CARD - Documentation")
print("=" * 80)

model_card = {
    "Model Information": {
        "Name": "Credit Risk Random Forest v1.0",
        "Version": "1.0",
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Type": "Binary Classification",
        "Algorithm": "Random Forest Classifier"
    },
    "Intended Use": {
        "Purpose": "Predict probability of loan default",
        "Scope": "Personal loans under $50,000",
        "Users": "Loan officers and credit analysts",
        "Decisions": "Decision support tool (not fully automated)"
    },
    "Training Data": {
        "Source": "Synthetic dataset",
        "Size": f"{n_samples} records",
        "Time Period": "Simulated 2020-2023",
        "Features": numerical_cols + categorical_cols
    },
    "Performance Metrics": {
        "Accuracy": f"{test_score:.3f}",
        "ROC-AUC": f"{auc_score:.3f}"
    },
    "Fairness Assessment": {
        "Gender Disparate Impact": f"{disparate_impact_gender:.3f}",
        "Age Disparate Impact": f"{disparate_impact_age:.3f}",
        "Status": "Pass" if len(bias_flags) == 0 else "Review Required"
    },
    "Known Limitations": [
        "May underperform for customers with thin credit files",
        "Not validated on economic recession data",
        "Uses features that may correlate with protected attributes"
    ],
    "Recommended Mitigations": [
        "Human review for cases with probability between 0.3-0.7",
        "Monthly fairness monitoring dashboard",
        "Quarterly model retraining with new data"
    ]
}

with open('model_card.json', 'w', encoding='utf-8') as f:
    json.dump(model_card, f, indent=2)

print("\n[OK] Model card saved as 'model_card.json'")


# ============================================================================
# SECTION 8: BUSINESS IMPACT ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: BUSINESS IMPACT ANALYSIS")
print("=" * 80)

cost_per_call = 2.50
deposit_value = 50.00
total_customers = len(X_test)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nCAMPAIGN COST-BENEFIT ANALYSIS")
print("-" * 60)

calls_all = total_customers
cost_all = calls_all * cost_per_call
successes_all = y_test.sum()
profit_all = successes_all * deposit_value - cost_all

print(f"\nStrategy 1: Call EVERYONE")
print(f"   Calls: {calls_all}")
print(f"   Cost: ${cost_all:,.2f}")
print(f"   Expected successes: {successes_all}")
print(f"   Profit: ${profit_all:,.2f}")

threshold_70 = np.percentile(y_proba, 70)
targeted_customers = y_proba >= threshold_70
calls_targeted = targeted_customers.sum()
cost_targeted = calls_targeted * cost_per_call
successes_targeted = y_test[targeted_customers].sum()
profit_targeted = successes_targeted * deposit_value - cost_targeted

print(f"\nStrategy 2: Model-targeted (Top 30%)")
print(f"   Calls: {calls_targeted}")
print(f"   Cost: ${cost_targeted:,.2f}")
print(f"   Expected successes: {successes_targeted}")
print(f"   Profit: ${profit_targeted:,.2f}")

print("\nBUSINESS IMPACT SUMMARY")
print("-" * 60)

improvement = profit_targeted - profit_all
roi_improvement = ((profit_targeted / cost_targeted) / (profit_all / cost_all) - 1) * 100

print(f"\n[OK] Best Strategy: Top 30% Targeting")
print(f"   Profit improvement: ${improvement:,.2f}")
print(f"   ROI improvement: {roi_improvement:.1f}%")
print(f"   Calls reduced: {calls_all - calls_targeted} ({(calls_all - calls_targeted)/calls_all*100:.1f}%)")


# ============================================================================
# SECTION 9: MONITORING PLAN
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: MONITORING PLAN - Ongoing Oversight")
print("=" * 80)

monitoring_plan = """
CREDIT RISK MODEL MONITORING PLAN v1.0
========================================

1. QUARTERLY METRICS TO TRACK
   ---------------------------
   a) Performance Metrics:
      - ROC-AUC (target > 0.85)
      - Accuracy (target > 0.80)
      - F1 Score (target > 0.80)
   
   b) Fairness Metrics:
      - Disparate Impact Ratio by gender (target > 0.8)
      - Disparate Impact Ratio by age (target > 0.8)
      - False Positive Rate parity (difference < 0.05)
      - False Negative Rate parity (difference < 0.05)
   
   c) Business Metrics:
      - Profit per campaign
      - Cost per acquisition
      - Conversion rate by segment

2. DATA COLLECTION
   ----------------
   [X] Log all predictions with timestamp
   [X] Store input features for drift detection
   [X] Track actual outcomes (loan performance)
   [X] Collect demographic data with consent

3. ALERT THRESHOLDS
   -----------------
   [GREEN] Green (Normal): All metrics within targets
   [YELLOW] Yellow (Warning): Any metric exceeds threshold by 10%
   [RED] Red (Critical): DIR < 0.7 or accuracy drop > 10%

4. ACTION PROTOCOLS
   -----------------
   Yellow Alert:
   - Investigate root cause
   - Review recent data changes
   - Increase monitoring frequency
   
   Red Alert:
   - Pause automated decisions
   - Convene fairness review committee
   - Retrain model with updated data
   - Conduct full bias audit

5. GOVERNANCE STRUCTURE
   ---------------------
   * Data Science Lead: Model performance monitoring
   * Compliance Officer: Fairness and regulatory review
   * Business Stakeholder: Business impact assessment
   * External Auditor: Annual independent review
"""

print(monitoring_plan)

with open('monitoring_plan.txt', 'w', encoding='utf-8') as f:
    f.write(monitoring_plan)
print("\n[OK] Monitoring plan saved as 'monitoring_plan.txt'")


# ============================================================================
# SECTION 10: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("EXERCISE COMPLETE - DELIVERABLES SUMMARY")
print("=" * 80)

deliverables = [
    ("credit_model.pkl", "Pickle-serialized model file"),
    ("credit_model.joblib", "Joblib-serialized model file"),
    ("scaler.joblib", "Feature scaler for preprocessing"),
    ("label_encoders.joblib", "Categorical encoders"),
    ("app.py", "Flask REST API for model deployment"),
    ("model_card.json", "Model documentation and limitations"),
    ("monitoring_plan.txt", "Quarterly fairness monitoring plan")
]

print("\nGenerated Files:")
for filename, description in deliverables:
    exists = "[X]" if os.path.exists(filename) else "[ ]"
    print(f"   {exists} {filename:30s} - {description}")

print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. DEPLOYMENT CREATES VALUE: A model in a notebook has zero value.
2. MODELS CAN PERPETUATE HARM: Always audit for fairness.
3. FAIRNESS IS MEASURABLE: Use Disparate Impact Ratio and error rates.
4. TRANSPARENCY BUILDS TRUST: Document limitations in model cards.
5. MONITORING IS CONTINUOUS: Models degrade over time.
6. ETHICS IS GOOD BUSINESS: Protects against regulatory risk.
""")

print("\n" + "=" * 80)
print(f"[OK] EXERCISE COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)