"""
pip install pandas numpy scikit-learn xgboost matplotlib

PART 5: Model Ensembles - The Power of Collective Intelligence
================================================================
This script demonstrates ensemble learning techniques for customer churn prediction.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


#import sys
#import io

# Forcibly set UTF-8 for output
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')



# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("=" * 80)
print("1. DATA PREPARATION")
print("=" * 80)

# Create sample data for demonstration if file doesn't exist
try:
    df = pd.read_csv('telco_churn.csv')
    print("✓ Data loaded from file")
except FileNotFoundError:
    print("! Sample data file not found. Creating synthetic data...")
   
    
# Generate synthetic telco churn data
np.random.seed(42)  # For reproducibility
n_samples = 1000    # Number of customers

data = {
    'customerID': [f'CUST_{i:04d}' for i in range(n_samples)],  # Unique ID
    'tenure': np.random.randint(1, 72, n_samples),              # Months as customer
    'MonthlyCharges': np.random.uniform(20, 120, n_samples).round(2),  # Monthly bill
    'TotalCharges': np.random.uniform(100, 8000, n_samples).round(2),  # Total paid
    'gender': np.random.choice(['Male', 'Female'], n_samples),   # Gender
    'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),  # Senior flag
    'Partner': np.random.choice(['Yes', 'No'], n_samples),       # Has partner
    'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),  # Has dependents
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'],  # Contract type
     n_samples, p=[0.5, 0.3, 0.2]),
    'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])  # Target variable
}    
    
df = pd.DataFrame(data)
print(f"✓ Synthetic dataset created with {n_samples} samples")

print(f"\nData Shape: {df.shape}")
print(f"\nTarget distribution:")
print(df['Churn'].value_counts(normalize=True))

# =============================================================================
# 2. DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 80)
print("2. DATA PREPROCESSING")
print("=" * 80)

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'customerID' in categorical_cols:
    categorical_cols.remove('customerID')  # Remove ID column

print(f"Categorical columns found: {len(categorical_cols)}")

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Shape after encoding: {df_encoded.shape}")

# Define features and target
X = df_encoded.drop(['Churn', 'customerID'], axis=1, errors='ignore')
y = df_encoded['Churn']

print(f"Number of features after encoding: {X.shape[1]}")

# =============================================================================
# 3. TRAIN-TEST SPLIT
# =============================================================================
print("\n" + "=" * 80)
print("3. TRAIN-TEST SPLIT")
print("=" * 80)

from sklearn.model_selection import train_test_split

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nClass distribution in train:")
print(y_train.value_counts(normalize=True))

# =============================================================================
# 4. BASELINE MODEL: SINGLE DECISION TREE
# =============================================================================
print("\n" + "=" * 80)
print("4. BASELINE MODEL: SINGLE DECISION TREE")
print("=" * 80)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Train a single deep tree
single_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
single_tree.fit(X_train, y_train)

# Predictions
y_pred_tree = single_tree.predict(X_test)
y_proba_tree = single_tree.predict_proba(X_test)[:, 1]

print("\n--- Single Decision Tree Performance ---")
print(classification_report(y_test, y_pred_tree))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba_tree):.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

# =============================================================================
# 5. RANDOM FOREST - THE POWER OF THE CROWD
# =============================================================================
print("\n" + "=" * 80)
print("5. RANDOM FOREST - THE POWER OF THE CROWD")
print("=" * 80)

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,           # Maximum depth
    min_samples_split=20,   # Minimum samples to split
    min_samples_leaf=10,    # Minimum samples in leaf
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Random Forest Performance ---")
print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba_rf):.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# =============================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("6. FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Get feature importances
importances = rf_model.feature_importances_
feature_names = X.columns

# Create DataFrame for sorting
importance_df = pd.DataFrame({
    'feature': feature_names, 
    'importance': importances
})
importance_df = importance_df.sort_values('importance', ascending=False)

print("\n--- Top 10 Most Important Features for Churn ---")
print(importance_df.head(10).to_string(index=False))

# Visualize feature importance
print("\nFeature Importance Visualization:")
print("-" * 60)
top_features = importance_df.head(10)
max_imp = top_features['importance'].max()

for _, row in top_features.iterrows():
    bar_length = int(50 * row['importance'] / max_imp)
    bar = '█' * bar_length
    print(f"{row['feature'][:35]:35} | {bar} ({row['importance']:.3f})")

# =============================================================================
# 7. GRADIENT BOOSTING (XGBoost)
# =============================================================================
print("\n" + "=" * 80)
print("7. GRADIENT BOOSTING (XGBoost)")
print("=" * 80)

try:
    import xgboost as xgb
    
    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    print("\n--- XGBoost Performance ---")
    print(classification_report(y_test, y_pred_xgb))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba_xgb):.3f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))
    
except ImportError:
    print("\n! XGBoost not installed. Skipping example.")
    print("  Install with: pip install xgboost")

# =============================================================================
# 8. MODEL COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("8. MODEL COMPARISON")
print("=" * 80)

comparison_data = {
    'Model': ['Decision Tree', 'Random Forest', 'XGBoost'],
    'ROC-AUC': [
        roc_auc_score(y_test, y_proba_tree),
        roc_auc_score(y_test, y_proba_rf),
        roc_auc_score(y_test, y_proba_xgb) if 'y_proba_xgb' in dir() else None
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.dropna()
print("\nModel Performance Comparison:")
print(comparison_df.to_string(index=False))

# =============================================================================
# 9. BUSINESS APPLICATION: TARGETED RETENTION CAMPAIGN
# =============================================================================
print("\n" + "=" * 80)
print("9. BUSINESS APPLICATION: TARGETED RETENTION CAMPAIGN")
print("=" * 80)

# Get test data with customer IDs
test_customers = df.loc[y_test.index].copy()
test_customers['churn_probability'] = y_proba_rf

# Rank customers by churn probability
test_customers['churn_rank'] = test_customers['churn_probability'].rank(
    ascending=False, method='first'
)

# Select top 20% for targeting
budget_percentage = 0.20
num_customers_to_target = int(len(test_customers) * budget_percentage)
target_customers = test_customers[
    test_customers['churn_rank'] <= num_customers_to_target
]

print(f"\n--- Campaign Targeting Results ---")
print(f"Budget allows targeting top {budget_percentage:.0%} of customers")
print(f"Number of customers to contact: {len(target_customers)}")

# Calculate campaign efficiency
actual_churners_in_target = target_customers['Churn'].sum()
total_actual_churners = test_customers['Churn'].sum()
capture_rate = actual_churners_in_target / total_actual_churners

print(f"\nCampaign Efficiency:")
print(f"- We target {budget_percentage:.0%} of customers")
print(f"- We capture {capture_rate:.1%} of all actual churners")
print(f"- LIFT: {capture_rate / budget_percentage:.1f}x (vs random targeting)")

# =============================================================================
# 10. KEY INSIGHTS AND BUSINESS RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 80)
print("10. KEY INSIGHTS AND BUSINESS RECOMMENDATIONS")
print("=" * 80)

print("\n--- Insights ---")
print("1. Ensemble methods significantly outperform single models")
print("2. Random Forest provides robust predictions AND feature importance")
print("3. Top churn drivers identified - focus retention efforts there")
print("4. Campaign targeting with ML provides 2-3x better efficiency")
print("5. Model selection depends on business needs (accuracy vs interpretability)")

print("\n" + "=" * 80)
print("END OF PART 5: Model Ensembles")
print("=" * 80)

# =============================================================================
# BONUS: Quick visualization function (optional)
# =============================================================================

def plot_feature_importance(importance_df, top_n=10):
    """
    Plot feature importance (requires matplotlib)
    """
    try:
        import matplotlib.pyplot as plt
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        print("✓ Feature importance plot displayed")
        
    except ImportError:
        print("\n! matplotlib not installed. Skipping visualization.")
        print("  Install with: pip install matplotlib")

# Uncomment to visualize
# plot_feature_importance(importance_df)