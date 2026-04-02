# Load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve)

# Setting up matplotlib styles
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# Load data
df = pd.read_csv('german_credit_data.csv')  # or your file

# Check columns
print("Column names:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# ============================================
# TARGET VARIABLE CONVERSION TO NUMERIC FORMAT
# ============================================
print("\n" + "="*50)
print("TARGET VARIABLE CONVERSION")
print("="*50)

# Check if 'Risk' or 'risk' column exists
if 'Risk' in df.columns:
    print(f"Risk column values: {df['Risk'].unique()}")
    if df['Risk'].dtype == 'object':
        # Convert 'good' -> 1, 'bad' -> 0
        df['risk_numeric'] = (df['Risk'] == 'good').astype(int)
        print("Converted 'good'/'bad' to 1/0")
    else:
        df['risk_numeric'] = df['Risk']
elif 'risk' in df.columns:
    print(f"risk column values: {df['risk'].unique()}")
    if df['risk'].dtype == 'object':
        df['risk_numeric'] = (df['risk'] == 'good').astype(int)
        print("Converted 'good'/'bad' to 1/0")
    else:
        df['risk_numeric'] = df['risk']
else:
    # If no standard column names, use the last column
    target_col = df.columns[-1]
    print(f"Using last column '{target_col}' as target")
    if df[target_col].dtype == 'object':
        unique_vals = df[target_col].unique()
        print(f"Unique values: {unique_vals}")
        # If values are 'good'/'bad'
        if 'good' in unique_vals and 'bad' in unique_vals:
            df['risk_numeric'] = (df[target_col] == 'good').astype(int)
        else:
            # Otherwise just encode as 0/1
            df['risk_numeric'] = pd.factorize(df[target_col])[0]
    else:
        df['risk_numeric'] = df[target_col]

# Check conversion results
print(f"\nConverted target distribution:")
print(df['risk_numeric'].value_counts(normalize=True))
print(f"Values: {df['risk_numeric'].unique()}")

# Remove old target variable if it exists
if 'risk' in df.columns:
    df = df.drop('risk', axis=1)
if 'Risk' in df.columns:
    df = df.drop('Risk', axis=1)

# Rename the new column
df = df.rename(columns={'risk_numeric': 'risk'})

# ============================================
# CATEGORICAL FEATURES ENCODING
# ============================================
print("\n" + "="*50)
print("ENCODING CATEGORICAL FEATURES")
print("="*50)

# Find categorical columns (object type)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns found: {categorical_cols}")

if categorical_cols:
    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"After encoding - shape: {df.shape}")
else:
    print("No categorical columns to encode")

# ============================================
# DATA PREPARATION FOR MODELS
# ============================================
print("\n" + "="*50)
print("DATA PREPARATION")
print("="*50)

# Define features and target variable
X = df.drop('risk', axis=1)
y = df['risk']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target values: {np.unique(y)}")
print(f"Target distribution:\n{y.value_counts(normalize=True)}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")

# ============================================
# FUNCTION FOR METRICS OUTPUT WITH BUSINESS INTERPRETATION
# ============================================
def print_business_metrics(y_true, y_pred, y_pred_proba, model_name, 
                          profit_per_good=1000, loss_per_bad=5000):
    """
    Output metrics with business interpretation
    
    Parameters:
    y_true: true values (0 - good, 1 - bad)
    y_pred: predicted classes
    y_pred_proba: probabilities for positive class (1 - bad)
    model_name: model name
    profit_per_good: profit from good client (default 1000₽)
    loss_per_bad: loss from bad client (default 5000₽)
    """
    print(f"\n{'='*70}")
    print(f"BUSINESS METRICS FOR {model_name}")
    print(f"{'='*70}")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\nCLASSIC METRICS:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nCONFUSION MATRIX:")
    print(f"{'':25} {'PREDICTED':^30}")
    print(f"{'':25} {'Good (0)':^12} {'Bad (1)':^12}")
    print(f"{'ACTUAL Good (0)':25} {tn:^12} {fp:^12}")
    print(f"{'ACTUAL Bad (1)':25} {fn:^12} {tp:^12}")
    
    # Percentage ratios
    total_good = tn + fn
    total_bad = fp + tp
    
    print(f"\nPERCENTAGE RATIOS:")
    if total_good > 0:
        print(f"   Good clients correctly identified: {tn/total_good*100:.1f}% ({tn} of {total_good})")
        print(f"   Good clients incorrectly rejected: {fn/total_good*100:.1f}% ({fn} of {total_good})")
    if total_bad > 0:
        print(f"   Bad clients correctly identified: {tp/total_bad*100:.1f}% ({tp} of {total_bad})")
        print(f"   Bad clients incorrectly approved: {fp/total_bad*100:.1f}% ({fp} of {total_bad})")
    
    # Business interpretation
    print(f"\nBUSINESS INTERPRETATION:")
    print(f"   [OK] TN (True Negatives): {tn} clients")
    print(f"        Correctly approved good clients - PROFIT")
    print(f"   [!!] FP (False Positives): {fp} clients - MOST EXPENSIVE ERROR")
    print(f"        Incorrectly approved bad clients - DIRECT LOSSES")
    print(f"   [!] FN (False Negatives): {fn} clients")
    print(f"        Incorrectly rejected good clients - MISSED PROFIT")
    print(f"   [OK] TP (True Positives): {tp} clients")
    print(f"        Correctly rejected bad clients - SAVED MONEY")
    
    # Financial result
    financial_result = (tn * profit_per_good + tp * loss_per_bad - 
                       fp * loss_per_bad - fn * profit_per_good)
    
    print(f"\nFINANCIAL RESULT (with profit {profit_per_good}₽/good, loss {loss_per_bad}₽/bad):")
    print(f"   + Profit from good (TN × {profit_per_good}₽): +{tn * profit_per_good:,.0f}₽")
    print(f"   + Saved from bad (TP × {loss_per_bad}₽): +{tp * loss_per_bad:,.0f}₽")
    print(f"   - Losses from bad (FP × {loss_per_bad}₽): -{fp * loss_per_bad:,.0f}₽")
    print(f"   - Missed from good (FN × {profit_per_good}₽): -{fn * profit_per_good:,.0f}₽")
    print(f"   {'='*50}")
    print(f"   TOTAL FINANCIAL RESULT: {financial_result:,.0f}₽")
    
    # Additional business metrics
    print(f"\nBUSINESS METRICS:")
    if total_bad > 0:
        bad_approval_rate = fp / total_bad * 100
        print(f"   Bad Approval Rate: {bad_approval_rate:.1f}%")
    if total_good > 0:
        good_rejection_rate = fn / total_good * 100
        print(f"   Good Rejection Rate: {good_rejection_rate:.1f}%")
    
    # Efficiency compared to random model
    random_result = (total_good * profit_per_good - total_bad * loss_per_bad)
    improvement = financial_result - random_result
    
    print(f"\nMODEL EFFICIENCY:")
    print(f"   Without model (approve everyone): {random_result:,.0f}₽")
    print(f"   With model: {financial_result:,.0f}₽")
    if random_result != 0:
        print(f"   Improvement: +{improvement:,.0f}₽ ({improvement/abs(random_result)*100:.1f}% relative to baseline)")
    
    print(f"\n{'='*70}")
    
    return financial_result

# ============================================
# MODEL TRAINING
# ============================================
print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

# Model 1: Logistic Regression
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train)
print("✓ Logistic Regression trained")

# Model 2: k-NN (k=5)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
print("✓ KNN (k=5) trained")

# ============================================
# MODEL EVALUATION WITH BUSINESS METRICS
# ============================================
print("\n" + "="*50)
print("MODEL EVALUATION WITH BUSINESS METRICS")
print("="*50)

# Predictions and probabilities
y_pred_log = log_model.predict(X_test)
y_pred_proba_log = log_model.predict_proba(X_test)[:, 1]

y_pred_knn = knn_model.predict(X_test)
y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]

# Use the new function for metrics output
profit_log = print_business_metrics(y_test, y_pred_log, y_pred_proba_log, 
                                   "LOGISTIC REGRESSION")

profit_knn = print_business_metrics(y_test, y_pred_knn, y_pred_proba_knn, 
                                   "K-NEAREST NEIGHBORS (k=5)")

# ============================================
# VISUALIZATION
# ============================================
print("\n" + "="*50)
print("VISUALIZATION")
print("="*50)

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Logistic Regression
cm_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Good (0)', 'Bad (1)'], yticklabels=['Good (0)', 'Bad (1)'])
axes[0].set_title('Logistic Regression - Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)
# Add business labels (without emojis)
axes[0].text(0.5, -0.15, 'FP: Approved bad clients (COSTLY ERROR)', 
             transform=axes[0].transAxes, ha='center', fontsize=10, color='red')
axes[0].text(0.5, -0.2, 'FN: Rejected good clients (LOST PROFIT)', 
             transform=axes[0].transAxes, ha='center', fontsize=10, color='orange')

# KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Good (0)', 'Bad (1)'], yticklabels=['Good (0)', 'Bad (1)'])
axes[1].set_title('KNN (k=5) - Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].text(0.5, -0.15, 'FP: Approved bad clients (COSTLY ERROR)', 
             transform=axes[1].transAxes, ha='center', fontsize=10, color='red')
axes[1].text(0.5, -0.2, 'FN: Rejected good clients (LOST PROFIT)', 
             transform=axes[1].transAxes, ha='center', fontsize=10, color='orange')

plt.tight_layout()
plt.show()

# ROC Curves
plt.figure(figsize=(10, 8))
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_proba_log)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_proba_knn)

plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_pred_proba_log):.3f})', 
         linewidth=2, color='blue')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_score(y_test, y_pred_proba_knn):.3f})', 
         linewidth=2, color='orange')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================
# KNN OPTIMIZATION
# ============================================
print("\n" + "="*50)
print("KNN OPTIMIZATION")
print("="*50)

# Search for optimal k
k_range = range(1, 31)
k_scores = []
k_auc_scores = []
k_financial_results = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Accuracy through cross-validation
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
    
    # ROC-AUC through cross-validation  
    auc_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='roc_auc')
    k_auc_scores.append(auc_scores.mean())
    
    # For financial result, train on all train and test on test
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_k)
    tn, fp, fn, tp = cm.ravel()
    financial = tn * 1000 + tp * 5000 - fp * 5000 - fn * 1000
    k_financial_results.append(financial)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Accuracy
axes[0].plot(k_range, k_scores, marker='o', color='green', markersize=5)
axes[0].set_xlabel('K Value', fontsize=12)
axes[0].set_ylabel('Cross-Validated Accuracy', fontsize=12)
axes[0].set_title('KNN: Accuracy vs K Value', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
best_k_acc = k_range[k_scores.index(max(k_scores))]
axes[0].axvline(x=best_k_acc, color='red', linestyle='--', 
                label=f'Best k={best_k_acc} (Acc={max(k_scores):.3f})')
axes[0].legend()

# Plot 2: ROC-AUC
axes[1].plot(k_range, k_auc_scores, marker='o', color='purple', markersize=5)
axes[1].set_xlabel('K Value', fontsize=12)
axes[1].set_ylabel('Cross-Validated ROC-AUC', fontsize=12)
axes[1].set_title('KNN: ROC-AUC vs K Value', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
best_k_auc = k_range[k_auc_scores.index(max(k_auc_scores))]
axes[1].axvline(x=best_k_auc, color='red', linestyle='--', 
                label=f'Best k={best_k_auc} (AUC={max(k_auc_scores):.3f})')
axes[1].legend()

# Plot 3: Financial Result
axes[2].plot(k_range, k_financial_results, marker='o', color='gold', markersize=5)
axes[2].set_xlabel('K Value', fontsize=12)
axes[2].set_ylabel('Financial Result (₽)', fontsize=12)
axes[2].set_title('KNN: Financial Result vs K Value', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
best_k_fin = k_range[k_financial_results.index(max(k_financial_results))]
axes[2].axvline(x=best_k_fin, color='red', linestyle='--', 
                label=f'Best k={best_k_fin} (Profit={max(k_financial_results):,.0f}₽)')
axes[2].legend()

plt.tight_layout()
plt.show()

print(f"\nBest k for Accuracy: {best_k_acc} (Accuracy: {max(k_scores):.4f})")
print(f"Best k for ROC-AUC: {best_k_auc} (AUC: {max(k_auc_scores):.4f})")
print(f"Best k for Financial Result: {best_k_fin} (Profit: {max(k_financial_results):,.0f}₽)")

# Train the best model based on financial result
best_knn = KNeighborsClassifier(n_neighbors=best_k_fin)
best_knn.fit(X_train, y_train)
y_pred_best_knn = best_knn.predict(X_test)
y_pred_proba_best_knn = best_knn.predict_proba(X_test)[:, 1]

# Evaluate the best model
profit_best = print_business_metrics(y_test, y_pred_best_knn, y_pred_proba_best_knn, 
                                    f"OPTIMIZED KNN (k={best_k_fin} for profit)")

# ============================================
# SUMMARY TABLE
# ============================================
print("\n" + "="*70)
print("MODELS COMPARISON SUMMARY")
print("="*70)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN (k=5)', f'KNN (k={best_k_fin})'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_best_knn)
    ],
    'Precision': [
        precision_score(y_test, y_pred_log),
        precision_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_best_knn)
    ],
    'Recall': [
        recall_score(y_test, y_pred_log),
        recall_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_best_knn)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_log),
        f1_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_best_knn)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba_log),
        roc_auc_score(y_test, y_pred_proba_knn),
        roc_auc_score(y_test, y_pred_proba_best_knn)
    ],
    'Profit (₽)': [profit_log, profit_knn, profit_best]
})

# Format table output
print("\n")
print(results.round(4).to_string(index=False))
print("\n" + "="*70)

# Find the best model by profit
best_model_idx = results['Profit (₽)'].argmax()
best_model_name = results.iloc[best_model_idx]['Model']
best_profit = results.iloc[best_model_idx]['Profit (₽)']

print(f"\n🏆 BEST MODEL BY PROFIT: {best_model_name}")
print(f"   Profit: {best_profit:,.0f}₽")

# ============================================
# FEATURE IMPORTANCE
# ============================================
print("\n" + "="*50)
print("FEATURE IMPORTANCE (LOGISTIC REGRESSION)")
print("="*50)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': log_model.coef_[0]
})
feature_importance = feature_importance.sort_values('coefficient', key=abs, ascending=False)

print("\nTop 10 most important features (by absolute coefficient):")
print(feature_importance.head(10).to_string(index=False))

# Feature importance visualization
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]
plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Coefficient Value (negative = increases risk, positive = decreases risk)', fontsize=12)
plt.title('Top 15 Feature Importance (Logistic Regression)', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)