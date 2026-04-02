"""
SECTION 2: REGRESSION MODELS - COMPLETE GUIDE
================================================
Linear Regression, Polynomial Regression, Ridge, Lasso, Cross-Validation
and Business Applications for Advertising Budget Optimization

Author: ML Course
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================================
# PART 2.1: INTRODUCTION TO REGRESSION
# =============================================================================

def regression_basics_explanation():
    """
    Simple explanation of regression concepts with analogies
    """
    print("\n" + "="*70)
    print("PART 2.1: WHAT IS REGRESSION?")
    print("="*70)
    
    print("""
📚 SIMPLE EXPLANATION WITH ANALOGIES:

Imagine you're a chef and your data are ingredients for a complex dish.
We'll learn to predict taste (target variable) from ingredients (features).

1️⃣ LINEAR REGRESSION (Simple straight-line relationship):
   • You believe the connection between ingredients and taste is straight.
   • "The more salt, the saltier the taste - always at the same rate."
   • On a graph: a straight line.
   • Simple and fast, but life is rarely perfectly straight.

2️⃣ POLYNOMIAL REGRESSION (Captures curves and bends):
   • You understand relationships can be more complex.
   • "A little salt enhances flavor, too much ruins it."
   • On a graph: a curved line that better fits the data.
   • Danger: can overfit by following noise instead of patterns.

3️⃣ RIDGE & LASSO REGRESSION (Penalize complexity):
   • Same as linear regression but with a "penalty" for complexity.
   • Like a chef on a diet: can use ingredients, but not too many.
   • Prevents overfitting by keeping the model simpler.
    """)

# =============================================================================
# PART 2.2: LOAD AND PREPARE DATA
# =============================================================================

def load_advertising_data():
    """
    Load and prepare advertising dataset for regression
    Dataset: Advertising spend vs Sales
    """
    print("\n" + "="*70)
    print("PART 2.2: ADVERTISING DATASET FOR REGRESSION")
    print("="*70)
    
    # Create sample advertising data
    np.random.seed(42)
    n_samples = 200
    
    # TV advertising (in $1000s)
    tv = np.random.uniform(0, 300, n_samples)
    
    # Radio advertising (in $1000s)
    radio = np.random.uniform(0, 50, n_samples)
    
    # Newspaper advertising (in $1000s)
    newspaper = np.random.uniform(0, 100, n_samples)
    
    # Sales (in thousands of units) with some noise
    # True relationship: sales = 2.5*tv + 3*radio + 1*newspaper + noise
    sales = (2.5 * tv + 3.0 * radio + 1.0 * newspaper + 
             np.random.normal(0, 20, n_samples) + 50)
    
    # Create dataframe
    df = pd.DataFrame({
        'TV': tv,
        'Radio': radio,
        'Newspaper': newspaper,
        'Sales': sales
    })
    
    print(f"📊 DATASET CREATED: {df.shape[0]} samples × {df.shape[1]} features")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\n📈 STATISTICAL SUMMARY:")
    print(df.describe().round(2))
    
    print("\n🔗 CORRELATION WITH SALES:")
    correlations = df.corr()['Sales'].sort_values(ascending=False)
    for feature, corr in correlations.items():
        strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
        print(f"  {feature:10} : {corr:.3f} ({strength} correlation)")
    
    return df

# =============================================================================
# PART 2.3: SIMPLE LINEAR REGRESSION
# =============================================================================

def simple_linear_regression_demo(df):
    """
    Demonstrate simple linear regression with one feature
    """
    print("\n" + "="*70)
    print("PART 2.3: SIMPLE LINEAR REGRESSION")
    print("="*70)
    
    print("""
📌 WHAT IS SIMPLE LINEAR REGRESSION?
   • Models relationship between ONE feature and target
   • Equation: y = mx + b (like you learned in school!)
   • 'm' is coefficient (slope) - how much y changes when x changes
   • 'b' is intercept - value of y when x is 0
    """)
    
    # Select TV as the feature
    X = df[['TV']]
    y = df['Sales']
    
    print(f"📊 Feature: TV advertising budget ($1000s)")
    print(f"🎯 Target: Sales (thousands of units)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 TRAIN/TEST SPLIT:")
    print(f"  • Training: {X_train.shape[0]} samples (80%) - model learns from these")
    print(f"  • Test:     {X_test.shape[0]} samples (20%) - model is evaluated on these")
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"\n🧠 MODEL TRAINED!")
    print(f"  • Coefficient (slope): {model.coef_[0]:.4f}")
    print(f"  • Intercept:           {model.intercept_:.4f}")
    print(f"  • Equation: Sales = {model.coef_[0]:.3f} × TV + {model.intercept_:.3f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 MODEL EVALUATION ON TEST SET:")
    print(f"  • Mean Absolute Error (MAE):  {mae:.2f}")
    print(f"    Average prediction error (in same units as target)")
    print(f"  • Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"    Penalizes large errors more than MAE")
    print(f"  • R-squared (R²):             {r2:.4f}")
    print(f"    {interpret_r2(r2)}")
    
    # Business interpretation
    print(f"\n💡 BUSINESS INTERPRETATION:")
    print(f"  • For every $1000 increase in TV advertising, sales increase by {model.coef_[0]:.1f} thousand units")
    print(f"  • With zero TV advertising, expected sales are {model.intercept_:.1f} thousand units")
    
    # Visualize
    visualize_simple_regression(X_train, y_train, X_test, y_test, model)
    
    return model, (X_train, X_test, y_train, y_test)

def visualize_simple_regression(X_train, y_train, X_test, y_test, model):
    """Helper function for visualization"""
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Training data with regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.5, label='Training data', color='blue')
    
    # Plot regression line
    x_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression line')
    
    plt.xlabel('TV Advertising ($1000s)')
    plt.ylabel('Sales (thousands of units)')
    plt.title('Simple Linear Regression: TV vs Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual
    plt.subplot(1, 2, 2)
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.5, color='green')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Predictions vs Actual (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# PART 2.4: MULTIPLE LINEAR REGRESSION
# =============================================================================

def multiple_linear_regression_demo(df):
    """
    Demonstrate multiple linear regression with all features
    """
    print("\n" + "="*70)
    print("PART 2.4: MULTIPLE LINEAR REGRESSION")
    print("="*70)
    
    print("""
📌 WHAT IS MULTIPLE LINEAR REGRESSION?
   • Models relationship between MULTIPLE features and target
   • Equation: y = b0 + b1*x1 + b2*x2 + b3*x3 + ...
   • Each coefficient shows the effect of that feature
   • More realistic than simple regression (real problems have many factors)
    """)
    
    # Use all features
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    print(f"📊 Features: TV, Radio, Newspaper advertising budgets")
    print(f"🎯 Target: Sales (thousands of units)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n⚖️ FEATURE SCALING APPLIED:")
    print(f"   StandardScaler transforms each feature to have mean=0, std=1")
    print(f"   This allows fair comparison of coefficients")
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    print(f"\n🧠 MODEL TRAINED!")
    print(f"\n📊 COEFFICIENTS (Feature Importance):")
    print(f"   (After scaling: coefficients show effect of 1 std dev increase)")
    for feature, coef in zip(X.columns, model.coef_):
        direction = "increases" if coef > 0 else "decreases"
        print(f"  • {feature:10} : {coef:8.3f} - {direction} sales")
    print(f"  • Intercept  : {model.intercept_:8.3f} (baseline sales)")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 MODEL EVALUATION ON TEST SET:")
    print(f"  • Mean Absolute Error (MAE):  {mae:.2f}")
    print(f"  • Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  • R-squared (R²):             {r2:.4f}")
    print(f"    {interpret_r2(r2)}")
    
    # Compare with simple regression
    print(f"\n🔍 COMPARISON WITH SIMPLE REGRESSION:")
    print(f"   Multiple regression typically performs better by considering")
    print(f"   all relevant factors that affect sales.")
    print(f"   For example, Radio advertising might be more effective than TV,")
    print(f"   which we couldn't see in the simple model.")
    
    return model, scaler, (X_train_scaled, X_test_scaled, y_train, y_test)

# =============================================================================
# PART 2.5: POLYNOMIAL REGRESSION
# =============================================================================

def polynomial_regression_demo(df):
    """
    Demonstrate polynomial regression for non-linear relationships
    """
    print("\n" + "="*70)
    print("PART 2.5: POLYNOMIAL REGRESSION")
    print("="*70)
    
    print("""
📌 WHAT IS POLYNOMIAL REGRESSION?
   • Captures CURVED relationships, not just straight lines
   • Adds terms like x², x³ to the equation
   • Example: y = b0 + b1*x + b2*x² (parabola)
   
🍳 CHEF ANALOGY:
   Linear: "More salt = saltier, always at same rate"
   Polynomial: "A little salt enhances flavor, too much ruins it"
   
⚠️ DANGER: Can OVERFIT - memorize noise instead of learning patterns
    """)
    
    # Use TV advertising with polynomial features
    X = df[['TV']]
    y = df['Sales']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n📊 CREATING POLYNOMIAL FEATURES (degree 2)...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    print(f"  • Original features: {X_train.shape[1]}")
    print(f"  • Polynomial features: {X_train_poly.shape[1]}")
    print(f"  • New features: {poly.get_feature_names_out(['TV'])}")
    
    # Train polynomial regression
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    print(f"\n🧠 POLYNOMIAL MODEL TRAINED!")
    print(f"Coefficients:")
    for feature, coef in zip(poly.get_feature_names_out(['TV']), model.coef_):
        print(f"  • {feature:10} : {coef:8.3f}")
    print(f"  • Intercept  : {model.intercept_:8.3f}")
    
    # Make predictions
    y_pred = model.predict(X_test_poly)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 MODEL EVALUATION:")
    print(f"  • R-squared (R²): {r2:.4f}")
    print(f"    {interpret_r2(r2)}")
    
    # Compare with linear model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    r2_linear = r2_score(y_test, y_pred_linear)
    
    print(f"  • Linear R² for comparison: {r2_linear:.4f}")
    print(f"  • Improvement: {r2 - r2_linear:.4f}")
    
    # Visualize
    visualize_polynomial_regression(X_test, y_test, X_train, y_train, 
                                   model, linear_model, poly)
    
    print(f"\n💡 WHEN TO USE POLYNOMIAL REGRESSION:")
    print("  • When relationship between features and target is NON-LINEAR")
    print("  • When you suspect DIMINISHING RETURNS (e.g., advertising saturation)")
    print("  • But beware of overfitting with high degrees!")
    
    return model, poly

def visualize_polynomial_regression(X_test, y_test, X_train, y_train, 
                                   poly_model, linear_model, poly):
    """Helper function for polynomial regression visualization"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Polynomial fit
    plt.subplot(1, 2, 1)
    
    # Sort for smooth curve
    X_sorted = X_test.sort_values('TV')
    X_sorted_poly = poly.transform(X_sorted)
    y_pred_poly = poly_model.predict(X_sorted_poly)
    
    # Linear predictions
    y_pred_linear = linear_model.predict(X_sorted)
    
    plt.scatter(X_test, y_test, alpha=0.5, label='Test data', color='blue')
    plt.plot(X_sorted, y_pred_poly, 'r-', linewidth=2, label='Polynomial (degree 2)')
    plt.plot(X_sorted, y_pred_linear, 'g--', linewidth=2, label='Linear')
    
    plt.xlabel('TV Advertising ($1000s)')
    plt.ylabel('Sales (thousands of units)')
    plt.title('Polynomial vs Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training vs Test performance
    plt.subplot(1, 2, 2)
    
    # Predict on train
    X_train_sorted = X_train.sort_values('TV')
    X_train_sorted_poly = poly.transform(X_train_sorted)
    y_train_pred = poly_model.predict(X_train_sorted_poly)
    y_train_actual = y_train.loc[X_train_sorted.index]
    
    plt.scatter(X_train, y_train, alpha=0.3, label='Training data', color='gray')
    plt.plot(X_train_sorted, y_train_pred, 'b-', linewidth=2, label='Model on train')
    plt.plot(X_sorted, y_pred_poly, 'r-', linewidth=2, label='Model on test')
    
    plt.xlabel('TV Advertising ($1000s)')
    plt.ylabel('Sales (thousands of units)')
    plt.title('Check for Overfitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# PART 2.6: RIDGE AND LASSO REGRESSION (REGULARIZATION)
# =============================================================================

def regularization_explanation():
    """
    Simple explanation of Ridge and Lasso
    """
    print("\n" + "="*70)
    print("PART 2.6: RIDGE AND LASSO REGRESSION")
    print("="*70)
    
    print("""
📌 WHAT IS REGULARIZATION?
   • Adding a PENALTY to prevent overfitting
   • Keeps model SIMPLE and GENERALIZABLE
   
🍳 CHEF ANALOGY WITH DIET:
   • Linear regression: Chef uses any ingredients freely
   • Ridge regression: Chef can use ingredients, but moderately (portion control)
   • Lasso regression: Chef must choose only ESSENTIAL ingredients (removes unnecessary ones)

🔍 RIDGE REGRESSION (L2 regularization):
   • Penalizes LARGE coefficients
   • SHRINKS all coefficients but keeps them > 0
   • All features remain, but their influence is smoothed
   • Best when most features have SOME effect

🔍 LASSO REGRESSION (L1 regularization):
   • Can set coefficients to ZERO
   • Performs FEATURE SELECTION
   • Removes completely useless features
   • Best when many features are irrelevant

⚖️ TRADE-OFF:
   • Simple model (high penalty) → May underfit (too simple)
   • Complex model (low penalty) → May overfit (too complex)
   • Need to find the right balance
    """)

def regularization_demo(df):
    """
    Demonstrate Ridge and Lasso regression for regularization
    """
    print("\n" + "="*70)
    print("REGULARIZATION DEMONSTRATION")
    print("="*70)
    
    # Use all features
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # Add random noise features to demonstrate overfitting
    np.random.seed(42)
    for i in range(5):
        X[f'noise_{i}'] = np.random.randn(len(X))
    
    print(f"📊 FEATURES: {X.shape[1]} (including 5 random noise features)")
    print(f"   These noise features are completely random and should be ignored")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train different models
    models = {
        'Linear Regression (No Regularization)': LinearRegression(),
        'Ridge Regression (alpha=1)': Ridge(alpha=1.0),
        'Lasso Regression (alpha=1)': Lasso(alpha=1.0),
        'Ridge Regression (alpha=10)': Ridge(alpha=10.0, max_iter=10000),
        'Lasso Regression (alpha=10)': Lasso(alpha=10.0, max_iter=10000)
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Get coefficients
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            non_zero = np.sum(np.abs(coefs) > 0.01)
            zero_count = len(coefs) - non_zero
        else:
            coefs = []
            non_zero = 0
            zero_count = 0
        
        results.append({
            'Model': name,
            'R²': r2,
            'MSE': mse,
            'Non-zero Coef': non_zero,
            'Zero Coef': zero_count
        })
        
        if 'Lasso' in name:
            print(f"\n{name}:")
            print(f"  • Non-zero coefficients: {non_zero}/{len(coefs)}")
            print(f"  • Zero coefficients: {zero_count}/{len(coefs)} (features eliminated!)")
            for feature, coef in zip(X.columns, coefs):
                if abs(coef) > 0.01:
                    print(f"    ✓ {feature:15} : {coef:8.3f}")
                else:
                    print(f"    ✗ {feature:15} : {coef:8.3f} (removed)")
    
    # Create comparison dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R²', ascending=False)
    
    print(f"\n📊 MODEL COMPARISON:")
    print(results_df.to_string(index=False))
    
    # Visualize coefficients
    visualize_coefficients(X, models)
    
    print(f"\n💡 REGULARIZATION INSIGHTS:")
    print("  • Ridge regression: Shrinks coefficients but keeps all features")
    print("  • Lasso regression: Can set coefficients to ZERO (feature selection)")
    print("  • Higher alpha = stronger regularization = simpler model")
    print("  • Regularization helps prevent overfitting, especially with many features")
    
    return models, results_df

def visualize_coefficients(X, models):
    """Helper function for coefficient visualization"""
    plt.figure(figsize=(15, 5))
    
    selected_models = ['Linear Regression (No Regularization)', 
                       'Ridge Regression (alpha=1)', 
                       'Lasso Regression (alpha=1)']
    
    for i, name in enumerate(selected_models, 1):
        if name in models:
            model = models[name]
            
            plt.subplot(1, 3, i)
            
            if hasattr(model, 'coef_'):
                coefs = model.coef_
                
                # Color code: red = noise (should be zero), blue = real features
                colors = ['red' if 'noise' in col else 'blue' for col in X.columns]
                bars = plt.bar(range(len(coefs)), coefs, color=colors, alpha=0.7)
                
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.xlabel('Features')
                plt.ylabel('Coefficient Value')
                plt.title(f'{name[:20]}...')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Real features'),
                                  Patch(facecolor='red', alpha=0.7, label='Noise (should be 0)')]
                plt.legend(handles=legend_elements, loc='upper right')
                
                # Rotate x labels
                plt.xticks(range(len(X.columns)), X.columns, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# PART 2.7: R² COEFFICIENT EXPLANATION
# =============================================================================

def interpret_r2(r2_score):
    """
    Provide interpretation of R² value
    """
    if r2_score >= 0.9:
        return "EXCELLENT: Model explains 90%+ of variance in data"
    elif r2_score >= 0.7:
        return "GOOD: Model explains 70-90% of variance"
    elif r2_score >= 0.5:
        return "MODERATE: Model explains 50-70% of variance"
    elif r2_score >= 0.3:
        return "WEAK: Model explains only 30-50% of variance"
    elif r2_score >= 0:
        return "VERY WEAK: Model barely better than guessing mean"
    else:
        return "WORSE THAN GUESSING: Model performs worse than simple mean"

def r2_explanation():
    """
    Simple explanation of R² coefficient
    """
    print("\n" + "="*70)
    print("PART 2.7: UNDERSTANDING R² COEFFICIENT")
    print("="*70)
    
    print("""
📊 WHAT IS R² (R-SQUARED)?

It answers: "How well does our model explain the data?"

🍳 CHEF ANALOGY:
   Imagine people in a room have different heights.
   
   • Without any model: We'd just guess the AVERAGE height (170 cm)
   • With our model: We predict height based on shoe size
   
   R² shows how much BETTER our model is than just guessing the average.

📈 INTERPRETATION:
   • R² = 1.0: PERFECT! Model explains ALL differences (never happens in reality)
   • R² = 0.8: Model explains 80% of why people have different heights
              20% is due to other factors (age, genetics)
   • R² = 0.0: Model is NO BETTER than just guessing the average
   • R² < 0.0: Model is WORSE than guessing average (very bad!)

🎯 TARGET VALUES:
   • R² > 0.7: Good model (explains most variance)
   • R² 0.5-0.7: Moderate model
   • R² 0.3-0.5: Weak but maybe useful
   • R² < 0.3: Very weak, need better features
    """)

# =============================================================================
# PART 2.8: CROSS-VALIDATION
# =============================================================================

def cross_validation_explanation():
    """
    Simple explanation of cross-validation
    """
    print("\n" + "="*70)
    print("PART 2.8: CROSS-VALIDATION")
    print("="*70)
    
    print("""
📌 WHAT IS CROSS-VALIDATION?

A method to HONESTLY evaluate your model and detect overfitting.

🍳 CHEF ANALOGY:
   You're cooking for a restaurant. You can't keep asking the same person 
   to taste your dish - they'll get used to it! You need NEW, independent 
   tasters each time.

🔍 HOW IT WORKS (K-FOLD method with K=5):

   Step 1: Split all data into 5 equal boxes
   
   Round 1: Train on boxes 1-4, Test on box 5
   Round 2: Train on boxes 1-3,5, Test on box 4
   Round 3: Train on boxes 1-2,4-5, Test on box 3
   Round 4: Train on boxes 1,3-5, Test on box 2
   Round 5: Train on boxes 2-5, Test on box 1
   
   Final: Average all 5 test results

✅ BENEFITS:
   • More reliable than single train/test split
   • Uses ALL data for both training AND testing
   • Helps detect overfitting
   • Better for model selection and tuning

⚠️ WARNING SIGNS:
   If model performs GREAT on training but POOR on cross-validation → OVERFITTING!
    """)

def cross_validation_demo(df):
    """
    Demonstrate cross-validation for more reliable evaluation
    """
    print("\n" + "="*70)
    print("CROSS-VALIDATION DEMONSTRATION")
    print("="*70)
    
    # Use all features
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    print(f"📊 DATASET: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (alpha=1)': Ridge(alpha=1.0),
        'Ridge (alpha=10)': Ridge(alpha=10.0),
        'Lasso (alpha=1)': Lasso(alpha=1.0, max_iter=10000),
        'Lasso (alpha=10)': Lasso(alpha=10.0, max_iter=10000)
    }
    
    cv_results = []
    
    print("\n📊 5-FOLD CROSS-VALIDATION RESULTS:")
    print("-" * 60)
    
    for name, model in models.items():
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, 
                                   cv=5, scoring='r2')
        
        cv_results.append({
            'Model': name,
            'Mean R²': cv_scores.mean(),
            'Std R²': cv_scores.std(),
            'Min R²': cv_scores.min(),
            'Max R²': cv_scores.max()
        })
        
        print(f"\n{name}:")
        print(f"  • CV R² scores: {cv_scores.round(4)}")
        print(f"  • Mean R²: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        print(f"  • Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
    
    # Create comparison dataframe
    cv_df = pd.DataFrame(cv_results)
    cv_df = cv_df.sort_values('Mean R²', ascending=False)
    
    print(f"\n📊 CROSS-VALIDATION SUMMARY:")
    print(cv_df.to_string(index=False))
    
    # Visualize CV results
    visualize_cv_results(models, X_scaled, y)
    
    print(f"\n💡 CROSS-VALIDATION BENEFITS:")
    print("  • More reliable performance estimate than single train/test split")
    print("  • Uses all data for both training and validation")
    print("  • Helps detect overfitting")
    print("  • Better for model selection and hyperparameter tuning")
    
    return cv_df

def visualize_cv_results(models, X_scaled, y):
    """Helper function for CV visualization"""
    plt.figure(figsize=(10, 6))
    
    # Prepare data for boxplot
    boxplot_data = []
    labels = []
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        boxplot_data.append(cv_scores)
        labels.append(name)
    
    plt.boxplot(boxplot_data, labels=labels)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='R²=0 (random guessing)')
    plt.ylabel('R² Score')
    plt.title('5-Fold Cross-Validation Results Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# PART 2.9: BUSINESS APPLICATION - BUDGET OPTIMIZATION
# =============================================================================

def business_application(df):
    """
    Business application: Budget allocation optimization
    """
    print("\n" + "="*70)
    print("PART 2.9: BUSINESS APPLICATION - ADVERTISING BUDGET OPTIMIZATION")
    print("="*70)
    
    print("""
🎯 BUSINESS SCENARIO:
   You have $200,000 total advertising budget to allocate
   across TV, Radio, and Newspaper channels.
   
   GOAL: Maximize sales with limited budget
   CHALLENGE: Different channels have different effectiveness
    """)
    
    # Train final model
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    print("\n📊 MODEL COEFFICIENTS (per standard deviation increase):")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  • {feature:10} : {coef:8.1f} thousand units increase in sales")
    
    # Business question 1: Fixed budget allocation
    print("\n" + "-" * 60)
    print("QUESTION 1: Fixed Budget Allocation")
    print("If we allocate $150K to TV, $30K to Radio, and $20K to Newspaper,")
    print("what sales can we expect?")
    
    budget_scenario = np.array([[150, 30, 20]])  # In $1000s
    budget_scaled = scaler.transform(budget_scenario)
    predicted_sales = model.predict(budget_scaled)
    
    print(f"💰 Predicted sales: {predicted_sales[0]:.1f} thousand units")
    
    # Business question 2: ROI per channel
    print("\n" + "-" * 60)
    print("QUESTION 2: Which channel gives the best return on investment?")
    
    # Calculate ROI per channel
    print("\n📈 RETURN ON INVESTMENT ANALYSIS:")
    print("(Based on one standard deviation increase in spending)")
    
    # Get one standard deviation in original units
    std_devs = df[['TV', 'Radio', 'Newspaper']].std()
    
    rois = []
    for feature in X.columns:
        # Cost of one std dev increase (in $1000s)
        cost_per_std = std_devs[feature]
        
        # Get scaled coefficient
        idx = list(X.columns).index(feature)
        effect_per_std = model.coef_[idx]  # Sales increase per std dev
        
        # ROI (sales increase per $1000 spent)
        roi = effect_per_std / cost_per_std
        
        rois.append(roi)
        print(f"  • {feature:10} : {roi:.3f} additional units per $1000")
    
    # Business question 3: Optimization
    print("\n" + "-" * 60)
    print("QUESTION 3: Optimal Budget Allocation")
    print("How should we allocate $200K for maximum sales?")
    
    # Test different allocations
    print("\n🔍 Testing different allocations...")
    
    allocations = [
        (100, 70, 30),   # Balanced with TV focus
        (70, 100, 30),   # Radio focus
        (80, 80, 40),    # Balanced
        (120, 60, 20),   # Heavy TV focus
        (60, 100, 40),   # Heavy Radio focus
        (50, 120, 30),   # Very heavy Radio
        (90, 90, 20),    # Balanced low newspaper
        (110, 70, 20),   # TV + Radio
    ]
    
    print("\nAllocation Results ($ in thousands):")
    print("TV     Radio   Newspaper   Predicted Sales")
    print("-" * 50)
    
    best_sales = 0
    best_allocation = None
    
    for tv, radio, newspaper in allocations:
        if tv + radio + newspaper == 200:  # Total budget constraint
            allocation = np.array([[tv, radio, newspaper]])
            allocation_scaled = scaler.transform(allocation)
            sales = model.predict(allocation_scaled)[0]
            
            # Highlight best
            if sales > best_sales:
                best_sales = sales
                best_allocation = (tv, radio, newspaper)
                marker = " ← BEST"
            else:
                marker = ""
            
            print(f"{tv:6} {radio:7} {newspaper:10} {sales:12.1f}{marker}")
    
    print(f"\n✅ RECOMMENDED OPTIMAL ALLOCATION:")
    print(f"  • TV:        ${best_allocation[0]}K")
    print(f"  • Radio:     ${best_allocation[1]}K")
    print(f"  • Newspaper: ${best_allocation[2]}K")
    print(f"  • Expected sales: {best_sales:.1f} thousand units")
    
    # Calculate improvement over equal allocation
    equal_allocation = np.array([[200/3, 200/3, 200/3]])
    equal_scaled = scaler.transform(equal_allocation)
    equal_sales = model.predict(equal_scaled)[0]
    
    improvement = ((best_sales - equal_sales) / equal_sales) * 100
    
    print(f"\n📊 COMPARED TO EQUAL ALLOCATION ($66.7K each):")
    print(f"  • Equal allocation sales: {equal_sales:.1f} thousand units")
    print(f"  • Improvement: +{improvement:.1f}%")
    print(f"  • Additional revenue: ${(best_sales - equal_sales)*1000:,.0f} units")
    
    return model, scaler

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_demos():
    """
    Run all regression demos
    """
    print("="*70)
    print("🎯 SECTION 2: REGRESSION MODELS - COMPLETE GUIDE")
    print("="*70)
    print("""
This comprehensive guide covers:
   • Simple Linear Regression (one feature)
   • Multiple Linear Regression (many features)
   • Polynomial Regression (curved relationships)
   • Ridge & Lasso Regularization (prevent overfitting)
   • R² Coefficient Interpretation
   • Cross-Validation (honest evaluation)
   • Business Application (budget optimization)
    """)
    
    # Load data
    df = load_advertising_data()
    
    # Explanations
    regression_basics_explanation()
    
    # Demos
    print("\n" + "="*70)
    print("STARTING REGRESSION DEMONSTRATIONS...")
    print("="*70)
    
    print("\n" + "🔷"*35)
    simple_model, simple_data = simple_linear_regression_demo(df)
    
    print("\n" + "🔷"*35)
    multi_model, multi_scaler, multi_data = multiple_linear_regression_demo(df)
    
    print("\n" + "🔷"*35)
    polynomial_regression_demo(df)
    
    print("\n" + "🔷"*35)
    regularization_explanation()
    reg_models, reg_results = regularization_demo(df)
    
    print("\n" + "🔷"*35)
    r2_explanation()
    
    print("\n" + "🔷"*35)
    cross_validation_explanation()
    cv_results = cross_validation_demo(df)
    
    print("\n" + "🔷"*35)
    business_model, business_scaler = business_application(df)
    
    print("\n" + "="*70)
    print("✅ ALL REGRESSION DEMOS COMPLETED!")
    print("="*70)
    
    print("\n📚 KEY TAKEAWAYS:")
    print("1. Start with simple models and add complexity as needed")
    print("2. Always evaluate models properly (train/test split, CV)")
    print("3. Consider regularization to prevent overfitting")
    print("4. Interpret results in business context (ROI, allocation)")
    print("5. Polynomial features can capture non-linear relationships")
    print("6. R² tells you how much variance your model explains")
    print("7. Cross-validation gives honest performance estimate")
    
    return {
        'simple_model': simple_model,
        'multi_model': multi_model,
        'reg_models': reg_models,
        'business_model': business_model,
        'data': df
    }

if __name__ == "__main__":
    results = run_all_demos()