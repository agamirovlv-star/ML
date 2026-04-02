# Load data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_advertising_data():
    """
    Load and prepare advertising dataset for regression
    Dataset: Advertising spend vs Sales
    """
    print("=" * 70)
    print("ADVERTISING DATASET FOR REGRESSION")
    print("=" * 70)
    
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
    
    print(f"Dataset created: {df.shape[0]} samples × {df.shape[1]} features")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\n📊 STATISTICAL SUMMARY:")
    print(df.describe().round(2))
    
    print("\n🔗 CORRELATION WITH SALES:")
    correlations = df.corr()['Sales'].sort_values(ascending=False)
    for feature, corr in correlations.items():
        print(f"  {feature:10} : {corr:.3f}")

    # We expect TV and Radio to have high positive correlation with Sales.    
    return df

#========================================================================

df =load_advertising_data()   #pd.read_csv('advertising.csv')

# Let's start with one feature: TV budget
X = df[['TV']] # Note: Double brackets to keep it as a DataFrame (2D)
y = df['Sales'] # Single brackets for a Series (1D)
print(f"\nFeature shape: {X.shape}, Target shape: {y.shape}")



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# test_size=0.25 -> 25% of data for testing, 75% for training.
# random_state ensures we get the same split every time (for reproducibility).

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")


# Create the model object
model = LinearRegression()

# Train the model on the TRAINING SET
model.fit(X_train, y_train)

# The model has learned its parameters:
print(f"\nModel Intercept (w0): {model.intercept_:.2f}")
print(f"Model Coefficient for 'TV' (w1): {model.coef_[0]:.4f}")
# Interpretation: For each additional $1000 spent on TV ads, sales increase by `coef` units.

# Predict on the test set (the unseen data)
y_pred = model.predict(X_test)

# Let's look at a few real vs predicted values
print("\nComparison for first 5 test samples:")
comparison_df = pd.DataFrame({'Actual Sales': y_test.values[:5],
'Predicted Sales': y_pred[:5].round(2)})
print(comparison_df)


# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R?) Score: {r2:.4f}")

# Interpretation:
# MAE: On average, our prediction is off by +- MAE units. Easy to understand.
# MSE: Penalizes large errors more heavily (because of squaring). Harder to interpret directly.
# R?: The proportion of variance in the target explained by the model. Range: 0 (bad) to 1 (perfect). 0.85 means 85% of sales variability is explained by TV ads.


# Let's calculate R? for both sets
y_pred_train = model.predict(X_train)
r2_train = r2_score(y_train, y_pred_train)

print(f"\nR? on Training Set: {r2_train:.4f}")
print(f"R? on Test Set: {r2:.4f}")

# Interpretation:
# - If both R? are low and similar -> Underfitting.
# - If R?(train) is high, but R?(test) is much lower -> Overfitting.
# - If both are reasonably high and close -> Good fit.

# Step 1 & 2: Prepare data with all features
X_multi = df[['TV', 'Radio', 'Newspaper']]
y_multi = df['Sales']

# Step 3: Split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.25, random_state=42)

# Step 4: Train
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

# Step 5: Predict & Evaluate
y_pred_multi = model_multi.predict(X_test_m)

print("\n--- Multiple Regression Results ---")
print(f"MAE: {mean_absolute_error(y_test_m, y_pred_multi):.2f}")
print(f"R?: {r2_score(y_test_m, y_pred_multi):.4f}")

# Interpret coefficients
coef_dict = dict(zip(X_multi.columns, model_multi.coef_))
print("\nFeature Coefficients (Impact on Sales):")
for feature, coef in coef_dict.items():
    print(f" {feature}: {coef:.4f}")

# Negative coefficient for 'Newspaper'? Might be an interesting business insight or a statistical quirk!


# Make a prediction for a specific scenario
new_campaign = np.array([[200, 40, 10]]) # Must be 2D array
predicted_sales = model_multi.predict(new_campaign)
print(f"\nPredicted sales for [TV=40k, Newspaper=$10k]: {predicted_sales[0]:.1f} units")

# Analyze channel effectiveness via coefficients
print("\nAccording to the model, spending $1k more yields:")
for feature, coef in coef_dict.items():
    print(f" {feature}: {coef:.2f} additional units sold")