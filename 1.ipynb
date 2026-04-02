"""
SECTION 1: INTRODUCTION TO MACHINE LEARNING & DATA PREPROCESSING
==================================================================
Complete guide covering ML basics, data exploration, cleaning, encoding,
outlier detection, and preprocessing pipelines.

Author: ML Course
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PART 0: INTRODUCTION TO MACHINE LEARNING
# =============================================================================

def ml_introduction_demo():
    """
    First ML model demonstration: Predicting sales from advertising spend
    Shows fundamental ML concepts: training, prediction, evaluation
    """
    print("\n" + "="*70)
    print("PART 0: INTRODUCTION TO MACHINE LEARNING")
    print("="*70)
    
    # Simple linear relationship: y = 1x + 50
    X = np.array([100, 200, 300, 400, 500]).reshape(-1, 1)
    y = np.array([150, 250, 350, 450, 550])
    
    print("\n📊 TRAINING DATA (Advertising → Sales):")
    print(f"   Ad Spend ($1000s): {X.flatten()}")
    print(f"   Sales (units):     {y}")
    
    # Train model
    print("\n🤖 TRAINING SIMPLE LINEAR MODEL...")
    model = LinearRegression()
    model.fit(X, y)
    
    # Show learned parameters
    print(f"\n🧠 MODEL LEARNED:")
    print(f"   Coefficient (slope): {model.coef_[0]:.2f}")
    print(f"   Intercept:           {model.intercept_:.2f}")
    print(f"   Formula: sales = {model.coef_[0]:.1f} × ad_spend + {model.intercept_:.1f}")
    
    # Make predictions
    print("\n🔮 PREDICTIONS ON NEW DATA:")
    test_values = np.array([[250], [350], [450], [600]])
    predictions = model.predict(test_values)
    
    for spend, pred in zip(test_values.flatten(), predictions):
        print(f"   ${spend}K ad spend → {pred:.1f} units predicted")
    
    # Evaluate
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n📈 MODEL EVALUATION:")
    print(f"   Mean Absolute Error: {mae:.2f} (average prediction error)")
    print(f"   R-squared Score:     {r2:.4f} (1.0 = perfect prediction)")
    
    return model

def ml_types_overview():
    """
    Overview of different ML types with business examples
    """
    print("\n" + "="*70)
    print("TYPES OF MACHINE LEARNING")
    print("="*70)
    
    ml_types = {
        "SUPERVISED LEARNING": {
            "description": "Learn from labeled data (input → output pairs)",
            "regression": "Predict continuous values (price, temperature, sales)",
            "classification": "Predict categories (churn: yes/no, spam/not spam)"
        },
        "UNSUPERVISED LEARNING": {
            "description": "Find patterns in unlabeled data",
            "clustering": "Group similar customers (market segmentation)",
            "dimensionality_reduction": "Reduce features while preserving information"
        },
        "REINFORCEMENT LEARNING": {
            "description": "Learn through trial and error (rewards/punishments)",
            "game_ai": "AlphaGo, chess engines",
            "optimization": "Dynamic pricing, inventory management"
        }
    }
    
    for ml_type, details in ml_types.items():
        print(f"\n🔹 {ml_type}:")
        print(f"   {details['description']}")
        for key, value in details.items():
            if key != 'description':
                print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    return ml_types

# =============================================================================
# PART 1.1: DATA EXPLORATION & BASIC PREPROCESSING
# =============================================================================

def exercise_basic_exploration():
    """
    Exercise 1: Basic Data Exploration
    Learn to understand dataset structure and characteristics
    """
    print("\n" + "="*70)
    print("EXERCISE 1: BASIC DATA EXPLORATION")
    print("="*70)
    
    # Create sample customer dataset
    data = {
        'customer_id': range(1, 11),
        'age': [25, 30, 35, 28, 40, 22, 45, 33, 29, 31],
        'income': [50000, 60000, 75000, 55000, 80000, 45000, 90000, 65000, 58000, 72000],
        'city': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles',
                'New York', 'Chicago', 'Los Angeles', 'New York', 'Chicago'],
        'purchase_category': ['Electronics', 'Clothing', 'Electronics', 'Home',
                             'Clothing', 'Books', 'Electronics', 'Home', 'Books', 'Clothing'],
        'last_visit_days': [5, 2, 15, 7, 3, 30, 1, 10, 20, 5],
        'total_spent': [1200.50, 450.00, 890.75, 2100.00, 320.25,
                       150.00, 1780.00, 920.50, 420.00, 280.75],
        'churn': [0, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    }
    
    df = pd.DataFrame(data)
    print("\n📋 DATASET OVERVIEW:")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n🔍 DATA TYPES:")
    for col in df.columns:
        dtype = df[col].dtype
        unique = df[col].nunique() if df[col].dtype == 'object' else 'N/A'
        print(f"   {col:20} | {str(dtype):12} | Unique: {unique}")
    
    print("\n📊 FIRST 5 ROWS:")
    print(df.head())
    
    print("\n📈 BASIC STATISTICS:")
    print(df.describe())
    
    print("\n✅ MISSING VALUES CHECK:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "   No missing values found!")
    
    return df

def exercise_missing_values():
    """
    Exercise 2: Handling Missing Values
    Different strategies for different data types
    """
    print("\n" + "="*70)
    print("EXERCISE 2: HANDLING MISSING VALUES")
    print("="*70)
    
    # Create data with missing values
    data = {
        'customer_id': range(1, 11),
        'age': [25, None, 35, 28, 40, 22, 45, None, 29, 31],
        'income': [50000, 60000, None, 55000, 80000, 45000, 90000, 65000, None, 72000],
        'city': ['New York', 'Los Angeles', 'New York', None, 'Los Angeles',
                'New York', 'Chicago', 'Los Angeles', 'New York', None],
        'purchase_amount': [1200.50, 450.00, 890.75, 2100.00, 320.25,
                           150.00, 1780.00, 920.50, 420.00, 280.75]
    }
    
    df = pd.DataFrame(data)
    
    print("\n📉 BEFORE HANDLING MISSING VALUES:")
    print(df.isnull().sum())
    
    # Strategy 1: Median for numeric (robust to outliers)
    age_median = df['age'].median()
    df['age'] = df['age'].fillna(age_median)
    print(f"\n📌 Strategy 1 - Median Imputation:")
    print(f"   Filled 'age' with median: {age_median}")
    
    # Strategy 2: Mean for numeric (sensitive to outliers)
    income_mean = df['income'].mean()
    df['income'] = df['income'].fillna(income_mean)
    print(f"\n📌 Strategy 2 - Mean Imputation:")
    print(f"   Filled 'income' with mean: {income_mean:.2f}")
    
    # Strategy 3: Mode for categorical
    city_mode = df['city'].mode()[0]
    df['city'] = df['city'].fillna(city_mode)
    print(f"\n📌 Strategy 3 - Mode Imputation:")
    print(f"   Filled 'city' with mode: '{city_mode}'")
    
    print(f"\n✅ AFTER HANDLING: {df.isnull().sum().sum()} missing values remain")
    
    return df

def exercise_categorical_encoding():
    """
    Exercise 3: Encoding Categorical Features
    Convert text data to numbers for ML algorithms
    """
    print("\n" + "="*70)
    print("EXERCISE 3: CATEGORICAL ENCODING")
    print("="*70)
    
    data = {
        'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'category': ['Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing'],
        'size': ['M', 'L', 'S', 'XL', 'M'],  # Ordinal (has order)
        'price_category': ['Low', 'Medium', 'High', 'Medium', 'Low']  # Ordinal
    }
    
    df = pd.DataFrame(data)
    print("📋 ORIGINAL DATA:")
    print(df)
    
    # Method 1: One-Hot Encoding (for nominal - no order)
    print("\n🔷 METHOD 1: ONE-HOT ENCODING (Nominal Features)")
    df_onehot = pd.get_dummies(df, columns=['category'], prefix='cat')
    print(df_onehot[['product_id', 'cat_Clothing', 'cat_Electronics', 'cat_Home']])
    
    # Method 2: Label Encoding (for ordinal - has order)
    print("\n🔶 METHOD 2: LABEL ENCODING (Ordinal Features)")
    
    # Size mapping (S < M < L < XL)
    size_order = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
    df['size_encoded'] = df['size'].map(size_order)
    print(f"   Size mapping: {size_order}")
    
    # Price mapping (Low < Medium < High)
    price_order = {'Low': 0, 'Medium': 1, 'High': 2}
    df['price_encoded'] = df['price_category'].map(price_order)
    print(f"   Price mapping: {price_order}")
    
    print("\n📊 ENCODED DATA:")
    print(df[['product_id', 'size', 'size_encoded', 'price_category', 'price_encoded']])
    
    return df

def exercise_outlier_detection():
    """
    Exercise 4: Outlier Detection and Treatment
    Identify and handle unusual values
    """
    print("\n" + "="*70)
    print("EXERCISE 4: OUTLIER DETECTION")
    print("="*70)
    
    # Create data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(1000, 200, 96)  # 96 normal values
    outliers = np.array([50, 100, 5000, 6000])     # 4 outliers
    purchase_amounts = np.concatenate([normal_data, outliers])
    
    df = pd.DataFrame({
        'customer_id': range(1, 101),
        'purchase_amount': purchase_amounts
    })
    
    print("\n📊 INITIAL STATISTICS:")
    print(f"   Mean: ${df['purchase_amount'].mean():.2f}")
    print(f"   Median: ${df['purchase_amount'].median():.2f}")
    print(f"   Min: ${df['purchase_amount'].min():.2f}")
    print(f"   Max: ${df['purchase_amount'].max():.2f}")
    
    # IQR Method
    Q1 = df['purchase_amount'].quantile(0.25)
    Q3 = df['purchase_amount'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\n📐 IQR CALCULATIONS:")
    print(f"   Q1 (25th percentile): ${Q1:.2f}")
    print(f"   Q3 (75th percentile): ${Q3:.2f}")
    print(f"   IQR: ${IQR:.2f}")
    print(f"   Lower bound: ${lower_bound:.2f}")
    print(f"   Upper bound: ${upper_bound:.2f}")
    
    outliers = df[(df['purchase_amount'] < lower_bound) | (df['purchase_amount'] > upper_bound)]
    print(f"\n⚠️ DETECTED OUTLIERS: {len(outliers)} customers")
    if len(outliers) > 0:
        print(outliers[['customer_id', 'purchase_amount']])
    
    # Capping (Winsorization)
    df['purchase_amount_capped'] = df['purchase_amount'].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"\n📊 AFTER CAPPING OUTLIERS:")
    print(f"   Original mean: ${df['purchase_amount'].mean():.2f}")
    print(f"   Capped mean:   ${df['purchase_amount_capped'].mean():.2f}")
    print(f"   Max value capped to: ${df['purchase_amount_capped'].max():.2f}")
    
    return df

def complete_preprocessing_pipeline():
    """
    Exercise 5: Complete Preprocessing Pipeline
    End-to-end data preparation for ML
    """
    print("\n" + "="*70)
    print("EXERCISE 5: COMPLETE PREPROCESSING PIPELINE")
    print("="*70)
    
    # Create comprehensive dataset
    data = {
        'customer_id': range(1, 21),
        'age': [25, None, 35, 28, 40, 22, 45, None, 29, 31,
               32, 27, 38, 41, 26, 33, 36, None, 44, 30],
        'income': [50000, 60000, None, 55000, 80000, 45000, 90000, 65000, None, 72000,
                  58000, 52000, 75000, 82000, 48000, 67000, 71000, 63000, 88000, None],
        'city': ['New York', 'Los Angeles', 'New York', None, 'Los Angeles',
                'New York', 'Chicago', 'Los Angeles', 'New York', 'Chicago',
                'New York', 'Los Angeles', 'Chicago', 'New York', 'Los Angeles',
                'Chicago', 'New York', 'Los Angeles', 'Chicago', None],
        'purchase_category': ['Electronics', 'Clothing', 'Electronics', 'Home',
                             'Clothing', 'Books', 'Electronics', 'Home', 'Books', 'Clothing',
                             'Electronics', 'Clothing', 'Home', 'Books', 'Electronics',
                             'Clothing', 'Home', 'Books', 'Electronics', 'Clothing'],
        'last_visit': [5, 2, 15, 7, 3, 30, 1, 10, 20, 5,
                      8, 12, 4, 25, 6, 14, 9, 18, 2, 11],
        'total_spent': [1200.50, 450.00, 890.75, 2100.00, 320.25,
                       150.00, 1780.00, 920.50, 420.00, 280.75,
                       1350.00, 600.00, 1950.00, 380.00, 1650.00,
                       550.00, 2050.00, 480.00, 1750.00, 620.00],
        'churn': [0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
                 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    print("\n📋 INITIAL DATASET:")
    print(f"   Shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    # STEP 1: Handle Missing Values
    print("\n🔧 STEP 1: HANDLING MISSING VALUES")
    
    # Numeric: median
    df['age'] = df['age'].fillna(df['age'].median())
    df['income'] = df['income'].fillna(df['income'].mean())
    
    # Categorical: mode
    df['city'] = df['city'].fillna(df['city'].mode()[0])
    
    print(f"   Missing values after: {df.isnull().sum().sum()}")
    
    # STEP 2: Feature Engineering
    print("\n➕ STEP 2: FEATURE ENGINEERING")
    
    # Create new features
    df['spent_per_visit'] = df['total_spent'] / (df['last_visit'] + 1)
    df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50], 
                              labels=['20-30', '30-40', '40-50'])
    
    print(f"   Created 'spent_per_visit' and 'age_group'")
    
    # STEP 3: Encode Categorical Features
    print("\n📝 STEP 3: ENCODING CATEGORICAL FEATURES")
    
    # One-hot encoding for nominal
    df = pd.get_dummies(df, columns=['city'], prefix='city', drop_first=True)
    df = pd.get_dummies(df, columns=['purchase_category'], prefix='cat', drop_first=True)
    
    # Label encoding for ordinal
    age_group_map = {'20-30': 0, '30-40': 1, '40-50': 2}
    df['age_group_encoded'] = df['age_group'].map(age_group_map)
    df = df.drop(columns=['age_group'])
    
    print(f"   New shape: {df.shape}")
    
    # STEP 4: Detect Outliers (simplified)
    print("\n📊 STEP 4: QUICK OUTLIER CHECK")
    for col in ['total_spent', 'spent_per_visit']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        if len(outliers) > 0:
            print(f"   {col}: {len(outliers)} outliers detected")
    
    # STEP 5: Prepare for ML
    print("\n🤖 STEP 5: PREPARE FOR MACHINE LEARNING")
    
    # Separate features and target
    X = df.drop(columns=['customer_id', 'churn'])
    y = df['churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n   FINAL DATASETS:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_test:  {y_test.shape}")
    
    print(f"\n   Class distribution in train:")
    print(f"   {y_train.value_counts(normalize=True).to_dict()}")
    
    return df, X_train, X_test, y_train, y_test

# =============================================================================
# PART 1.2: ADVANCED PREPROCESSING UTILITIES
# =============================================================================

def automatic_preprocessing_pipeline(df, target_column, test_size=0.2):
    """
    Automated preprocessing pipeline that handles:
    - Missing values (numeric: median, categorical: mode)
    - Encoding (one-hot for low cardinality, label for high)
    - Train-test split with stratification
    - Feature scaling (optional)
    """
    print("\n" + "="*70)
    print("AUTOMATIC PREPROCESSING PIPELINE")
    print("="*70)
    
    df_clean = df.copy()
    
    # Identify column types
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    print(f"\n📊 COLUMN TYPES:")
    print(f"   Numeric: {len(numeric_cols)} columns")
    print(f"   Categorical: {len(categorical_cols)} columns")
    
    # Step 1: Handle missing values
    print("\n🔧 STEP 1: HANDLING MISSING VALUES")
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            print(f"   {col}: filled with median")
    
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            print(f"   {col}: filled with mode")
    
    # Step 2: Encode categorical features
    print("\n📝 STEP 2: ENCODING CATEGORICAL FEATURES")
    for col in categorical_cols:
        unique_count = df_clean[col].nunique()
        
        if unique_count <= 5:  # Few categories → One-hot
            dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
            df_clean = pd.concat([df_clean, dummies], axis=1)
            df_clean = df_clean.drop(columns=[col])
            print(f"   {col}: one-hot encoding ({unique_count} categories)")
        
        else:  # Many categories → Label encoding
            le = LabelEncoder()
            df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col])
            df_clean = df_clean.drop(columns=[col])
            print(f"   {col}: label encoding ({unique_count} categories)")
    
    # Step 3: Prepare for ML
    print("\n🤖 STEP 3: TRAIN-TEST SPLIT")
    
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n   FINAL SHAPES:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def environment_check():
    """
    Check if required packages are installed
    """
    print("\n" + "="*70)
    print("ENVIRONMENT CHECK")
    print("="*70)
    
    required = {
        'numpy': 'Numerical computing',
        'pandas': 'Data manipulation',
        'sklearn': 'Machine learning',
        'matplotlib': 'Visualization',
        'seaborn': 'Statistical plots'
    }
    
    print("\nChecking installed packages...")
    for package, description in required.items():
        try:
            __import__(package)
            print(f"   ✅ {package:12} - {description}")
        except ImportError:
            print(f"   ❌ {package:12} - NOT INSTALLED")
            print(f"      Install: pip install {package}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SECTION 1: MACHINE LEARNING INTRODUCTION & DATA PREPROCESSING")
    print("="*70)
    print("\nThis script contains all code from the introductory ML sections.")
    print("Each function demonstrates key concepts:")
    print("   • Part 0: ML Basics")
    print("   • Part 1.1: Data Exploration & Basic Preprocessing")
    print("   • Part 1.2: Advanced Preprocessing Utilities")
    
    # Run all demonstrations
    ml_introduction_demo()
    ml_types_overview()
    exercise_basic_exploration()
    exercise_missing_values()
    exercise_categorical_encoding()
    exercise_outlier_detection()
    complete_preprocessing_pipeline()
    environment_check()
    
    print("\n" + "="*70)
    print("✅ SECTION 1 COMPLETE - READY FOR MODELING!")
    print("="*70)
    print("\nNext steps:")
    print("   1. Understand your data (exploration)")
    print("   2. Clean and preprocess (this section)")
    print("   3. Build models (Section 2+)")
    print("   4. Evaluate and iterate")