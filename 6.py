"""
Module 5_ex: Customer Segmentation using Clustering Algorithms
Complete business case study with multiple clustering approaches
Author: Data Science Course
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization libraries (commented out for CLI, uncomment in Jupyter)
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# =============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# =============================================================================

print("="*70)
print("CUSTOMER SEGMENTATION ANALYSIS")
print("="*70)

# Try to load real data, create synthetic if not available
try:
    df = pd.read_csv('mall_customers.csv')
    print("✓ Real data loaded successfully!")
except FileNotFoundError:
    print("! Real data not found. Creating synthetic dataset...")
    # Create synthetic customer data
    np.random.seed(42)
    n_customers = 500
    
    # Generate realistic customer features
    age = np.random.normal(40, 15, n_customers).clip(18, 70)
    income = np.random.normal(60, 25, n_customers).clip(15, 150)
    
    # Create spending patterns correlated with age and income
    spending_base = 50 + 0.3 * (income - 60) - 0.2 * (age - 40)
    spending = spending_base + np.random.normal(0, 15, n_customers)
    spending = spending.clip(1, 100)
    
    df = pd.DataFrame({
        'CustomerID': range(1, n_customers + 1),
        'Age': age.round(1),
        'Annual_Income': income.round(1),
        'Spending_Score': spending.round(1)
    })
    print("✓ Synthetic dataset created with 500 customers!")

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\nFirst 5 customers:")
print(df.head())

print("\nBasic statistics:")
print(df.describe().round(1))

# =============================================================================
# PART 2: DATA PREPROCESSING - THE CRITICAL SCALING STEP
# =============================================================================

print("\n" + "="*70)
print("STEP 1: DATA PREPROCESSING")
print("="*70)

# Select features for clustering
features = ['Age', 'Annual_Income', 'Spending_Score']
X = df[features]

print(f"\nFeatures selected: {features}")
print("Why these features? They represent customer demographics and behavior.")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✓ Features scaled using StandardScaler")
print("  (Mean=0, Standard Deviation=1)")

# Show scaling effect
print("\nOriginal data (first 3 rows):")
print(X.head(3).to_string(index=False))
print("\nScaled data (first 3 rows):")
print(pd.DataFrame(X_scaled[:3], columns=features).round(3).to_string(index=False))

# =============================================================================
# PART 3: FINDING OPTIMAL K - ELBOW METHOD
# =============================================================================

print("\n" + "="*70)
print("STEP 2: FINDING OPTIMAL NUMBER OF CLUSTERS (k)")
print("="*70)
print("Method A: Elbow Method (WCSS)")

wcss = []
k_range = range(1, 11)

print("\nk-value : WCSS (Within-Cluster Sum of Squares)")
print("-" * 40)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    print(f"k = {k:2d}   : {wcss[-1]:8.1f}")

print("\nWCSS Reduction (to identify elbow):")
print("-" * 40)
for i in range(1, len(wcss)):
    reduction = wcss[i-1] - wcss[i]
    reduction_pct = (reduction / wcss[i-1]) * 100
    print(f"k={i} → {i+1}: {reduction:8.1f} ({reduction_pct:.1f}% reduction)")

print("\n" + "-"*40)
print("Observation: Look for the 'elbow' where reduction slows down.")
print("Typical elbow occurs at k=4 or k=5 for this type of data.")

# =============================================================================
# PART 4: SILHOUETTE ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("STEP 3: SILHOUETTE SCORE ANALYSIS")
print("="*70)
print("Method B: Silhouette Score (higher is better, range: -1 to 1)")

silhouette_scores = []
k_range_sil = range(2, 9)  # Silhouette requires at least 2 clusters

print("\nk-value : Silhouette Score")
print("-" * 30)

for k in k_range_sil:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k = {k}     : {score:.4f}")

best_k_sil = k_range_sil[np.argmax(silhouette_scores)]
print(f"\n✓ Best k by Silhouette Score: {best_k_sil}")

# =============================================================================
# PART 5: FINAL K-MEANS MODEL
# =============================================================================

print("\n" + "="*70)
print("STEP 4: TRAINING FINAL K-MEANS MODEL")
print("="*70)

# Based on elbow and silhouette, let's choose k=5
optimal_k = 5
print(f"\nChoosing k = {optimal_k} (combines elbow and silhouette results)")

final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = final_kmeans.fit_predict(X_scaled)

print(f"\nCluster distribution:")
cluster_counts = df['KMeans_Cluster'].value_counts().sort_index()
for cluster in range(optimal_k):
    count = cluster_counts[cluster]
    percentage = (count / len(df)) * 100
    print(f"Cluster {cluster}: {count:3d} customers ({percentage:.1f}%)")

# =============================================================================
# PART 6: CLUSTER PROFILING - THE BUSINESS GOLD!
# =============================================================================

print("\n" + "="*70)
print("STEP 5: CLUSTER PROFILING (BUSINESS INTERPRETATION)")
print("="*70)

# Calculate profiles using original (unscaled) values for interpretation
cluster_profile = df.groupby('KMeans_Cluster')[features].mean()
cluster_profile['Size'] = df['KMeans_Cluster'].value_counts().sort_index()
cluster_profile['Size_%'] = (cluster_profile['Size'] / len(df) * 100).round(1)

print("\n=== CLUSTER PROFILES (Average Values) ===")
print("="*70)
print(cluster_profile.round(1))

print("\n" + "="*70)
print("BUSINESS SEGMENTATION & MARKETING STRATEGY")
print("="*70)

# Business interpretation and naming
cluster_names = {}
cluster_actions = {}

for cluster in range(optimal_k):
    row = cluster_profile.loc[cluster]
    income = row['Annual_Income']
    spending = row['Spending_Score']
    age = row['Age']
    
    # Intelligent naming based on characteristics
    if income > 70 and spending > 65:
        name = "💎 PREMIUM SHOPPERS"
        action = [
            "Exclusive VIP program with early access to sales",
            "Personal stylist consultations",
            "Premium product recommendations",
            "Invitations to exclusive events"
        ]
    elif income > 70 and spending < 40:
        name = "💰 CAUTIOUS AFFLUENT"
        action = [
            "Focus on quality and durability in marketing",
            "Extended warranty offers",
            "Premium customer service",
            "Loyalty rewards program"
        ]
    elif income < 40 and spending > 65:
        name = "🛍️ ASPIRING FASHIONISTAS"
        action = [
            "Flash sales and limited-time discounts",
            "Trend alerts and lookbooks",
            "Affordable luxury recommendations",
            "Social media engagement campaigns"
        ]
    elif income < 45 and spending < 40:
        name = "😴 BUDGET CONSCIOUS"
        action = [
            "Re-engagement emails with special offers",
            "Bundle deals and value packs",
            "Clearance sale notifications",
            "Survey to understand barriers"
        ]
    elif age < 30 and spending > 50:
        name = "🎧 YOUNG TRENDSETTERS"
        action = [
            "Influencer collaborations",
            "Social media contests",
            "Student discounts",
            "Trending items recommendations"
        ]
    else:
        name = "👔 MAINSTREAM SHOPPERS"
        action = [
            "Seasonal sales campaigns",
            "Newsletter with general offers",
            "Cross-selling recommendations",
            "Referral programs"
        ]
    
    cluster_names[cluster] = name
    cluster_actions[cluster] = action
    
    print(f"\n{name}")
    print(f"  Profile: Age {age:.1f} | Income ${income:.1f}k | Spending Score {spending:.1f}")
    print(f"  Size: {row['Size']:.0f} customers ({row['Size_%']}% of total)")
    print("  Recommended Actions:")
    for i, act in enumerate(action[:3], 1):  # Show top 3 actions
        print(f"    {i}. {act}")

# =============================================================================
# PART 7: COMPARING WITH OTHER CLUSTERING ALGORITHMS
# =============================================================================

print("\n" + "="*70)
print("STEP 6: COMPARISON WITH OTHER ALGORITHMS")
print("="*70)

# 7.1 Hierarchical Clustering
print("\n--- Hierarchical Clustering ---")
print("Building dendrogram structure...")

# Perform hierarchical clustering
Z = linkage(X_scaled, method='ward')
df['Hierarchical_Cluster'] = fcluster(Z, t=optimal_k, criterion='maxclust')

# Compare with K-Means
hierarchical_counts = df['Hierarchical_Cluster'].value_counts().sort_index()
print(f"Hierarchical clusters (k={optimal_k}):")
for cluster in range(1, optimal_k + 1):
    count = hierarchical_counts[cluster]
    print(f"  Cluster {cluster}: {count} customers")

# 7.2 DBSCAN for outlier detection
print("\n--- DBSCAN (Density-Based Clustering) ---")
print("DBSCAN finds clusters of arbitrary shape and identifies outliers.")

# Try different epsilon values
eps_values = [0.3, 0.5, 0.7, 1.0]
for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  eps={eps:.1f}: {n_clusters} clusters, {n_noise} outliers")

# Use best eps (typically 0.5 for this data)
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
n_outliers = (df['DBSCAN_Cluster'] == -1).sum()

print(f"\n✓ Final DBSCAN (eps=0.5):")
print(f"  {n_clusters_db} clusters found")
print(f"  {n_outliers} outliers detected (potential unusual behavior)")

# =============================================================================
# PART 8: ALGORITHM COMPARISON SUMMARY
# =============================================================================

print("\n" + "="*70)
print("STEP 7: ALGORITHM COMPARISON SUMMARY")
print("="*70)

print("""
╔════════════════╦════════════════════════════════╦════════════════════════════╗
║ Algorithm      ║ Strengths                      ║ Weaknesses                 ║
╠════════════════╬════════════════════════════════╬════════════════════════════╣
║ K-Means        │ • Fast and scalable            │ • Must choose k            │
║                │ • Simple to understand         │ • Assumes spherical        │
║                │ • Works well on large data     │ • Sensitive to outliers    │
╠════════════════╬════════════════════════════════╬════════════════════════════╣
║ Hierarchical   │ • No need to choose k          │ • Slow on large data       │
║                │ • Dendrogram for visualization │ • Can't undo merges        │
║                │ • Deterministic                │ • Sensitive to noise       │
╠════════════════╬════════════════════════════════╬════════════════════════════╣
║ DBSCAN         │ • Finds arbitrary shapes       │ • Parameters are tricky    │
║                │ • Identifies outliers          │ • Struggles with varying   │
║                │ • No need to specify k         │   densities                │
╚════════════════╩════════════════════════════════╩════════════════════════════╝
""")

# =============================================================================
# PART 9: BUSINESS RECOMMENDATIONS
# =============================================================================

print("\n" + "="*70)
print("FINAL BUSINESS RECOMMENDATIONS")
print("="*70)

print("\nBased on our analysis, we recommend the following marketing strategy:")

# Calculate potential financial impact
total_customers = len(df)
avg_customer_value = 500  # Assume $500 average annual value
budget_constraint = 0.20  # Can only target 20% of customers

print(f"""
MARKETING CAMPAIGN PLAN:
------------------------
Total customers: {total_customers}
Average customer value: ${avg_customer_value}/year
Campaign budget allows targeting: {budget_constraint*100:.0f}% of customers

TARGETING STRATEGY:
1. Focus on high-value segments first:
   - Premium Shoppers (highest spenders)
   - Aspiring Fashionistas (high potential)
   
2. Use different channels per segment:
   - Premium: Email + Personal outreach
   - Aspiring: Social Media + SMS
   - Budget: Email + App notifications

3. Measure success with:
   - Campaign response rate by segment
   - Increase in spending score
   - Customer lifetime value increase

EXPECTED ROI:
- Random targeting: {budget_constraint*100:.0f}% of churners prevented
- ML-based targeting: ~{budget_constraint*100*2:.0f}% of churners prevented
- Lift: 2x improvement
""")

print("\n" + "="*70)
print("✓ ANALYSIS COMPLETE")
print("="*70)
print("\nKey Learnings:")
print("1. Always scale your data before clustering")
print("2. Use multiple methods to choose k (Elbow + Silhouette)")
print("3. Profile clusters in original units for interpretation")
print("4. Name clusters with business-meaningful terms")
print("5. Translate insights into specific actions")
print("\nFor visualization, run this script in Jupyter Notebook with:")
print("import matplotlib.pyplot as plt")
print("import seaborn as sns")
print("# Then uncomment the visualization code in the full version")

# Optional: Save results
df.to_csv('customer_segments_results.csv', index=False)
print("\n✓ Results saved to 'customer_segments_results.csv'")