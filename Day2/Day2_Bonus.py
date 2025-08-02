# CUSTOMER ANALYTICS MAGIC - PREDICTIVE INSIGHTS & ADVANCED ANALYSIS
# This script demonstrates advanced analytics techniques for customer data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🎯 CUSTOMER ANALYTICS MAGIC - ADVANCED INSIGHTS")
print("=" * 60)

# Load your data (assuming you have the CSV from Power BI export)
try:
    df = pd.read_csv('customer_data_for_powerbi.csv')
    print(f"✅ Data loaded successfully: {len(df)} customers")
except:
    print("⚠️  Please ensure 'customer_data_for_powerbi.csv' is in the same folder")
    print("   Creating demo data for this example...")
    # Create demo data if file not found
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.normal(40, 12, 1000).astype(int),
        'balance': np.random.exponential(1000, 1000),
        'estimated_clv': np.random.normal(500, 200, 1000),
        'risk_score': np.random.beta(2, 5, 1000),
        'deposit': np.random.choice(['yes', 'no'], 1000, p=[0.3, 0.7]),
        'job': np.random.choice(['management', 'technician', 'entrepreneur', 'blue-collar', 'retired'], 1000),
        'education': np.random.choice(['primary', 'secondary', 'tertiary'], 1000),
        'marital': np.random.choice(['married', 'single', 'divorced'], 1000)
    })

print("\n🔮 MAGIC #1: PREDICTIVE DEPOSIT MODEL")
print("-" * 40)

# Prepare data for machine learning
def prepare_ml_data(df):
    """Prepare data for machine learning"""
    df_ml = df.copy()
    
    # Encode categorical variables
    le_dict = {}
    categorical_cols = ['job', 'education', 'marital']
    
    for col in categorical_cols:
        if col in df_ml.columns:
            le = LabelEncoder()
            df_ml[col + '_encoded'] = le.fit_transform(df_ml[col])
            le_dict[col] = le
    
    # Select features for prediction
    feature_cols = ['age', 'balance', 'estimated_clv', 'risk_score']
    if 'job_encoded' in df_ml.columns:
        feature_cols.extend(['job_encoded', 'education_encoded', 'marital_encoded'])
    
    X = df_ml[feature_cols]
    y = (df_ml['deposit'] == 'yes').astype(int)
    
    return X, y, le_dict

# Prepare data
X, y, encoders = prepare_ml_data(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

model_results = {}
best_model = None
best_score = 0

print("Training predictive models...")
for name, model in models.items():
    # Scale data for Logistic Regression
    if name == 'Logistic Regression':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    if name == 'Logistic Regression':
        accuracy = model.score(X_test_scaled, y_test)
    else:
        accuracy = model.score(X_test, y_test)
    
    model_results[name] = {
        'model': model,
        'auc_score': auc_score,
        'accuracy': accuracy,
        'predictions': y_pred_proba
    }
    
    if auc_score > best_score:
        best_score = auc_score
        best_model = name
    
    print(f"  {name}: AUC = {auc_score:.3f}, Accuracy = {accuracy:.3f}")

print(f"\n🏆 Best Model: {best_model} (AUC: {best_score:.3f})")

# Feature importance for best model
if best_model == 'Random Forest' or best_model == 'Gradient Boosting':
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': models[best_model].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Most Important Features for Predicting Deposits:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

print("\n🎯 MAGIC #2: CUSTOMER LIFETIME VALUE INSIGHTS")
print("-" * 40)

# CLV Analysis
clv_stats = df['estimated_clv'].describe()
print(f"Average CLV: ${clv_stats['mean']:.2f}")
print(f"CLV Range: ${clv_stats['min']:.2f} - ${clv_stats['max']:.2f}")

# Find high-value customers
high_value_threshold = df['estimated_clv'].quantile(0.8)
high_value_customers = df[df['estimated_clv'] >= high_value_threshold]
print(f"\n💎 Top 20% of customers (CLV >= ${high_value_threshold:.2f}):")
print(f"  Count: {len(high_value_customers)} customers")
print(f"  Average CLV: ${high_value_customers['estimated_clv'].mean():.2f}")
if 'job' in df.columns:
    top_jobs = high_value_customers['job'].value_counts().head(3)
    print(f"  Top jobs: {', '.join([f'{job} ({count})' for job, count in top_jobs.items()])}")

print("\n🔍 MAGIC #3: ADVANCED CUSTOMER SEGMENTATION")
print("-" * 40)

# Advanced K-Means clustering on numerical features
numerical_features = ['age', 'balance', 'estimated_clv', 'risk_score']
X_cluster = df[numerical_features].fillna(df[numerical_features].mean())

# Standardize features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Find optimal number of clusters using elbow method
inertias = []
K_range = range(2, 8)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Use 4 clusters for analysis
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['ml_cluster'] = cluster_labels

# Analyze clusters
print(f"Created {optimal_k} data-driven customer clusters:")
for i in range(optimal_k):
    cluster_data = df[df['ml_cluster'] == i]
    print(f"\n  📊 Cluster {i+1} ({len(cluster_data)} customers):")
    print(f"     Avg Age: {cluster_data['age'].mean():.1f}")
    print(f"     Avg Balance: ${cluster_data['balance'].mean():.2f}")
    print(f"     Avg CLV: ${cluster_data['estimated_clv'].mean():.2f}")
    print(f"     Avg Risk: {cluster_data['risk_score'].mean():.3f}")
    if 'deposit' in df.columns:
        deposit_rate = (cluster_data['deposit'] == 'yes').mean() * 100
        print(f"     Deposit Rate: {deposit_rate:.1f}%")

print("\n📈 MAGIC #4: STATISTICAL INSIGHTS")
print("-" * 40)

# Statistical tests and correlations
print("Key Correlations:")
correlation_matrix = df[numerical_features].corr()
# Find strongest correlations (excluding self-correlations)
correlations = []
for i, col1 in enumerate(numerical_features):
    for j, col2 in enumerate(numerical_features):
        if i < j:  # Avoid duplicates and self-correlation
            corr_value = correlation_matrix.loc[col1, col2]
            correlations.append((col1, col2, abs(corr_value), corr_value))

# Sort by absolute correlation strength
correlations.sort(key=lambda x: x[2], reverse=True)

for col1, col2, abs_corr, corr in correlations[:3]:
    direction = "positively" if corr > 0 else "negatively"
    print(f"  {col1} & {col2}: {direction} correlated (r={corr:.3f})")

# Age group analysis
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                        labels=['Young', 'Middle-aged', 'Senior', 'Elder'])

age_analysis = df.groupby('age_group').agg({
    'estimated_clv': 'mean',
    'balance': 'mean',
    'deposit': lambda x: (x == 'yes').mean() if 'deposit' in df.columns else 0
}).round(2)

print(f"\n👥 Age Group Insights:")
for age_group, data in age_analysis.iterrows():
    print(f"  {age_group}: CLV=${data['estimated_clv']:.2f}, Balance=${data['balance']:.2f}")

print("\n🎨 MAGIC #5: CREATING ADVANCED VISUALIZATIONS")
print("-" * 40)

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('🎯 Advanced Customer Analytics Dashboard', fontsize=16, fontweight='bold')

# Plot 1: CLV vs Risk Score by ML Cluster
scatter = axes[0,0].scatter(df['estimated_clv'], df['risk_score'], 
                           c=df['ml_cluster'], cmap='viridis', alpha=0.6)
axes[0,0].set_xlabel('Estimated CLV ($)')
axes[0,0].set_ylabel('Risk Score')
axes[0,0].set_title('CLV vs Risk Score (ML Clusters)')
plt.colorbar(scatter, ax=axes[0,0])

# Plot 2: Age Distribution by Deposit Status
if 'deposit' in df.columns:
    df[df['deposit'] == 'yes']['age'].hist(alpha=0.7, bins=20, label='Deposit: Yes', ax=axes[0,1])
    df[df['deposit'] == 'no']['age'].hist(alpha=0.7, bins=20, label='Deposit: No', ax=axes[0,1])
    axes[0,1].set_xlabel('Age')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Age Distribution by Deposit Status')
    axes[0,1].legend()

# Plot 3: CLV by ML Cluster
cluster_clv = df.groupby('ml_cluster')['estimated_clv'].mean()
axes[1,0].bar(range(len(cluster_clv)), cluster_clv.values, color=plt.cm.viridis(np.linspace(0, 1, len(cluster_clv))))
axes[1,0].set_xlabel('ML Cluster')
axes[1,0].set_ylabel('Average CLV ($)')
axes[1,0].set_title('Average CLV by ML Cluster')
axes[1,0].set_xticks(range(len(cluster_clv)))
axes[1,0].set_xticklabels([f'Cluster {i+1}' for i in range(len(cluster_clv))])

# Plot 4: Feature Importance (if available)
if best_model in ['Random Forest', 'Gradient Boosting'] and 'feature_importance' in locals():
    top_features = feature_importance.head(6)
    axes[1,1].barh(range(len(top_features)), top_features['importance'].values)
    axes[1,1].set_yticks(range(len(top_features)))
    axes[1,1].set_yticklabels(top_features['feature'].values)
    axes[1,1].set_xlabel('Feature Importance')
    axes[1,1].set_title(f'{best_model} - Feature Importance')
else:
    # Alternative: Correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Feature Correlation Heatmap')

plt.tight_layout()
plt.show()

print("\n🎯 MAGIC #6: ACTIONABLE BUSINESS RECOMMENDATIONS")
print("-" * 40)

# Generate actionable insights
print("💡 Key Recommendations:")

# High-value, low-risk customers
if 'risk_score' in df.columns and 'estimated_clv' in df.columns:
    sweet_spot = df[(df['estimated_clv'] > df['estimated_clv'].quantile(0.7)) & 
                    (df['risk_score'] < df['risk_score'].quantile(0.3))]
    print(f"1. 🎯 Target {len(sweet_spot)} high-CLV, low-risk customers for premium campaigns")

# Identify improvement opportunities
if 'deposit' in df.columns:
    non_depositors = df[df['deposit'] == 'no']
    high_potential = non_depositors[non_depositors['estimated_clv'] > non_depositors['estimated_clv'].median()]
    print(f"2. 📈 Focus on {len(high_potential)} high-potential non-depositors")

# Risk management
high_risk_valuable = df[(df['risk_score'] > df['risk_score'].quantile(0.8)) & 
                       (df['estimated_clv'] > df['estimated_clv'].median())]
print(f"3. ⚠️  Monitor {len(high_risk_valuable)} high-risk but valuable customers")

# Cluster-based recommendations
best_cluster = df.groupby('ml_cluster')['estimated_clv'].mean().idxmax()
print(f"4. 🏆 Replicate strategies from Cluster {best_cluster+1} (highest avg CLV)")

print(f"\n5. 🤖 Use the {best_model} model for future deposit predictions")

print("\n" + "="*60)
print("🎉 ANALYTICS MAGIC COMPLETE!")
print("Your data now has:")
print("  ✅ Predictive deposit probability scores")
print("  ✅ Advanced ML-based customer clusters") 
print("  ✅ Statistical insights and correlations")
print("  ✅ Actionable business recommendations")
print("  ✅ Beautiful visualizations")
print("\n💡 Next steps: Use these insights to optimize your marketing campaigns!")