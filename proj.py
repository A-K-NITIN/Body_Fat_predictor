import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('bodyfat.csv')

# Display basic information

print(df.info())
print(df.isnull().sum())

# Check for missing values

missing_values = df.isnull().sum()
print("Missing values per column:", missing_values)

# Handle any missing values (if found)
df = df.dropna()  # or use appropriate imputation

# Distribution of target variable

plt.figure(figsize=(8, 6))
sns.histplot(df['BodyFat'], kde=True)
plt.title('Distribution of Body Fat Percentage')
plt.xlabel('Body Fat %')
# plt.show()



X = df.drop('BodyFat', axis=1)  # Features
y = df['BodyFat']                 # Target



# 3. SIMPLE LINEAR REGRESSION WITH CROSS-VALIDATION
model = LinearRegression()


# ============================================
print("-"*40)
print("\n📊 METHOD 1: WITHOUT CROSS-VALIDATION")
print("-"*40)

# Single train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# Predict and get accuracy
y_pred = model.predict(X_test)
simple_accuracy = r2_score(y_test, y_pred)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"ACCURACY (R²): {simple_accuracy}")


cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

# 4. RESULTS IN SIMPLE TERMS
print("\n" + "="*50)
print("📊 CROSS-VALIDATION RESULTS")
print("="*50)

for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.2%} accuracy")

print("\n" + "="*50)
print(f"🎯 FINAL MODEL ACCURACY: {cv_scores.mean():.2%}")
print(f"   (variation: ±{cv_scores.std():.2%})")
print("="*50)

# 6. SIMPLE ACCURACY INTERPRETATION
accuracy = cv_scores.mean()

print(f"ACuraccy: {accuracy}")

# ============================================
print("\n" + "="*60)
print("📊 GENERATING INDIVIDUAL VISUALIZATIONS")
print("="*60)

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ========== 2. Cross-Validation Scores Bar Chart ==========
plt.figure(figsize=(14, 8))
bars = plt.bar(range(1, 11), cv_scores * 100, color=['#2E86AB' if x >= cv_scores.mean() else '#F24236' for x in cv_scores], edgecolor='black', linewidth=1.5)
plt.axhline(y=cv_scores.mean() * 100, color='#FFB140', linestyle='--', linewidth=3, label=f"Mean Accuracy: {cv_scores.mean():.2%}")

# Add value labels on bars
for bar, score in zip(bars, cv_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{score:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xlabel('Fold Number', fontsize=14)
plt.ylabel('Accuracy (R² Score %)', fontsize=14)
plt.title('10-Fold Cross-Validation Results', fontsize=18, fontweight='bold', pad=20)
plt.xticks(range(1, 11))
plt.ylim(0, 100)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3, axis='y')



# ========== 3. Feature Correlation with Body Fat ==========
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
target_corr = correlation_matrix['BodyFat'].sort_values(ascending=False)

# Create horizontal bar chart
colors = ['#2E86AB' if x > 0 else '#F24236' for x in target_corr.values]
bars = plt.barh(range(len(target_corr)), target_corr.values, color=colors, edgecolor='black', linewidth=1)
plt.yticks(range(len(target_corr)), target_corr.index, fontsize=12)
plt.xlabel('Correlation with Body Fat', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.title('Feature Correlation with Body Fat Percentage', fontsize=18, fontweight='bold', pad=20)
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, target_corr.values)):
    if val > 0:
        plt.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    else:
        plt.text(val - 0.08, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10, fontweight='bold')

plt.grid(True, alpha=0.3, axis='x')



# ========== 4. Complete Correlation Heatmap (More Readable) ==========
plt.figure(figsize=(18, 14))

# Create mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix), k=1)

# Create heatmap with better formatting
ax = sns.heatmap(correlation_matrix, mask=mask,annot=True,  cmap='coolwarm', center=0,  square=True, fmt='.2f', linewidths=1,
                 cbar_kws={"shrink": 0.6, "label": "Correlation Coefficient", "aspect": 30},
                 annot_kws={"size": 9, "weight": "bold"}, xticklabels=True,yticklabels=True)

# Rotate and adjust labels
plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold')
plt.yticks(rotation=0, fontsize=11, fontweight='bold')

# Add more spacing at the bottom
plt.subplots_adjust(bottom=0.15, left=0.15)

plt.title('Complete Feature Correlation Matrix', fontsize=20, fontweight='bold', pad=30)




# ========== 5. Actual vs Predicted (All Data) ==========
# Train on all data for visualization
model.fit(X, y)
y_pred_all = model.predict(X)

plt.figure(figsize=(12, 10))
plt.scatter(y, y_pred_all, alpha=0.6, c='#2E86AB', s=100, edgecolors='black', linewidth=1)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=3, label='Perfect Prediction Line')
plt.fill_between([y.min(), y.max()], [y.min(), y.max()], [y.min()-5, y.max()-5], 
                 alpha=0.1, color='red', label='Under预测')
plt.fill_between([y.min(), y.max()], [y.min(), y.max()], [y.min()+5, y.max()+5], 
                 alpha=0.1, color='blue', label='Over预测')

plt.xlabel('Actual Body Fat Percentage (%)', fontsize=14)
plt.ylabel('Predicted Body Fat Percentage (%)', fontsize=14)
plt.title(f'Actual vs Predicted Body Fat\nR² Score: {r2_score(y, y_pred_all):.3f}', 
          fontsize=18, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)
plt.axis('equal')



# ========== 6. Residual Analysis ==========
residuals = y - y_pred_all

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Residual plot
axes[0].scatter(y_pred_all, residuals, alpha=0.6, c='#2E86AB', s=80, edgecolors='black', linewidth=1)
axes[0].axhline(y=0, color='#F24236', linestyle='--', linewidth=3)
axes[0].set_xlabel('Predicted Body Fat (%)', fontsize=14)
axes[0].set_ylabel('Residuals (Actual - Predicted)', fontsize=14)
axes[0].set_title('Residual Plot\n(Homogeneous spread indicates good model)', fontsize=16, fontweight='bold', pad=20)
axes[0].grid(True, alpha=0.3)

# Residual histogram
axes[1].hist(residuals, bins=20, color='#2E86AB', edgecolor='black', alpha=0.7, linewidth=1.5)
axes[1].axvline(x=0, color='#F24236', linestyle='--', linewidth=3, label='Zero Error')
axes[1].axvline(x=residuals.mean(), color='#FFB140', linestyle='--', linewidth=3, label=f'Mean Error: {residuals.mean():.2f}')
axes[1].set_xlabel('Prediction Error (%)', fontsize=14)
axes[1].set_ylabel('Frequency', fontsize=14)
axes[1].set_title(f'Distribution of Prediction Errors\nRMSE: {np.sqrt(mean_squared_error(y, y_pred_all)):.2f}%', 
                  fontsize=16, fontweight='bold', pad=20)
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)



# ========== 10. Learning Curve ==========
from sklearn.model_selection import learning_curve

plt.figure(figsize=(14, 8))

train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(), X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#2E86AB')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='#F24236')
plt.plot(train_sizes, train_mean, 'o-', color='#2E86AB', linewidth=3, markersize=8, label='Training Score')
plt.plot(train_sizes, test_mean, 'o-', color='#F24236', linewidth=3, markersize=8, label='Cross-Validation Score')

plt.xlabel('Training Examples', fontsize=14)
plt.ylabel('R² Score', fontsize=14)
plt.title('Learning Curve - Model Performance vs Training Data Size', fontsize=18, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3)

plt.show()
# ============================================
# TRAINING & TESTING PERFORMANCE METRICS
# ============================================

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Training Performance
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Testing Performance
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print Results
print("\n" + "="*50)
print("📊 TRAINING PERFORMANCE")
print("="*50)
print(f"R² Score: {train_r2:.2%}")
print(f"MAE: {train_mae:.2f} % body fat")
print(f"RMSE: {train_rmse:.2f} % body fat")

print("\n" + "="*50)
print("📊 TESTING PERFORMANCE")
print("="*50)
print(f"R² Score: {test_r2:.2%}")
print(f"MAE: {test_mae:.2f} % body fat")
print(f"RMSE: {test_rmse:.2f} % body fat")
print(X_train.head())
