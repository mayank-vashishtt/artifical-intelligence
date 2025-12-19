# House Rent Prediction with Linear Regression - Complete Guide

A detailed walkthrough of building a Linear Regression model to predict house rental prices, from data loading to predictions, with explanations at every step.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Libraries & Imports](#libraries--imports)
3. [Data Loading & EDA](#data-loading--eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering & Encoding](#feature-engineering--encoding)
6. [Train-Test Split](#train-test-split)
7. [Feature Scaling](#feature-scaling)
8. [Model Training](#model-training)
9. [Model Evaluation](#model-evaluation)
10. [Visualizations](#visualizations)
11. [Making Predictions](#making-predictions)
12. [Insights & Improvements](#insights--improvements)

---

## Project Overview

**Objective**: Build a machine learning model to predict house rental prices in India using Linear Regression.

**Dataset**: Kaggle House Rent Prediction Dataset
- **Records**: ~5000+ house rental listings
- **Features**: BHK, Size, Bathrooms, Location, Furnishing Status, etc.
- **Target**: Rent (monthly rental price in INR)

**Why Linear Regression?**
- Simple, interpretable baseline model
- Good for understanding feature-target relationships
- Fast to train and deploy
- Serves as a reference for comparing with advanced models

---

## Libraries & Imports

```python
import pandas as pd           # Data manipulation
import numpy as np            # Numerical computations
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns         # Enhanced visualizations
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler       # Feature scaling
from sklearn.linear_model import LinearRegression      # Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Metrics
import warnings
warnings.filterwarnings('ignore')
```

**Why each library?**
- **pandas**: Load, manipulate, and explore CSV data efficiently.
- **numpy**: Fast numerical operations on arrays.
- **matplotlib/seaborn**: Create insightful visualizations.
- **scikit-learn**: Industry-standard ML library with built-in functions.

---

## Data Loading & EDA

### Step 1: Load Data from Kaggle

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "House_Rent_Dataset.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "iamsouravbanerjee/house-rent-prediction-dataset",
    file_path
)
```

**Why Kaggle Hub?**
- Direct access to Kaggle datasets without manual download.
- Automatic caching and version management.
- Keeps notebooks reproducible across machines.

### Step 2: Initial Exploration

```python
print(f"Dataset Shape: {df.shape}")           # Rows and columns
print(f"Column Names: {df.dtypes}")           # Data types
print(f"Missing Values: {df.isnull().sum()}")  # Null counts
print(f"Basic Stats: {df.describe()}")        # Mean, std, min, max, etc.
```

**What we learn:**
- **Shape**: (5000, 12) means 5000 houses, 12 features
- **Data Types**: Identify numeric vs categorical columns
- **Missing Values**: Plan imputation strategy
- **Statistics**: Check for outliers, ranges, distributions

**Example Output Understanding:**
```
BHK: ranges 1-4 (apartments are typically 1-4 bedroom)
Size: 300-6000 sqft (wide range → needs scaling)
Rent: 1000-100000+ (target variable)
```

---

## Data Preprocessing

### Why Preprocessing?

Raw data is messy:
- Different scales (Size: 300-6000 vs Bathroom: 1-5)
- Text/categories (City names, Area Type) must be numeric
- Missing values need handling
- Inconsistent formats

Without preprocessing, models perform poorly and training is slow.

### Step 1: Separate Features and Target

```python
X = df.drop('Rent', axis=1)  # Features (inputs)
y = df['Rent']                # Target (output to predict)
```

**Why?**
- Model learns relationship: X → y
- Clean separation ensures we don't use target in features (leakage).

### Step 2: Drop Irrelevant Columns

```python
X = X.drop(['Posted On'], axis=1, errors='ignore')
```

**Why drop "Posted On"?**
- Date information doesn't directly predict rent (unless you engineer it into temporal features).
- Having too many features causes overfitting and increases dimensionality.
- Later, you could extract: Day of Week, Month, Season if seasonality matters.

### Step 3: Identify Column Types

```python
numeric_cols = ['BHK', 'Rent', 'Size', 'Bathroom']
categorical_cols = [col for col in X.columns if col not in numeric_cols]
```

**Why separate?**
- Numeric columns: Ready to use directly (or scale).
- Categorical columns: Need encoding (convert text to numbers).

---

## Feature Engineering & Encoding

### What is Encoding?

Machine learning models work with numbers, not text. Encoding converts:
- "Bangalore" → numeric representation
- "Furnished" → numeric representation

### One-Hot Encoding

```python
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
```

**How it works:**

Before:
```
City         Furnishing Status
Bangalore    Furnished
Mumbai       Unfurnished
Bangalore    Furnished
```

After (drop_first=True):
```
City_Mumbai  Furnishing_Unfurnished
0            0
1            1
0            1
```

**Why drop_first=True?**

For k categories, create k-1 binary columns (not k).

Example: 3 cities → create 2 columns
- If both are 0 → it's the dropped city (Bangalore)
- If City_Mumbai=1, City_Delhi=0 → it's Mumbai
- If City_Mumbai=0, City_Delhi=1 → it's Delhi

**Why necessary?**
- Avoids **multicollinearity**: Having all k columns makes features perfectly correlated.
- Linear Regression requires features to be independent; redundant columns confuse the model.
- Exception: Tree-based models don't require this.

**Real-world impact:**
Without drop_first, model may fail with singular matrix error or give unreliable coefficients.

---

## Train-Test Split

### Why Split Data?

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # Reproducibility
)
```

**The Problem Without Split:**
- Train on all data → model memorizes everything (overfitting).
- When you test, it already saw those examples → unfair evaluation.
- You overestimate performance; real-world usage will disappoint.

**The Solution:**
- **Train Set (80%)**: Model learns patterns from this data.
- **Test Set (20%)**: Model predicts on unseen data; measures true generalization.

**Why 80/20?**
- Common heuristic; provides enough training samples while meaningful test size.
- For smaller datasets: use 70/30 or Cross-Validation.

**Why random_state=42?**
- Ensures reproducibility (same split every time).
- Helps compare models: same data split means fair comparison.
- 42 is arbitrary but conventional.

**Train/Test Size:**
```
Total: 5000 samples
Train: 4000 samples
Test:  1000 samples
```

---

## Feature Scaling

### The Problem Without Scaling

Imagine two features:
- **Size**: ranges 300–6000 (scale of thousands)
- **Bathroom**: ranges 1–5 (scale of single digits)

Gradient-based algorithms step in direction of steepest descent. Without scaling:
- Large-scale feature (Size) dominates gradients.
- Small-scale feature (Bathroom) is "drowned out."
- Model heavily weights Size, ignores Bathroom.

### StandardScaler: Normalization

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Mathematical Transformation (z-score):**

For each feature j:
```
scaled_value = (value - mean) / standard_deviation
```

**Example:**
- Size: mean=1500, std=800
- Original house size: 2000
- Scaled: (2000 - 1500) / 800 = 0.625

After scaling:
- All features have mean ≈ 0 and std ≈ 1.
- Features are on the same scale.
- Gradients update features fairly.

**Critical: Fit on Train, Apply to Test**

```python
# ✓ CORRECT:
scaler.fit(X_train)       # Learn statistics from train only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✗ WRONG:
scaler.fit(X_train + X_test)  # Leaks test data statistics!
```

**Why?**
- Scaler computes mean/std from training data only.
- When test arrives in production, you won't know its statistics.
- Using test statistics during training = data leakage = optimistic metrics.

**When Scaling Isn't Needed:**
- Tree-based models (Decision Trees, Random Forests): scale-invariant.
- Linear models, SVM, KNN, Neural Nets: scaling helps.

---

## Model Training

### What is Linear Regression?

Linear Regression finds the best-fit line through data:

```
y = intercept + slope₁*feature₁ + slope₂*feature₂ + ... + noise
```

In matrix form:
```
y = w₀ + w₁*X₁ + w₂*X₂ + ... + wₙ*Xₙ
```

Where:
- **w₀**: Intercept (baseline rent when all features are 0)
- **w₁, w₂, ..., wₙ**: Slopes/weights (how much each feature affects rent)

### Training the Model

```python
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

**What fit() Does:**
1. Receives training features (X_train_scaled) and labels (y_train).
2. Computes optimal weights (w₀, w₁, ..., wₙ) to minimize error.
3. Stores these weights internally.

**Optimization Approach:**
- Uses **Ordinary Least Squares (OLS)**: Minimize sum of squared errors.
- Solves: min Σ(y_true - y_pred)² using linear algebra (not gradient descent).
- Closed-form solution exists → fast, deterministic.

**Model Parameters After Training:**

```python
print(f"Intercept: {model.intercept_}")  # ~500000 (base rent)
print(f"Coefficients: {model.coef_}")    # [2000, -50000, 10000, ...]
```

**Interpretation Example:**
- If coefficient for "Size" is 50: every additional sqft adds ₹50 to rent.
- If coefficient for "City_Mumbai" is 20000: Mumbai is ₹20K more expensive (vs baseline city).
- If coefficient for "Bathroom" is -5000: having more bathrooms decreases rent? (strange—likely due to confounding).

### Making Predictions

```python
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
```

**What predict() Does:**
1. Takes new feature data.
2. Applies learned weights: y_pred = intercept + sum(weights * features).
3. Returns predicted rent values.

---

## Model Evaluation

### Why Multiple Metrics?

A single metric hides problems. Together, they tell the full story.

### MAE (Mean Absolute Error)

```python
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
```

**Formula:**
```
MAE = (1/n) * Σ|y_true - y_pred|
```

**Interpretation:**
- MAE ≈ ₹15,000 means predictions are off by ₹15K on average.
- Same units as target → directly interpretable.
- Less sensitive to outliers than RMSE.

**Example:**
```
True: [100, 120, 130]
Pred: [110, 115, 125]
Errors: [10, 5, 5]
MAE = (10 + 5 + 5) / 3 = 6.67
```

### RMSE (Root Mean Squared Error)

```python
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
```

**Formula:**
```
RMSE = √[(1/n) * Σ(y_true - y_pred)²]
```

**Why sqrt()?**
- MSE has units²; sqrt brings it back to original units.
- RMSE ≈ ₹20,000 is interpretable; MSE = 400M is not.

**Difference from MAE:**
- RMSE penalizes large errors more heavily.
- MAE treats all errors equally.

**When to use:**
- **MAE**: Equally important all errors (e.g., small order mix-ups cost same as large).
- **RMSE**: Large errors are costly (e.g., over-predicting energy demand → wasted resources).

**Example:**
```
Errors: [10, 5, 5]
MAE = 6.67
MSE = (100 + 25 + 25) / 3 = 50
RMSE = √50 = 7.07

Errors: [2, 2, 20]
MAE = 8
MSE = (4 + 4 + 400) / 3 = 136
RMSE = √136 = 11.66  (much higher due to 20)
```

### R² (Coefficient of Determination)

```python
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
```

**Formula:**
```
R² = 1 - (SS_res / SS_tot)

where:
SS_res = Σ(y_true - y_pred)²    [residual sum of squares]
SS_tot = Σ(y_true - mean(y))²   [total sum of squares]
```

**Intuition:**
- R² = fraction of variance explained by the model.
- R² = 0.7 means "model explains 70% of price variation."
- R² = 0 means model is as good as predicting the mean.
- R² < 0 means model is worse than predicting the mean (very bad).

**Interpretation:**
```
R² = 1.0   → Perfect fit (unlikely, overfitting)
R² = 0.8   → Strong fit (80% variance explained)
R² = 0.5   → Moderate fit (50% variance explained)
R² = 0.3   → Weak fit (30% variance explained)
R² < 0     → Model is useless
```

**Range Issue:**
- In this notebook, test R² is **negative** → model is worse than predicting mean rent.
- This indicates **poor generalization** and likely **overfitting**.

---

## Visualizations

### Plot 1: Actual vs Predicted (Test Set)

```python
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
```

**What it shows:**
- X-axis: True rent values.
- Y-axis: Model's predictions.
- Red line: Perfect predictions (y_pred = y_true).

**Good vs Bad:**
- **Good**: Points cluster tightly around red line → accurate predictions.
- **Bad**: Points scattered far from line → poor predictions.
- **Bad**: Points follow a curved pattern → non-linear relationship not captured.

**In this notebook:**
- Points scatter widely → poor fit, high errors.

### Plot 2: Residuals vs Predicted

```python
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5, s=20, color='green')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
```

**What it shows:**
- X-axis: Model's predictions.
- Y-axis: Residuals (errors = true - predicted).
- Red line: Zero error.

**Good vs Bad:**
- **Good**: Residuals scattered randomly around zero, no pattern.
  - Indicates linear assumptions are valid.
  - Errors are unbiased (not systematically over/under-predicting).
- **Bad**: Residuals form a pattern (curve, increasing spread).
  - Indicates non-linearity or heteroskedasticity (non-constant variance).

**In this notebook:**
- Random scatter → linear model is reasonable, but fit is just poor.

### Plot 3: Training vs Test Error

```python
metrics = ['MAE', 'RMSE']
train_errors = [train_mae, train_rmse]
test_errors = [test_mae, test_rmse]
axes[1, 0].bar(x_pos - width/2, train_errors, width, label='Training', color='skyblue')
axes[1, 0].bar(x_pos + width/2, test_errors, width, label='Testing', color='orange')
```

**What it shows:**
- Compares training vs test error for each metric.

**Good vs Bad:**
- **Good**: Train and test bars are similar height.
  - Model generalizes well; no overfitting.
- **Bad**: Test errors >> train errors.
  - Classic overfitting: model memorized training data but doesn't generalize.

**In this notebook:**
- Test errors much higher than train → **overfitting detected**.
- This is the model's main issue.

### Plot 4: R² Score Comparison

```python
scores = ['Training R²', 'Testing R²']
r2_values = [train_r2, test_r2]
axes[1, 1].bar(scores, r2_values, color=colors, alpha=0.7)
```

**What it shows:**
- Side-by-side R² scores for train vs test.

**Good vs Bad:**
- **Good**: Both scores high and similar.
- **Bad**: Train R² high but test R² low or negative.
  - Overfitting.

**In this notebook:**
- Train R² ≈ 0.65 (decent).
- Test R² < 0 (terrible).
- Clear overfitting.

---

## Making Predictions

### Create New Sample Data

```python
sample_houses = pd.DataFrame({
    'BHK': [2, 3, 1, 4],
    'Size': [1200, 2000, 800, 3000],
    'City': ['Bangalore', 'Bangalore', 'Bangalore', 'Bangalore'],
    ...
})
```

**Why create samples?**
- Demonstrate real-world usage: given house features, predict rent.
- Validate model makes sensible predictions.

### Preprocess Samples (Apply Same Transformations)

```python
sample_encoded = pd.get_dummies(sample_processed, columns=categorical_cols_sample, drop_first=True)

# Ensure columns match training data
for col in X_final.columns:
    if col not in sample_encoded.columns:
        sample_encoded[col] = 0

sample_final = sample_encoded[X_final.columns]
sample_scaled = scaler.transform(sample_final)
```

**Why match columns?**
- Training data had certain categories; test data might have new categories or missing ones.
- Add missing columns as 0 (feature is absent).
- Reorder columns to match training data exactly.

**Why use same scaler?**
- Scaler learned mean/std from training data.
- Apply same transformation to new data (not fit a new scaler).
- Ensures consistency.

### Predict and Display Results

```python
predictions = model.predict(sample_scaled)

for idx, pred in enumerate(predictions, 1):
    print(f"House {idx}: Predicted Monthly Rent: ₹{pred:,.2f}")
    print(f"Prediction Range: ₹{pred-test_rmse:,.2f} to ₹{pred+test_rmse:,.2f}")
```

**Prediction Range (Uncertainty):**
- pred ± test_rmse gives a rough confidence interval.
- More rigorously, you'd use bootstrap or probabilistic models.

---

## Insights & Improvements

### Identified Problems

**1. Overfitting**
- Training R² ≈ 0.65, Test R² < 0.
- Model memorized training noise.

**2. High Error Rate**
- Test RMSE ≈ ₹113K on test set (very high).
- Predictions are unreliable for business use.

**3. Feature Dimensionality**
- One-hot encoding creates many sparse columns.
- Curse of dimensionality: more features, more overfitting with same sample size.

### Recommended Improvements

#### 1. Feature Selection/Dimensionality Reduction

```python
# Remove low-variance features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_train_selected = selector.fit_transform(X_train_scaled)

# Or use PCA to reduce dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
```

**Why?**
- Fewer features = simpler model = less overfitting.
- Removes noise-only columns.

#### 2. Use Regularization

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge: penalizes large weights
ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)

# Lasso: forces some weights to zero
lasso = Lasso(alpha=1000).fit(X_train_scaled, y_train)
```

**Why?**
- Regularization reduces overfitting by penalizing complexity.
- Ridge: small penalties on all features.
- Lasso: aggressive, can eliminate features entirely.

**Hyperparameter alpha:**
- alpha=0 → no regularization (vanilla Linear Regression).
- alpha=∞ → extreme regularization (all weights → 0).
- Find optimal alpha via Cross-Validation.

#### 3. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    LinearRegression(), 
    X_train_scaled, 
    y_train,
    cv=5,  # 5-fold CV
    scoring='r2'
)

print(f"CV R² scores: {scores}")
print(f"Mean CV R²: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

**Why?**
- Single train/test split is noisy; CV averages over multiple splits.
- More robust estimate of generalization performance.

#### 4. Polynomial/Non-linear Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Now train on polynomial features
model_poly = LinearRegression().fit(X_train_poly, y_train)
```

**Why?**
- Linear model assumes linear relationship.
- If relationship is curved, polynomial features capture it.
- Caution: increases dimensions significantly; prone to overfitting.

#### 5. Try Ensemble Methods

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # No scaling needed for trees!

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

# Compare metrics
print(f"RF Test R²: {rf.score(X_test, y_test):.4f}")
print(f"GB Test R²: {gb.score(X_test, y_test):.4f}")
```

**Why?**
- Trees are robust to non-linearity, outliers, and mixed feature scales.
- Ensembles reduce overfitting via averaging.
- Often outperform linear models.

#### 6. Feature Engineering

```python
# Extract temporal features from date
df['Posted_Month'] = pd.to_datetime(df['Posted On']).dt.month
df['Posted_Season'] = df['Posted_Month'].apply(lambda m: 'Winter' if m in [12,1,2] else ...)

# Create interaction features
df['BHK_per_Bathroom'] = df['BHK'] / df['Bathroom']
df['Price_per_Sqft'] = df['Rent'] / df['Size']  # Only on train for EDA!

# Log-transform skewed target
y_log = np.log1p(y)  # Predict log(Rent), then inverse-transform
```

**Why?**
- Domain knowledge: seasonal trends, locality interactions matter.
- Log-transform: if Rent is right-skewed, log makes it more normal.
- Improves model's ability to capture relationships.

#### 7. Handle Outliers

```python
# Identify outliers (e.g., >3 std from mean)
threshold = 3
outlier_mask = np.abs(y_train - y_train.mean()) > threshold * y_train.std()

# Remove or down-weight outliers
X_train_no_outliers = X_train_scaled[~outlier_mask]
y_train_no_outliers = y_train[~outlier_mask]

# Train on clean data
model = LinearRegression().fit(X_train_no_outliers, y_train_no_outliers)
```

**Why?**
- Outliers pull the regression line, increase error.
- Removing extreme outliers can improve fit (but check if they're real).

#### 8. Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('model', Ridge())
])

param_grid = {
    'model__alpha': [0.1, 1, 10, 100, 1000]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='r2')
grid.fit(X_train_scaled, y_train)

print(f"Best alpha: {grid.best_params_['model__alpha']}")
print(f"Best CV R²: {grid.best_score_:.4f}")

model = grid.best_estimator_
y_test_pred = model.predict(X_test_scaled)
```

**Why?**
- Systematically test parameter combinations.
- Avoids manual guessing.
- CV ensures fair selection (no test data leak).

---

## Advanced Topics

### Why Linear Regression Underperforms Here

1. **Non-linear Relationships**: Rent depends on location + size in complex ways (maybe certain neighborhoods have premium pricing). Linear can't capture this.
2. **Interaction Effects**: BHK × Size interaction (small 4-bedroom vs large 2-bedroom). Linear Regression doesn't capture.
3. **Categorical Interactions**: Furnishing status might matter differently for Mumbai vs Bangalore. Linear doesn't capture.
4. **Sparse One-Hot Features**: Too many dummy variables, too little data relative to features.

### Bias-Variance Tradeoff

- **High Bias** (underfitting): Linear model on non-linear data → systematic error.
- **High Variance** (overfitting): Too many features → fits noise, fails on test.

**In this notebook**: Likely both issues—linear model can't fit training data well (bias), and with many features, it overfits the little it can (variance).

**Solution**: Regularization + feature selection + better features balances bias-variance.

### Cross-Validation Deep Dive

**K-Fold (e.g., K=5):**
1. Split data into 5 folds.
2. Train on fold 1-4, test on fold 5.
3. Train on fold 1-3 + 5, test on fold 4.
4. ... (5 iterations total).
5. Average scores.

**Why 5-fold?**
- Smaller K (e.g., 3): more bias (fewer train samples per fold).
- Larger K (e.g., 10): more variance (overlapping train sets), slower.
- 5 is balanced default.

**Time Series Split:**
- Don't shuffle chronologically.
- Train on past, test on future.

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

### Residual Analysis

For valid Linear Regression:
- Residuals should be **normally distributed** (bell curve).
- Residuals should have **constant variance** (homoskedastic).
- Residuals should have **no correlation** with predictions or features.

```python
from scipy import stats

# Q-Q plot (check normality)
stats.probplot(residuals, dist="norm", plot=plt)
plt.show()

# Breusch-Pagan test (check heteroskedasticity)
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, X_train_scaled)
print(f"BP p-value: {bp_pval}")  # p > 0.05 → constant variance ✓
```

### Feature Importance (Linear Regression)

```python
# Absolute coefficients (magnitude of effect)
feature_importance = np.abs(model.coef_)
feature_names = X_final.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(importance_df.head(10))

# Plot top features
plt.barh(importance_df['Feature'].head(10), importance_df['Importance'].head(10))
plt.xlabel('|Coefficient|')
plt.title('Top 10 Most Important Features')
plt.show()
```

**Caution:**
- Only interpretable if features are scaled (StandardScaler).
- Unscaled: large-scale features have large coefficients even if weak relationship.

---

## Summary & Next Steps

### What We Learned

1. **Data Loading**: Use kagglehub for reproducibility.
2. **EDA**: Understand data shape, types, ranges, missingness.
3. **Preprocessing**: Handle categoricals (one-hot), separate features/target.
4. **Feature Scaling**: Standardize numeric features for fair optimization.
5. **Train-Test Split**: Prevent overfitting evaluation via held-out test set.
6. **Model Training**: LinearRegression uses OLS to find best weights.
7. **Evaluation**: Use MAE, RMSE, R² to assess fit.
8. **Visualization**: Plot predictions, residuals, errors to diagnose issues.
9. **Inference**: Preprocess and predict on new data consistently.

### Key Takeaways

- Linear Regression is interpretable but assumes linearity.
- Overfitting is detected by train >> test error gap.
- Regularization, feature selection, and better features reduce overfitting.
- Cross-Validation provides robust performance estimates.
- Tree-based and ensemble models often outperform linear for complex data.

### Next Projects

1. **Try Ridge/Lasso Regression**: Add regularization to the pipeline.
2. **Implement Random Forest**: Compare non-linear model performance.
3. **Feature Engineering**: Extract temporal features, interactions.
4. **Cross-Validation**: Replace single train-test with 5-fold CV.
5. **Ensemble Stacking**: Combine multiple models' predictions.
6. **Time Series Forecasting**: If temporal patterns exist, use ARIMA/Prophet.
7. **Hyperparameter Tuning**: GridSearchCV over alpha, max_depth, etc.

---

## Code Checklist (Best Practices)

- ✓ Set `random_state` everywhere (train_test_split, models, cross-val).
- ✓ Fit scaler/encoder on train only; apply to test.
- ✓ Use Pipelines for consistency.
- ✓ Compare train vs test metrics to detect overfitting.
- ✓ Plot residuals and actual-vs-predicted.
- ✓ Handle categorical encoding correctly (drop_first for linear models).
- ✓ Use cross-validation for robust estimates.
- ✓ Document model coefficients and predictions.
- ✓ Create reproducible samples for new predictions.

---