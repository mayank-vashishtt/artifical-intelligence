# Practical ML Concepts & Workflow Guide

This guide explains each item in your sheet with concise theory, safety checks, real-world examples, and short code snippets.

---

## Baseline

- What: A simple model or rule of thumb to set a reference performance before complex models.
- Why: Detect data issues early and ensure any model is truly better than “obvious” strategies.
- Examples:
  - Regression: predict the training mean price for every house.
  - Classification: always predict the majority class (e.g., “not churn”).
- Quick code (scikit-learn):
  ```python
  from sklearn.dummy import DummyRegressor, DummyClassifier
  # Regression baseline
  reg_base = DummyRegressor(strategy="mean").fit(X_train, y_train)
  # Classification baseline
  clf_base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
  ```
- Safety checks:
  - Compare your model to the baseline on the test set.
  - If your model barely beats baseline, revisit features/labels/processing.

---

## Missing Values (NaNs)

- Types:
  - MCAR: Missing completely at random (e.g., sensor glitch).
  - MAR: Missing depends on observed data (e.g., income missing more for certain groups).
  - MNAR: Missing depends on the missing value itself (e.g., high income not disclosed).
- Strategies:
  - Drop rows/columns (only if small impact).
  - Simple impute: mean/median for numeric, mode for categorical.
  - Add “missing indicator” column to flag NaNs.
  - Advanced impute: KNNImputer, IterativeImputer.
  - Domain-specific rules (e.g., medical: lab not ordered vs test failed).
- Real-world:
  - IoT sensors: occasional null readings → median impute + indicator.
  - Health records: lab not taken → encode “not measured” explicitly, not 0.
- Safety checks:
  - Fit imputer on training data only; transform train and test consistently.
  - Do not peek into test data to compute impute statistics (avoid leakage).

---

## Exploratory Data Analysis (EDA)

- Goals: Understand shape, types, distributions, missingness, outliers, correlations, leakage risks.
- Steps:
  - Inspect `df.shape`, `df.dtypes`, missing % per column.
  - Plot distributions for numeric, bar charts for categorical.
  - Check target distribution; class imbalance; time trends.
  - Correlations; duplicates; ID-like columns; near-constant features.
- Real-world:
  - Housing: price is skewed → apply log transform to stabilize variance.
  - Credit risk: minority “default” class → stratified split and suitable metrics.

---

## Encoding (What is encoding?)

- What: Turn non-numeric data (strings/categories) into numeric features for models.
- Common types:
  - One-Hot Encoding: binary columns per category (for nominal categories).
  - Ordinal Encoding: integer map for ordered categories (e.g., small < medium < large).
  - Target/Frequency Encoding: use aggregated stats; handle with care to avoid leakage.
- Use cases:
  - Product category, city names → one-hot.
  - Education level (ordered) → ordinal.

---

## One-Hot Encoding with `pd.get_dummies`

Your snippet:
```python
X_encoded = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)
```
- `X`: your feature DataFrame.
- `columns=categorical_cols`: list of columns to one-hot encode.
- `drop_first=True`: for each categorical variable, drop the first dummy to avoid perfect multicollinearity (dummy variable trap), useful for linear models. You get k-1 columns for k categories.

- Example:
  ```python
  import pandas as pd
  X = pd.DataFrame({"color": ["red", "blue", "blue", "green"], "size": ["S","M","S","L"]})
  pd.get_dummies(X, columns=["color", "size"], drop_first=True)
  # Result columns (example):
  # color_green, color_red, size_M, size_S
  ```

- Caveats & safety:
  - Train/test alignment: Ensure the same dummy columns exist in train and test. Unseen categories in test won’t get columns automatically; consider sklearn’s `OneHotEncoder(handle_unknown="ignore")`.
  - High cardinality: Many categories → many columns, memory blow-up. Consider hashing or target/frequency encoding (with CV).
  - Consistency: Fit encoders on training, apply to test via Pipeline.

- Preferred approach in sklearn:
  ```python
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.pipeline import Pipeline
  from sklearn.linear_model import LogisticRegression

  cat_cols = categorical_cols
  num_cols = [c for c in X.columns if c not in cat_cols]

  preproc = ColumnTransformer(
      transformers=[
          ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
          ("num", "passthrough", num_cols),
      ]
  )

  model = Pipeline(steps=[("preproc", preproc), ("clf", LogisticRegression())])
  model.fit(X_train, y_train)
  ```

---

## Safety Checks (General)

- No target leakage: Features must not contain future info or direct transformations of the target.
- Proper split: Do not compute statistics on the full dataset; fit transformers on train only.
- Time-aware split: For time series, split chronologically (no random shuffle).
- Duplicates and leakage via IDs: Remove duplicates; do not use unique identifiers as features.
- Reproducibility: Set random seeds; save versions of data and code.

---

## Why Split into Train and Test?

- Purpose: Estimate how well your model generalizes to new data.
- If you train on the entire dataset, you will overestimate performance (model sees everything).
- Test set acts as an unbiased proxy for future data.
- Use validation or cross-validation to tune hyperparameters; keep the final test untouched.

- Time series:
  - Use `TimeSeriesSplit` or chronological split to avoid future leakage.

---

## Why Not Train on the Whole Dataset?

- Without a held-out set, you cannot detect overfitting or data leakage.
- You may tune decisions to fit noise, leading to poor real-world performance.
- Always keep a final untouched test set, or use nested CV for robust estimates.

---

## Why `random_state=42`?

- Ensures reproducible randomness (same split, same initialization).
- The value 42 is arbitrary but conventional; any fixed integer works.
- Set seeds in libraries that use randomness (NumPy, PyTorch, TensorFlow), if applicable.

---

## Key Concept: Generalization

- ML goal: Perform well on unseen data from the same (or similar) distribution.
- Good generalization:
  - Clean features, robust preprocessing, appropriate model complexity.
  - Proper evaluation (CV, test set).
- Beware domain shift: If production data differs from training data, performance can drop; monitor and retrain.

---

## Overfitting vs Underfitting

- Overfitting:
  - Model memorizes noise; high train score, low test score.
  - Example: Deep decision tree fits every training point but fails on new data.
  - Fix: Regularization, simpler model, more data, early stopping, dropout, pruning.
- Underfitting:
  - Model too simple; low train and test scores.
  - Example: Linear model for highly non-linear pattern.
  - Fix: Add features, use more expressive models, reduce regularization.

- Tip: Plot learning curves (train vs validation error) to diagnose.

---

## Feature Scaling

- Why:
  - Distance-based models (KNN, SVM with RBF), gradient-based methods (linear/logistic regression, neural nets) benefit from scaled features.
  - Prevents one large-scale feature from dominating gradients/distances.
- Not always needed:
  - Tree-based models (Decision Trees, Random Forests, Gradient Boosting) are scale-invariant.
- Common scalers:
  - StandardScaler (z-score).
  - MinMaxScaler (0–1 range).
  - RobustScaler (median/IQR; resilient to outliers).

---

## What StandardScaler Does (Math)

- For each numeric feature j:
  - Compute mean μ_j and standard deviation σ_j on the TRAIN set only.
  - Transform: z = (x - μ_j) / σ_j
- Effects:
  - Features centered around 0 with unit variance.
  - Helps optimization and comparability across features.

- Example:
  - Heights: [160, 170, 180] cm (train) → μ=170, σ≈8.165
  - Test value 175 → z ≈ (175 - 170) / 8.165 ≈ 0.612

- Safe usage with Pipeline:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import Ridge

  pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_test)
  ```

---

## c

- MAE (Mean Absolute Error):
  - Average absolute difference |y_true - y_pred|.
  - Robust to outliers compared to MSE.
  - Unit: same as target (e.g., dollars).
- MSE (Mean Squared Error):
  - Average squared error (emphasizes larger errors).
  - Good for penalizing big mistakes.
- RMSE (Root Mean Squared Error):
  - sqrt(MSE); same unit as target, intuitive scale.
- R² (Coefficient of Determination):
  - Fraction of variance explained by the model; ranges typically from negative (bad) to 1.0 (perfect).
  - 0 means predicting the mean is as good as your model.

- Example:
  - True: [100, 120, 130], Pred: [110, 115, 125]
  - Errors: [10, -5, -5]
  - MAE = (|10|+|5|+|5|)/3 ≈ 6.67
  - MSE = (100 + 25 + 25)/3 ≈ 50
  - RMSE ≈ 7.07
  - R² depends on variance of true values; use `sklearn.metrics.r2_score`.

- Code:
  ```python
  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
  mae = mean_absolute_error(y_test, y_pred)
  mse = mean_squared_error(y_test, y_pred)
  rmse = mse ** 0.5
  r2 = r2_score(y_test, y_pred)
  ```

---

## Putting It Together: Minimal Workflow

1. EDA + safety checks (types, missingness, leakage risks).
2. Split data (train/validation/test or cross-validation).
3. Build Pipeline:
   - Impute missing values.
   - Encode categoricals (OneHotEncoder with handle_unknown).
   - Scale numerics (if needed).
   - Fit model.
4. Establish baseline; compare metrics.
5. Tune hyperparameters with CV; select model.
6. Final evaluation on test set; set seeds for reproducibility.
7. Monitor in production; retrain as data shifts.

---