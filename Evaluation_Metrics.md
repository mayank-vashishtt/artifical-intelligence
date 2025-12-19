# Regression Metrics: An Intuitive, Practical Guide

This guide explains regression metrics with intuition, when to use them, pitfalls, and code examples from scratch and with scikit-learn.

Contents
- Core metrics: MAE, MSE, RMSE, R², Adjusted R²
- Robust/scale-aware metrics: MedAE, MAPE, sMAPE, MSLE/RMSLE, Explained Variance
- Advanced/when costs differ: Huber, Quantile (Pinball), Poisson Deviance
- Weighted, multi-output, and time-series considerations
- Confidence intervals with bootstrap
- End-to-end examples: NumPy, scikit-learn, cross-validation, pipelines
- Visual diagnostics for errors

---

## 1) Core Metrics

### Mean Absolute Error (MAE)
- What: Average absolute error |y - ŷ|.
- Intuition: “Typical” error size. Linear penalty for mistakes.
- Pros: Robust to outliers vs MSE, interpretable (same units as target).
- Cons: Gradient is less smooth at 0 (but fine for modern optimizers).
- Use when:
  - You care equally about all errors.
  - Outliers exist, but you don’t want them to dominate.

From scratch (NumPy):
```python
import numpy as np

def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))
```

scikit-learn:
```python
from sklearn.metrics import mean_absolute_error
mae_val = mean_absolute_error(y_test, y_pred)
```

Real-world example:
- Food delivery ETA: MAE directly reads as “average minutes off,” which is interpretable to ops teams.

---

### Mean Squared Error (MSE)
- What: Average squared error (y - ŷ)².
- Intuition: Penalizes large errors quadratically; pushes the model to avoid big mistakes.
- Pros: Smooth gradients; common for optimization.
- Cons: Sensitive to outliers.
- Use when:
  - Very large errors are disproportionately costly (e.g., energy demand overestimation leads to massive procurement costs).

From scratch:
```python
def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)
```

scikit-learn:
```python
from sklearn.metrics import mean_squared_error
mse_val = mean_squared_error(y_test, y_pred)
```

---

### Root Mean Squared Error (RMSE)
- What: sqrt(MSE).
- Intuition: Like MSE but back in target units; still penalizes large errors.
- Use when:
  - You want interpretability in the original units + sensitivity to large errors.

From scratch:
```python
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))
```

---

### R² (Coefficient of Determination)
- What: 1 - SS_res / SS_tot, where SS_res = Σ(y - ŷ)², SS_tot = Σ(y - ȳ)².
- Intuition: Fraction of variance explained by the model vs predicting the mean.
- Range: Can be negative (worse than predicting the mean) up to 1.0 (perfect).
- Cautions:
  - R² alone is not comparable across different target scales.
  - On small or low-variance datasets, R² can be misleading.

From scratch:
```python
def r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
```

scikit-learn:
```python
from sklearn.metrics import r2_score
r2_val = r2_score(y_test, y_pred)
```

---

### Adjusted R²
- What: Penalized R² that accounts for number of features p and samples n.
- Formula: 1 - (1 - R²) * (n - 1) / (n - p - 1)
- Intuition: Prevents inflated R² from adding unnecessary features.

From scratch:
```python
def adjusted_r2(y_true, y_pred, n_samples, n_features):
    r2_val = r2(y_true, y_pred)
    denom = max(n_samples - n_features - 1, 1)
    return 1 - (1 - r2_val) * (n_samples - 1) / denom
```

---

## 2) Robust and Scale-Sensitive Metrics

### Median Absolute Error (MedAE)
- What: Median(|y - ŷ|).
- Intuition: Very robust to outliers. Represents the “typical” error of the median case.
- Use when: Strong outliers or heavy-tailed error distributions.

scikit-learn:
```python
from sklearn.metrics import median_absolute_error
medae_val = median_absolute_error(y_test, y_pred)
```

---

### MAPE (Mean Absolute Percentage Error)
- What: Average of |(y - ŷ) / y| expressed in %.
- Intuition: Percent-based error; scale-free.
- Pitfalls:
  - Undefined/infinite near y ≈ 0; bias towards over-forecasting.
- Use when: Targets are strictly positive and not near zero.

Safe implementation:
```python
def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100
```

---

### sMAPE (Symmetric MAPE)
- What: 2|y - ŷ| / (|y| + |ŷ|).
- Intuition: Reduces asymmetry of MAPE; still issues when both near zero.

From scratch:
```python
def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)), eps)
    return np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100
```

---

### MSLE and RMSLE (Log-Scaled)
- What: MSE between log(1 + y) and log(1 + ŷ); RMSLE = sqrt(MSLE).
- Intuition: Penalizes underestimates more than overestimates; handles multiplicative errors and skew.
- Use when: Targets are positive and vary across orders of magnitude (e.g., sales, counts).

scikit-learn:
```python
from sklearn.metrics import mean_squared_log_error
msle_val = mean_squared_log_error(y_test, y_pred)         # beware negative targets
rmsle_val = msle_val ** 0.5
```

---

### Explained Variance
- What: 1 - Var(y - ŷ) / Var(y).
- Intuition: Similar to R² but uses variance of residuals; less sensitive to bias in predictions.

scikit-learn:
```python
from sklearn.metrics import explained_variance_score
evs = explained_variance_score(y_test, y_pred)
```

---

## 3) Advanced / Alternative Losses

### Huber Loss (and Huber Regressor)
- Intuition: Quadratic for small errors, linear for large errors—robust to outliers but smooth near zero.
- Use when: You want a compromise between MAE (robust) and MSE (smooth gradients).

scikit-learn model:
```python
from sklearn.linear_model import HuberRegressor
hub = HuberRegressor().fit(X_train, y_train)
y_pred = hub.predict(X_test)
```

---

### Quantile Loss (Pinball)
- Intuition: Optimize conditional quantiles (e.g., 90th percentile). Useful for prediction intervals and asymmetric costs.
- Loss for quantile q: max(q*(y - ŷ), (q - 1)*(y - ŷ)).

scikit-learn:
```python
from sklearn.linear_model import QuantileRegressor
q90 = QuantileRegressor(quantile=0.9, alpha=0.0).fit(X_train, y_train)
y_p90 = q90.predict(X_test)
```

---

### Poisson/Gamma Deviance (GLMs for positive targets)
- Intuition: For counts (Poisson) or strictly positive continuous (Gamma); models multiplicative noise.

scikit-learn (metrics):
```python
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
mpd = mean_poisson_deviance(y_test, y_pred_poisson)    # y_pred must be positive
mgd = mean_gamma_deviance(y_test, y_pred_gamma)        # y_pred must be positive
```

---

## 4) Weighted, Multi-Output, and Time-Series

### Sample weights
- Use when some observations are more important or to correct sampling bias.

scikit-learn:
```python
from sklearn.metrics import mean_absolute_error
w = np.linspace(1, 2, len(y_test))  # example weights
mae_w = mean_absolute_error(y_test, y_pred, sample_weight=w)
```

Many sklearn regressors accept `sample_weight` in `.fit()`.

---

### Multi-output regression
- Metrics can be averaged or returned per output.

scikit-learn:
```python
from sklearn.metrics import mean_squared_error
# raw values per output
mse_per_target = mean_squared_error(Y_test, Y_pred, multioutput='raw_values')
# uniform or weighted average:
mse_avg = mean_squared_error(Y_test, Y_pred, multioutput='uniform_average')
```

---

### Time-series considerations
- Always use chronological splits (no random shuffle).
- Evaluate with `TimeSeriesSplit`; metrics remain the same.

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.linear_model import Ridge

tscv = TimeSeriesSplit(n_splits=5)
scorer = make_scorer(mean_absolute_error, greater_is_better=False)
model = Ridge()

scores = cross_val_score(model, X, y, scoring=scorer, cv=tscv)
mae_cv = -scores.mean()
```

---

## 5) Confidence Intervals via Bootstrap

Bootstrap for metric uncertainty:
```python
rng = np.random.default_rng(42)

def bootstrap_metric(y_true, y_pred, metric_fn, n_boot=2000, ci=0.95):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    stats = np.sort(stats)
    lo = np.percentile(stats, (1-ci)/2*100)
    hi = np.percentile(stats, (1+ci)/2*100)
    return np.mean(stats), (lo, hi)

# Example:
mean_mae, (lo, hi) = bootstrap_metric(y_test, y_pred, mae, n_boot=1000, ci=0.95)
print(f"MAE ~ {mean_mae:.3f} (95% CI {lo:.3f}, {hi:.3f})")
```

---

## 6) End-to-End: From Scratch and With scikit-learn

### Synthetic example and metrics (NumPy + sklearn)
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data
X, y = make_regression(n_samples=1500, n_features=10, noise=15.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Metrics (library)
mae_val = mean_absolute_error(y_test, y_pred)
mse_val = mean_squared_error(y_test, y_pred)
rmse_val = mse_val ** 0.5
r2_val = r2_score(y_test, y_pred)

# Metrics (from scratch)
def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def mse(y_true, y_pred): return np.mean((y_true - y_pred)**2)
def rmse(y_true, y_pred): return np.sqrt(mse(y_true, y_pred))
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

print(f"MAE:  {mae_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")
print(f"R2:   {r2_val:.3f}")
```

---

### Pipelines, scaling, encoding, and cross-validation
```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

# Example heterogeneous dataset
df = pd.DataFrame({
    "sqft": np.random.normal(1200, 300, 1000),
    "bed": np.random.randint(1, 5, 1000),
    "city": np.random.choice(["NY", "SF", "LA"], 1000),
    "price": np.random.normal(350000, 80000, 1000)
})

X = df.drop(columns=["price"])
y = df["price"]

num_cols = ["sqft", "bed"]
cat_cols = ["city"]

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
])

model = Ridge(alpha=1.0)

pipe = Pipeline([
    ("preproc", preproc),
    ("model", model),
])

scoring = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(lambda yt, yp: mean_squared_error(yt, yp, squared=False), greater_is_better=False),
    "R2": make_scorer(r2_score),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_res = cross_validate(pipe, X, y, scoring=scoring, cv=cv, return_train_score=False)

# Note MAE/RMSE are negative because greater_is_better=False. Negate to read.
mae_cv = -cv_res["test_MAE"].mean()
rmse_cv = -cv_res["test_RMSE"].mean()
r2_cv = cv_res["test_R2"].mean()

print(f"CV MAE:  {mae_cv:,.0f}")
print(f"CV RMSE: {rmse_cv:,.0f}")
print(f"CV R2:   {r2_cv:.3f}")
```

---

## 7) Visual Diagnostics

Residual analysis helps verify assumptions and find issues like heteroskedasticity, nonlinearity, or outliers.

```python
import matplotlib.pyplot as plt
import numpy as np

def residual_plots(y_true, y_pred):
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    # Predicted vs True
    axes[0].scatter(y_true, y_pred, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--')
    axes[0].set_title("Predicted vs True")
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Predicted")

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.4)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")

    # Residuals histogram
    axes[2].hist(residuals, bins=30, alpha=0.7)
    axes[2].set_title("Residuals Distribution")
    axes[2].set_xlabel("Residual")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

# Usage:
# residual_plots(y_test, y_pred)
```

---

## 8) Choosing the Right Metric

- Use MAE when:
  - Interpretability in the original units is key; robust to outliers.
- Use RMSE when:
  - Large errors should be penalized more; smooth optimization.
- Use RMSLE/MSLE when:
  - Targets are positive and multiplicative errors matter; underestimates are worse.
- Use MAPE/sMAPE when:
  - Stakeholders expect percent error, and y is not near zero.
- Use MedAE/Huber when:
  - Outliers exist but you want stability.
- Use Quantile loss when:
  - You need prediction intervals or asymmetric cost handling.
- Report uncertainty:
  - Add bootstrap confidence intervals and show residual diagnostics.

Quick reference:
- Typical “balanced” choice: MAE + RMSE + R².
- Skewed/positive targets: add RMSLE.
- With outliers: add MedAE or Huber.
- Business targets in %: add sMAPE/MAPE (with zero-handling).

---

## 9) Common Pitfalls

- Data leakage inflates metrics. Keep a held-out test set.
- Report training metrics only → misleading; always use validation/test.
- Using MAPE with zeros/near-zeros → infinite or unstable.
- Comparing raw MAE/RMSE across different datasets/units → not meaningful.
- Ignoring temporal order in time series → optimistic results.
- Not aligning preprocessing between train/test (fit scalers/encoders on train only).

---

## 10) Practical CLI and VS Code Tips (macOS)

- Create and open a Jupyter Notebook to iterate on metrics:
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip scikit-learn numpy pandas matplotlib
python -m pip install jupyterlab
jupyter lab
```

- In VS Code:
  - Use the Python extension.
  - Run cells inline and view plots in the “Plot Viewer”.
  - Use the Testing panel to run unit tests for custom metric functions.

---