"""
Regenerates a handful of key preview charts from the notebook analysis
and saves them to outputs/ as PNGs, for use in the README.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set_theme(style="whitegrid")
OUT = "outputs"

# --- Load & clean (mirrors notebook Part 1) ---
df = pd.read_csv("data/diamonds.csv")
cleaned_df = df[
    (df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0) &
    (df['y'] < 30) & (df['z'] < 30)
].copy()

cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

cleaned_df['cut'] = cleaned_df['cut'].map(cut_mapping)
cleaned_df['color'] = cleaned_df['color'].map(color_mapping)
cleaned_df['clarity'] = cleaned_df['clarity'].map(clarity_mapping)

# --- 1. Correlation heatmap ---
plt.figure(figsize=(9, 7))
correlation = cleaned_df.select_dtypes(include='number').corr()
sns.heatmap(correlation, center=0, square=True, annot=True, fmt=".2f", cmap="RdBu_r")
plt.title("Correlation Heatmap — Diamond Features vs Price", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=150)
plt.close()

# --- 2. Price & carat distributions ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(cleaned_df['carat'], bins=40, ax=axes[0], color="#5B8FF9")
axes[0].set_title("Carat Distribution (Right-Skewed)")
sns.histplot(cleaned_df['price'], bins=40, ax=axes[1], color="#61DDAA")
axes[1].set_title("Price Distribution (Right-Skewed)")
plt.tight_layout()
plt.savefig(f"{OUT}/distributions.png", dpi=150)
plt.close()

# --- 3. Baseline OLS residuals (raw price scale) ---
X = cleaned_df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
y = cleaned_df['price']
X_const = sm.add_constant(X)
baseline_model = sm.OLS(y, X_const).fit()

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(baseline_model.fittedvalues, baseline_model.resid, alpha=0.15, s=8, color="#F6903D")
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals")
ax.set_title("Baseline Model Residuals — Heteroscedasticity Pattern", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/baseline_residuals.png", dpi=150)
plt.close()

# --- 4. Log-log model: actual vs predicted (best model) ---
log_X = X[['carat']].apply(np.log).rename(columns={'carat': 'log_carat'})
log_X = pd.concat([log_X, X[['cut', 'color', 'clarity', 'depth', 'table']]], axis=1)
log_X_const = sm.add_constant(log_X)
log_y = np.log(y)
log_model = sm.OLS(log_y, log_X_const).fit()

pred_log = log_model.fittedvalues
correction = np.mean(np.exp(log_model.resid))
pred_price = np.exp(pred_log) * correction

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(y, pred_price, alpha=0.15, s=8, color="#5AD8A6")
lims = [0, y.max()]
ax.plot(lims, lims, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Actual Price ($)")
ax.set_ylabel("Predicted Price ($)")
ax.set_title(f"Log-Log Model: Actual vs Predicted (R²={log_model.rsquared:.3f})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/logmodel_actual_vs_predicted.png", dpi=150)
plt.close()

print("Saved 4 preview charts to outputs/")
