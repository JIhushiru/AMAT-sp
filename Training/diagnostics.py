"""
Statistical diagnostics: OLS regression, VIF scores, multicollinearity checks.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_ols(X, y, alpha=0.05):
    """Perform OLS regression and print summary and significant predictors."""
    X_ols = sm.add_constant(X)
    model = sm.OLS(y, X_ols).fit()
    print("\n--- OLS Regression Summary ---")
    print(model.summary())

    print(f"\n--- Predictors Significant at alpha = {alpha} ---")
    sig = model.pvalues[model.pvalues < alpha]
    print(sig)

    return model


def calculate_vif(X):
    """Calculate VIF for each feature to detect multicollinearity."""
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print("\n--- VIF Scores ---")
    print(vif_data)
    return vif_data


def check_multicollinearity(X, vif_threshold=10, corr_threshold=0.95):
    """Check multicollinearity via VIF and correlation matrix."""
    vif_df = pd.DataFrame()
    vif_df['Feature'] = X.columns
    vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    high_vif = vif_df[vif_df['VIF'] > vif_threshold]

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(col, row, upper.loc[row, col])
                 for col in upper.columns
                 for row in upper.index
                 if upper.loc[row, col] > corr_threshold]

    if not high_vif.empty:
        print(f"\nFeatures with VIF > {vif_threshold}:\n{high_vif}")
    else:
        print(f"\nNo features exceed VIF threshold of {vif_threshold}")

    if high_corr:
        print(f"\nHighly correlated feature pairs (corr > {corr_threshold}):")
        for a, b, val in high_corr:
            print(f"   {a} & {b} -> correlation = {val:.3f}")
    else:
        print(f"\nNo feature pairs exceed correlation threshold of {corr_threshold}")

    return high_vif, high_corr
