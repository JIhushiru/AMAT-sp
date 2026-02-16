"""
Feature selection: VIF filtering + Boruta selection + temperature rule.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from statsmodels.stats.outliers_influence import variance_inflation_factor


def feature_selection_workflow(X, y, vif_threshold=5.0):
    """Standardize, check VIF, then apply Boruta for feature selection."""
    print("Starting feature selection workflow...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    print(f"Features standardized: {X_scaled_df.shape[1]} features")

    filtered_features, removed_by_vif = drop_high_vif_features(
        X_scaled_df, thresh=vif_threshold, protected_features=["tmp"]
    )
    X_filtered = X_scaled_df[filtered_features]
    print(f"---- After VIF filtering: {len(filtered_features)} features remaining")

    model = RandomForestRegressor(n_jobs=-1, random_state=42)
    selector = BorutaPy(model, n_estimators=1000, max_iter=50, perc=90, alpha=0.05, two_step=False, verbose=0)

    y_np = y.to_numpy() if isinstance(y, pd.Series) else np.array(y)
    selector.fit(X_filtered.values, y_np)

    boruta_selected = X_filtered.columns[selector.support_].tolist()
    boruta_rejected = X_filtered.columns[~selector.support_].tolist()
    print(f"---- After Boruta: {len(boruta_selected)} features selected")

    final_features = temperature_rule(boruta_selected)
    print("Selected features:", final_features)

    return {
        'selected_features': final_features,
        'removed_by_vif': removed_by_vif,
        'rejected_by_boruta': boruta_rejected,
        'original_count': X.shape[1],
        'final_count': len(final_features)
    }


def drop_high_vif_features(X, thresh=5.0, protected_features=None):
    """Drop features with high VIF iteratively, preserving protected features."""
    if protected_features is None:
        protected_features = []

    X_copy = X.copy()
    removed_features = []

    while True:
        vif = pd.DataFrame()
        vif["feature"] = X_copy.columns
        vif["VIF"] = [variance_inflation_factor(X_copy.values, i) for i in range(X_copy.shape[1])]

        candidates = vif[~vif["feature"].isin(protected_features)]

        if candidates.empty:
            break

        max_vif = candidates["VIF"].max()
        if max_vif > thresh:
            drop_feature = candidates.loc[candidates["VIF"].idxmax(), "feature"]
            removed_features.append({'feature': drop_feature, 'vif': max_vif, 'reason': 'High VIF'})
            print(f"Dropping '{drop_feature}' (VIF = {max_vif:.2f})")
            X_copy = X_copy.drop(columns=[drop_feature])
        else:
            break

    final_vif = pd.DataFrame()
    final_vif["feature"] = X_copy.columns
    final_vif["VIF"] = [variance_inflation_factor(X_copy.values, i) for i in range(X_copy.shape[1])]
    print("\nVIF scores after dropping high VIF features:")
    print(final_vif)

    if removed_features:
        print(f"\nRemoved {len(removed_features)} features due to high VIF:")
        for item in removed_features:
            print(f"  - {item['feature']}: VIF = {item['vif']:.2f}")

    return X_copy.columns.tolist(), removed_features


def temperature_rule(selected_features):
    """Force tmp to replace individual temperature indicators (tmn, tmx, dtr)."""
    temp_indicators = ['tmn', 'tmx', 'dtr']
    removed = [feature for feature in temp_indicators if feature in selected_features]

    selected_features = [f for f in selected_features if f not in temp_indicators]

    if removed and 'tmp' not in selected_features:
        print("Removed:", removed, "| Added: tmp")
        selected_features.append('tmp')

    return selected_features
