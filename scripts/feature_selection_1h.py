#!/usr/bin/env python3
"""
Feature selection analysis for 1h XGBoost model.

Performs:
1. Baseline feature importance (gain-based)
2. Permutation importance on validation set
3. Sequential backward elimination to find optimal feature subset
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / "data/processed/features_hourly_full.parquet"
TARGET_PATH = PROJECT_ROOT / "data/processed/targets_hourly_1h.parquet"
RESULTS_PATH = PROJECT_ROOT / "results/model_comparison/feature_selection_1h.json"

# XGBoost parameters (reduced for memory efficiency)
XGBOOST_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 150,  # Reduced from 200
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": 1,  # Single thread to reduce memory
    "tree_method": "hist",  # Memory-efficient histogram method
}

# Data split dates
TRAIN_START = "2017-01-01"
TRAIN_END = "2020-12-31"
VAL_START = "2021-01-01"
VAL_END = "2022-12-31"

# Feature elimination settings
MIN_FEATURES = 5


def load_and_prepare_data():
    """Load features and targets, align, clean, and split by timestamp."""
    print("Loading data...")
    features = pd.read_parquet(FEATURES_PATH)
    targets = pd.read_parquet(TARGET_PATH)

    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")

    # Convert features to float32 to reduce memory
    features = features.astype(np.float32)

    # Align on timestamp
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]

    print(f"After alignment: {features.shape}")

    # Replace inf with NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    # Drop rows with any NaN
    mask = ~(features.isna().any(axis=1) | targets.isna())
    features = features[mask]
    targets = targets[mask]

    print(f"After cleaning: {features.shape}")

    # Split by timestamp
    train_mask = (features.index >= TRAIN_START) & (features.index <= TRAIN_END)
    val_mask = (features.index >= VAL_START) & (features.index <= VAL_END)

    X_train = features[train_mask]
    y_train = targets[train_mask]
    X_val = features[val_mask]
    y_val = targets[val_mask]

    print(f"\nTrain: {X_train.shape}, period: {X_train.index.min()} to {X_train.index.max()}")
    print(f"Val:   {X_val.shape}, period: {X_val.index.min()} to {X_val.index.max()}")
    print(f"Train target distribution: {y_train.value_counts().to_dict()}")
    print(f"Val target distribution: {y_val.value_counts().to_dict()}")

    return X_train, y_train, X_val, y_val


def train_and_evaluate(X_train, y_train, X_val, y_val, feature_subset=None):
    """Train XGBoost and return validation AUC."""
    if feature_subset is not None:
        X_train = X_train[feature_subset]
        X_val = X_val[feature_subset]

    # Convert to float32 to reduce memory
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_proba)

    return model, val_auc


def get_feature_importance_ranking(model, feature_names):
    """Get feature importance ranking based on gain."""
    importances = model.get_booster().get_score(importance_type="gain")

    # XGBoost uses f0, f1, ... as feature names internally
    # Map back to original feature names
    feature_importance = {}
    for i, fname in enumerate(feature_names):
        key = f"f{i}"
        feature_importance[fname] = importances.get(key, 0.0)

    # Sort by importance descending
    ranking = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    return ranking


def get_permutation_importance_ranking(model, X_val, y_val, feature_names):
    """Get permutation importance ranking on validation set."""
    print("\nComputing permutation importance (5 repeats for memory efficiency)...")

    # Convert to float32
    X_val = X_val.astype(np.float32)

    perm_result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=5,  # Reduced from 10
        random_state=42,
        n_jobs=1,  # Single thread
        scoring="roc_auc",
    )

    # Create ranking
    perm_importances = list(zip(feature_names, perm_result.importances_mean))
    perm_ranking = sorted(perm_importances, key=lambda x: x[1], reverse=True)

    return perm_ranking


def sequential_backward_elimination(X_train, y_train, X_val, y_val):
    """
    Sequential backward elimination: remove least important feature at each step.

    Returns:
        elimination_history: list of (n_features, val_auc, removed_feature)
    """
    print("\n" + "=" * 80)
    print("SEQUENTIAL BACKWARD ELIMINATION")
    print("=" * 80)

    current_features = list(X_train.columns)
    elimination_history = []

    # Baseline with all features
    print(f"\nBaseline: {len(current_features)} features")
    model, val_auc = train_and_evaluate(X_train, y_train, X_val, y_val, current_features)
    elimination_history.append((len(current_features), val_auc, "(baseline)"))
    print(f"Val AUC: {val_auc:.6f}")

    # Eliminate features one by one
    while len(current_features) > MIN_FEATURES:
        # Get importance of current features
        importances = model.get_booster().get_score(importance_type="gain")

        # Map to current feature names
        feature_importance = {}
        for i, fname in enumerate(current_features):
            key = f"f{i}"
            feature_importance[fname] = importances.get(key, 0.0)

        # Find least important feature
        least_important = min(feature_importance.items(), key=lambda x: x[1])[0]

        # Remove it
        current_features.remove(least_important)

        # Retrain and evaluate
        print(f"\nRemoving '{least_important}', {len(current_features)} features remaining")
        model, val_auc = train_and_evaluate(X_train, y_train, X_val, y_val, current_features)
        elimination_history.append((len(current_features), val_auc, least_important))
        print(f"Val AUC: {val_auc:.6f}")

    return elimination_history


def find_optimal_subset(elimination_history):
    """Find the feature count with highest validation AUC."""
    best_idx = max(range(len(elimination_history)), key=lambda i: elimination_history[i][1])
    best_n_features, best_val_auc, _ = elimination_history[best_idx]
    return best_n_features, best_val_auc


def get_optimal_feature_names(X_train, y_train, X_val, y_val, n_features):
    """Retrain with all features and keep top n_features by importance."""
    print(f"\nRetraining with all features to identify optimal {n_features} features...")
    model, _ = train_and_evaluate(X_train, y_train, X_val, y_val)

    ranking = get_feature_importance_ranking(model, X_train.columns.tolist())
    optimal_features = [fname for fname, _ in ranking[:n_features]]

    return optimal_features


def print_results(gain_ranking, perm_ranking, elimination_history, optimal_n_features, optimal_features):
    """Print formatted tables to console."""
    print("\n" + "=" * 100)
    print("FEATURE IMPORTANCE RANKING (Gain vs Permutation)")
    print("=" * 100)
    print(f"{'Rank':<6} {'Feature (Gain)':<30} {'Gain':<12} {'Feature (Perm)':<30} {'Perm Imp':<12}")
    print("-" * 100)

    max_rank = max(len(gain_ranking), len(perm_ranking))
    for i in range(max_rank):
        gain_feat, gain_val = gain_ranking[i] if i < len(gain_ranking) else ("", 0.0)
        perm_feat, perm_val = perm_ranking[i] if i < len(perm_ranking) else ("", 0.0)
        print(f"{i + 1:<6} {gain_feat:<30} {gain_val:<12.6f} {perm_feat:<30} {perm_val:<12.6f}")

    print("\n" + "=" * 80)
    print("BACKWARD ELIMINATION")
    print("=" * 80)
    print(f"{'N Features':<12} {'Val AUC':<12} {'Removed Feature':<40}")
    print("-" * 80)

    for n_features, val_auc, removed in elimination_history:
        print(f"{n_features:<12} {val_auc:<12.6f} {removed:<40}")

    print("\n" + "=" * 80)
    print(
        f"OPTIMAL: N={optimal_n_features} features, Val AUC={elimination_history[[h[0] for h in elimination_history].index(optimal_n_features)][1]:.6f}"
    )
    print("=" * 80)
    print("Optimal feature subset:")
    for i, feat in enumerate(optimal_features, 1):
        print(f"  {i}. {feat}")

    # Check if removing features improved AUC
    baseline_auc = elimination_history[0][1]
    best_auc = elimination_history[[h[0] for h in elimination_history].index(optimal_n_features)][1]

    print("\n" + "=" * 80)
    if best_auc > baseline_auc:
        improvement = best_auc - baseline_auc
        print(f"✓ Removing features IMPROVED AUC by {improvement:.6f}")
        print("  This suggests the full feature set may be overfitting.")
    else:
        degradation = baseline_auc - best_auc
        print(f"✗ Removing features DEGRADED AUC by {degradation:.6f}")
        print("  The full feature set performs best on validation.")
    print("=" * 80)


def save_results(gain_ranking, perm_ranking, elimination_history, optimal_n_features, optimal_features):
    """Save results to JSON file."""
    results = {
        "feature_importance_gain": [{"feature": feat, "importance": float(imp)} for feat, imp in gain_ranking],
        "feature_importance_permutation": [{"feature": feat, "importance": float(imp)} for feat, imp in perm_ranking],
        "elimination_history": [
            {"n_features": n_feat, "val_auc": float(auc), "removed_feature": removed}
            for n_feat, auc, removed in elimination_history
        ],
        "optimal_n_features": optimal_n_features,
        "optimal_features": optimal_features,
        "baseline_val_auc": float(elimination_history[0][1]),
        "optimal_val_auc": float(elimination_history[[h[0] for h in elimination_history].index(optimal_n_features)][1]),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")


def main():
    print("=" * 80)
    print("FEATURE SELECTION ANALYSIS: 1H XGBOOST MODEL")
    print("=" * 80)

    # Load and prepare data
    X_train, y_train, X_val, y_val = load_and_prepare_data()

    # Train baseline model
    print("\n" + "=" * 80)
    print("BASELINE MODEL (ALL FEATURES)")
    print("=" * 80)
    baseline_model, baseline_val_auc = train_and_evaluate(X_train, y_train, X_val, y_val)
    print(f"Val AUC: {baseline_val_auc:.6f}")

    # Feature importance rankings
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE (GAIN-BASED)")
    print("=" * 80)
    gain_ranking = get_feature_importance_ranking(baseline_model, X_train.columns.tolist())
    for i, (feat, imp) in enumerate(gain_ranking[:10], 1):
        print(f"{i:2d}. {feat:30s} {imp:.6f}")
    print(f"... (showing top 10 of {len(gain_ranking)})")

    # Permutation importance
    perm_ranking = get_permutation_importance_ranking(baseline_model, X_val, y_val, X_train.columns.tolist())
    print("\nTop 10 by permutation importance:")
    for i, (feat, imp) in enumerate(perm_ranking[:10], 1):
        print(f"{i:2d}. {feat:30s} {imp:.6f}")

    # Sequential backward elimination
    elimination_history = sequential_backward_elimination(X_train, y_train, X_val, y_val)

    # Find optimal subset
    optimal_n_features, optimal_val_auc = find_optimal_subset(elimination_history)
    optimal_features = get_optimal_feature_names(X_train, y_train, X_val, y_val, optimal_n_features)

    # Print results
    print_results(gain_ranking, perm_ranking, elimination_history, optimal_n_features, optimal_features)

    # Save results
    save_results(gain_ranking, perm_ranking, elimination_history, optimal_n_features, optimal_features)

    print("\n✓ Feature selection analysis complete!")


if __name__ == "__main__":
    main()
