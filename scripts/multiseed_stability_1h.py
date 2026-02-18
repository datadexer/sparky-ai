#!/usr/bin/env python3
"""
Multi-seed stability test for 1h XGBoost model.

Tests robustness of signal across different random seeds.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from sparky.data.loader import load


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load features and target, align, clean."""
    print("Loading data...")
    X = load("features_hourly_full", purpose="training")
    y_df = load("targets_hourly_1h", purpose="training")

    # Ensure datetime index
    if not isinstance(X.index, pd.DatetimeIndex):
        X.index = pd.to_datetime(X.index)
    if not isinstance(y_df.index, pd.DatetimeIndex):
        y_df.index = pd.to_datetime(y_df.index)

    # Extract target column
    if "target_1h" in y_df.columns:
        y = y_df["target_1h"]
    elif "target" in y_df.columns:
        y = y_df["target"]
    else:
        y = y_df.iloc[:, 0]

    # Align
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    print(f"Initial shape: X={X.shape}, y={y.shape}")

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    print(f"After cleaning: X={X.shape}, y={y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


def split_by_timestamp(X: pd.DataFrame, y: pd.Series) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Split data by timestamp."""
    print("\nSplitting by timestamp...")

    train_mask = (X.index >= "2017-01-01") & (X.index < "2021-01-01")
    val_mask = (X.index >= "2021-01-01") & (X.index < "2023-01-01")
    test_mask = (X.index >= "2023-01-01") & (X.index < "2024-01-01")

    splits = {
        "train": (X[train_mask], y[train_mask]),
        "val": (X[val_mask], y[val_mask]),
        "test": (X[test_mask], y[test_mask]),
    }

    for name, (X_split, y_split) in splits.items():
        print(f"{name:5s}: {len(X_split):6d} samples, pos={y_split.sum()}/{len(y_split)} ({100 * y_split.mean():.1f}%)")

    return splits


def train_and_evaluate_seed(
    seed: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """Train XGBoost with given seed and evaluate."""
    print(f"\n{'=' * 60}")
    print(f"Training with seed={seed}")
    print(f"{'=' * 60}")

    params = dict(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=2,
        random_state=seed,
    )

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    # Validation metrics
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)

    # Test metrics
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    top5_features = importances.head(5).to_dict()

    print(f"Val  metrics: acc={val_acc:.4f}, auc={val_auc:.4f}")
    print(f"Test metrics: acc={test_acc:.4f}, auc={test_auc:.4f}")
    print("Top-5 features:")
    for feat, imp in list(top5_features.items())[:5]:
        print(f"  {feat:30s}: {imp:.4f}")

    return {
        "seed": seed,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "feature_importances": importances.to_dict(),
        "top5_features": top5_features,
    }


def compute_stability_metrics(results: List[Dict]) -> Dict:
    """Compute stability statistics across seeds."""
    print(f"\n{'=' * 60}")
    print("STABILITY ANALYSIS")
    print(f"{'=' * 60}")

    # Extract metrics
    val_aucs = [r["val_auc"] for r in results]
    test_aucs = [r["test_auc"] for r in results]
    val_accs = [r["val_acc"] for r in results]
    test_accs = [r["test_acc"] for r in results]

    # Compute statistics
    stats = {
        "val_auc_mean": float(np.mean(val_aucs)),
        "val_auc_std": float(np.std(val_aucs, ddof=1)),
        "val_auc_min": float(np.min(val_aucs)),
        "val_auc_max": float(np.max(val_aucs)),
        "test_auc_mean": float(np.mean(test_aucs)),
        "test_auc_std": float(np.std(test_aucs, ddof=1)),
        "test_auc_min": float(np.min(test_aucs)),
        "test_auc_max": float(np.max(test_aucs)),
        "val_acc_mean": float(np.mean(val_accs)),
        "val_acc_std": float(np.std(val_accs, ddof=1)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs, ddof=1)),
    }

    # Feature importance stability (Kendall's tau)
    feature_rankings = []
    for r in results:
        # Sort features by importance for this seed
        sorted_features = sorted(r["feature_importances"].items(), key=lambda x: x[1], reverse=True)
        feature_rankings.append([f[0] for f in sorted_features])

    # Compute pairwise Kendall's tau
    taus = []
    for i in range(len(feature_rankings)):
        for j in range(i + 1, len(feature_rankings)):
            # Create ranking indices
            all_features = feature_rankings[0]  # Use first seed's feature order as reference
            rank_i = {f: idx for idx, f in enumerate(feature_rankings[i])}
            rank_j = {f: idx for idx, f in enumerate(feature_rankings[j])}

            ranks_i = [rank_i[f] for f in all_features]
            ranks_j = [rank_j[f] for f in all_features]

            tau, _ = kendalltau(ranks_i, ranks_j)
            taus.append(tau)

    stats["feature_importance_kendall_tau_mean"] = float(np.mean(taus))
    stats["feature_importance_kendall_tau_std"] = float(np.std(taus, ddof=1))
    stats["feature_importance_kendall_tau_min"] = float(np.min(taus))

    # Top-3 feature consistency
    top3_counts = {}
    for r in results:
        top3 = list(r["top5_features"].keys())[:3]
        for feat in top3:
            top3_counts[feat] = top3_counts.get(feat, 0) + 1

    stats["top3_feature_counts"] = top3_counts

    return stats


def check_pass_criteria(stats: Dict, results: List[Dict]) -> Dict[str, bool]:
    """Check if stability criteria are met."""
    print(f"\n{'=' * 60}")
    print("PASS/FAIL CRITERIA")
    print(f"{'=' * 60}")

    criteria = {}

    # Val AUC std < 0.005
    val_auc_stable = stats["val_auc_std"] < 0.005
    criteria["val_auc_stable"] = val_auc_stable
    print(f"Val AUC std < 0.005: {stats['val_auc_std']:.6f} -> {'PASS' if val_auc_stable else 'FAIL'}")

    # Test AUC std < 0.01
    test_auc_stable = stats["test_auc_std"] < 0.01
    criteria["test_auc_stable"] = test_auc_stable
    print(f"Test AUC std < 0.01: {stats['test_auc_std']:.6f} -> {'PASS' if test_auc_stable else 'FAIL'}")

    # Top-3 features consistent
    top3_consistent = all(count >= 3 for count in list(stats["top3_feature_counts"].values())[:3])
    criteria["top3_consistent"] = top3_consistent
    print(f"Top-3 features appear in >=3 seeds: {'PASS' if top3_consistent else 'FAIL'}")
    print(f"  Top-3 feature counts: {stats['top3_feature_counts']}")

    # Mean val AUC matches single-seed (~0.555) within 0.01
    expected_auc = 0.555
    val_auc_matches = abs(stats["val_auc_mean"] - expected_auc) < 0.01
    criteria["val_auc_matches_baseline"] = val_auc_matches
    print(
        f"Mean val AUC ~{expected_auc} (±0.01): {stats['val_auc_mean']:.4f} -> {'PASS' if val_auc_matches else 'FAIL'}"
    )

    # Overall pass
    all_pass = all(criteria.values())
    criteria["overall"] = all_pass

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'=' * 60}")

    return criteria


def print_results_table(results: List[Dict], stats: Dict):
    """Print formatted results table."""
    print(f"\n{'=' * 60}")
    print("RESULTS TABLE")
    print(f"{'=' * 60}")
    print(f"{'Seed':<8} {'Val AUC':<10} {'Test AUC':<10} {'Top Feature':<30}")
    print("-" * 60)

    for r in results:
        top_feat = list(r["top5_features"].keys())[0]
        print(f"{r['seed']:<8} {r['val_auc']:<10.4f} {r['test_auc']:<10.4f} {top_feat:<30}")

    print("-" * 60)
    mean_val = stats["val_auc_mean"]
    std_val = stats["val_auc_std"]
    mean_test = stats["test_auc_mean"]
    std_test = stats["test_auc_std"]
    print(f"{'Mean':<8} {mean_val:.4f}±{std_val:.4f}  {mean_test:.4f}±{std_test:.4f}")
    print("=" * 60)


def main():
    """Main execution."""
    print("=" * 60)
    print("Multi-Seed Stability Test for 1h XGBoost Model")
    print("=" * 60)

    # Load and prepare data
    X, y = load_and_prepare_data()
    splits = split_by_timestamp(X, y)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # Train with multiple seeds
    seeds = [42, 123, 456, 789, 1337]
    results = []

    for seed in seeds:
        result = train_and_evaluate_seed(seed, X_train, y_train, X_val, y_val, X_test, y_test)
        results.append(result)

    # Compute stability metrics
    stats = compute_stability_metrics(results)

    # Check pass criteria
    criteria = check_pass_criteria(stats, results)

    # Print results table
    print_results_table(results, stats)

    # Save results
    output = {
        "seeds": seeds,
        "results": results,
        "statistics": stats,
        "pass_criteria": criteria,
    }

    output_path = Path("results/model_comparison/multiseed_1h_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0 if criteria["overall"] else 1


if __name__ == "__main__":
    exit(main())
