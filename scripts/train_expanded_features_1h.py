#!/usr/bin/env python3
"""
Train CatBoost and XGBoost on EXPANDED feature set (base + macro + onchain) for 1h BTC prediction.

Compares:
1. CatBoost with base features only (baseline)
2. CatBoost with all features (expanded)
3. XGBoost with all features (cross-model comparison)

Key question: Do macro/on-chain features improve beyond the base 23 features?
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss
from catboost import CatBoostClassifier

from sparky.models.xgboost_model import XGBoostModel
from sparky.backtest.leakage_detector import LeakageDetector


def load_and_join_features():
    """Load all three feature sets and join them."""
    print("=" * 80)
    print("LOADING FEATURE SETS")
    print("=" * 80)

    # Load all feature sets
    features_base = pd.read_parquet("data/processed/features_hourly_full.parquet")
    features_macro = pd.read_parquet("data/processed/macro_features_hourly.parquet")
    features_onchain = pd.read_parquet("data/processed/onchain_features_hourly.parquet")
    targets = pd.read_parquet("data/processed/targets_hourly_1h.parquet")

    print(f"Base features: {features_base.shape} ({len(features_base.columns)} columns)")
    print(f"Macro features: {features_macro.shape} ({len(features_macro.columns)} columns)")
    print(f"Onchain features: {features_onchain.shape} ({len(features_onchain.columns)} columns)")
    print(f"Targets: {targets.shape}")
    print()

    # Join all features (inner join to handle NaN alignment)
    print("Joining feature sets...")
    features_all = features_base.join(features_macro, how='inner').join(features_onchain, how='inner')
    print(f"Combined features (after inner join): {features_all.shape}")

    # Drop any remaining NaN rows
    print(f"NaN rows before dropna: {features_all.isna().any(axis=1).sum()}")
    features_all = features_all.dropna()
    print(f"Features after dropna: {features_all.shape}")

    # Align targets
    targets = targets.loc[features_all.index]
    print(f"Targets aligned: {targets.shape}")
    print(f"Targets NaN: {targets.isna().sum().sum()}")
    targets = targets.dropna()

    # Final alignment
    common_idx = features_all.index.intersection(targets.index)
    features_all = features_all.loc[common_idx]
    targets = targets.loc[common_idx]

    # Clean inf values (replace with NaN, then drop)
    print(f"Checking for inf values...")
    inf_mask = np.isinf(features_all).any(axis=1)
    print(f"Rows with inf: {inf_mask.sum()}")
    features_all = features_all[~inf_mask]
    targets = targets.loc[features_all.index]

    # Final NaN check
    nan_mask = features_all.isna().any(axis=1)
    print(f"Rows with NaN after inf removal: {nan_mask.sum()}")
    features_all = features_all[~nan_mask]
    targets = targets.loc[features_all.index]

    print(f"\nFinal cleaned data: {features_all.shape} features, {targets.shape} targets")
    print(f"Date range: {features_all.index.min()} to {features_all.index.max()}")
    print(f"Total columns: {len(features_all.columns)}")
    print()

    return features_all, features_base.loc[features_all.index], targets


def split_data(features, targets):
    """Split into train/val/test by date."""
    print("=" * 80)
    print("SPLITTING DATA")
    print("=" * 80)

    # Define split dates
    train_start = "2017-01-01"
    train_end = "2020-12-31"
    val_start = "2021-01-01"
    val_end = "2022-12-31"
    test_start = "2023-01-01"
    test_end = "2023-12-31"

    # Split data
    train_mask = (features.index >= train_start) & (features.index <= train_end)
    val_mask = (features.index >= val_start) & (features.index <= val_end)
    test_mask = (features.index >= test_start) & (features.index <= test_end)

    X_train = features.loc[train_mask]
    y_train = targets.loc[train_mask].values.ravel()

    X_val = features.loc[val_mask]
    y_val = targets.loc[val_mask].values.ravel()

    X_test = features.loc[test_mask]
    y_test = targets.loc[test_mask].values.ravel()

    print(f"Train: {X_train.shape} ({train_start} to {train_end})")
    print(f"  Positive rate: {y_train.mean():.3f}")
    print(f"Val: {X_val.shape} ({val_start} to {val_end})")
    print(f"  Positive rate: {y_val.mean():.3f}")
    print(f"Test: {X_test.shape} ({test_start} to {test_end})")
    print(f"  Positive rate: {y_test.mean():.3f}")
    print()

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    """Train and evaluate a model."""
    print("=" * 80)
    print(f"TRAINING: {model_name}")
    print("=" * 80)

    # Train
    print("Fitting model...")
    if isinstance(model, CatBoostClassifier):
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    else:
        model.fit(X_train, y_train)

    # Predictions
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    results = {
        "model": model_name,
        "features": X_train.shape[1],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "val_auc": float(roc_auc_score(y_val, y_val_proba)),
        "val_precision": float(precision_score(y_val, y_val_pred, zero_division=0)),
        "val_recall": float(recall_score(y_val, y_val_pred, zero_division=0)),
        "val_f1": float(f1_score(y_val, y_val_pred, zero_division=0)),
        "val_logloss": float(log_loss(y_val, y_val_proba)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_auc": float(roc_auc_score(y_test, y_test_proba)),
        "test_precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "test_logloss": float(log_loss(y_test, y_test_proba)),
    }

    # Feature importance
    if isinstance(model, CatBoostClassifier):
        importance = model.get_feature_importance()
        feature_names = X_train.columns.tolist()
    else:
        importance = model.model.feature_importances_
        feature_names = X_train.columns.tolist()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    results['feature_importance_top15'] = importance_df.head(15).to_dict('records')

    # Print results
    print(f"\nValidation Results:")
    print(f"  Accuracy: {results['val_accuracy']:.4f}")
    print(f"  AUC: {results['val_auc']:.4f}")
    print(f"  Precision: {results['val_precision']:.4f}")
    print(f"  Recall: {results['val_recall']:.4f}")
    print(f"  F1: {results['val_f1']:.4f}")
    print(f"  Log Loss: {results['val_logloss']:.4f}")

    print(f"\nTest Results:")
    print(f"  Accuracy: {results['test_accuracy']:.4f}")
    print(f"  AUC: {results['test_auc']:.4f}")
    print(f"  Precision: {results['test_precision']:.4f}")
    print(f"  Recall: {results['test_recall']:.4f}")
    print(f"  F1: {results['test_f1']:.4f}")
    print(f"  Log Loss: {results['test_logloss']:.4f}")

    print(f"\nTop 15 Features:")
    for i, row in enumerate(importance_df.head(15).itertuples(), 1):
        print(f"  {i:2d}. {row.feature:30s} {row.importance:10.4f}")
    print()

    return results, model


def run_leakage_detection(model, X_train, y_train, X_val, y_val, model_name):
    """Run leakage detection on a model."""
    print("=" * 80)
    print(f"LEAKAGE DETECTION: {model_name}")
    print("=" * 80)

    detector = LeakageDetector(n_shuffle_trials=5)
    result = detector.detect(model, X_train, y_train, X_val, y_val)

    print(f"Leakage detected: {result.leakage_detected}")
    print(f"Real AUC: {result.real_auc:.4f}")
    print(f"Random mean AUC: {result.random_auc_mean:.4f}")
    print(f"Random std AUC: {result.random_auc_std:.4f}")
    print(f"P-value: {result.p_value:.4f}")
    print()

    return {
        "leakage_detected": result.leakage_detected,
        "real_auc": float(result.real_auc),
        "random_auc_mean": float(result.random_auc_mean),
        "random_auc_std": float(result.random_auc_std),
        "p_value": float(result.p_value),
    }


def print_comparison_table(results_list):
    """Print a comparison table of all models."""
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Table header
    print(f"{'Model':<30s} {'Features':>10s} {'Val AUC':>10s} {'Test AUC':>10s} {'Delta vs Base':>15s}")
    print("-" * 80)

    # Baseline AUC
    baseline_val_auc = None
    baseline_test_auc = None
    for r in results_list:
        if "base" in r["model"].lower():
            baseline_val_auc = r["val_auc"]
            baseline_test_auc = r["test_auc"]
            break

    # Print rows
    for r in results_list:
        model_name = r["model"]
        features = r["features"]
        val_auc = r["val_auc"]
        test_auc = r["test_auc"]

        if baseline_val_auc is not None:
            delta_val = val_auc - baseline_val_auc
            delta_test = test_auc - baseline_test_auc
            delta_str = f"+{delta_test:.4f}" if delta_test >= 0 else f"{delta_test:.4f}"
        else:
            delta_str = "BASELINE"

        print(f"{model_name:<30s} {features:>10d} {val_auc:>10.4f} {test_auc:>10.4f} {delta_str:>15s}")

    print()


def main():
    """Main training pipeline."""
    # Create results directory
    results_dir = Path("results/expanded_features")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load and join features
    features_all, features_base, targets = load_and_join_features()

    # Split data for all features
    X_train_all, y_train, X_val_all, y_val, X_test_all, y_test = split_data(features_all, targets)

    # Split data for base features only (same samples, different features)
    X_train_base = features_base.loc[X_train_all.index]
    X_val_base = features_base.loc[X_val_all.index]
    X_test_base = features_base.loc[X_test_all.index]

    all_results = []

    # ========================================
    # 1. CatBoost with base features only
    # ========================================
    catboost_base = CatBoostClassifier(
        depth=5,
        learning_rate=0.05,
        iterations=200,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        subsample=0.8,
        rsm=0.8,
        task_type="GPU",
        devices="0",
    )

    results_base, model_base = evaluate_model(
        catboost_base, X_train_base, y_train, X_val_base, y_val, X_test_base, y_test,
        "CatBoost (base features)"
    )
    all_results.append(results_base)

    # Save results
    with open(results_dir / "catboost_base_results.json", "w") as f:
        json.dump(results_base, f, indent=2)

    # ========================================
    # 2. CatBoost with all features
    # ========================================
    catboost_expanded = CatBoostClassifier(
        depth=5,
        learning_rate=0.05,
        iterations=200,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        subsample=0.8,
        rsm=0.8,
        task_type="GPU",
        devices="0",
    )

    results_catboost, model_catboost = evaluate_model(
        catboost_expanded, X_train_all, y_train, X_val_all, y_val, X_test_all, y_test,
        "CatBoost (expanded features)"
    )
    all_results.append(results_catboost)

    # Save results
    with open(results_dir / "catboost_expanded_results.json", "w") as f:
        json.dump(results_catboost, f, indent=2)

    # ========================================
    # 3. XGBoost with all features
    # ========================================
    xgboost_expanded = XGBoostModel(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42
    )

    results_xgboost, model_xgboost = evaluate_model(
        xgboost_expanded, X_train_all, y_train, X_val_all, y_val, X_test_all, y_test,
        "XGBoost (expanded features)"
    )
    all_results.append(results_xgboost)

    # Save results
    with open(results_dir / "xgboost_expanded_results.json", "w") as f:
        json.dump(results_xgboost, f, indent=2)

    # ========================================
    # Comparison table
    # ========================================
    print_comparison_table(all_results)

    # ========================================
    # Feature importance comparison
    # ========================================
    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    print("\nCatBoost (base features) - Top 10:")
    for i, feat in enumerate(results_base['feature_importance_top15'][:10], 1):
        print(f"  {i:2d}. {feat['feature']:30s} {feat['importance']:10.4f}")

    print("\nCatBoost (expanded features) - Top 10:")
    for i, feat in enumerate(results_catboost['feature_importance_top15'][:10], 1):
        print(f"  {i:2d}. {feat['feature']:30s} {feat['importance']:10.4f}")

    print("\nXGBoost (expanded features) - Top 10:")
    for i, feat in enumerate(results_xgboost['feature_importance_top15'][:10], 1):
        print(f"  {i:2d}. {feat['feature']:30s} {feat['importance']:10.4f}")

    # Count macro/onchain features in top 15
    print("\n" + "=" * 80)
    print("MACRO/ONCHAIN FEATURE IMPACT")
    print("=" * 80)

    macro_features = [
        'dxy_return_1d', 'dxy_return_5d', 'dxy_sma_ratio_20d',
        'gold_return_1d', 'gold_return_5d', 'gold_sma_ratio_20d',
        'spx_return_1d', 'spx_return_5d', 'spx_vol_5d',
        'vix_level', 'vix_change_1d', 'vix_sma_ratio_10d', 'btc_gold_corr_30d'
    ]

    onchain_features = [
        'mvrv_ratio', 'mvrv_zscore', 'active_addresses_change_7d',
        'hash_rate_change_30d', 'exchange_net_flow_7d',
        'fee_ratio_change_7d', 'tx_count_change_7d', 'nvt_ratio'
    ]

    for results in [results_catboost, results_xgboost]:
        model_name = results['model']
        top15_features = [f['feature'] for f in results['feature_importance_top15']]

        macro_count = sum(1 for f in top15_features if f in macro_features)
        onchain_count = sum(1 for f in top15_features if f in onchain_features)

        print(f"\n{model_name}:")
        print(f"  Macro features in top 15: {macro_count}/13")
        print(f"  Onchain features in top 15: {onchain_count}/8")

        if macro_count > 0:
            print(f"  Top macro features:")
            for f in top15_features:
                if f in macro_features:
                    importance = next(item['importance'] for item in results['feature_importance_top15'] if item['feature'] == f)
                    print(f"    - {f:30s} {importance:10.4f}")

        if onchain_count > 0:
            print(f"  Top onchain features:")
            for f in top15_features:
                if f in onchain_features:
                    importance = next(item['importance'] for item in results['feature_importance_top15'] if item['feature'] == f)
                    print(f"    - {f:30s} {importance:10.4f}")

    # ========================================
    # Leakage detection on best model
    # ========================================
    # Determine best model by test AUC
    best_result = max(all_results, key=lambda x: x['test_auc'])
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_result['model']} (Test AUC: {best_result['test_auc']:.4f})")
    print("=" * 80)

    # Run leakage detection on the best model
    if "CatBoost (expanded" in best_result['model']:
        best_model = model_catboost
    elif "XGBoost" in best_result['model']:
        best_model = model_xgboost
    else:
        best_model = model_base

    leakage_results = run_leakage_detection(
        best_model, X_train_all, y_train, X_val_all, y_val,
        best_result['model']
    )

    # Add leakage results to best model's results file
    best_result['leakage_detection'] = leakage_results

    # Determine which file to update
    if "CatBoost (expanded" in best_result['model']:
        output_file = results_dir / "catboost_expanded_results.json"
    elif "XGBoost" in best_result['model']:
        output_file = results_dir / "xgboost_expanded_results.json"
    else:
        output_file = results_dir / "catboost_base_results.json"

    with open(output_file, "w") as f:
        json.dump(best_result, f, indent=2)

    # ========================================
    # Final summary
    # ========================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    baseline_auc = results_base['test_auc']
    catboost_delta = results_catboost['test_auc'] - baseline_auc
    xgboost_delta = results_xgboost['test_auc'] - baseline_auc

    print(f"\nBaseline (CatBoost, base features): {baseline_auc:.4f} Test AUC")
    print(f"CatBoost with expanded features: {results_catboost['test_auc']:.4f} Test AUC ({catboost_delta:+.4f})")
    print(f"XGBoost with expanded features: {results_xgboost['test_auc']:.4f} Test AUC ({xgboost_delta:+.4f})")

    print(f"\nResults saved to: {results_dir}/")
    print("  - catboost_base_results.json")
    print("  - catboost_expanded_results.json")
    print("  - xgboost_expanded_results.json")
    print()

    # Answer the key question
    print("=" * 80)
    print("KEY QUESTION: Do macro/on-chain features improve the model?")
    print("=" * 80)

    if catboost_delta > 0.01:
        print(f"YES - CatBoost improved by {catboost_delta:.4f} AUC ({catboost_delta/baseline_auc*100:.1f}%)")
    elif catboost_delta > 0:
        print(f"MARGINAL - CatBoost improved by {catboost_delta:.4f} AUC (small gain)")
    else:
        print(f"NO - CatBoost declined by {catboost_delta:.4f} AUC (no improvement)")

    if xgboost_delta > 0.01:
        print(f"YES - XGBoost improved by {xgboost_delta:.4f} AUC ({xgboost_delta/baseline_auc*100:.1f}%)")
    elif xgboost_delta > 0:
        print(f"MARGINAL - XGBoost improved by {xgboost_delta:.4f} AUC (small gain)")
    else:
        print(f"NO - XGBoost declined by {xgboost_delta:.4f} AUC (no improvement)")

    print()


if __name__ == "__main__":
    main()
