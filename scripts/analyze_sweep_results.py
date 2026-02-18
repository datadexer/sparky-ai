#!/usr/bin/env python3
"""Analyze 58-feature hyperparameter sweep results."""

import json
import sys
from pathlib import Path

import numpy as np


def load_results():
    """Load sweep results."""
    results_path = Path("results/validation/sweep_58_features.json")
    if not results_path.exists():
        print(f"ERROR: Results not found at {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    return results


def analyze(results):
    """Analyze sweep results."""
    print("=" * 80)
    print("58-FEATURE HYPERPARAMETER SWEEP ANALYSIS")
    print("=" * 80)
    print()

    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    print(f"Total configs: {len(results)}")
    print(f"Valid results: {len(valid_results)}")
    print(f"Errors: {len(error_results)}")
    print()

    if not valid_results:
        print("No valid results to analyze!")
        return

    # Sort by Sharpe
    valid_results.sort(key=lambda x: x["mean_sharpe"], reverse=True)

    # Statistics
    sharpes = [r["mean_sharpe"] for r in valid_results]
    accuracies = [r["mean_accuracy"] for r in valid_results]
    aucs = [r["mean_auc"] for r in valid_results]

    print("OVERALL STATISTICS")
    print("-" * 80)
    print(
        f"Sharpe:   mean={np.mean(sharpes):.3f}, std={np.std(sharpes):.3f}, "
        f"min={np.min(sharpes):.3f}, max={np.max(sharpes):.3f}"
    )
    print(f"Accuracy: mean={np.mean(accuracies):.3f}, std={np.std(accuracies):.3f}")
    print(f"AUC:      mean={np.mean(aucs):.3f}, std={np.std(aucs):.3f}")
    print()

    # Baseline comparison
    baseline_sharpe = 1.062  # Multi-TF Donchian (corrected)
    single_tf_sharpe = 1.243  # Single-TF (40/20)

    beats_multi = sum(1 for s in sharpes if s > baseline_sharpe)
    beats_single = sum(1 for s in sharpes if s > single_tf_sharpe)

    print("BASELINE COMPARISON")
    print("-" * 80)
    print(
        f"Configs beating Multi-TF baseline (1.062): {beats_multi}/{len(valid_results)} "
        f"({100 * beats_multi / len(valid_results):.1f}%)"
    )
    print(
        f"Configs beating Single-TF baseline (1.243): {beats_single}/{len(valid_results)} "
        f"({100 * beats_single / len(valid_results):.1f}%)"
    )
    print()

    # Top 10
    print("TOP 10 CONFIGS")
    print("-" * 80)
    for i, r in enumerate(valid_results[:10], 1):
        model = r["config"]["model"]
        params = r["config"]["params"]
        sharpe = r["mean_sharpe"]
        acc = r["mean_accuracy"]
        auc = r["mean_auc"]

        # Extract key params
        if model == "CatBoost":
            depth = params["depth"]
            lr = params["learning_rate"]
            l2 = params["l2_leaf_reg"]
            param_str = f"d={depth}, lr={lr}, l2={l2}"
        else:  # LightGBM
            depth = params["max_depth"]
            lr = params["learning_rate"]
            l1 = params["reg_alpha"]
            param_str = f"d={depth}, lr={lr}, l1={l1}"

        print(f"{i:2d}. {model:8s} - Sharpe={sharpe:.3f}, Acc={acc:.3f}, AUC={auc:.3f}")
        print(f"    {param_str}")

    print()

    # Year-by-year for best model
    best = valid_results[0]
    print("BEST MODEL YEAR-BY-YEAR BREAKDOWN")
    print("-" * 80)
    print(f"Model: {best['config']['model']}")
    print(f"Params: {best['config']['params']}")
    print()

    for yr in best["yearly_results"]:
        print(
            f"  {yr['year']}: Sharpe={yr['sharpe']:.3f}, Acc={yr['accuracy']:.3f}, "
            f"AUC={yr['auc']:.3f}, N={yr['test_samples']}"
        )

    print()

    # Verdict
    print("VERDICT")
    print("-" * 80)
    best_sharpe = valid_results[0]["mean_sharpe"]

    if best_sharpe > single_tf_sharpe:
        print(f"✅ ML BEATS BEST BASELINE (Sharpe {best_sharpe:.3f} vs {single_tf_sharpe:.3f})")
        print("   Recommendation: Run feature ablation to identify key features")
        print("   Then train ensemble and request OOS validation")
    elif best_sharpe > baseline_sharpe:
        print(f"⚠️  ML BEATS MULTI-TF but not Single-TF ({best_sharpe:.3f} vs {single_tf_sharpe:.3f})")
        print("   Recommendation: Try ensemble methods or different model families")
    elif best_sharpe > 0.7:
        print(f"⚠️  ML shows promise ({best_sharpe:.3f}) but below baselines")
        print("   Recommendation: Consider alternative approaches or accept baselines")
    else:
        print(f"❌ ML does not show alpha ({best_sharpe:.3f} << baseline {baseline_sharpe:.3f})")
        print("   Recommendation: Accept that simple baselines are superior")

    print()


if __name__ == "__main__":
    results = load_results()
    analyze(results)
