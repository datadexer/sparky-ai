#!/usr/bin/env python3
"""VALIDATION 2: Leakage Re-Audit

Re-run leakage detector SPECIFICALLY on technical-only, 30d configuration
with increased trials (n=20) to confirm no hidden leakage.

Holdout test already FAILED (Sharpe -1.48), but we need to diagnose WHY.
Possible causes:
1. Data leakage (shuffled-label test should fail)
2. Implementation bug (sanity checks should fail)
3. Genuine overfitting (leakage passes, but model learned noise)
"""

import logging

import pandas as pd

from sparky.backtest.leakage_detector import LeakageDetector
from sparky.models.xgboost_model import XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("VALIDATION 2: LEAKAGE RE-AUDIT")
    logger.info("=" * 80)
    logger.info("Configuration: Technical-only features, 30d horizon")
    logger.info("Trials: 20 (increased from default 10 for confidence)")
    logger.info("=" * 80)

    # Load data
    logger.info("Loading data...")
    X = pd.read_parquet("data/processed/feature_matrix_btc.parquet")
    targets_df = pd.read_parquet("data/processed/targets_btc_30d.parquet")
    y = targets_df["target"]

    # Technical-only features
    technical_features = ["rsi_14", "momentum_30d", "ema_ratio_20d"]
    X_technical = X[technical_features]

    # Align
    common_index = X_technical.index.intersection(y.index)
    X_technical = X_technical.loc[common_index]
    y = y.loc[common_index]

    logger.info(f"Total samples: {len(X_technical)}")
    logger.info(f"Features: {list(X_technical.columns)}")

    # Split train/test (using same split as Phase 2-3)
    split_date = pd.Timestamp("2022-01-01", tz="UTC")
    train_mask = X_technical.index < split_date
    test_mask = X_technical.index >= split_date

    X_train = X_technical[train_mask]
    y_train = y[train_mask]
    X_test = X_technical[test_mask]
    y_test = y[test_mask]

    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Train model
    logger.info("\nTraining XGBoost...")
    model = XGBoostModel(random_state=0)
    model.fit(X_train, y_train)
    logger.info("✓ Model trained")

    # Run leakage detector with increased trials
    logger.info("\nRunning leakage detector (n_trials=20)...")
    detector = LeakageDetector(n_shuffle_trials=20)
    report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)

    logger.info("\n" + "=" * 80)
    logger.info("LEAKAGE DETECTION RESULTS")
    logger.info("=" * 80)

    for check in report.checks:
        status = "✅ PASS" if check.passed else "❌ FAIL"
        logger.info(f"{check.check_name}: {status}")
        logger.info(f"  {check.detail}")

    logger.info("=" * 80)
    logger.info(f"Overall: {'✅ ALL CHECKS PASSED' if report.passed else '❌ LEAKAGE DETECTED'}")
    logger.info("=" * 80)

    # Interpretation
    logger.info("\nINTERPRETATION:")
    if report.passed:
        logger.info("Leakage detector PASSED all checks.")
        logger.info("This means:")
        logger.info("- Shuffled labels show ~50% accuracy (random performance) ✓")
        logger.info("- No temporal boundary violations ✓")
        logger.info("- No index overlap issues ✓")
        logger.info("")
        logger.info("CONCLUSION: Holdout failure is due to OVERFITTING, not leakage.")
        logger.info("The model learned patterns specific to train/test split that don't generalize.")
        verdict = "OVERFITTING"
    else:
        logger.info("Leakage detector FAILED.")
        logger.info("This means:")
        logger.info("- Features contain target information (data leakage)")
        logger.info("- Holdout failure is due to BOTH overfitting AND leakage")
        logger.info("")
        logger.info("CONCLUSION: Must fix leakage before proceeding.")
        verdict = "LEAKAGE"

    # Save results
    import json
    from datetime import datetime, timezone

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "features": technical_features,
            "horizon": 30,
            "seed": 0,
            "n_trials": 20,
        },
        "leakage_report": {
            "passed": report.passed,
            "checks": [
                {
                    "check_name": check.check_name,
                    "passed": bool(check.passed),
                    "detail": check.detail,
                }
                for check in report.checks
            ],
        },
        "verdict": verdict,
    }

    output_path = "results/experiments/leakage_reaudit_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Append to RESEARCH_LOG.md
    log_entry = f"""
---
## VALIDATION 2: Leakage Re-Audit — {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC

**Configuration**: Technical-only (RSI, Momentum, EMA), 30d horizon, n_trials=20

**Results**:
{chr(10).join(f"- {check.check_name}: {'PASS' if check.passed else 'FAIL'} - {check.detail}" for check in report.checks)}

**Overall**: {"✅ ALL CHECKS PASSED" if report.passed else "❌ LEAKAGE DETECTED"}

**Verdict**: [{verdict}]
{"Holdout failure is due to OVERFITTING, not leakage. Model learned noise in train/test split." if verdict == "OVERFITTING" else "Holdout failure is due to LEAKAGE. Must fix data pipeline."}
"""

    with open("roadmap/RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    logger.info("Results logged to roadmap/RESEARCH_LOG.md")

    return results


if __name__ == "__main__":
    results = main()
