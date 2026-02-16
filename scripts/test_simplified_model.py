#!/usr/bin/env python3
"""Test simplified model WITHOUT returns_1d to check if leakage persists.

This is Phase 0, Step 2 of the debugging protocol:
- Load real BTC data
- Remove returns_1d feature
- Train XGBoost
- Run leakage detector
- Expected: PASS if returns_1d was the culprit
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from sparky.backtest.costs import TransactionCostModel
from sparky.backtest.engine import WalkForwardBacktester
from sparky.backtest.leakage_detector import LeakageDetector
from sparky.models.xgboost_model import XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("PHASE 0 STEP 2: Test Simplified Model (WITHOUT returns_1d)")
print("=" * 80)

# Load feature matrix and 7d targets
logger.info("Loading data...")
X = pd.read_parquet("data/processed/feature_matrix_btc.parquet")
y_7d = pd.read_parquet("data/processed/targets_btc_7d.parquet")["target"]

logger.info(f"Feature matrix shape: {X.shape}")
logger.info(f"Features: {list(X.columns)}")
logger.info(f"Target shape: {y_7d.shape}")

# Remove returns_1d feature (testing hypothesis that this is the leakage source)
if 'returns_1d' in X.columns:
    X_simple = X.drop(columns=['returns_1d'])
    logger.info("✓ Removed 'returns_1d' feature")
else:
    X_simple = X.copy()
    logger.warning("'returns_1d' not found in features")

logger.info(f"Simplified feature set: {list(X_simple.columns)}")

# Split data
IS_END = "2022-01-01"
in_sample_mask = X_simple.index < IS_END
X_train = X_simple[in_sample_mask]
y_train = y_7d[in_sample_mask]
X_test = X_simple[~in_sample_mask]
y_test = y_7d[~in_sample_mask]

logger.info(f"Train: {len(X_train)} samples")
logger.info(f"Test: {len(X_test)} samples")

# Train XGBoost
logger.info("\nTraining XGBoost (simplified features, 7d horizon)...")
model = XGBoostModel(random_state=42)
model.fit(X_train, y_train)

# Run leakage detector with INCREASED trials
logger.info("\nRunning leakage detector (n_shuffle_trials=10)...")
detector = LeakageDetector(n_shuffle_trials=10)

try:
    report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)

    print("\n" + "=" * 80)
    print("LEAKAGE DETECTION RESULTS")
    print("=" * 80)
    print(f"\nOverall: {'✅ PASSED' if report.passed else '❌ FAILED'}")
    print(f"\nChecks:")
    for check in report.checks:
        status = "✅ PASS" if check.passed else "❌ FAIL"
        print(f"  {check.check_name}: {status}")
        print(f"    {check.detail}")
        if hasattr(check, 'metric_value') and check.metric_value > 0:
            print(f"    Metric: {check.metric_value:.4f}")

    if report.failed_checks:
        print(f"\n❌ FAILED CHECKS: {report.failed_checks}")
        print("\nConclusion: Leakage persists even without returns_1d")
        print("  → Issue is deeper (backtester timing? other features?)")
        sys.exit(1)
    else:
        print("\n✅ ALL CHECKS PASSED")
        print("\nConclusion: returns_1d was likely the leakage source")
        print("  → Proceed to fix by removing or lagging returns_1d")
        sys.exit(0)

except Exception as e:
    logger.error(f"Leakage detector failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
