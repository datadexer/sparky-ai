#!/usr/bin/env python3
"""Feature importance analysis tool.

Loads feature matrix via sparky.data.loader, trains a default XGBoost on
GPU, prints ranked feature importance, and saves to JSON.

Usage:
    PYTHONPATH=. python3 scripts/analyze_features.py [dataset_name]
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sparky.data.loader import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("results/feature_importance.json")


def main(dataset: str = "btc_1h_features"):
    """Analyze feature importance for a dataset."""
    logger.info(f"Loading dataset: {dataset}")
    df = load(dataset, purpose="training")
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Find target column
    target_col = [c for c in df.columns if "target" in c.lower() or "direction" in c.lower()]
    if not target_col:
        logger.error("No target column found. Expected 'target' or 'direction' in column name.")
        sys.exit(1)

    target_col = target_col[0]
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Features: {len(X.columns)}, Target: {target_col}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    # Train default XGBoost on GPU
    from xgboost import XGBClassifier

    logger.info("Training XGBoost (default params, GPU)...")
    model = XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)

    # Extract importance
    importance = dict(zip(X.columns, model.feature_importances_))
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Print ranked features
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE RANKING")
    print("=" * 60)
    for i, (name, score) in enumerate(ranked, 1):
        bar = "#" * int(score * 200)
        print(f"  {i:3d}. {name:40s} {score:.4f} {bar}")

    # Save to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": dataset,
        "n_features": len(X.columns),
        "n_samples": len(X),
        "ranked_features": [
            {"rank": i + 1, "name": name, "importance": float(score)} for i, (name, score) in enumerate(ranked)
        ],
        "top_20": [name for name, _ in ranked[:20]],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to {OUTPUT_PATH}")

    # Summary
    print("\nTop 20 features for sweep (copy to config):")
    print(f"  {[name for name, _ in ranked[:20]]}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "btc_1h_features"
    main(dataset)
