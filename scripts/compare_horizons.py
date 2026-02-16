#!/usr/bin/env python3
"""Compare Multi-Horizon XGBoost Results

Loads results from all 4 horizon training runs and generates a comparison
table with a data-driven recommendation.

Usage:
    python scripts/compare_horizons.py

Requires:
    results/hourly_horizon_experiments/{1h,4h,24h,exec24h}_results.json

Outputs:
    results/hourly_horizon_experiments/final_recommendation.md
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

HORIZONS = ["1h", "4h", "24h", "exec24h"]
EFFECTIVE_SAMPLES = {"1h": 35000, "4h": 8766, "24h": 1461, "exec24h": 1461}

# Decision matrix weights
WEIGHTS = {
    "nonoverlap_val_auc": 0.30,
    "effective_samples": 0.25,
    "val_test_consistency": 0.15,
    "feature_coherence": 0.15,
    "trading_utility": 0.15,
}

TRADING_UTILITY = {"1h": 0.3, "4h": 0.6, "24h": 0.9, "exec24h": 1.0}


def load_results() -> dict:
    """Load results JSON for all horizons."""
    results_dir = Path("results/hourly_horizon_experiments")
    results = {}
    for h in HORIZONS:
        path = results_dir / f"{h}_results.json"
        if path.exists():
            results[h] = json.loads(path.read_text())
            logger.info(f"Loaded {h} results")
        else:
            logger.warning(f"Missing results for {h}: {path}")
    return results


def score_horizon(h: str, data: dict) -> dict:
    """Score a horizon across all criteria.

    Returns dict with individual scores (0-1) and weighted total.
    """
    scores = {}

    # 1. Non-overlapping val AUC (0-1 scale, 0.5=random → 0, 0.6+ → 1)
    val_auc = data["val"]["nonoverlap"]["roc_auc"]
    scores["nonoverlap_val_auc"] = max(0, min(1, (val_auc - 0.5) / 0.1))

    # 2. Effective independent samples (10K+ → 1.0, <1K → 0)
    n_eff = EFFECTIVE_SAMPLES[h]
    scores["effective_samples"] = max(0, min(1, n_eff / 10000))

    # 3. Val-test consistency (<5% gap → 1.0, >15% → 0)
    gap = data["val_test_auc_gap"]
    scores["val_test_consistency"] = max(0, min(1, 1 - gap / 0.15))

    # 4. Feature coherence (heuristic: top feature importance > 0.05 → good)
    top_imp = data["feature_importance"][0]["importance"] if data["feature_importance"] else 0
    scores["feature_coherence"] = min(1, top_imp / 0.15)

    # 5. Trading utility (fixed per horizon)
    scores["trading_utility"] = TRADING_UTILITY[h]

    # Weighted total
    total = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    scores["weighted_total"] = total

    return scores


def determine_scenario(results: dict) -> str:
    """Determine which scenario we're in based on AUC values."""
    aucs = {}
    for h, data in results.items():
        aucs[h] = data["val"]["nonoverlap"]["roc_auc"]

    best_h = max(aucs, key=aucs.get)
    best_auc = aucs[best_h]

    if best_auc < 0.52:
        return "D"  # All near-random
    elif best_h == "1h":
        return "A"
    elif best_h == "4h":
        return "B"
    else:
        return "C"


def generate_report(results: dict) -> str:
    """Generate the final comparison and recommendation report."""
    lines = []
    lines.append("# Multi-Horizon XGBoost Comparison Report")
    lines.append("")
    lines.append("## Experiment Summary")
    lines.append("")
    lines.append("Trained XGBoost (max_depth=5, n_estimators=200, lr=0.05) on hourly BTC features")
    lines.append("across 4 target horizons. Same hyperparameters for all horizons to isolate target effect.")
    lines.append("")

    # Metrics table
    lines.append("## Results Table (Non-Overlapping Evaluation)")
    lines.append("")
    lines.append("| Metric | 1h | 4h | 24h | exec24h |")
    lines.append("|--------|----|----|-----|---------|")

    metrics_rows = [
        ("Val Accuracy", lambda h, d: f"{d['val']['nonoverlap']['accuracy']:.4f}"),
        ("Val ROC-AUC", lambda h, d: f"{d['val']['nonoverlap']['roc_auc']:.4f}"),
        ("Val F1", lambda h, d: f"{d['val']['nonoverlap']['f1']:.4f}"),
        ("Test Accuracy", lambda h, d: f"{d['test']['nonoverlap']['accuracy']:.4f}"),
        ("Test ROC-AUC", lambda h, d: f"{d['test']['nonoverlap']['roc_auc']:.4f}"),
        ("Val-Test AUC Gap", lambda h, d: f"{d['val_test_auc_gap']:.4f}"),
        ("Effective Train Samples", lambda h, d: f"{EFFECTIVE_SAMPLES[h]:,}"),
        ("Leakage Check", lambda h, d: "PASS" if d["leakage_passed"] else "FAIL"),
        ("Val Class Balance", lambda h, d: f"{d['val']['nonoverlap']['class_balance']:.3f}"),
    ]

    for metric_name, getter in metrics_rows:
        row = f"| {metric_name} |"
        for h in HORIZONS:
            if h in results:
                row += f" {getter(h, results[h])} |"
            else:
                row += " N/A |"
        lines.append(row)

    # Standard vs non-overlapping comparison
    lines.append("")
    lines.append("## Standard vs Non-Overlapping Evaluation")
    lines.append("")
    lines.append("| Horizon | Val AUC (standard) | Val AUC (nonoverlap) | Inflation |")
    lines.append("|---------|-------------------|---------------------|-----------|")
    for h in HORIZONS:
        if h in results:
            std_auc = results[h]["val"]["standard"]["roc_auc"]
            no_auc = results[h]["val"]["nonoverlap"]["roc_auc"]
            inflation = std_auc - no_auc
            lines.append(f"| {h} | {std_auc:.4f} | {no_auc:.4f} | {inflation:+.4f} |")

    # Feature importance comparison
    lines.append("")
    lines.append("## Top-5 Features by Horizon")
    lines.append("")
    for h in HORIZONS:
        if h not in results:
            continue
        lines.append(f"### {h}")
        lines.append("")
        for i, item in enumerate(results[h]["feature_importance"][:5]):
            lines.append(f"{i+1}. `{item['feature']}` ({item['importance']:.4f})")
        lines.append("")

    # Check for hour_of_day dominance
    for h in HORIZONS:
        if h not in results:
            continue
        top_feat = results[h]["feature_importance"][0]["feature"]
        if top_feat == "hour_of_day":
            lines.append(f"**WARNING**: `hour_of_day` dominates {h} model — may be learning time-of-day noise.")
            lines.append("")

    # Scoring
    lines.append("## Decision Matrix Scores")
    lines.append("")
    lines.append(f"Weights: {WEIGHTS}")
    lines.append("")
    lines.append("| Criterion | Weight | " + " | ".join(HORIZONS) + " |")
    lines.append("|-----------|--------| " + " | ".join(["---"] * len(HORIZONS)) + " |")

    all_scores = {}
    for h in HORIZONS:
        if h in results:
            all_scores[h] = score_horizon(h, results[h])

    for criterion in WEIGHTS:
        row = f"| {criterion} | {WEIGHTS[criterion]:.0%} |"
        for h in HORIZONS:
            if h in all_scores:
                row += f" {all_scores[h][criterion]:.3f} |"
            else:
                row += " N/A |"
        lines.append(row)

    row = "| **TOTAL** | |"
    for h in HORIZONS:
        if h in all_scores:
            row += f" **{all_scores[h]['weighted_total']:.3f}** |"
        else:
            row += " N/A |"
    lines.append(row)

    # Recommendation
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")

    scenario = determine_scenario(results)
    best_h = max(all_scores, key=lambda h: all_scores[h]["weighted_total"])
    best_score = all_scores[best_h]["weighted_total"]

    lines.append(f"**Scenario: {scenario}**")
    lines.append(f"**Winner: {best_h}** (weighted score: {best_score:.3f})")
    lines.append("")

    if scenario == "A":
        lines.append("1h horizon wins on ROC-AUC with the most independent training samples (~35K).")
        lines.append("Recommendation: Use 1h model. Aggregate 24 hourly predictions into daily confidence.")
        lines.append("Daily signal = mean(P(up) for last 24 hours) > 0.5 → LONG")
    elif scenario == "B":
        lines.append("4h horizon balances signal quality and sample count (~8.7K independent).")
        lines.append("Recommendation: Use 4h model. Aggregate 6 predictions per day into daily signal.")
    elif scenario == "C":
        lines.append("24h/exec horizon has best AUC despite fewer independent samples (~1.4K).")
        lines.append("Note: Effective sample count does NOT meet 10K+ audit goal.")
        lines.append("Recommendation: Use with caution, or combine with cross-asset expansion.")
    elif scenario == "D":
        lines.append("All horizons show ROC-AUC < 0.52 — no learnable signal at any hourly horizon.")
        lines.append("Recommendation: Pivot to APPROACH 2 (cross-asset training) or APPROACH 3 (on-chain features).")

    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    if scenario != "D":
        lines.append(f"1. Run holdout evaluation on {best_h} model (ONE test only)")
        lines.append(f"2. If holdout confirms, build signal aggregation pipeline ({best_h} → daily)")
        lines.append("3. Proceed with cross-asset expansion for robustness")
    else:
        lines.append("1. Investigate feature quality — are hourly features too noisy?")
        lines.append("2. Try APPROACH 2: cross-asset training with ETH, SOL, etc.")
        lines.append("3. Consider alternative model architectures (LSTM, attention)")

    return "\n".join(lines)


def main():
    logger.info("=" * 80)
    logger.info("COMPARE MULTI-HORIZON RESULTS")
    logger.info("=" * 80)

    results = load_results()

    if len(results) == 0:
        logger.error("No results found. Run train_hourly_horizon.py for each horizon first.")
        return

    if len(results) < 4:
        logger.warning(f"Only {len(results)}/4 horizons have results. Proceeding with partial comparison.")

    report = generate_report(results)

    # Save report
    output_dir = Path("results/hourly_horizon_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "final_recommendation.md"
    report_path.write_text(report)
    logger.info(f"\nSaved report: {report_path}")

    # Print to console
    print("\n" + report)


if __name__ == "__main__":
    main()
