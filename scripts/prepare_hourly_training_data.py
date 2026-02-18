#!/usr/bin/env python3
"""Prepare Hourly Training Data — Multi-Horizon Targets

Generates hourly feature matrix (115K rows) and 4 target variants for
the multi-horizon XGBoost experiment.

Outputs:
    data/processed/features_hourly_full.parquet      # ~115K × 23 hourly features
    data/processed/targets_hourly_1h.parquet          # 1h-ahead binary target
    data/processed/targets_hourly_4h.parquet          # 4h-ahead binary target
    data/processed/targets_hourly_24h.parquet         # 24h-ahead binary target
    data/processed/targets_hourly_exec24h.parquet     # execution-adjusted 24h target

Target definitions:
    1h:      close(T+1) > close(T)
    4h:      close(T+4) > close(T)
    24h:     close(T+24) > close(T)
    exec24h: close(T+25) > open(T+1)  (signal → next-bar execution → 24h eval)

Autocorrelation analysis determines effective independent samples per horizon.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_targets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Generate 4 target variants from hourly OHLCV.

    Args:
        df: Hourly OHLCV DataFrame with 'close' and 'open' columns.

    Returns:
        Dict mapping horizon name to binary target Series.
    """
    close = df["close"]
    open_ = df["open"]

    targets = {}

    # 1h: close(T+1) > close(T)
    targets["1h"] = (close.shift(-1) > close).astype(int)

    # 4h: close(T+4) > close(T)
    targets["4h"] = (close.shift(-4) > close).astype(int)

    # 24h: close(T+24) > close(T)
    targets["24h"] = (close.shift(-24) > close).astype(int)

    # exec24h: signal at T, execute at T+1 open, evaluate at T+25 close
    next_open = open_.shift(-1)
    target_close_exec = close.shift(-25)
    targets["exec24h"] = (target_close_exec > next_open).astype(int)

    return targets


def compute_autocorrelation_report(
    targets: dict[str, pd.Series],
    train_start: str = "2017-01-01",
    train_end: str = "2021-01-01",
) -> str:
    """Compute autocorrelation and effective sample count for each target.

    Args:
        targets: Dict of horizon name → binary target Series.
        train_start: Train period start (inclusive).
        train_end: Train period end (exclusive).

    Returns:
        Formatted report string.
    """
    horizon_hours = {"1h": 1, "4h": 4, "24h": 24, "exec24h": 24}

    lines = [
        "=" * 70,
        "AUTOCORRELATION & EFFECTIVE SAMPLE REPORT",
        "=" * 70,
        "",
    ]

    for name, target in targets.items():
        train_target = target.loc[(target.index >= train_start) & (target.index < train_end)].dropna()

        n_nominal = len(train_target)
        ac_lag1 = train_target.autocorr(lag=1)
        h = horizon_hours[name]
        ac_lag_h = train_target.autocorr(lag=h)

        # Effective independent samples (Newey-West approximation)
        # n_eff ≈ n / (1 + 2 * sum of autocorrelations)
        # Simplified: n_eff ≈ n * (1 - ac_lag1) / (1 + ac_lag1) for AR(1)
        if abs(ac_lag1) < 1.0:
            n_effective = n_nominal * (1 - ac_lag1) / (1 + ac_lag1)
        else:
            n_effective = n_nominal

        # Non-overlapping sample count (stride by horizon)
        n_nonoverlap = len(train_target.iloc[::h])

        balance = train_target.mean()

        lines.append(f"Horizon: {name}")
        lines.append(f"  Nominal train samples:     {n_nominal:,}")
        lines.append(f"  Autocorr(lag=1):           {ac_lag1:.4f}")
        lines.append(f"  Autocorr(lag={h}):          {ac_lag_h:.4f}")
        lines.append(f"  Effective independent:     {n_effective:,.0f}")
        lines.append(f"  Non-overlapping (stride={h}): {n_nonoverlap:,}")
        lines.append(f"  Class balance:             {balance:.3f} ({balance:.1%} positive)")
        meets = "YES" if n_effective >= 10000 else "NO"
        lines.append(f"  Meets 10K audit goal:      {meets}")
        lines.append("")

    return "\n".join(lines)


def main():
    """Generate hourly features and multi-horizon targets."""
    from scripts.prepare_hourly_features import compute_hourly_features, load_hourly_data

    logger.info("=" * 80)
    logger.info("PREPARE HOURLY TRAINING DATA — MULTI-HORIZON")
    logger.info("=" * 80)

    # Load raw hourly OHLCV
    df_hourly = load_hourly_data()
    logger.info(f"Raw data: {len(df_hourly):,} hourly candles")
    logger.info(f"Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")

    # Compute hourly features (NO resampling)
    features = compute_hourly_features(df_hourly)
    logger.info(f"Hourly features shape: {features.shape}")

    # Drop NaN rows (warmup period)
    n_before = len(features)
    features_clean = features.dropna()
    n_dropped = n_before - len(features_clean)
    logger.info(f"Dropped {n_dropped} NaN rows (warmup), {len(features_clean):,} remain")

    # Generate targets
    targets = generate_targets(df_hourly)

    # Save features
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "features_hourly_full.parquet"
    features_clean.to_parquet(features_path)
    logger.info(f"Saved features: {features_path} ({len(features_clean):,} × {features_clean.shape[1]})")

    # Save each target (aligned to clean features index)
    for name, target in targets.items():
        # Align to feature index and drop NaN (end-of-series)
        aligned = target.reindex(features_clean.index).dropna().astype(int)
        target_df = pd.DataFrame({"target": aligned})
        target_path = output_dir / f"targets_hourly_{name}.parquet"
        target_df.to_parquet(target_path)
        logger.info(f"Saved target {name}: {target_path} ({len(aligned):,} samples, balance={aligned.mean():.3f})")

    # Autocorrelation report
    # Re-align targets to clean features for report
    aligned_targets = {}
    for name, target in targets.items():
        aligned_targets[name] = target.reindex(features_clean.index).dropna()

    report = compute_autocorrelation_report(aligned_targets)
    logger.info("\n" + report)

    # Save report
    report_path = Path("results/hourly_horizon_experiments")
    report_path.mkdir(parents=True, exist_ok=True)
    (report_path / "autocorrelation_report.txt").write_text(report)
    logger.info(f"Saved autocorrelation report: {report_path / 'autocorrelation_report.txt'}")

    # Validation summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)

    # Verify hourly frequency
    time_diffs = features_clean.index.to_series().diff().dropna()
    median_diff = time_diffs.median()
    logger.info(f"Median time step: {median_diff} (should be ~1h)")
    logger.info(f"Feature matrix rows: {len(features_clean):,} (should be ~115K, NOT ~4.8K)")

    assert len(features_clean) > 50000, (
        f"Feature matrix has {len(features_clean)} rows — expected 100K+. Check for accidental daily resampling."
    )

    logger.info("\nSUCCESS — Hourly training data ready for multi-horizon experiment")
    logger.info("Next: python scripts/train_hourly_horizon.py --horizon 1h")


if __name__ == "__main__":
    main()
