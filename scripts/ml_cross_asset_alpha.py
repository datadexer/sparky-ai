#!/usr/bin/env python3
"""
ML CROSS-ASSET ALPHA SEARCH
===========================

Train CatBoost on 364,830 cross-asset hourly samples, aggregate to daily signals,
validate with yearly walk-forward backtesting.

OBJECTIVE: Beat Multi-Timeframe Ensemble (0.772 Sharpe) with ML model.

Target: Sharpe ≥0.85 (10% improvement)

Strategy:
1. Load hourly data for BTC, ETH, SOL (3 assets with longest history)
2. Compute 23 base technical features per asset
3. Add asset_id as categorical feature
4. Pool all assets (364,830 hourly samples total)
5. Train CatBoost with GPU acceleration
6. Aggregate hourly predictions to daily signals (≥60% positive → LONG)
7. Backtest on BTC daily with yearly walk-forward (6 folds: 2018-2023)
8. Compare to Multi-Timeframe baseline (0.772 Sharpe)

Splits:
- Training: 6 yearly expanding windows (2017-2022)
- Test: 6 yearly holdouts (2018-2023)
- Transaction costs: 0.26% round-trip

Success Criteria:
- ✅ Sharpe ≥0.85 (10% improvement over Multi-TF 0.772)
- ✅ Monte Carlo ≥75%
- ✅ Beats Multi-TF in ≥4/6 years

Failure Criteria:
- ❌ Sharpe ≤0.77 (no improvement)
- ❌ Overfitting (train Sharpe >> test Sharpe)
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparky.features.technical import rsi, ema, macd, momentum
from sparky.features.returns import simple_returns
from sparky.features.advanced import (
    bollinger_bandwidth,
    bollinger_position,
    atr,
    intraday_range,
    volume_momentum,
    volume_ma_ratio,
    vwap_deviation,
    higher_highs_lower_lows,
    volatility_clustering,
    price_distance_from_sma,
    momentum_quality,
    session_hour,
    day_of_week,
    price_acceleration,
)
from sparky.backtest.costs import TransactionCostModel

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_asset_hourly(asset: str) -> pd.DataFrame:
    """Load hourly OHLCV for asset."""
    path = Path(f"data/raw/{asset}/ohlcv_hourly.parquet")

    if not path.exists():
        raise FileNotFoundError(f"Missing {asset} hourly data: {path}")

    df = pd.read_parquet(path)
    logger.info(f"  {asset.upper()}: {len(df):,} candles ({df.index.min().date()} to {df.index.max().date()})")

    return df


def compute_features(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Compute 23 base technical features."""

    features = pd.DataFrame(index=df.index)

    # Technical indicators
    features["rsi_14h"] = rsi(df["close"], period=14)
    features["rsi_6h"] = rsi(df["close"], period=6)

    macd_line, _, histogram = macd(df["close"])
    features["macd_line"] = macd_line
    features["macd_histogram"] = histogram

    ema_fast = ema(df["close"], span=10)
    ema_slow = ema(df["close"], span=20)
    features["ema_ratio_20h"] = ema_fast / ema_slow - 1.0

    features["bb_bandwidth_20h"] = bollinger_bandwidth(df["close"], period=20)
    features["bb_position_20h"] = bollinger_position(df["close"], period=20)

    # Momentum
    features["momentum_4h"] = momentum(df["close"], period=4)
    features["momentum_24h"] = momentum(df["close"], period=24)
    features["momentum_168h"] = momentum(df["close"], period=168)
    features["momentum_quality_30h"] = momentum_quality(df["close"], period=30)
    features["price_acceleration_10h"] = price_acceleration(df["close"], period=10)

    # Volatility
    features["atr_14h"] = atr(df["high"], df["low"], df["close"], period=14)
    features["intraday_range"] = intraday_range(df["high"], df["low"], df["close"])

    hourly_returns = simple_returns(df["close"])
    features["vol_clustering_24h"] = volatility_clustering(hourly_returns, period=24)
    features["realized_vol_24h"] = hourly_returns.rolling(window=24).std()

    # Volume
    features["volume_momentum_30h"] = volume_momentum(df["volume"], period=30)
    features["volume_ma_ratio_20h"] = volume_ma_ratio(df["volume"], period=20)
    features["vwap_deviation_24h"] = vwap_deviation(df["close"], df["volume"], df["high"], df["low"], period=24)

    # Market microstructure
    features["distance_from_sma_200h"] = price_distance_from_sma(df["close"], period=200)
    features["higher_highs_lower_lows_5h"] = higher_highs_lower_lows(df["high"], df["low"], period=5)

    # Temporal
    features["hour_of_day"] = session_hour(df.index)
    features["day_of_week"] = day_of_week(df.index)

    # Clean
    features = features.replace([np.inf, -np.inf], np.nan)

    logger.info(f"  {asset.upper()}: {features.shape[1]} features computed")

    return features


def load_and_pool_assets():
    """Load and pool cross-asset hourly data."""

    logger.info("=" * 80)
    logger.info("LOADING CROSS-ASSET HOURLY DATA")
    logger.info("=" * 80)

    assets = ["btc", "eth", "sol"]  # Top 3 by market cap with longest history

    all_features = []
    all_targets = []

    for asset in assets:
        logger.info(f"\nProcessing {asset.upper()}...")

        # Load OHLCV
        df = load_asset_hourly(asset)

        # Compute features
        features = compute_features(df, asset)

        # Target: 1h ahead direction
        targets = (df["close"].shift(-1) > df["close"]).astype(int)
        targets.name = "target"

        # Add asset_id
        features["asset_id"] = asset

        # Align
        common_idx = features.index.intersection(targets.index)
        features = features.loc[common_idx]
        targets = targets.loc[common_idx]

        # Drop NaN
        valid = ~features.drop(["asset_id"], axis=1).isna().any(axis=1)
        features = features[valid]
        targets = targets.loc[features.index]

        logger.info(f"  {asset.upper()}: {len(features):,} valid hourly samples")

        all_features.append(features)
        all_targets.append(targets)

    # Pool
    logger.info("\nPooling assets...")
    X = pd.concat(all_features, axis=0).sort_index()
    y = pd.concat(all_targets, axis=0).sort_index()

    logger.info(f"Total pooled: {len(X):,} hourly samples")
    logger.info(f"Features: {X.shape[1]} (23 base + asset_id)")
    logger.info(f"Date range: {X.index.min().date()} to {X.index.max().date()}")
    logger.info(f"Target balance: {y.mean():.2%} positive")

    logger.info("\nAsset distribution:")
    for asset in assets:
        count = (X["asset_id"] == asset).sum()
        logger.info(f"  {asset.upper()}: {count:,} ({count/len(X)*100:.1f}%)")

    return X, y


def train_catboost(X_train, y_train, X_val, y_val, seed=42):
    """Train CatBoost classifier."""

    logger.info("\nTraining CatBoost...")
    logger.info(f"  Train: {len(X_train):,} samples")
    logger.info(f"  Val: {len(X_val):,} samples")

    # Separate categorical features
    cat_features = ["asset_id"]

    model = CatBoostClassifier(
        iterations=1000,
        depth=4,
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=seed,
        verbose=False,
        cat_features=cat_features,
        task_type='GPU',  # Use GPU if available
        devices='0',
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=100,
    )

    # Metrics
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    logger.info(f"\n  Train AUC: {train_auc:.4f}")
    logger.info(f"  Val AUC: {val_auc:.4f}")
    logger.info(f"  Overfitting: {train_auc - val_auc:.4f}")

    return model


def aggregate_hourly_to_daily_signals(hourly_probs: pd.Series, hourly_index: pd.DatetimeIndex) -> pd.Series:
    """Aggregate hourly predictions to daily signals.

    Rule: LONG if ≥60% of daily hours predict UP, else FLAT.

    Args:
        hourly_probs: Hourly probability of UP (from model)
        hourly_index: Hourly datetime index

    Returns:
        Daily signals (1=LONG, 0=FLAT)
    """
    # Create DataFrame with hourly predictions
    df = pd.DataFrame({
        'prob': hourly_probs,
        'pred': (hourly_probs > 0.5).astype(int)
    }, index=hourly_index)

    # Resample to daily
    daily = df.resample('D').agg({
        'prob': 'mean',  # Average probability
        'pred': 'mean'   # Fraction of hours predicting UP
    })

    # Signal: LONG if ≥60% of hours predict UP
    daily_signals = (daily['pred'] >= 0.6).astype(int)

    return daily_signals


def backtest_daily_signals(signals: pd.Series, prices_daily: pd.Series, year: int) -> dict:
    """Backtest daily signals on BTC with transaction costs.

    Args:
        signals: Daily signals (1=LONG, 0=FLAT)
        prices_daily: Daily BTC close prices
        year: Test year

    Returns:
        dict with Sharpe, return, trades
    """
    from sparky.features.returns import annualized_sharpe, max_drawdown

    # Align signals and prices
    common_dates = signals.index.intersection(prices_daily.index)
    signals = signals.loc[common_dates]
    prices = prices_daily.loc[common_dates]

    # Returns
    daily_returns = prices.pct_change()

    # Strategy returns (with 1-day lag for next-day execution)
    positions = signals.shift(1)  # Enter at tomorrow's open
    strategy_returns = positions * daily_returns

    # Transaction costs (0.26% round-trip)
    cost_model = TransactionCostModel.for_btc()
    position_changes = positions.diff().abs()
    costs = position_changes * cost_model.total_cost_pct  # Use total_cost_pct, not total_cost
    strategy_returns = strategy_returns - costs

    # Metrics
    cumulative_return = (1 + strategy_returns).prod() - 1
    sharpe = annualized_sharpe(strategy_returns, periods_per_year=365)  # Use 365 for crypto (24/7 trading)

    # Max drawdown from equity curve
    equity_curve = (1 + strategy_returns).cumprod()
    max_dd = max_drawdown(equity_curve)

    n_trades = position_changes.sum()

    return {
        "sharpe": sharpe,
        "total_return": cumulative_return,
        "max_drawdown": max_dd,
        "trades": int(n_trades),
        "mean_daily_return": strategy_returns.mean(),
        "std_daily_return": strategy_returns.std(),
    }


def yearly_walk_forward_validation(X_hourly, y_hourly, btc_prices_daily):
    """Yearly walk-forward validation with expanding window.

    Train on expanding window, test on each year 2018-2023.

    Returns:
        dict with yearly results
    """

    logger.info("\n" + "=" * 80)
    logger.info("YEARLY WALK-FORWARD VALIDATION")
    logger.info("=" * 80)

    test_years = [2018, 2019, 2020, 2021, 2022, 2023]
    results = []

    for year in test_years:
        logger.info(f"\n--- YEAR {year} ---")

        # Splits
        train_end = f"{year-1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        # Train: All data before test year
        X_train = X_hourly[X_hourly.index < test_start]
        y_train = y_hourly[y_hourly.index < test_start]

        # Use previous year as validation
        val_start = f"{year-1}-01-01"
        X_val = X_hourly[(X_hourly.index >= val_start) & (X_hourly.index < test_start)]
        y_val = y_hourly[(y_hourly.index >= val_start) & (y_hourly.index < test_start)]

        # Test: Current year (hourly)
        X_test_hourly = X_hourly[(X_hourly.index >= test_start) & (X_hourly.index <= test_end)]
        y_test_hourly = y_hourly[(y_hourly.index >= test_start) & (y_hourly.index <= test_end)]

        # Filter to BTC only for test
        btc_mask = X_test_hourly["asset_id"] == "btc"
        X_test_btc = X_test_hourly[btc_mask]

        logger.info(f"  Train: {len(X_train):,} hourly (all assets, up to {train_end})")
        logger.info(f"  Val: {len(X_val):,} hourly (year {year-1})")
        logger.info(f"  Test: {len(X_test_btc):,} hourly (BTC {year})")

        # Train model
        model = train_catboost(X_train, y_train, X_val, y_val, seed=42)

        # Predict on test (hourly)
        hourly_probs = model.predict_proba(X_test_btc)[:, 1]
        hourly_probs = pd.Series(hourly_probs, index=X_test_btc.index)

        # Aggregate to daily signals
        daily_signals = aggregate_hourly_to_daily_signals(hourly_probs, X_test_btc.index)

        logger.info(f"  Daily signals: {len(daily_signals)} days, {daily_signals.mean():.1%} LONG")

        # Backtest daily signals on BTC
        test_prices = btc_prices_daily[(btc_prices_daily.index >= test_start) & (btc_prices_daily.index <= test_end)]

        metrics = backtest_daily_signals(daily_signals, test_prices, year)

        logger.info(f"  Sharpe: {metrics['sharpe']:.3f}")
        logger.info(f"  Return: {metrics['total_return']*100:+.1f}%")
        logger.info(f"  Max DD: {metrics['max_drawdown']*100:.1f}%")
        logger.info(f"  Trades: {metrics['trades']}")

        results.append({
            "year": year,
            **metrics
        })

    return results


def summarize_results(yearly_results):
    """Summarize yearly walk-forward results."""

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: ML CROSS-ASSET vs MULTI-TIMEFRAME BASELINE")
    logger.info("=" * 80)

    sharpes = [r["sharpe"] for r in yearly_results]
    returns = [r["total_return"] for r in yearly_results]

    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)
    min_sharpe = np.min(sharpes)
    max_sharpe = np.max(sharpes)
    median_sharpe = np.median(sharpes)

    positive_years = sum(1 for s in sharpes if s > 0)

    logger.info(f"\nML CROSS-ASSET (6 years: 2018-2023):")
    logger.info(f"  Mean Sharpe: {mean_sharpe:.3f}")
    logger.info(f"  Std Sharpe: {std_sharpe:.3f}")
    logger.info(f"  Median Sharpe: {median_sharpe:.3f}")
    logger.info(f"  Min Sharpe: {min_sharpe:.3f}")
    logger.info(f"  Max Sharpe: {max_sharpe:.3f}")
    logger.info(f"  Positive years: {positive_years}/6")

    # Baseline
    baseline_sharpe = 0.772

    logger.info(f"\nBASELINE (Multi-Timeframe Ensemble):")
    logger.info(f"  Mean Sharpe: {baseline_sharpe:.3f}")

    delta = mean_sharpe - baseline_sharpe
    pct_improvement = (delta / baseline_sharpe) * 100

    logger.info(f"\nCOMPARISON:")
    logger.info(f"  Delta: {delta:+.3f} ({pct_improvement:+.1f}%)")

    # Verdict
    logger.info("\n" + "=" * 80)
    logger.info("VERDICT")
    logger.info("=" * 80)

    if mean_sharpe >= 0.85:
        verdict = "✅ SUCCESS — ML beats baseline by ≥10%"
        logger.info(verdict)
    elif mean_sharpe >= 0.77:
        verdict = "⚠️ MARGINAL — ML shows improvement but <10%"
        logger.warning(verdict)
    else:
        verdict = "❌ FAILED — ML does not improve on baseline"
        logger.error(verdict)

    logger.info("=" * 80)

    return {
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "median_sharpe": median_sharpe,
        "min_sharpe": min_sharpe,
        "max_sharpe": max_sharpe,
        "positive_years": f"{positive_years}/6",
        "baseline_sharpe": baseline_sharpe,
        "delta": delta,
        "pct_improvement": pct_improvement,
        "verdict": verdict,
        "yearly_results": yearly_results,
    }


def save_results(results: dict):
    """Save results to JSON."""

    output_dir = Path("results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "ml_cross_asset_validation.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


def main():
    """Main execution."""

    logger.info("=" * 80)
    logger.info("ML CROSS-ASSET ALPHA SEARCH")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
    logger.info(f"Objective: Beat Multi-Timeframe (0.772 Sharpe) with ML")
    logger.info(f"Target: Sharpe ≥0.85 (10% improvement)")

    # 1. Load and pool cross-asset hourly data
    X_hourly, y_hourly = load_and_pool_assets()

    # 2. Load BTC daily prices for backtesting
    logger.info("\nLoading BTC daily prices for backtesting...")
    btc_daily_path = Path("data/raw/btc/ohlcv.parquet")
    if not btc_daily_path.exists():
        raise FileNotFoundError(f"BTC daily data not found: {btc_daily_path}")

    btc_daily = pd.read_parquet(btc_daily_path)
    btc_prices_daily = btc_daily["close"]
    logger.info(f"  BTC daily: {len(btc_prices_daily):,} days ({btc_prices_daily.index.min().date()} to {btc_prices_daily.index.max().date()})")

    # 3. Yearly walk-forward validation
    yearly_results = yearly_walk_forward_validation(X_hourly, y_hourly, btc_prices_daily)

    # 4. Summarize results
    summary = summarize_results(yearly_results)

    # 5. Save results
    save_results(summary)

    logger.info("\nExecution complete!")


if __name__ == "__main__":
    main()
