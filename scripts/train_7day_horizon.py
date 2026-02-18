#!/usr/bin/env python3
"""
Train 7-day prediction horizon model.
Uses hourly aggregated to daily via resampling, predicts 7-day forward returns.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime, UTC

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sparky.backtest.engine import WalkForwardBacktester
from src.sparky.oversight.time_tracker import TaskTimer

SELECTED_FEATURES = [
    "volume_surge_4h", "tick_direction_ratio_24h", "rsi_divergence_14h_168h",
    "momentum_24h", "volatility_regime", "breakout_proximity_upper",
    "intraday_range", "vwap_deviation_24h", "macd_histogram",
    "intraday_momentum_reversal", "higher_highs_lower_lows_5h",
    "high_low_ratio_20h", "distance_from_sma_200h", "obv",
    "price_acceleration_10h", "momentum_168h", "mfi_14h",
    "momentum_divergence_24h_168h", "rsi_6h", "volume_momentum_30h",
]

def main():
    timer = TaskTimer(agent_id="research")
    timer.start("train_7day_horizon")

    print("="*80)
    print("7-DAY HORIZON (aggregated hourly → daily)")
    print(f"Started: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*80)

    # Load hourly data
    print("\nLoading hourly features...")
    hourly = pd.read_parquet("data/processed/feature_matrix_btc_hourly.parquet")
    
    # Resample to daily (take last hour of each day)
    daily = hourly.resample('D').last().dropna()
    
    # Load daily prices for target creation
    print("Loading prices...")
    import ccxt
    exchange = ccxt.okx()
    since_ms = int(daily.index[0].timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', since=since_ms, limit=5000)
    prices = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms', utc=True)
    prices.set_index('timestamp', inplace=True)
    
    # Align
    common_idx = daily.index.intersection(prices.index)
    daily = daily.loc[common_idx]
    prices = prices.loc[common_idx]
    
    # Create 7-day target
    fwd_7d = prices['close'].shift(-7)
    y = (fwd_7d > prices['close']).astype(int)
    
    # Drop last 7 days
    valid = ~y.isna()
    X = daily[valid][SELECTED_FEATURES]
    y = y[valid]
    
    print(f"\nSamples: {len(X)}")
    print(f"Date range: {X.index[0]} to {X.index[-1]}")
    print(f"Target balance: {y.mean()*100:.1f}% positive")

    # Walk-forward
    backtester = WalkForwardBacktester(
        initial_train_years=3,
        test_period_days=365,
        step_size_days=365,
    )

    config = {
        "iterations": 200, "depth": 4, "learning_rate": 0.01,
        "l2_leaf_reg": 1.0, "task_type": "GPU", "devices": "0", "verbose": 0,
    }

    print(f"\nWalk-forward validation...")
    sharpes, accs = [], []

    for train_dates, test_dates in backtester.generate_folds(X.index):
        model = CatBoostClassifier(**config)
        model.fit(X.loc[train_dates], y.loc[train_dates])
        
        y_pred_proba = model.predict_proba(X.loc[test_dates])[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        acc = accuracy_score(y.loc[test_dates], y_pred)
        auc = roc_auc_score(y.loc[test_dates], y_pred_proba)
        
        # Backtest
        signals = pd.Series(2 * y_pred - 1, index=test_dates)
        returns = prices.loc[test_dates, 'close'].pct_change()
        strat = signals.shift(1).fillna(0) * returns
        strat = strat.replace([np.inf, -np.inf], 0).fillna(0)
        
        sharpe = strat.mean() / strat.std() * np.sqrt(365) if strat.std() > 0 else 0.0
        
        print(f"  {test_dates[0].year}: Sharpe={sharpe:.3f} Acc={acc:.3f} AUC={auc:.3f}")
        sharpes.append(sharpe)
        accs.append(acc)

    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)

    print(f"\n{'='*80}")
    print(f"7-day horizon: Sharpe={mean_sharpe:.3f} ± {std_sharpe:.3f}, Acc={np.mean(accs):.3f}")
    print(f"1-day baseline: Sharpe=0.982")
    print(f"Donchian: Sharpe=1.062")
    
    if mean_sharpe > 1.062:
        print(f"✓ BEATS by {((mean_sharpe/1.062-1)*100):.1f}%")
    else:
        print(f"✗ Below by {((1-mean_sharpe/1.062)*100):.1f}%")

    timer.end(claimed_duration_minutes=10)

if __name__ == "__main__":
    main()
