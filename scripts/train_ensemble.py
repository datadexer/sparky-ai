#!/usr/bin/env python3
"""
Ensemble of CatBoost + XGBoost (voting).
Hypothesis: Averaging predictions reduces overfitting.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
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
    timer = TaskTimer(agent_id="ceo")
    timer.start("train_ensemble")

    print("="*80)
    print("ENSEMBLE: CatBoost + XGBoost (soft voting)")
    print(f"Started: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*80)

    # Load from sweep results
    features = pd.read_parquet("data/processed/feature_matrix_btc_hourly.parquet")
    targets = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")
    
    # Resample to daily
    features_daily = features.resample('D').last().dropna()
    targets_daily = targets.resample('D').last().dropna()
    
    # Align
    common = features_daily.index.intersection(targets_daily.index)
    X = features_daily.loc[common][SELECTED_FEATURES]
    y = targets_daily.loc[common]['target']
    
    print(f"\nSamples: {len(X)}")
    print(f"Date range: {X.index[0]} to {X.index[-1]}")
    print(f"Target: {y.mean()*100:.1f}% positive")

    # Configs from sweep
    cat_config = {
        "iterations": 200, "depth": 4, "learning_rate": 0.01,
        "l2_leaf_reg": 1.0, "task_type": "GPU", "devices": "0", "verbose": 0,
    }
    xgb_config = {
        "n_estimators": 200, "max_depth": 3, "learning_rate": 0.01,
        "reg_lambda": 1.0, "tree_method": "hist", "device": "cuda",
    }

    backtester = WalkForwardBacktester(
        initial_train_years=3, test_period_days=365, step_size_days=365,
    )

    print("\nWalk-forward validation...")
    sharpes, accs = [], []

    for train_dates, test_dates in backtester.generate_folds(X.index):
        X_train, y_train = X.loc[train_dates], y.loc[train_dates]
        X_test, y_test = X.loc[test_dates], y.loc[test_dates]
        
        # Train both
        cat = CatBoostClassifier(**cat_config)
        cat.fit(X_train, y_train)
        
        xgb = XGBClassifier(**xgb_config)
        xgb.fit(X_train, y_train)
        
        # Predict proba
        cat_proba = cat.predict_proba(X_test)[:, 1]
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        
        # Average (soft voting)
        ensemble_proba = (cat_proba + xgb_proba) / 2
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        acc = accuracy_score(y_test, ensemble_pred)
        auc = roc_auc_score(y_test, ensemble_proba)
        
        # Backtest
        signals = pd.Series(2 * ensemble_pred - 1, index=test_dates)
        
        # Load prices for this fold
        test_prices = pd.read_parquet("data/raw/btc_hourly_okx.parquet")
        test_prices = test_prices[test_prices.index.isin(test_dates)]
        returns = test_prices['close'].pct_change()
        
        strat = signals.shift(1).fillna(0) * returns
        strat = strat.replace([np.inf, -np.inf], 0).fillna(0)
        
        sharpe = strat.mean() / strat.std() * np.sqrt(365) if strat.std() > 0 else 0.0
        
        print(f"  {test_dates[0].year}: Sharpe={sharpe:.3f} Acc={acc:.3f} AUC={auc:.3f}")
        sharpes.append(sharpe)
        accs.append(acc)

    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)

    print(f"\n{'='*80}")
    print(f"Ensemble: Sharpe={mean_sharpe:.3f} ± {std_sharpe:.3f}, Acc={np.mean(accs):.3f}")
    print(f"CatBoost alone: 0.982")
    print(f"XGBoost alone: 0.940")
    print(f"Donchian: 1.062")
    
    if mean_sharpe > 1.062:
        print(f"✓ BEATS baseline by {((mean_sharpe/1.062-1)*100):.1f}%")
    else:
        print(f"✗ Below baseline by {((1-mean_sharpe/1.062)*100):.1f}%")

    timer.end(claimed_duration_minutes=20)

if __name__ == "__main__":
    main()
