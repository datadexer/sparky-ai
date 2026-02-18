#!/usr/bin/env python3
"""
Try deeper trees (depth=6,7,8) with stronger regularization.
Hypothesis: Sweep stopped at depth=5, maybe depth=6+ helps.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime, UTC
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
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

def walk_forward_split(index, train_min=1095, test_len=365, step=365):
    folds = []
    for i in range(0, len(index), step):
        test_start = train_min + i
        test_end = test_start + test_len
        if test_end > len(index):
            break
        train_idx = index[:test_start]
        test_idx = index[test_start:test_end]
        if len(train_idx) >= train_min and len(test_idx) == test_len:
            folds.append((train_idx, test_idx))
    return folds

def main():
    timer = TaskTimer(agent_id="research")
    timer.start("train_deeper_trees")

    print("="*80)
    print("DEEPER TREES: CatBoost depth 6, 7, 8")
    print(f"Started: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*80)

    # Load
    features = pd.read_parquet("data/processed/feature_matrix_btc_hourly.parquet")
    targets = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")
    
    features_daily = features.resample('D').last().dropna()
    targets_daily = targets.resample('D').last().dropna()
    
    common = features_daily.index.intersection(targets_daily.index)
    X = features_daily.loc[common][SELECTED_FEATURES]
    y = targets_daily.loc[common]['target']
    
    prices = pd.read_parquet("data/raw/btc/ohlcv_hourly.parquet")
    prices_daily = prices.resample('D').last().dropna()
    prices_common = prices_daily[prices_daily.index.isin(X.index)].loc[X.index]
    
    print(f"\nSamples: {len(X)}, Features: {X.shape[1]}")
    
    configs = [
        {"depth": 6, "l2_leaf_reg": 3.0},
        {"depth": 7, "l2_leaf_reg": 5.0},
        {"depth": 8, "l2_leaf_reg": 10.0},
    ]
    
    all_results = []
    
    for cfg in configs:
        print(f"\n--- Testing depth={cfg['depth']}, l2={cfg['l2_leaf_reg']} ---")
        
        model_cfg = {
            "iterations": 200,
            "depth": cfg["depth"],
            "learning_rate": 0.01,
            "l2_leaf_reg": cfg["l2_leaf_reg"],
            "task_type": "GPU",
            "devices": "0",
            "verbose": 0,
        }
        
        sharpes, accs = [], []
        
        for train_dates, test_dates in walk_forward_split(X.index):
            model = CatBoostClassifier(**model_cfg)
            model.fit(X.loc[train_dates], y.loc[train_dates])
            
            y_pred_proba = model.predict_proba(X.loc[test_dates])[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            acc = accuracy_score(y.loc[test_dates], y_pred)
            
            signals = pd.Series(2 * y_pred - 1, index=test_dates)
            returns = prices_common.loc[test_dates, 'close'].pct_change()
            strat = signals.shift(1).fillna(0) * returns
            strat = strat.replace([np.inf, -np.inf], 0).fillna(0)
            
            sharpe = strat.mean() / strat.std() * np.sqrt(365) if strat.std() > 0 else 0.0
            
            print(f"  {test_dates[0].year}: Sharpe={sharpe:.3f} Acc={acc:.3f}")
            sharpes.append(sharpe)
            accs.append(acc)
        
        mean_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        
        all_results.append({
            "depth": cfg["depth"],
            "l2_leaf_reg": cfg["l2_leaf_reg"],
            "mean_sharpe": mean_sharpe,
            "std_sharpe": std_sharpe,
            "mean_acc": np.mean(accs),
        })
        
        print(f"  → Mean Sharpe={mean_sharpe:.3f} ± {std_sharpe:.3f}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in sorted(all_results, key=lambda x: x['mean_sharpe'], reverse=True):
        print(f"depth={r['depth']} l2={r['l2_leaf_reg']}: Sharpe={r['mean_sharpe']:.3f}")
    
    print(f"\nBaseline (depth=4): Sharpe=0.982")
    print(f"Donchian: Sharpe=1.062")
    
    best = max(all_results, key=lambda x: x['mean_sharpe'])
    
    with open("results/validation/deeper_trees.json", "w") as f:
        json.dump({"results": all_results, "best": best}, f, indent=2)
    
    if best['mean_sharpe'] > 1.062:
        print(f"\n✓ BEATS baseline by {((best['mean_sharpe']/1.062-1)*100):.1f}%")
    else:
        print(f"\n✗ Best still below baseline by {((1-best['mean_sharpe']/1.062)*100):.1f}%")

    timer.end(claimed_duration_minutes=25)

if __name__ == "__main__":
    main()
