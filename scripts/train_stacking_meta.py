#!/usr/bin/env python3
"""
Stacking: Train Cat/XGB/Light base models, then meta-model on their predictions.
Hypothesis: Meta-learner finds optimal combination weights.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
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
    timer.start("train_stacking_meta")

    print("="*80)
    print("STACKING: Cat + XGB + Light → Logistic meta-model")
    print(f"Started: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*80)

    features = pd.read_parquet("data/processed/feature_matrix_btc_hourly_clean.parquet")
    targets = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")
    
    features_daily = features.resample('D').last().dropna()
    targets_daily = targets.resample('D').last().dropna()
    
    common = features_daily.index.intersection(targets_daily.index)
    X = features_daily.loc[common][SELECTED_FEATURES]
    y = targets_daily.loc[common]['target']
    
    print(f"\nSamples: {len(X)}, Features: {X.shape[1]}")

    # Base models
    cat_cfg = {"iterations": 200, "depth": 4, "learning_rate": 0.01, "l2_leaf_reg": 1.0, "task_type": "GPU", "devices": "0", "verbose": 0}
    xgb_cfg = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.01, "reg_lambda": 1.0, "tree_method": "hist", "device": "cuda"}
    lgb_cfg = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.01, "reg_lambda": 1.0, "device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0, "verbose": -1}

    print("\nWalk-forward stacking...")
    accs = []

    for train_dates, test_dates in walk_forward_split(X.index):
        X_train, y_train = X.loc[train_dates], y.loc[train_dates]
        X_test, y_test = X.loc[test_dates], y.loc[test_dates]
        
        # Split train into base_train (60%) and meta_train (40%)
        split_idx = int(len(X_train) * 0.6)
        X_base = X_train.iloc[:split_idx]
        y_base = y_train.iloc[:split_idx]
        X_meta_train = X_train.iloc[split_idx:]
        y_meta_train = y_train.iloc[split_idx:]
        
        # Train base models
        cat = CatBoostClassifier(**cat_cfg)
        cat.fit(X_base, y_base)
        
        xgb = XGBClassifier(**xgb_cfg)
        xgb.fit(X_base, y_base)
        
        lgb = LGBMClassifier(**lgb_cfg)
        lgb.fit(X_base, y_base)
        
        # Generate meta-features on meta_train
        cat_meta = cat.predict_proba(X_meta_train)[:, 1].reshape(-1, 1)
        xgb_meta = xgb.predict_proba(X_meta_train)[:, 1].reshape(-1, 1)
        lgb_meta = lgb.predict_proba(X_meta_train)[:, 1].reshape(-1, 1)
        
        X_meta = np.hstack([cat_meta, xgb_meta, lgb_meta])
        
        # Train meta-model
        meta = LogisticRegression(max_iter=1000)
        meta.fit(X_meta, y_meta_train)
        
        # Predict on test
        cat_test = cat.predict_proba(X_test)[:, 1].reshape(-1, 1)
        xgb_test = xgb.predict_proba(X_test)[:, 1].reshape(-1, 1)
        lgb_test = lgb.predict_proba(X_test)[:, 1].reshape(-1, 1)
        
        X_test_meta = np.hstack([cat_test, xgb_test, lgb_test])
        
        y_pred_proba = meta.predict_proba(X_test_meta)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"  {test_dates[0].year}: Acc={acc:.3f} AUC={auc:.3f}")
        accs.append(acc)

    mean_acc = np.mean(accs)

    print(f"\n{'='*80}")
    print(f"Stacking: Mean Acc={mean_acc:.3f}")
    print(f"CatBoost alone: 0.530")
    print(f"XGBoost alone: 0.518")
    print(f"Donchian baseline: Sharpe 1.062")
    
    result = {"mean_acc": mean_acc, "yearly_accs": accs}
    
    with open("results/validation/stacking_meta.json", "w") as f:
        json.dump(result, f, indent=2)

    if mean_acc > 0.530:
        print(f"\n✓ Stacking improves over single CatBoost by {((mean_acc/0.530-1)*100):.1f}%")
    else:
        print(f"\n✗ Stacking underperforms single CatBoost by {((1-mean_acc/0.530)*100):.1f}%")

    timer.end(claimed_duration_minutes=30)

if __name__ == "__main__":
    main()
