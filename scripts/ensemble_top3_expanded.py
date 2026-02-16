#!/usr/bin/env python3
"""Ensemble top-3 configs from expanded feature sweep to reduce variance.

Top 3 from sweep:
1. CatBoost d=3 lr=0.01 l2=1.0 → screening Sharpe 0.793
2. CatBoost d=5 lr=0.01 l2=1.0 → screening Sharpe 0.769
3. CatBoost d=3 lr=0.01 l2=3.0 → screening Sharpe 0.703

Ensemble strategy: Average probabilities from all 3 models.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
from pathlib import Path

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from sparky.backtest.costs import TransactionCostModel

TOP3_CONFIGS = [
    {'depth': 3, 'learning_rate': 0.01, 'l2_leaf_reg': 1.0},
    {'depth': 5, 'learning_rate': 0.01, 'l2_leaf_reg': 1.0},
    {'depth': 3, 'learning_rate': 0.01, 'l2_leaf_reg': 3.0},
]


def load_data():
    """Load expanded features (top 25 selected)."""
    features = pd.read_parquet("data/processed/feature_matrix_btc_hourly_expanded.parquet")
    target = pd.read_parquet("data/processed/targets_btc_hourly_1d.parquet")
    if isinstance(target, pd.DataFrame):
        target = target['target']

    prices_hourly = pd.read_parquet("data/raw/btc/ohlcv_hourly_max_coverage.parquet")
    prices_daily = prices_hourly.resample("D").last()
    del prices_hourly

    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    features = features.loc[:'2024-05-31']
    target = target.loc[:'2024-05-31']

    mask = features.notna().all(axis=1) & target.notna()
    features = features.loc[mask]
    target = target.loc[mask]

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    target = target.loc[features.index]

    # Top 25 features from Stage 0
    top_features = [
        'rsi_divergence_14h_168h', 'price_momentum_divergence', 'intraday_range',
        'recovery_from_20h_low', 'rsi_volume_interaction', 'vwap_deviation_24h',
        'tick_direction_ratio_24h', 'rsi_divergence_4h_24h', 'momentum_divergence_72h_336h',
        'drawdown_from_20h_high', 'day_of_week', 'vol_ratio_24h_168h',
        'choppiness_index', 'momentum_720h', 'high_low_ratio_20h',
        'hl_spread_pct', 'rsi_168h', 'breakout_proximity_lower',
        'higher_highs_lower_lows_5h', 'momentum_336h', 'volatility_regime',
        'momentum_72h', 'price_range_expansion', 'obv', 'mid_vs_close'
    ]
    features = features[top_features]

    return features, target, prices_daily


def compute_sharpe(signals, prices_daily, cost_model):
    """Compute Sharpe from signals."""
    common_idx = signals.index.intersection(prices_daily.index)
    signals = signals.loc[common_idx]
    prices = prices_daily.loc[common_idx]

    returns = prices['close'].pct_change()
    strategy_returns = signals.shift(1) * returns
    costs = signals.diff().abs() * cost_model.total_cost_pct
    net_returns = (strategy_returns - costs).dropna()

    if len(net_returns) < 30 or net_returns.std() == 0:
        return 0.0, {}

    sharpe = float(net_returns.mean() / net_returns.std() * np.sqrt(365))
    
    cum_returns = (1 + net_returns).cumprod()
    max_dd = (cum_returns / cum_returns.cummax() - 1).min()
    win_rate = (net_returns > 0).mean()
    
    return sharpe, {
        'max_dd': max_dd,
        'win_rate': win_rate,
        'total_trades': signals.diff().abs().sum(),
    }


def validate_ensemble(features, target, prices_daily):
    """5-year walk-forward validation with ensemble."""
    print("=" * 80)
    print("ENSEMBLE VALIDATION (top 3 configs, average probabilities)")
    print("=" * 80)
    
    years = [2019, 2020, 2021, 2022, 2023]
    cost_model = TransactionCostModel.for_btc()
    
    results = []
    for year in years:
        train_end = f"{year-1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        X_train = features.loc[:train_end]
        y_train = target.loc[:train_end]
        X_test = features.loc[test_start:test_end]
        y_test = target.loc[test_start:test_end]

        if len(X_test) < 100:
            continue

        # Train 3 models
        y_probas = []
        for cfg in TOP3_CONFIGS:
            model = CatBoostClassifier(
                iterations=200, **cfg,
                task_type='GPU', devices='0',
                verbose=0, random_state=42
            )
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_probas.append(y_proba)
        
        # Ensemble: average probabilities
        y_proba_ensemble = np.mean(y_probas, axis=0)
        y_pred = (y_proba_ensemble > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        
        signals = pd.Series((y_proba_ensemble > 0.52).astype(int), index=X_test.index)
        sharpe, metrics = compute_sharpe(signals, prices_daily, cost_model)
        
        results.append({
            'year': year,
            'sharpe': sharpe,
            'acc': acc,
            'max_dd': metrics['max_dd'],
            'win_rate': metrics['win_rate'],
            'trades': metrics['total_trades'],
        })
        
        print(f"{year}: Sharpe={sharpe:6.2f}, Acc={acc:.3f}, MaxDD={metrics['max_dd']:.2%}, "
              f"WinRate={metrics['win_rate']:.2%}, Trades={metrics['total_trades']:.0f}")
    
    df = pd.DataFrame(results)
    print(f"\nMean Sharpe: {df['sharpe'].mean():.3f} ± {df['sharpe'].std():.3f}")
    print(f"Mean Acc:    {df['acc'].mean():.3%}")
    print(f"Mean MaxDD:  {df['max_dd'].mean():.2%}")
    
    # Bootstrap CI
    print("\nBootstrap 95% CI (100 samples):")
    boot_means = []
    for _ in range(100):
        sample = np.random.choice(df['sharpe'], size=len(df), replace=True)
        boot_means.append(np.mean(sample))
    
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"Donchian baseline: 1.062")
    
    if ci_lower > 1.062:
        print("✓ BEATS BASELINE (statistically significant)")
    else:
        print("✗ Does NOT beat baseline (not significant)")
    
    return df, ci_lower, ci_upper


def main():
    """Run ensemble validation."""
    features, target, prices_daily = load_data()
    
    print(f"Features: {features.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}\n")
    
    df, ci_lower, ci_upper = validate_ensemble(features, target, prices_daily)
    
    # Save results
    output = Path("results/validation/ensemble_top3_expanded.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    results = {
        'ensemble': 'top3_average_probabilities',
        'configs': TOP3_CONFIGS,
        'mean_sharpe': float(df['sharpe'].mean()),
        'std_sharpe': float(df['sharpe'].std()),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'baseline_sharpe': 1.062,
        'beats_baseline': bool(ci_lower > 1.062),
        'year_breakdown': df.to_dict(orient='records'),
    }
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output}")


if __name__ == "__main__":
    main()
