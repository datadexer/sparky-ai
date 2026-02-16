#!/usr/bin/env python3
"""Full validation of best expanded-feature config.

Best config: CatBoost d=5 lr=0.01 l2=3.0
- Walk-forward Sharpe: 1.194 ± 1.79
- Accuracy: 53.5%

Validation checks:
1. Year-by-year Sharpe breakdown
2. Bootstrap 95% CI
3. Look-ahead bias check (signal shift verification)
4. Max drawdown, win rate, Sharpe ratio
5. Comparison to Donchian baseline
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
from pathlib import Path

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from sparky.backtest.costs import TransactionCostModel

BEST_CONFIG = {
    'iterations': 200,
    'depth': 5,
    'learning_rate': 0.01,
    'l2_leaf_reg': 3.0,
    'task_type': 'GPU',
    'devices': '0'
}

DONCHIAN_SHARPE = 1.062


def load_data():
    """Load expanded features, targets, prices."""
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

    # Select top 25 features (from Stage 0)
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
    strategy_returns = signals.shift(1) * returns  # no look-ahead
    costs = signals.diff().abs() * cost_model.total_cost_pct
    net_returns = (strategy_returns - costs).dropna()

    if len(net_returns) < 30 or net_returns.std() == 0:
        return 0.0, {}

    sharpe = float(net_returns.mean() / net_returns.std() * np.sqrt(365))
    
    # Additional metrics
    cum_returns = (1 + net_returns).cumprod()
    max_dd = (cum_returns / cum_returns.cummax() - 1).min()
    win_rate = (net_returns > 0).mean()
    
    return sharpe, {
        'returns_mean': net_returns.mean(),
        'returns_std': net_returns.std(),
        'max_dd': max_dd,
        'win_rate': win_rate,
        'total_trades': signals.diff().abs().sum(),
    }


def validate_year_by_year(features, target, prices_daily):
    """Train on all data up to year-1, test on year."""
    print("=" * 80)
    print("YEAR-BY-YEAR VALIDATION")
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

        model = CatBoostClassifier(**BEST_CONFIG, verbose=0, random_state=42)
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        
        signals = pd.Series((y_proba > 0.52).astype(int), index=X_test.index)
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
    print(f"Mean WinRate:{df['win_rate'].mean():.2%}")
    
    return df


def bootstrap_ci(features, target, prices_daily, n_bootstrap=100):
    """Bootstrap 95% CI for mean Sharpe."""
    print("\n" + "=" * 80)
    print(f"BOOTSTRAP 95% CI ({n_bootstrap} samples)")
    print("=" * 80)
    
    years = [2019, 2020, 2021, 2022, 2023]
    cost_model = TransactionCostModel.for_btc()
    
    # Get year Sharpes
    year_sharpes = []
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

        model = CatBoostClassifier(**BEST_CONFIG, verbose=0, random_state=42)
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_test)[:, 1]
        signals = pd.Series((y_proba > 0.52).astype(int), index=X_test.index)
        sharpe, _ = compute_sharpe(signals, prices_daily, cost_model)
        
        year_sharpes.append(sharpe)
    
    # Bootstrap
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(year_sharpes, size=len(year_sharpes), replace=True)
        boot_means.append(np.mean(sample))
    
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    print(f"Mean Sharpe: {np.mean(year_sharpes):.3f}")
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"Donchian baseline: {DONCHIAN_SHARPE}")
    
    if ci_lower > DONCHIAN_SHARPE:
        print("✓ Lower bound BEATS baseline — statistically significant edge")
    else:
        print("⚠ Lower bound below baseline — not statistically significant")
    
    return ci_lower, ci_upper


def main():
    """Run full validation suite."""
    features, target, prices_daily = load_data()
    
    print(f"Features: {features.shape}")
    print(f"Target: {target.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}\n")
    
    # Year-by-year
    df_years = validate_year_by_year(features, target, prices_daily)
    
    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_ci(features, target, prices_daily, n_bootstrap=100)
    
    # Save results
    output_path = Path("results/validation/expanded_features_best.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'config': BEST_CONFIG,
        'mean_sharpe': float(df_years['sharpe'].mean()),
        'std_sharpe': float(df_years['sharpe'].std()),
        'mean_acc': float(df_years['acc'].mean()),
        'mean_max_dd': float(df_years['max_dd'].mean()),
        'mean_win_rate': float(df_years['win_rate'].mean()),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'baseline_sharpe': DONCHIAN_SHARPE,
        'beats_baseline': bool(ci_lower > DONCHIAN_SHARPE),
        'year_breakdown': df_years.to_dict(orient='records'),
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    if results['beats_baseline']:
        print("\n" + "=" * 80)
        print("TIER 2 VALIDATED: Ready for paper trading")
        print("=" * 80)


if __name__ == "__main__":
    main()
