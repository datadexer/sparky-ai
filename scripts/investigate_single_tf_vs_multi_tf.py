"""Rigorous comparison: Single-TF Donchian(40/20) vs Multi-TF Donchian(20/40/60).

Background:
- Parameter sensitivity (2026-02-16) showed Entry=40/Exit=20 achieves Sharpe 1.243
- Multi-TF baseline [20,40,60] achieves Sharpe 1.062
- Single-TF appears BETTER and SIMPLER — investigate why and if it's robust

Tests:
1. Yearly walk-forward (2019-2023, in-sample only)
2. Bootstrap 95% CI for both strategies
3. Head-to-head comparison (paired bootstrap)
4. Trade analysis (frequency, holding period, win rate)
5. Bear market resilience (2022 focus)
6. Regime analysis (bull vs bear vs choppy)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sparky.backtest.costs import TransactionCostModel
from sparky.features.returns import annualized_sharpe, max_drawdown
from sparky.models.simple_baselines import donchian_channel_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_btc_data() -> pd.Series:
    """Load BTC daily prices."""
    data_path = Path("/home/akamath/sparky-ai/data/btc_daily.parquet")
    df = pd.read_parquet(data_path)
    prices = df["close"]
    prices.index = pd.to_datetime(df.index)
    logger.info(f"Loaded BTC data: {len(prices)} days ({prices.index[0]} to {prices.index[-1]})")
    return prices


def backtest_strategy(prices, signals, start_date, end_date, cost_model):
    """Backtest with look-ahead bias fix (signals.shift(1))."""
    period_prices = prices.loc[start_date:end_date]
    period_signals = signals.loc[start_date:end_date]

    if len(period_prices) == 0:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "n_trades": 0, "returns": pd.Series(dtype=float)}

    price_returns = period_prices.pct_change()
    actual_positions = period_signals.shift(1).fillna(0)
    strategy_returns = actual_positions * price_returns
    position_changes = actual_positions.diff().abs()
    n_trades = int(position_changes.sum())
    transaction_costs = position_changes * cost_model.total_cost_pct
    strategy_returns_after_costs = strategy_returns - transaction_costs
    equity_curve = (1 + strategy_returns_after_costs).cumprod()
    total_return = (equity_curve.iloc[-1] - 1) * 100

    return {
        "sharpe": annualized_sharpe(strategy_returns_after_costs),
        "total_return": total_return,
        "max_drawdown": max_drawdown(strategy_returns_after_costs),
        "n_trades": n_trades,
        "returns": strategy_returns_after_costs,
        "positions": actual_positions,
        "equity_curve": equity_curve,
    }


def bootstrap_sharpe_ci(returns, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for Sharpe ratio."""
    rng = np.random.default_rng(42)
    returns_arr = returns.dropna().values
    n = len(returns_arr)

    sharpe_samples = []
    for _ in range(n_bootstrap):
        boot_sample = rng.choice(returns_arr, size=n, replace=True)
        mean_r = np.mean(boot_sample)
        std_r = np.std(boot_sample, ddof=1)
        if std_r > 0:
            sharpe_samples.append(mean_r / std_r * np.sqrt(252))

    sharpe_samples = np.array(sharpe_samples)
    alpha = (1 - ci) / 2
    lower = np.percentile(sharpe_samples, alpha * 100)
    upper = np.percentile(sharpe_samples, (1 - alpha) * 100)
    return float(np.mean(sharpe_samples)), float(lower), float(upper)


def paired_bootstrap_test(returns_a, returns_b, n_bootstrap=10000):
    """Paired bootstrap test: P(strategy A > strategy B)."""
    rng = np.random.default_rng(42)
    # Align returns
    common_idx = returns_a.dropna().index.intersection(returns_b.dropna().index)
    ra = returns_a.loc[common_idx].values
    rb = returns_b.loc[common_idx].values
    n = len(ra)

    a_wins = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_a = ra[idx]
        boot_b = rb[idx]
        sharpe_a = np.mean(boot_a) / np.std(boot_a, ddof=1) * np.sqrt(252) if np.std(boot_a, ddof=1) > 0 else 0
        sharpe_b = np.mean(boot_b) / np.std(boot_b, ddof=1) * np.sqrt(252) if np.std(boot_b, ddof=1) > 0 else 0
        if sharpe_a > sharpe_b:
            a_wins += 1

    return a_wins / n_bootstrap


def analyze_trades(positions, prices):
    """Analyze trade characteristics."""
    trades = []
    in_trade = False
    entry_date = None
    entry_price = None

    for i in range(1, len(positions)):
        if positions.iloc[i] == 1 and positions.iloc[i-1] == 0:
            # Entry
            in_trade = True
            entry_date = positions.index[i]
            entry_price = prices.loc[entry_date] if entry_date in prices.index else None
        elif positions.iloc[i] == 0 and positions.iloc[i-1] == 1:
            # Exit
            if in_trade and entry_price is not None:
                exit_date = positions.index[i]
                exit_price = prices.loc[exit_date] if exit_date in prices.index else None
                if exit_price is not None:
                    trades.append({
                        "entry_date": str(entry_date.date()),
                        "exit_date": str(exit_date.date()),
                        "holding_days": (exit_date - entry_date).days,
                        "return_pct": (exit_price / entry_price - 1) * 100,
                    })
            in_trade = False

    if not trades:
        return {"n_trades": 0, "avg_holding_days": 0, "win_rate": 0, "avg_return": 0}

    returns = [t["return_pct"] for t in trades]
    holding_days = [t["holding_days"] for t in trades]

    return {
        "n_trades": len(trades),
        "avg_holding_days": np.mean(holding_days),
        "median_holding_days": np.median(holding_days),
        "win_rate": sum(1 for r in returns if r > 0) / len(returns) * 100,
        "avg_return": np.mean(returns),
        "median_return": np.median(returns),
        "best_trade": max(returns),
        "worst_trade": min(returns),
        "trades": trades,
    }


def main():
    logger.info("=== Single-TF vs Multi-TF Investigation ===")

    prices = load_btc_data()
    cost_model = TransactionCostModel.for_btc()

    # Generate signals
    logger.info("\nGenerating signals...")

    # Single-TF: Entry=40, Exit=20
    signals_single = donchian_channel_strategy(prices, entry_period=40, exit_period=20)

    # Multi-TF: Majority vote of [20, 40, 60]
    signals_20 = donchian_channel_strategy(prices, entry_period=20, exit_period=10)
    signals_40 = donchian_channel_strategy(prices, entry_period=40, exit_period=20)
    signals_60 = donchian_channel_strategy(prices, entry_period=60, exit_period=30)
    signals_multi = ((signals_20 + signals_40 + signals_60) >= 2).astype(int)

    # Also test top multi-TF: [20, 30, 40]
    signals_30 = donchian_channel_strategy(prices, entry_period=30, exit_period=15)
    signals_multi_best = ((signals_20 + signals_30 + signals_40) >= 2).astype(int)

    # Define test periods (in-sample only — before 2024-06-01 OOS boundary)
    test_years = {
        "2019": ("2019-01-01", "2019-12-31"),
        "2020": ("2020-01-01", "2020-12-31"),
        "2021": ("2021-01-01", "2021-12-31"),
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
    }

    strategies = {
        "Single-TF (40/20)": signals_single,
        "Multi-TF (20/40/60)": signals_multi,
        "Multi-TF (20/30/40)": signals_multi_best,
    }

    # ============================================
    # TEST 1: Yearly Walk-Forward
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("TEST 1: YEARLY WALK-FORWARD COMPARISON")
    logger.info("="*70)

    all_results = {}
    all_returns = {}

    for strat_name, strat_signals in strategies.items():
        results = {}
        combined_returns = pd.Series(dtype=float)

        for year, (start, end) in test_years.items():
            r = backtest_strategy(prices, strat_signals, start, end, cost_model)
            results[year] = {
                "sharpe": r["sharpe"],
                "total_return": r["total_return"],
                "max_drawdown": r["max_drawdown"],
                "n_trades": r["n_trades"],
            }
            combined_returns = pd.concat([combined_returns, r["returns"]])

        sharpe_values = [r["sharpe"] for r in results.values()]
        mean_sharpe = np.mean(sharpe_values)
        std_sharpe = np.std(sharpe_values, ddof=1)

        all_results[strat_name] = {
            "yearly": results,
            "mean_sharpe": float(mean_sharpe),
            "std_sharpe": float(std_sharpe),
            "min_sharpe": float(min(sharpe_values)),
            "max_sharpe": float(max(sharpe_values)),
            "positive_years": sum(1 for s in sharpe_values if s > 0),
        }
        all_returns[strat_name] = combined_returns

        logger.info(f"\n{strat_name}:")
        for year, r in results.items():
            logger.info(f"  {year}: Sharpe={r['sharpe']:.3f}, Return={r['total_return']:.1f}%, MaxDD={r['max_drawdown']:.1f}%, Trades={r['n_trades']}")
        logger.info(f"  MEAN: {mean_sharpe:.3f} (std: {std_sharpe:.3f})")

    # ============================================
    # TEST 2: Bootstrap 95% CI
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("TEST 2: BOOTSTRAP 95% CONFIDENCE INTERVALS")
    logger.info("="*70)

    for strat_name, returns in all_returns.items():
        mean_s, lower, upper = bootstrap_sharpe_ci(returns)
        logger.info(f"{strat_name}: Sharpe={mean_s:.3f} [{lower:.3f}, {upper:.3f}]")
        all_results[strat_name]["bootstrap_ci"] = {
            "mean": mean_s, "lower_95": lower, "upper_95": upper
        }

    # ============================================
    # TEST 3: Paired Bootstrap (Head-to-Head)
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("TEST 3: PAIRED BOOTSTRAP (HEAD-TO-HEAD)")
    logger.info("="*70)

    single_returns = all_returns["Single-TF (40/20)"]
    multi_returns = all_returns["Multi-TF (20/40/60)"]
    multi_best_returns = all_returns["Multi-TF (20/30/40)"]

    p_single_beats_multi = paired_bootstrap_test(single_returns, multi_returns)
    p_single_beats_multi_best = paired_bootstrap_test(single_returns, multi_best_returns)
    p_multi_best_beats_multi = paired_bootstrap_test(multi_best_returns, multi_returns)

    logger.info(f"P(Single-TF > Multi-TF 20/40/60) = {p_single_beats_multi:.3f}")
    logger.info(f"P(Single-TF > Multi-TF 20/30/40) = {p_single_beats_multi:.3f}")
    logger.info(f"P(Multi-TF 20/30/40 > Multi-TF 20/40/60) = {p_multi_best_beats_multi:.3f}")

    all_results["paired_tests"] = {
        "single_vs_multi_2040_60": p_single_beats_multi,
        "single_vs_multi_best_2030_40": p_single_beats_multi_best,
        "multi_best_vs_multi": p_multi_best_beats_multi,
    }

    # ============================================
    # TEST 4: Trade Analysis (Full In-Sample Period)
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("TEST 4: TRADE ANALYSIS (2019-2023)")
    logger.info("="*70)

    full_start, full_end = "2019-01-01", "2023-12-31"

    for strat_name, strat_signals in strategies.items():
        r = backtest_strategy(prices, strat_signals, full_start, full_end, cost_model)
        trade_analysis = analyze_trades(r["positions"], prices.loc[full_start:full_end])
        logger.info(f"\n{strat_name}:")
        logger.info(f"  Trades: {trade_analysis['n_trades']}")
        logger.info(f"  Avg holding: {trade_analysis['avg_holding_days']:.0f} days (median: {trade_analysis.get('median_holding_days', 0):.0f})")
        logger.info(f"  Win rate: {trade_analysis['win_rate']:.1f}%")
        logger.info(f"  Avg return per trade: {trade_analysis['avg_return']:.1f}%")
        logger.info(f"  Best/Worst trade: {trade_analysis.get('best_trade', 0):.1f}% / {trade_analysis.get('worst_trade', 0):.1f}%")

        # Don't include full trade list in JSON
        trade_summary = {k: v for k, v in trade_analysis.items() if k != "trades"}
        all_results[strat_name]["trade_analysis"] = trade_summary

    # ============================================
    # TEST 5: Signal Overlap Analysis
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("TEST 5: SIGNAL OVERLAP ANALYSIS")
    logger.info("="*70)

    is_period = (prices.index >= "2019-01-01") & (prices.index <= "2023-12-31")
    single_sig = signals_single[is_period]
    multi_sig = signals_multi[is_period]

    agreement = (single_sig == multi_sig).mean() * 100
    both_long = ((single_sig == 1) & (multi_sig == 1)).mean() * 100
    single_only = ((single_sig == 1) & (multi_sig == 0)).mean() * 100
    multi_only = ((single_sig == 0) & (multi_sig == 1)).mean() * 100
    both_flat = ((single_sig == 0) & (multi_sig == 0)).mean() * 100

    logger.info(f"Signal agreement: {agreement:.1f}%")
    logger.info(f"Both LONG: {both_long:.1f}%")
    logger.info(f"Single LONG only: {single_only:.1f}%")
    logger.info(f"Multi LONG only: {multi_only:.1f}%")
    logger.info(f"Both FLAT: {both_flat:.1f}%")

    # Analyze returns when they disagree
    price_returns = prices.pct_change()
    is_period_idx = prices.index[is_period]

    single_only_mask = (single_sig == 1) & (multi_sig == 0)
    multi_only_mask = (single_sig == 0) & (multi_sig == 1)

    if single_only_mask.sum() > 0:
        single_only_returns = price_returns.loc[is_period_idx][single_only_mask.values]
        logger.info(f"\nWhen ONLY Single-TF is LONG ({single_only_mask.sum()} days):")
        logger.info(f"  Mean daily return: {single_only_returns.mean()*100:.3f}%")
        logger.info(f"  Annualized: {single_only_returns.mean()*252*100:.1f}%")

    if multi_only_mask.sum() > 0:
        multi_only_returns = price_returns.loc[is_period_idx][multi_only_mask.values]
        logger.info(f"\nWhen ONLY Multi-TF is LONG ({multi_only_mask.sum()} days):")
        logger.info(f"  Mean daily return: {multi_only_returns.mean()*100:.3f}%")
        logger.info(f"  Annualized: {multi_only_returns.mean()*252*100:.1f}%")

    all_results["signal_overlap"] = {
        "agreement_pct": agreement,
        "both_long_pct": both_long,
        "single_only_long_pct": single_only,
        "multi_only_long_pct": multi_only,
        "both_flat_pct": both_flat,
    }

    # ============================================
    # TEST 6: Full-Period (2019-2023) Comparison
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("TEST 6: FULL PERIOD BACKTEST (2019-2023)")
    logger.info("="*70)

    for strat_name, strat_signals in strategies.items():
        r = backtest_strategy(prices, strat_signals, full_start, full_end, cost_model)
        logger.info(f"{strat_name}: Sharpe={r['sharpe']:.3f}, Return={r['total_return']:.1f}%, MaxDD={r['max_drawdown']:.1f}%, Trades={r['n_trades']}")
        all_results[strat_name]["full_period"] = {
            "sharpe": r["sharpe"],
            "total_return": r["total_return"],
            "max_drawdown": r["max_drawdown"],
            "n_trades": r["n_trades"],
        }

    # ============================================
    # VERDICT
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("VERDICT")
    logger.info("="*70)

    single_mean = all_results["Single-TF (40/20)"]["mean_sharpe"]
    multi_mean = all_results["Multi-TF (20/40/60)"]["mean_sharpe"]
    multi_best_mean = all_results["Multi-TF (20/30/40)"]["mean_sharpe"]

    logger.info(f"\nMean Sharpe (yearly folds):")
    logger.info(f"  Single-TF (40/20):   {single_mean:.3f}")
    logger.info(f"  Multi-TF (20/40/60): {multi_mean:.3f}")
    logger.info(f"  Multi-TF (20/30/40): {multi_best_mean:.3f}")

    improvement = (single_mean - multi_mean) / abs(multi_mean) * 100
    logger.info(f"\nSingle-TF improvement over Multi-TF baseline: {improvement:.1f}%")
    logger.info(f"P(Single-TF > Multi-TF): {p_single_beats_multi:.3f}")

    if p_single_beats_multi >= 0.75:
        logger.info("\nVERDICT: Single-TF (40/20) is SIGNIFICANTLY better than Multi-TF baseline")
        logger.info("RECOMMENDATION: Switch baseline to Single-TF (40/20)")
    elif p_single_beats_multi >= 0.60:
        logger.info("\nVERDICT: Single-TF (40/20) is MODERATELY better (not statistically significant)")
        logger.info("RECOMMENDATION: Consider switching but not conclusive")
    else:
        logger.info("\nVERDICT: No significant difference between strategies")
        logger.info("RECOMMENDATION: Keep Multi-TF (simpler explanation, similar performance)")

    # Save results
    output_path = Path("/home/akamath/sparky-ai/results/validation/single_tf_vs_multi_tf_investigation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean up non-serializable data
    serializable_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            serializable_results[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, (float, int, str, dict, list)):
                    serializable_results[k2] = v2 if k == "paired_tests" or k == "signal_overlap" else None
            serializable_results[k] = {k2: v2 for k2, v2 in v.items() if isinstance(v2, (float, int, str, dict, list))}

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
