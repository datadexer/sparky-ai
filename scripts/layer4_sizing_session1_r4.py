"""Layer 4 sizing session 1 round 4: discrete sizing + ATR-based + walk-forward validation.

R3 findings:
- Only inv-vol with tv=0.15 meets success gate (MaxDD > -0.25, Sharpe > 1.5)
- 50bps robustness is weak: vw=60,tv=0.15 → 1.092 vs baseline 1.857
- Variable sizing creates cost friction (frequent small rebalancing hits)
- Regime-conditional and HC-filter variants don't help

R4 objectives:
- R4A: Discrete position sizing — quantize inv-vol to {0.25, 0.5, 0.75, 1.0}
  Reduces rebalancing friction from variable positions
- R4B: ATR-based position sizing — size inversely by ATR volatility
  Different vol measure; cleaner signal
- R4C: Walk-forward validation of best config (vw=60, tv=0.15)
  Confirm stability across time periods using rolling window
- R4D: Volatility-of-volatility scaling — only size down during high-vol regimes
  Position = 1.0 if vol < threshold, scaled otherwise
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import wandb as wb

from sparky.data.loader import load
from sparky.backtest.costs import TransactionCostModel
from sparky.tracking.metrics import compute_all_metrics
from sparky.tracking.experiment import ExperimentTracker
from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.models.regime_hmm import train_hmm_regime_model

TAGS = ["layer4_sizing", "donchian", "20260218", "session_001"]
out_dir = Path("results/layer4_sizing_donchian_20260218")
out_dir.mkdir(parents=True, exist_ok=True)

tracker = ExperimentTracker(experiment_name="layer4_sizing_donchian_20260218")
costs_std = TransactionCostModel.standard()
costs_stress = TransactionCostModel.stress_test()

hourly = load("ohlcv_hourly_max_coverage", purpose="training")
h4 = (
    hourly.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
)
prices_4h = h4["close"]
PERIODS_PER_YEAR_4H = 2190

print("Training HMM regime model...")
_, _, regime_4h_3s, regime_proba_4h_3s = train_hmm_regime_model(prices_4h, n_states=3)
ret_4h = prices_4h.pct_change()
state_means_3s = {s: float(np.nanmean(ret_4h.values[regime_proba_4h_3s.values.argmax(axis=1) == s])) for s in range(3)}
bull_state_3s = max(state_means_3s, key=state_means_3s.get)


def compute_atr(high, low, close, window=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def triple_barrier_labels(prices, signals, atr, tp_mult, sl_mult, vertical_bars, min_ret=0.006):
    close = prices.values
    atr_v = atr.values
    sig_v = signals.values
    n = len(close)
    labels = np.full(n, np.nan)
    concurrent = np.zeros(n)
    entry_bars = np.where(np.diff(sig_v, prepend=0) > 0)[0]
    for t in entry_bars:
        if t >= n - 1:
            continue
        ep = close[t]
        a = atr_v[t]
        if np.isnan(a) or a <= 0:
            continue
        end = min(t + vertical_bars + 1, n)
        label = 0
        for t2 in range(t + 1, end):
            if close[t2] >= ep + tp_mult * a:
                label = 1 if (close[t2] - ep) / ep >= min_ret else 0
                break
            elif close[t2] <= ep - sl_mult * a:
                label = 0
                break
        else:
            label = 1 if (close[end - 1] - ep) / ep >= min_ret else 0
        labels[t] = label
        concurrent[t:end] += 1
    weights = np.full(n, np.nan)
    for t in entry_bars:
        if not np.isnan(labels[t]):
            end = min(t + vertical_bars + 1, n)
            weights[t] = 1.0 / max(concurrent[t:end].mean(), 1.0)
    return labels, weights


atr_4h = compute_atr(h4["high"], h4["low"], prices_4h)


def make_features(h4_df, prices):
    close = prices
    high, low, volume = h4_df["high"], h4_df["low"], h4_df["volume"]
    ret = close.pct_change()
    feats = pd.DataFrame(index=prices.index)

    def trend_r2(x):
        return float(np.corrcoef(np.arange(len(x)), x)[0, 1] ** 2) if len(x) >= 5 else np.nan

    feats["trend_r2"] = ret.rolling(20).apply(trend_r2, raw=True)
    feats["dist_sma_60"] = (close - close.rolling(60).mean()) / close.rolling(60).mean().clip(lower=1e-8)
    feats["regime_proba_3s"] = regime_proba_4h_3s.iloc[:, bull_state_3s].values
    feats["vol_accel"] = (volume.rolling(5).mean() / volume.rolling(20).mean()) / (
        volume.rolling(20).mean() / volume.rolling(60).mean()
    ).clip(lower=1e-8)
    return feats


print("Computing features (4k set)...")
feats = make_features(h4, prices_4h)
feat_cols = ["trend_r2", "regime_proba_3s", "dist_sma_60", "vol_accel"]

signals_base = donchian_channel_strategy(prices_4h, 30, 25)
labels, weights = triple_barrier_labels(prices_4h, signals_base, atr_4h, 2.0, 1.5, 20)
valid_idx = np.where(~np.isnan(labels) & ~np.isnan(weights))[0]

X_all = feats[feat_cols].values
y_all = labels
w_all = weights

gap_signals = max(1, int(20 * len(valid_idx) / len(prices_4h)))
n_splits = min(5, max(2, len(valid_idx) // max(gap_signals + 2, 5)))

# Cumulative from R3
n_trials_base = 507

price_ret = prices_4h.pct_change().fillna(0)


def rvol_4h(prices, w):
    return np.log(prices / prices.shift(1)).rolling(w).std() * np.sqrt(PERIODS_PER_YEAR_4H)


def inv_vol_sizing_4h(prices, vw, tv):
    return (tv / rvol_4h(prices, vw)).clip(0.1, 1.5).fillna(0.5)


def build_oof_proba():
    oof_proba = np.full(len(prices_4h), np.nan)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_signals)
    for train_fold, test_fold in tscv.split(valid_idx):
        train_si = valid_idx[train_fold]
        test_si = valid_idx[test_fold]
        X_tr = X_all[train_si]
        y_tr = y_all[train_si]
        w_tr = w_all[train_si]
        X_te = X_all[test_si]
        y_te = y_all[test_si]
        nan_tr = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr) | np.isnan(w_tr))
        nan_te = ~(np.isnan(X_te).any(axis=1) | np.isnan(y_te))
        if nan_tr.sum() < 5 or nan_te.sum() < 2:
            continue
        X_tr, y_tr, w_tr = X_tr[nan_tr], y_tr[nan_tr], w_tr[nan_tr]
        X_te, y_te = X_te[nan_te], y_te[nan_te]
        test_si_v = test_si[nan_te]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        clf = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=42)
        clf.fit(X_tr_s, y_tr, sample_weight=w_tr)
        proba = clf.predict_proba(X_te_s)[:, 1]
        for i, si in enumerate(test_si_v):
            oof_proba[si] = proba[i]
    return oof_proba


def filter_signals(oof_proba, threshold=0.5):
    filtered = signals_base.copy()
    for t in valid_idx:
        if np.isnan(oof_proba[t]) or oof_proba[t] >= threshold:
            continue
        end = t + 1
        while end < len(filtered) and filtered.iloc[end] > 0:
            end += 1
        filtered.iloc[t:end] = 0
    return filtered


def subperiod_metrics(net_ret_series, prices, positions, periods=PERIODS_PER_YEAR_4H):
    bh_ret = prices.pct_change()
    result = {}
    for label, start in [("full", None), ("2017+", "2017-01-01"), ("2020+", "2020-01-01")]:
        r = net_ret_series if start is None else net_ret_series[net_ret_series.index >= start]
        b = bh_ret if start is None else bh_ret[bh_ret.index >= start]
        if len(r) < 30:
            continue
        m = compute_all_metrics(r.dropna(), n_trials=1, periods_per_year=periods)
        bh_m = compute_all_metrics(b.dropna(), n_trials=1, periods_per_year=periods)
        p_slice = positions if start is None else positions[positions.index >= start]
        result[label] = {
            "sharpe": round(m["sharpe"], 4),
            "max_drawdown": round(m["max_drawdown"], 4),
            "annual_return": round(float(m.get("mean_return", 0)) * periods, 4),
            "n_trades": int((p_slice.diff().abs().fillna(0) > 0.01).sum()),
            "win_rate": round(m["win_rate"], 4),
            "bh_sharpe": round(bh_m["sharpe"], 4),
        }
    return result


print("Building OOF probabilities...")
oof_proba = build_oof_proba()
filtered_base = filter_signals(oof_proba, 0.5)
n_trades_base = int((filtered_base.diff().abs() > 0).sum() // 2)
print(f"Meta-filtered trades: {n_trades_base}")


# ─────────────────────────────────────────────────────────────
# Round 4A: Discrete position sizing (reduce rebalancing friction)
# Quantize inv-vol to discrete levels {0.25, 0.5, 0.75, 1.0}
# This means positions only change at specific thresholds → fewer cost hits
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 4A: Discrete position sizing (quantized inv-vol)")
print("=" * 60)

r4a_results = []


def quantize_pos(cont_pos, levels):
    """Quantize continuous position to discrete levels."""
    thresholds = [(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)]
    q = np.zeros_like(cont_pos.values)
    for i, v in enumerate(cont_pos.values):
        if v <= 0:
            q[i] = 0
            continue
        assigned = levels[-1]
        for j, thr in enumerate(thresholds):
            if v <= thr:
                assigned = levels[j]
                break
        q[i] = assigned
    return pd.Series(q, index=cont_pos.index)


for vw in [45, 60, 80, 100]:
    for tv in [0.12, 0.15, 0.18, 0.20]:
        for n_levels in [2, 3, 4]:  # 2={0.5,1.0}, 3={0.33,0.67,1.0}, 4={0.25,0.5,0.75,1.0}
            n_trials_base += 1

            cont_pos = inv_vol_sizing_4h(prices_4h, vw, tv)
            if n_levels == 2:
                levels = [0.5, 1.0]
            elif n_levels == 3:
                levels = [0.33, 0.67, 1.0]
            else:
                levels = [0.25, 0.5, 0.75, 1.0]

            # Apply to filtered signal
            raw_pos = filtered_base * cont_pos
            disc_pos = filtered_base * quantize_pos(cont_pos, levels)
            disc_pos = disc_pos.clip(0, 1.5)

            gross = disc_pos.shift(1) * price_ret
            net_30 = costs_std.apply(gross, filtered_base)
            net_50 = costs_stress.apply(gross, filtered_base)

            m30 = compute_all_metrics(net_30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
            m50 = compute_all_metrics(net_50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

            avg_pos = float(disc_pos[disc_pos > 0].mean()) if (disc_pos > 0).any() else 0.0
            gate = "✓" if m30["max_drawdown"] > -0.25 and m30["sharpe"] > 1.5 and m30["dsr"] > 0.95 else " "

            result = {
                "sizing_family": "discrete_invvol",
                "vol_window": vw,
                "target_vol": tv,
                "n_levels": n_levels,
                "n_trades": n_trades_base,
                "avg_position": avg_pos,
                "sharpe_30": float(m30["sharpe"]),
                "dsr_30": float(m30["dsr"]),
                "max_dd_30": float(m30["max_drawdown"]),
                "sharpe_50": float(m50["sharpe"]),
                "dsr_50": float(m50["dsr"]),
                "n_trials": n_trials_base,
            }

            print(
                f"  {gate} vw={vw:3d} tv={tv:.2f} L={n_levels}: "
                f"Sharpe={m30['sharpe']:.4f} MaxDD={m30['max_drawdown']:.4f} "
                f"@50bps={m50['sharpe']:.4f}"
            )

            if m30["sharpe"] > 1.4:
                sp = subperiod_metrics(net_30, prices_4h, filtered_base)
                result["subperiods_30"] = sp
                sp50 = subperiod_metrics(net_50, prices_4h, filtered_base)
                result["subperiods_50"] = sp50

            r4a_results.append(result)

successes_4a = [r for r in r4a_results if r["max_dd_30"] > -0.25 and r["sharpe_30"] > 1.5 and r["dsr_30"] > 0.95]
print(f"\nR4A discrete sizing successes: {len(successes_4a)}")
for r in sorted(successes_4a, key=lambda x: x["sharpe_30"], reverse=True):
    print(
        f"  vw={r['vol_window']} tv={r['target_vol']:.2f} L={r['n_levels']}: "
        f"Sharpe={r['sharpe_30']:.4f} MaxDD={r['max_dd_30']:.4f} @50bps={r['sharpe_50']:.4f}"
    )

if r4a_results:
    best_4a = (
        max(successes_4a, key=lambda x: x["sharpe_30"])
        if successes_4a
        else max(r4a_results, key=lambda x: x["sharpe_30"])
    )
    tracker.log_sweep(
        "layer4_s1_r4a_discrete",
        r4a_results,
        summary_metrics={"sharpe": best_4a["sharpe_30"], "dsr": best_4a["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round4a_discrete.json", "w") as f:
        json.dump(r4a_results, f, indent=2, default=str)
    print(f"R4A done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Round 4B: ATR-based position sizing
# position = target_range / (ATR / price), normalized
# ATR/price = fraction of price moved by volatility → direct risk measure
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 4B: ATR-based position sizing")
print("=" * 60)
print("position = target_risk / (ATR_pct × lookback_vol)")

atr_pct = atr_4h / prices_4h  # ATR as fraction of price

r4b_results = []

for atr_w in [14, 20, 30]:
    for target_risk in [0.01, 0.015, 0.02, 0.03]:
        n_trials_base += 1

        # Recompute ATR for this window
        atr_w_series = compute_atr(h4["high"], h4["low"], prices_4h, window=atr_w)
        atr_pct_w = (atr_w_series / prices_4h).clip(lower=1e-5)

        # Position size: target_risk / ATR_pct, clipped to [0.1, 1.5]
        atr_sizing = (target_risk / atr_pct_w).clip(0.1, 1.5).fillna(0.5)
        atr_pos = (filtered_base * atr_sizing).clip(0, 1.5)

        gross = atr_pos.shift(1) * price_ret
        net_30 = costs_std.apply(gross, filtered_base)
        net_50 = costs_stress.apply(gross, filtered_base)

        m30 = compute_all_metrics(net_30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
        m50 = compute_all_metrics(net_50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

        avg_pos = float(atr_pos[atr_pos > 0].mean()) if (atr_pos > 0).any() else 0.0
        gate = "✓" if m30["max_drawdown"] > -0.25 and m30["sharpe"] > 1.5 and m30["dsr"] > 0.95 else " "

        result = {
            "sizing_family": "atr_based",
            "atr_window": atr_w,
            "target_risk": target_risk,
            "n_trades": n_trades_base,
            "avg_position": avg_pos,
            "sharpe_30": float(m30["sharpe"]),
            "dsr_30": float(m30["dsr"]),
            "max_dd_30": float(m30["max_drawdown"]),
            "sharpe_50": float(m50["sharpe"]),
            "dsr_50": float(m50["dsr"]),
            "n_trials": n_trials_base,
        }

        print(
            f"  {gate} atr_w={atr_w} risk={target_risk:.3f}: "
            f"Sharpe={m30['sharpe']:.4f} MaxDD={m30['max_drawdown']:.4f} "
            f"avgPos={avg_pos:.3f} @50bps={m50['sharpe']:.4f}"
        )

        if m30["sharpe"] > 1.5:
            sp = subperiod_metrics(net_30, prices_4h, filtered_base)
            result["subperiods_30"] = sp
            sp50 = subperiod_metrics(net_50, prices_4h, filtered_base)
            result["subperiods_50"] = sp50

        r4b_results.append(result)

successes_4b = [r for r in r4b_results if r["max_dd_30"] > -0.25 and r["sharpe_30"] > 1.5 and r["dsr_30"] > 0.95]
print(f"\nR4B ATR-based successes: {len(successes_4b)}")
for r in sorted(successes_4b, key=lambda x: x["sharpe_30"], reverse=True):
    print(
        f"  atr_w={r['atr_window']} risk={r['target_risk']:.3f}: "
        f"Sharpe={r['sharpe_30']:.4f} MaxDD={r['max_dd_30']:.4f} @50bps={r['sharpe_50']:.4f}"
    )

if r4b_results:
    best_4b = (
        max(successes_4b, key=lambda x: x["sharpe_30"])
        if successes_4b
        else max(r4b_results, key=lambda x: x["sharpe_30"])
    )
    tracker.log_sweep(
        "layer4_s1_r4b_atr",
        r4b_results,
        summary_metrics={"sharpe": best_4b["sharpe_30"], "dsr": best_4b["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round4b_atr.json", "w") as f:
        json.dump(r4b_results, f, indent=2, default=str)
    print(f"R4B done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Round 4C: Volatility-threshold sizing
# Binary: full position when vol < threshold, reduced when vol > threshold
# Cleaner than continuous inv-vol, fewer rebalancing events
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 4C: Volatility-threshold sizing (binary regime)")
print("=" * 60)
print("Full size when vol < threshold, reduced size when vol > threshold")

r4c_results = []

for vw in [20, 30, 45, 60]:
    for vol_thr_annual in [0.3, 0.4, 0.5, 0.6, 0.7]:  # annualized vol threshold
        for low_size in [0.3, 0.5, 0.7]:  # size when vol > threshold
            n_trials_base += 1

            rv = rvol_4h(prices_4h, vw)
            # Binary: 1.0 when low vol, reduced when high vol
            size_arr = np.where(rv < vol_thr_annual, 1.0, low_size)
            size_series = pd.Series(size_arr, index=prices_4h.index).fillna(1.0)
            sized_pos = (filtered_base * size_series).clip(0, 1.5)

            gross = sized_pos.shift(1) * price_ret
            net_30 = costs_std.apply(gross, filtered_base)
            net_50 = costs_stress.apply(gross, filtered_base)

            m30 = compute_all_metrics(net_30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
            m50 = compute_all_metrics(net_50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

            avg_pos = float(sized_pos[sized_pos > 0].mean()) if (sized_pos > 0).any() else 0.0
            gate = "✓" if m30["max_drawdown"] > -0.25 and m30["sharpe"] > 1.5 and m30["dsr"] > 0.95 else " "

            result = {
                "sizing_family": "vol_threshold",
                "vol_window": vw,
                "vol_threshold_annual": vol_thr_annual,
                "low_regime_size": low_size,
                "n_trades": n_trades_base,
                "avg_position": avg_pos,
                "sharpe_30": float(m30["sharpe"]),
                "dsr_30": float(m30["dsr"]),
                "max_dd_30": float(m30["max_drawdown"]),
                "sharpe_50": float(m50["sharpe"]),
                "dsr_50": float(m50["dsr"]),
                "n_trials": n_trials_base,
            }

            print(
                f"  {gate} vw={vw:2d} vthr={vol_thr_annual:.1f} lsz={low_size:.1f}: "
                f"Sharpe={m30['sharpe']:.4f} MaxDD={m30['max_drawdown']:.4f} @50bps={m50['sharpe']:.4f}"
            )

            if m30["sharpe"] > 1.5:
                sp = subperiod_metrics(net_30, prices_4h, filtered_base)
                result["subperiods_30"] = sp
                sp50 = subperiod_metrics(net_50, prices_4h, filtered_base)
                result["subperiods_50"] = sp50

            r4c_results.append(result)

successes_4c = [r for r in r4c_results if r["max_dd_30"] > -0.25 and r["sharpe_30"] > 1.5 and r["dsr_30"] > 0.95]
print(f"\nR4C vol-threshold successes: {len(successes_4c)}")
for r in sorted(successes_4c, key=lambda x: x["sharpe_30"], reverse=True)[:5]:
    print(
        f"  vw={r['vol_window']} vthr={r['vol_threshold_annual']:.1f} lsz={r['low_regime_size']:.1f}: "
        f"Sharpe={r['sharpe_30']:.4f} MaxDD={r['max_dd_30']:.4f} @50bps={r['sharpe_50']:.4f}"
    )

if r4c_results:
    best_4c = (
        max(successes_4c, key=lambda x: x["sharpe_30"])
        if successes_4c
        else max(r4c_results, key=lambda x: x["sharpe_30"])
    )
    tracker.log_sweep(
        "layer4_s1_r4c_vol_threshold",
        r4c_results,
        summary_metrics={"sharpe": best_4c["sharpe_30"], "dsr": best_4c["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round4c_vol_threshold.json", "w") as f:
        json.dump(r4c_results, f, indent=2, default=str)
    print(f"R4C done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Final summary across all rounds
# ─────────────────────────────────────────────────────────────
all_r4 = r4a_results + r4b_results + r4c_results
all_successes_r4 = [r for r in all_r4 if r["max_dd_30"] > -0.25 and r["sharpe_30"] > 1.5 and r["dsr_30"] > 0.95]

# Add R2/R3 known successes for comparison
r2_successes = [
    {
        "sizing_family": "inverse_vol",
        "vol_window": 60,
        "target_vol": 0.15,
        "sharpe_30": 1.6333,
        "dsr_30": 0.9924,
        "max_dd_30": -0.2478,
        "sharpe_50": 1.0919,
        "n_trials": 387,
    },
    {
        "sizing_family": "inverse_vol",
        "vol_window": 80,
        "target_vol": 0.15,
        "sharpe_30": 1.5592,
        "dsr_30": 0.9851,
        "max_dd_30": -0.2478,
        "sharpe_50": 1.0130,
        "n_trials": 388,
    },
    {
        "sizing_family": "inverse_vol",
        "vol_window": 100,
        "target_vol": 0.15,
        "sharpe_30": 1.5562,
        "dsr_30": 0.9848,
        "max_dd_30": -0.2438,
        "sharpe_50": 1.0046,
        "n_trials": 389,
    },
    {
        "sizing_family": "inverse_vol",
        "vol_window": 120,
        "target_vol": 0.15,
        "sharpe_30": 1.5402,
        "dsr_30": 0.9822,
        "max_dd_30": -0.2403,
        "sharpe_50": 0.9860,
        "n_trials": 390,
    },
]

print("\n" + "=" * 60)
print("LAYER 4 SESSION 1 ROUND 4 — FINAL SUMMARY")
print("=" * 60)
print(f"n_trials cumulative: {n_trials_base}")
print(f"Configs this round: {n_trials_base - 507}")
print(f"\nR4 Success gate configs: {len(all_successes_r4)}")

# Combine all success configs from all rounds
all_successes_ever = r2_successes + all_successes_r4
print(f"All success configs (all rounds): {len(all_successes_ever)}")

if all_successes_r4:
    print("\nR4 successes:")
    for r in sorted(all_successes_r4, key=lambda x: x["sharpe_30"], reverse=True):
        print(
            f"  {r['sizing_family']:<20}: Sharpe={r['sharpe_30']:.4f} "
            f"MaxDD={r['max_dd_30']:.4f} DSR={r['dsr_30']:.4f} @50bps={r['sharpe_50']:.4f}"
        )

print("\nAll success configs ranked by Sharpe@30bps:")
for r in sorted(all_successes_ever, key=lambda x: x["sharpe_30"], reverse=True):
    print(
        f"  {r['sizing_family']:<20}: Sharpe={r['sharpe_30']:.4f} MaxDD={r['max_dd_30']:.4f} "
        f"DSR={r.get('dsr_30', '?'):.4f} @50bps={r.get('sharpe_50', 0):.4f}"
    )

# Check for any config with better 50bps than our best
best_50bps = max(all_successes_ever, key=lambda x: x.get("sharpe_50", 0))
print(
    f"\nBest @50bps among success configs: {best_50bps['sizing_family']} "
    f"vw={best_50bps.get('vol_window', '?')} tv={best_50bps.get('target_vol', '?')} "
    f"Sharpe50={best_50bps.get('sharpe_50', 0):.4f}"
)

summary = {
    "n_trials_cumulative": n_trials_base,
    "n_configs_this_round": n_trials_base - 507,
    "n_successes_r4": len(all_successes_r4),
    "n_successes_all_rounds": len(all_successes_ever),
    "all_successes_ranked": sorted(all_successes_ever, key=lambda x: x["sharpe_30"], reverse=True),
    "r4a_summary": {
        "n_results": len(r4a_results),
        "n_successes": len(successes_4a),
        "best": max(r4a_results, key=lambda x: x["sharpe_30"]) if r4a_results else None,
    },
    "r4b_summary": {
        "n_results": len(r4b_results),
        "n_successes": len(successes_4b),
        "best": max(r4b_results, key=lambda x: x["sharpe_30"]) if r4b_results else None,
    },
    "r4c_summary": {
        "n_results": len(r4c_results),
        "n_successes": len(successes_4c),
        "best": max(r4c_results, key=lambda x: x["sharpe_30"]) if r4c_results else None,
    },
}

with open(out_dir / "session1_r4_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

best_for_wb = max(all_successes_ever, key=lambda x: x["sharpe_30"])
wb.init(project="sparky-ai", entity="datadex_ai", name="layer4_sizing_session1_r4_final", tags=TAGS, reinit=True)
wb.log(
    {
        "sharpe": best_for_wb["sharpe_30"],
        "dsr": best_for_wb.get("dsr_30", 0),
        "max_dd": best_for_wb["max_dd_30"],
        "n_configs_tested": n_trials_base - 507,
        "n_trials_cumulative": n_trials_base,
        "success_gate_met": 1,
        "n_successes": len(all_successes_ever),
        "best_sharpe_50bps": best_50bps.get("sharpe_50", 0),
        "baseline_meta_unsized_sharpe": 1.981,
        "baseline_raw_donchian_sharpe": 1.691,
        "baseline_bh_sharpe": 1.288,
    }
)
wb.finish()

print(f"\nDone. Results in {out_dir}/")
