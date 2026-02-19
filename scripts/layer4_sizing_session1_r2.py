"""Layer 4 sizing session 1 round 2: redesigned after calibration collapse.

Findings from R1: Platt calibration collapses proba spread (std=0.032, mean=0.468),
making 2p-1 near-zero and Kelly positions too small to overcome costs.
Calibrated Kelly is non-viable at this sample size.

New approach:
- R2A: Kelly with UNCALIBRATED probabilities (better spread, proven OOF accuracy 0.561)
- R2B: Inverse vol on meta-filtered signals (4 windows x 4 target vols)
- R2C: Combined best uncal-Kelly + best inv-vol if both show independent benefit
- R2D: High-confidence filter variants (thr>0.5) to reduce trade count / drawdown
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
print(f"Valid label count: {len(valid_idx)}, pos_rate: {np.nanmean(labels[valid_idx]):.3f}")

X_all = feats[feat_cols].values
y_all = labels
w_all = weights

gap_signals = max(1, int(20 * len(valid_idx) / len(prices_4h)))
max_splits = max(2, len(valid_idx) // max(gap_signals + 2, 5))
n_splits = min(5, max_splits)

# n_trials: R1 tested 4 Kelly configs (all bad) + baseline = 5
# Continue from 360 + 5 = 365
n_trials_base = 365

price_ret = prices_4h.pct_change().fillna(0)


def rvol_4h(prices, w):
    return np.log(prices / prices.shift(1)).rolling(w).std() * np.sqrt(PERIODS_PER_YEAR_4H)


def inv_vol_sizing_4h(prices, vw, tv):
    return (tv / rvol_4h(prices, vw)).clip(0.1, 1.5).fillna(0.5)


def build_oof_proba_uncal():
    oof_proba = np.full(len(prices_4h), np.nan)
    fold_accs = []
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
        fold_accs.append(float(((proba > 0.5) == y_te).mean()))
        for i, si in enumerate(test_si_v):
            oof_proba[si] = proba[i]
    return oof_proba, fold_accs


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


print("Building uncalibrated OOF probabilities...")
oof_proba_uncal, fold_accs_uncal = build_oof_proba_uncal()
uncal_valid = oof_proba_uncal[valid_idx[~np.isnan(oof_proba_uncal[valid_idx])]]
print(f"Uncal OOF accuracy: {np.mean(fold_accs_uncal):.3f}")
print(
    f"Uncal prob stats: mean={uncal_valid.mean():.3f} std={uncal_valid.std():.3f} "
    f"min={uncal_valid.min():.3f} max={uncal_valid.max():.3f}"
)

# Baseline meta-filter at thr=0.5 (matches session 2 best: Sharpe=1.981)
filtered_base = filter_signals(oof_proba_uncal, 0.5)
gross_base = filtered_base.shift(1) * price_ret
net_base_30 = costs_std.apply(gross_base, filtered_base)
n_trades_base_filter = int((filtered_base.diff().abs() > 0).sum() // 2)
m_base = compute_all_metrics(net_base_30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
print(
    f"Baseline confirmed: Sharpe={m_base['sharpe']:.4f} DSR={m_base['dsr']:.4f} "
    f"MaxDD={m_base['max_drawdown']:.4f} trades={n_trades_base_filter}"
)


# ─────────────────────────────────────────────────────────────
# Round 2A: Kelly with UNCALIBRATED probabilities
# The uncal proba has spread: mean~0.55, std~0.12 → edge = 2p-1 in [-0.5, +0.5]
# Only apply Kelly to accepted trades (oof_proba >= 0.5)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 2A: Kelly with uncalibrated probabilities")
print("=" * 60)

r2a_results = []

for kelly_frac in [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]:
    n_trials_base += 1

    kelly_pos = pd.Series(0.0, index=prices_4h.index)
    for t in valid_idx:
        p = oof_proba_uncal[t]
        if np.isnan(p) or p < 0.5:  # only size accepted trades
            continue
        edge = 2 * p - 1  # positive since p >= 0.5
        kelly_pos.iloc[t] = min(1.0, kelly_frac * edge)

    # Forward-fill within trade windows
    sig_arr = filtered_base.values
    kp_arr = kelly_pos.values.copy()
    for t in range(1, len(sig_arr)):
        if sig_arr[t] > 0 and sig_arr[t - 1] > 0 and kp_arr[t] == 0.0:
            kp_arr[t] = kp_arr[t - 1]
    kelly_pos = pd.Series(kp_arr.clip(0, 1), index=prices_4h.index)

    gross_k = kelly_pos.shift(1) * price_ret
    net_k30 = costs_std.apply(gross_k, filtered_base)
    net_k50 = costs_stress.apply(gross_k, filtered_base)
    n_trades_k = int((filtered_base.diff().abs() > 0).sum() // 2)

    if n_trades_k < 5:
        continue

    m30 = compute_all_metrics(net_k30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
    m50 = compute_all_metrics(net_k50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

    result = {
        "sizing_family": "fractional_kelly_uncal",
        "kelly_fraction": kelly_frac,
        "use_calibration": False,
        "n_trades": n_trades_k,
        "avg_position": float(kelly_pos[kelly_pos > 0].mean()),
        "sharpe_30": float(m30["sharpe"]),
        "dsr_30": float(m30["dsr"]),
        "max_dd_30": float(m30["max_drawdown"]),
        "sharpe_50": float(m50["sharpe"]),
        "dsr_50": float(m50["dsr"]),
        "n_trials": n_trials_base,
    }

    print(
        f"  Kelly={kelly_frac}: Sharpe={m30['sharpe']:.4f} DSR={m30['dsr']:.4f} "
        f"MaxDD={m30['max_drawdown']:.4f} avgPos={result['avg_position']:.3f}"
    )

    if m30["sharpe"] > 1.691:
        sp = subperiod_metrics(net_k30, prices_4h, filtered_base)
        result["subperiods_30"] = sp
        sp50 = subperiod_metrics(net_k50, prices_4h, filtered_base)
        result["subperiods_50"] = sp50
        print(
            f"    2017+: Sharpe={sp.get('2017+', {}).get('sharpe', 'N/A')} "
            f"MaxDD={sp.get('2017+', {}).get('max_drawdown', 'N/A')}"
        )
        print(
            f"    2020+: Sharpe={sp.get('2020+', {}).get('sharpe', 'N/A')} "
            f"MaxDD={sp.get('2020+', {}).get('max_drawdown', 'N/A')}"
        )

    r2a_results.append(result)

if r2a_results:
    best_r2a = max(r2a_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR2A best: Kelly={best_r2a['kelly_fraction']} Sharpe={best_r2a['sharpe_30']:.4f} "
        f"MaxDD={best_r2a['max_dd_30']:.4f}"
    )
    tracker.log_sweep(
        "layer4_s1_r2a_kelly_uncal",
        r2a_results,
        summary_metrics={"sharpe": best_r2a["sharpe_30"], "dsr": best_r2a["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round2a_kelly_uncal.json", "w") as f:
        json.dump(r2a_results, f, indent=2, default=str)
    print(f"R2A done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Round 2B: Inverse Volatility Sizing on meta-filtered signals
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 2B: Inverse Volatility Sizing")
print("=" * 60)

r2b_results = []

for vw in [20, 30, 45, 60]:
    for tv in [0.15, 0.2, 0.3, 0.4]:
        n_trials_base += 1

        vol_scale = inv_vol_sizing_4h(prices_4h, vw, tv)
        invvol_pos = filtered_base * vol_scale
        invvol_pos = invvol_pos.clip(0, 1.5)

        gross_iv = invvol_pos.shift(1) * price_ret
        net_iv30 = costs_std.apply(gross_iv, filtered_base)
        net_iv50 = costs_stress.apply(gross_iv, filtered_base)
        n_trades_iv = int((filtered_base.diff().abs() > 0).sum() // 2)

        if n_trades_iv < 5:
            continue

        m30 = compute_all_metrics(net_iv30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
        m50 = compute_all_metrics(net_iv50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

        result = {
            "sizing_family": "inverse_vol",
            "vol_window": vw,
            "target_vol": tv,
            "n_trades": n_trades_iv,
            "avg_position": float(invvol_pos[invvol_pos > 0].mean()),
            "sharpe_30": float(m30["sharpe"]),
            "dsr_30": float(m30["dsr"]),
            "max_dd_30": float(m30["max_drawdown"]),
            "sharpe_50": float(m50["sharpe"]),
            "dsr_50": float(m50["dsr"]),
            "n_trials": n_trials_base,
        }

        print(
            f"  vw={vw} tv={tv}: Sharpe={m30['sharpe']:.4f} DSR={m30['dsr']:.4f} "
            f"MaxDD={m30['max_drawdown']:.4f} avgPos={result['avg_position']:.3f}"
        )

        if m30["sharpe"] > 1.691:
            sp = subperiod_metrics(net_iv30, prices_4h, filtered_base)
            result["subperiods_30"] = sp
            sp50 = subperiod_metrics(net_iv50, prices_4h, filtered_base)
            result["subperiods_50"] = sp50
            print(
                f"    2017+: Sharpe={sp.get('2017+', {}).get('sharpe', 'N/A')} "
                f"MaxDD={sp.get('2017+', {}).get('max_drawdown', 'N/A')}"
            )
            print(
                f"    2020+: Sharpe={sp.get('2020+', {}).get('sharpe', 'N/A')} "
                f"MaxDD={sp.get('2020+', {}).get('max_drawdown', 'N/A')}"
            )

        r2b_results.append(result)

if r2b_results:
    best_r2b = max(r2b_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR2B best: vw={best_r2b['vol_window']} tv={best_r2b['target_vol']} "
        f"Sharpe={best_r2b['sharpe_30']:.4f} MaxDD={best_r2b['max_dd_30']:.4f}"
    )

    tracker.log_sweep(
        "layer4_s1_r2b_inv_vol",
        r2b_results,
        summary_metrics={"sharpe": best_r2b["sharpe_30"], "dsr": best_r2b["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round2b_invvol.json", "w") as f:
        json.dump(r2b_results, f, indent=2, default=str)
    print(f"R2B done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Round 2C: High-confidence filter (thr > 0.5) to reduce drawdown
# Filter trades more aggressively — only take highest-confidence signals
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 2C: High-confidence filter variants (thr > 0.5)")
print("=" * 60)

r2c_results = []

for thr in [0.55, 0.60, 0.65, 0.70]:
    n_trials_base += 1

    filtered_hc = filter_signals(oof_proba_uncal, thr)
    gross_hc = filtered_hc.shift(1) * price_ret
    net_hc30 = costs_std.apply(gross_hc, filtered_hc)
    net_hc50 = costs_stress.apply(gross_hc, filtered_hc)
    n_trades_hc = int((filtered_hc.diff().abs() > 0).sum() // 2)

    if n_trades_hc < 5:
        print(f"  thr={thr}: too few trades ({n_trades_hc}), skip")
        continue

    m30 = compute_all_metrics(net_hc30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
    m50 = compute_all_metrics(net_hc50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

    result = {
        "sizing_family": "high_confidence_filter",
        "threshold": thr,
        "n_trades": n_trades_hc,
        "sharpe_30": float(m30["sharpe"]),
        "dsr_30": float(m30["dsr"]),
        "max_dd_30": float(m30["max_drawdown"]),
        "sharpe_50": float(m50["sharpe"]),
        "dsr_50": float(m50["dsr"]),
        "n_trials": n_trials_base,
    }

    print(
        f"  thr={thr}: Sharpe={m30['sharpe']:.4f} DSR={m30['dsr']:.4f} "
        f"MaxDD={m30['max_drawdown']:.4f} trades={n_trades_hc}"
    )

    if m30["sharpe"] > 1.691:
        sp = subperiod_metrics(net_hc30, prices_4h, filtered_hc)
        result["subperiods_30"] = sp
        sp50 = subperiod_metrics(net_hc50, prices_4h, filtered_hc)
        result["subperiods_50"] = sp50
        print(
            f"    2017+: Sharpe={sp.get('2017+', {}).get('sharpe', 'N/A')} "
            f"MaxDD={sp.get('2017+', {}).get('max_drawdown', 'N/A')}"
        )
        print(
            f"    2020+: Sharpe={sp.get('2020+', {}).get('sharpe', 'N/A')} "
            f"MaxDD={sp.get('2020+', {}).get('max_drawdown', 'N/A')}"
        )

    r2c_results.append(result)

if r2c_results:
    best_r2c = max(r2c_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR2C best: thr={best_r2c['threshold']} Sharpe={best_r2c['sharpe_30']:.4f} "
        f"MaxDD={best_r2c['max_dd_30']:.4f} trades={best_r2c['n_trades']}"
    )
    tracker.log_sweep(
        "layer4_s1_r2c_high_confidence",
        r2c_results,
        summary_metrics={"sharpe": best_r2c["sharpe_30"], "dsr": best_r2c["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round2c_highconf.json", "w") as f:
        json.dump(r2c_results, f, indent=2, default=str)
    print(f"R2C done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Round 2D: Combined Kelly (uncal) + Inverse Vol
# Best combos from R2A and R2B
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 2D: Combined Kelly (uncal) + Inverse Vol")
print("=" * 60)

r2d_results = []

# Focused search: pick 3 best Kelly fracs x 3 best vol params
kelly_fracs = [0.25, 0.35, 0.5, 0.75, 1.0]
invvol_params = [(20, 0.15), (20, 0.2), (30, 0.15), (30, 0.2), (45, 0.15), (45, 0.2)]

for kf in kelly_fracs:
    for vw, tv in invvol_params:
        n_trials_base += 1

        # Kelly positions on accepted trades
        kelly_pos = pd.Series(0.0, index=prices_4h.index)
        for t in valid_idx:
            p = oof_proba_uncal[t]
            if np.isnan(p) or p < 0.5:
                continue
            edge = 2 * p - 1
            kelly_pos.iloc[t] = min(1.0, kf * edge)

        sig_arr = filtered_base.values
        kp_arr = kelly_pos.values.copy()
        for t in range(1, len(sig_arr)):
            if sig_arr[t] > 0 and sig_arr[t - 1] > 0 and kp_arr[t] == 0.0:
                kp_arr[t] = kp_arr[t - 1]
        kelly_pos = pd.Series(kp_arr.clip(0, 1), index=prices_4h.index)

        # Scale Kelly positions by inverse vol
        vol_scale = inv_vol_sizing_4h(prices_4h, vw, tv)
        combined_pos = (kelly_pos * vol_scale).clip(0, 1.5)

        gross_c = combined_pos.shift(1) * price_ret
        net_c30 = costs_std.apply(gross_c, filtered_base)
        net_c50 = costs_stress.apply(gross_c, filtered_base)
        n_trades_c = int((filtered_base.diff().abs() > 0).sum() // 2)

        if n_trades_c < 5:
            continue

        m30 = compute_all_metrics(net_c30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
        m50 = compute_all_metrics(net_c50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

        result = {
            "sizing_family": "combined_kelly_invvol",
            "kelly_fraction": kf,
            "vol_window": vw,
            "target_vol": tv,
            "use_calibration": False,
            "n_trades": n_trades_c,
            "avg_position": float(combined_pos[combined_pos > 0].mean()),
            "sharpe_30": float(m30["sharpe"]),
            "dsr_30": float(m30["dsr"]),
            "max_dd_30": float(m30["max_drawdown"]),
            "sharpe_50": float(m50["sharpe"]),
            "dsr_50": float(m50["dsr"]),
            "n_trials": n_trials_base,
        }

        print(
            f"  Kelly={kf} vw={vw} tv={tv}: Sharpe={m30['sharpe']:.4f} "
            f"MaxDD={m30['max_drawdown']:.4f} DSR={m30['dsr']:.4f}"
        )

        if m30["sharpe"] > 1.691:
            sp = subperiod_metrics(net_c30, prices_4h, filtered_base)
            result["subperiods_30"] = sp
            sp50 = subperiod_metrics(net_c50, prices_4h, filtered_base)
            result["subperiods_50"] = sp50
            print(
                f"    2017+: Sharpe={sp.get('2017+', {}).get('sharpe', 'N/A')} "
                f"MaxDD={sp.get('2017+', {}).get('max_drawdown', 'N/A')}"
            )
            print(
                f"    2020+: Sharpe={sp.get('2020+', {}).get('sharpe', 'N/A')} "
                f"MaxDD={sp.get('2020+', {}).get('max_drawdown', 'N/A')}"
            )

        r2d_results.append(result)

if r2d_results:
    best_r2d = max(r2d_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR2D best: Kelly={best_r2d['kelly_fraction']} vw={best_r2d['vol_window']} tv={best_r2d['target_vol']} "
        f"Sharpe={best_r2d['sharpe_30']:.4f} MaxDD={best_r2d['max_dd_30']:.4f}"
    )
    tracker.log_sweep(
        "layer4_s1_r2d_combined",
        r2d_results,
        summary_metrics={"sharpe": best_r2d["sharpe_30"], "dsr": best_r2d["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round2d_combined.json", "w") as f:
        json.dump(r2d_results, f, indent=2, default=str)
    print(f"R2D done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────
all_results = r2a_results + r2b_results + r2c_results + r2d_results

print("\n" + "=" * 60)
print("FINAL SUMMARY — Layer 4 Session 1")
print("=" * 60)
print(f"Baselines: raw={1.691:.4f}, meta-unsized={1.981:.4f}, B&H={1.288:.4f}")
print(f"n_trials cumulative: {n_trials_base}")
print(f"Configs tested this session: {n_trials_base - 360}")

successes = [r for r in all_results if r["max_dd_30"] > -0.25 and r["sharpe_30"] > 1.5 and r["dsr_30"] > 0.95]
print(f"Success gate (MaxDD>-0.25, Sharpe>1.5, DSR>0.95): {len(successes)} configs")

if all_results:
    best_overall = max(all_results, key=lambda x: x["sharpe_30"])
    best_maxdd = max(all_results, key=lambda x: x["max_dd_30"])  # least negative
    print(f"\nBest Sharpe: {best_overall['sharpe_30']:.4f} (family={best_overall['sizing_family']})")
    print(
        f"Best MaxDD: {best_maxdd['max_dd_30']:.4f} (Sharpe={best_maxdd['sharpe_30']:.4f}, family={best_maxdd['sizing_family']})"
    )

    if successes:
        best_s = max(successes, key=lambda x: x["sharpe_30"])
        print("\n*** SUCCESS GATE MET ***")
        print(f"  Sharpe={best_s['sharpe_30']:.4f} DSR={best_s['dsr_30']:.4f} MaxDD={best_s['max_dd_30']:.4f}")
        print(f"  Family={best_s['sizing_family']}")
        if "subperiods_30" in best_s:
            for k, v in best_s["subperiods_30"].items():
                print(f"  {k}: Sharpe={v.get('sharpe', 'N/A')} MaxDD={v.get('max_drawdown', 'N/A')}")

# Rank all by MaxDD then Sharpe (to find best trade-off)
print("\nTop 10 by MaxDD (descending = least negative, i.e. best risk control):")
for r in sorted(all_results, key=lambda x: x["max_dd_30"], reverse=True)[:10]:
    print(f"  {r['sizing_family']:<30} Sharpe={r['sharpe_30']:.4f} MaxDD={r['max_dd_30']:.4f} DSR={r['dsr_30']:.4f}")

summary = {
    "n_configs_session": n_trials_base - 360,
    "n_trials_cumulative": n_trials_base,
    "success_gate_met": len(successes) > 0,
    "n_successes": len(successes),
    "successes": successes,
    "r2a_kelly_best": max(r2a_results, key=lambda x: x["sharpe_30"]) if r2a_results else None,
    "r2b_invvol_best": max(r2b_results, key=lambda x: x["sharpe_30"]) if r2b_results else None,
    "r2c_highconf_best": max(r2c_results, key=lambda x: x["sharpe_30"]) if r2c_results else None,
    "r2d_combined_best": max(r2d_results, key=lambda x: x["sharpe_30"]) if r2d_results else None,
    "best_maxdd_config": max(all_results, key=lambda x: x["max_dd_30"]) if all_results else None,
    "findings": {
        "r1_finding": "Calibrated Kelly (Platt sigmoid) fails: collapses proba spread (std=0.032), positions near-zero",
        "r2a_finding": "TBD",
        "r2b_finding": "TBD",
        "r2c_finding": "TBD",
        "r2d_finding": "TBD",
    },
}

with open(out_dir / "session1_r2_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

best_s_or_overall = (
    max(successes, key=lambda x: x["sharpe_30"])
    if successes
    else (max(all_results, key=lambda x: x["sharpe_30"]) if all_results else {"sharpe_30": 0, "dsr_30": 0})
)
wb.init(project="sparky-ai", entity="datadex_ai", name="layer4_sizing_session1_r2_final", tags=TAGS, reinit=True)
wb.log(
    {
        "sharpe": best_s_or_overall["sharpe_30"],
        "dsr": best_s_or_overall["dsr_30"],
        "max_dd": max(all_results, key=lambda x: x["max_dd_30"])["max_dd_30"] if all_results else 0,
        "n_configs_tested": n_trials_base - 360,
        "n_trials_cumulative": n_trials_base,
        "success_gate_met": int(len(successes) > 0),
        "n_successes": len(successes),
        "baseline_meta_unsized_sharpe": 1.981,
        "baseline_raw_donchian_sharpe": 1.691,
        "baseline_bh_sharpe": 1.288,
    }
)
wb.finish()

print(f"\nDone. Results saved to {out_dir}/")
