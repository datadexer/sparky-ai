"""Layer 4 sizing session 1: Kelly + inv-vol sizing on best meta-labeled Donchian(30,25)."""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
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

# Base Donchian signals — FROZEN
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
print(f"CV: {n_splits} splits, gap={gap_signals}")

# n_trials starts at 360 (cumulative from Layer 3)
n_trials_base = 360

price_ret = prices_4h.pct_change().fillna(0)


def rvol_4h(prices, w):
    return np.log(prices / prices.shift(1)).rolling(w).std() * np.sqrt(PERIODS_PER_YEAR_4H)


def inv_vol_sizing_4h(prices, vw, tv):
    return (tv / rvol_4h(prices, vw)).clip(0.1, 1.5).fillna(0.5)


def subperiod_metrics(net_ret_series, prices, positions, periods=PERIODS_PER_YEAR_4H):
    """Compute sub-period metrics for 2017+ and 2020+ sub-periods."""
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


# ─────────────────────────────────────────────────────────────
# Build OOF probabilities with base LogReg (uncalibrated) — used as reference
# ─────────────────────────────────────────────────────────────
def build_oof_proba(use_calibration=False, calibration_method="sigmoid", calibration_cv=3):
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

        base_clf = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=42)

        if use_calibration:
            # Calibrate using inner CV on the training fold
            clf = CalibratedClassifierCV(base_clf, method=calibration_method, cv=calibration_cv)
            clf.fit(X_tr_s, y_tr)
        else:
            base_clf.fit(X_tr_s, y_tr, sample_weight=w_tr)
            clf = base_clf

        proba = clf.predict_proba(X_te_s)[:, 1]
        fold_accs.append(float(((proba > 0.5) == y_te).mean()))
        for i, si in enumerate(test_si_v):
            oof_proba[si] = proba[i]

    return oof_proba, fold_accs


# ─────────────────────────────────────────────────────────────
# Build filtered signals from oof_proba at threshold
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Baseline: uncalibrated binary filter (matches session 2 best)
# ─────────────────────────────────────────────────────────────
print("\nBuilding baseline OOF probabilities (uncalibrated)...")
oof_proba_uncal, fold_accs_uncal = build_oof_proba(use_calibration=False)
print(f"Uncalibrated OOF accuracy: {np.mean(fold_accs_uncal):.3f}")

filtered_base = filter_signals(oof_proba_uncal, 0.5)
gross_base = filtered_base.shift(1) * price_ret
net_base_30 = costs_std.apply(gross_base, filtered_base)
net_base_50 = costs_stress.apply(gross_base, filtered_base)
n_trades_base = int((filtered_base.diff().abs() > 0).sum() // 2)

n_trials_base += 1
m_base = compute_all_metrics(net_base_30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
print(
    f"Baseline (uncal binary filter): Sharpe={m_base['sharpe']:.4f} DSR={m_base['dsr']:.4f} "
    f"MaxDD={m_base['max_drawdown']:.4f} trades={n_trades_base}"
)


# ─────────────────────────────────────────────────────────────
# Round 1: Fractional Kelly with Calibrated Probabilities
# Priority 1 — CalibratedClassifierCV(sigmoid, cv=3)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 1: Fractional Kelly with Calibrated Probabilities")
print("=" * 60)

print("Building calibrated OOF probabilities (Platt sigmoid, cv=3)...")
oof_proba_cal, fold_accs_cal = build_oof_proba(use_calibration=True, calibration_method="sigmoid", calibration_cv=3)
print(f"Calibrated OOF accuracy: {np.mean(fold_accs_cal):.3f}")
cal_valid = (~np.isnan(oof_proba_cal[valid_idx])).sum()
cal_proba_vals = oof_proba_cal[valid_idx[~np.isnan(oof_proba_cal[valid_idx])]]
print(
    f"Calibrated prob stats: mean={cal_proba_vals.mean():.3f} std={cal_proba_vals.std():.3f} "
    f"min={cal_proba_vals.min():.3f} max={cal_proba_vals.max():.3f}"
)

r1_results = []

for kelly_frac in [0.15, 0.25, 0.35, 0.5]:
    n_trials_base += 1

    # Build Kelly-sized positions from calibrated probabilities
    # Position = kelly_frac * (2p - 1), applied at entry bars, forward-filled in trade
    filtered_cal = filter_signals(oof_proba_cal, 0.5)

    kelly_pos = pd.Series(0.0, index=prices_4h.index)
    for t in valid_idx:
        p = oof_proba_cal[t]
        if np.isnan(p) or filtered_cal.iloc[t] == 0:
            continue
        edge = 2 * p - 1  # ranges -1 to +1; positive = model thinks profitable
        kelly_pos.iloc[t] = max(0.0, kelly_frac * edge)

    # Forward-fill within trade windows
    sig_arr = filtered_cal.values
    kp_arr = kelly_pos.values.copy()
    for t in range(1, len(sig_arr)):
        if sig_arr[t] > 0 and sig_arr[t - 1] > 0 and kp_arr[t] == 0.0:
            kp_arr[t] = kp_arr[t - 1]
    kelly_pos = pd.Series(kp_arr.clip(0, 1), index=prices_4h.index)

    gross_k = kelly_pos.shift(1) * price_ret
    net_k30 = costs_std.apply(gross_k, filtered_cal)
    net_k50 = costs_stress.apply(gross_k, filtered_cal)
    n_trades_k = int((filtered_cal.diff().abs() > 0).sum() // 2)

    if n_trades_k < 5:
        continue

    m30 = compute_all_metrics(net_k30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
    m50 = compute_all_metrics(net_k50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

    result = {
        "sizing_family": "fractional_kelly",
        "kelly_fraction": kelly_frac,
        "use_calibration": True,
        "calibration_method": "sigmoid",
        "calibration_cv": 3,
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

    # Sub-period analysis for any config that beats baseline Sharpe
    if m30["sharpe"] > 1.691:
        sp = subperiod_metrics(net_k30, prices_4h, filtered_cal)
        result["subperiods_30"] = sp
        sp50 = subperiod_metrics(net_k50, prices_4h, filtered_cal)
        result["subperiods_50"] = sp50
        print(
            f"    Sub-periods: 2017+={sp.get('2017+', {}).get('sharpe', 'N/A')} "
            f"2020+={sp.get('2020+', {}).get('sharpe', 'N/A')}"
        )
        print(f"    MaxDD 2020+={sp.get('2020+', {}).get('max_drawdown', 'N/A')}")

    r1_results.append(result)

if r1_results:
    best_r1 = max(r1_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR1 best: Kelly={best_r1['kelly_fraction']} Sharpe={best_r1['sharpe_30']:.4f} "
        f"MaxDD={best_r1['max_dd_30']:.4f} DSR={best_r1['dsr_30']:.4f}"
    )

    tracker.log_sweep(
        "layer4_s1_r1_fractional_kelly",
        r1_results,
        summary_metrics={"sharpe": best_r1["sharpe_30"], "dsr": best_r1["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round1_kelly_results.json", "w") as f:
        json.dump(r1_results, f, indent=2, default=str)
    print(f"Round 1 done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Round 2: Inverse Volatility Sizing
# Priority 2 — applied on top of meta-label filter (binary, uncalibrated)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 2: Inverse Volatility Sizing on meta-labeled returns")
print("=" * 60)

r2_results = []

for vw in [20, 30, 45, 60]:
    for tv in [0.15, 0.2, 0.3, 0.4]:
        n_trials_base += 1

        # Inv-vol scales on the META-filtered positions
        vol_scale = inv_vol_sizing_4h(prices_4h, vw, tv)
        invvol_pos = filtered_base * vol_scale  # element-wise: 0 or 1 * vol_scale
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
                f"    Sub-periods: 2017+={sp.get('2017+', {}).get('sharpe', 'N/A')} "
                f"2020+={sp.get('2020+', {}).get('sharpe', 'N/A')}"
            )
            print(f"    MaxDD 2020+={sp.get('2020+', {}).get('max_drawdown', 'N/A')}")

        r2_results.append(result)

if r2_results:
    best_r2 = max(r2_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR2 best: vw={best_r2['vol_window']} tv={best_r2['target_vol']} "
        f"Sharpe={best_r2['sharpe_30']:.4f} MaxDD={best_r2['max_dd_30']:.4f}"
    )

    tracker.log_sweep(
        "layer4_s1_r2_inv_vol",
        r2_results,
        summary_metrics={"sharpe": best_r2["sharpe_30"], "dsr": best_r2["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round2_invvol_results.json", "w") as f:
        json.dump(r2_results, f, indent=2, default=str)
    print(f"Round 2 done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Round 3: Combined Kelly (calibrated) + Inverse Vol
# Priority 3 — only if families 1 and 2 each show independent improvement
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 3: Combined Kelly + Inverse Vol")
print("=" * 60)

r3_results = []

# Pick best params from R1 and R2 to combine
best_kelly_params = [(0.15,), (0.25,), (0.35,)]
best_invvol_params = [(20, 0.15), (20, 0.2), (30, 0.15), (30, 0.2), (45, 0.15), (45, 0.2)]

for (kf,) in best_kelly_params:
    for vw, tv in best_invvol_params:
        n_trials_base += 1

        # Build Kelly-calibrated positions
        filtered_cal = filter_signals(oof_proba_cal, 0.5)
        kelly_pos = pd.Series(0.0, index=prices_4h.index)
        for t in valid_idx:
            p = oof_proba_cal[t]
            if np.isnan(p) or filtered_cal.iloc[t] == 0:
                continue
            edge = 2 * p - 1
            kelly_pos.iloc[t] = max(0.0, kf * edge)

        sig_arr = filtered_cal.values
        kp_arr = kelly_pos.values.copy()
        for t in range(1, len(sig_arr)):
            if sig_arr[t] > 0 and sig_arr[t - 1] > 0 and kp_arr[t] == 0.0:
                kp_arr[t] = kp_arr[t - 1]
        kelly_pos = pd.Series(kp_arr.clip(0, 1), index=prices_4h.index)

        # Scale Kelly positions by inverse vol
        vol_scale = inv_vol_sizing_4h(prices_4h, vw, tv)
        combined_pos = (kelly_pos * vol_scale).clip(0, 1.5)

        gross_c = combined_pos.shift(1) * price_ret
        net_c30 = costs_std.apply(gross_c, filtered_cal)
        net_c50 = costs_stress.apply(gross_c, filtered_cal)
        n_trades_c = int((filtered_cal.diff().abs() > 0).sum() // 2)

        if n_trades_c < 5:
            continue

        m30 = compute_all_metrics(net_c30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
        m50 = compute_all_metrics(net_c50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

        result = {
            "sizing_family": "combined_kelly_invvol",
            "kelly_fraction": kf,
            "vol_window": vw,
            "target_vol": tv,
            "use_calibration": True,
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
            sp = subperiod_metrics(net_c30, prices_4h, filtered_cal)
            result["subperiods_30"] = sp
            sp50 = subperiod_metrics(net_c50, prices_4h, filtered_cal)
            result["subperiods_50"] = sp50
            print(
                f"    Sub-periods: 2017+={sp.get('2017+', {}).get('sharpe', 'N/A')} "
                f"2020+={sp.get('2020+', {}).get('sharpe', 'N/A')}"
            )
            print(f"    MaxDD 2020+={sp.get('2020+', {}).get('max_drawdown', 'N/A')}")

        r3_results.append(result)

if r3_results:
    best_r3 = max(r3_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR3 best: Kelly={best_r3['kelly_fraction']} vw={best_r3['vol_window']} tv={best_r3['target_vol']} "
        f"Sharpe={best_r3['sharpe_30']:.4f} MaxDD={best_r3['max_dd_30']:.4f}"
    )

    tracker.log_sweep(
        "layer4_s1_r3_combined",
        r3_results,
        summary_metrics={"sharpe": best_r3["sharpe_30"], "dsr": best_r3["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round3_combined_results.json", "w") as f:
        json.dump(r3_results, f, indent=2, default=str)
    print(f"Round 3 done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Final summary & wandb log
# ─────────────────────────────────────────────────────────────
all_results = r1_results + r2_results + r3_results

best_overall = max(all_results, key=lambda x: x["sharpe_30"]) if all_results else None
best_maxdd = min(all_results, key=lambda x: x["max_dd_30"]) if all_results else None

# Success gate: MaxDD > -0.25 AND Sharpe > 1.5 AND DSR > 0.95
successes = [r for r in all_results if r["max_dd_30"] > -0.25 and r["sharpe_30"] > 1.5 and r["dsr_30"] > 0.95]

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Total configs tested this session: {n_trials_base - 360}")
print(f"Total cumulative n_trials: {n_trials_base}")
print(f"Configs meeting success gate (MaxDD>-0.25, Sharpe>1.5, DSR>0.95): {len(successes)}")

if successes:
    best_success = max(successes, key=lambda x: x["sharpe_30"])
    print("\n*** SUCCESS GATE MET ***")
    print(
        f"  Sharpe={best_success['sharpe_30']:.4f} DSR={best_success['dsr_30']:.4f} "
        f"MaxDD={best_success['max_dd_30']:.4f}"
    )
    print(f"  Family: {best_success['sizing_family']}")

if best_overall:
    print(f"\nBest Sharpe: {best_overall['sharpe_30']:.4f} (family={best_overall['sizing_family']})")
    print(
        f"Best MaxDD (least negative): {best_maxdd['max_dd_30']:.4f} "
        f"(Sharpe={best_maxdd['sharpe_30']:.4f}, family={best_maxdd['sizing_family']})"
    )

summary = {
    "n_configs_this_session": n_trials_base - 360,
    "n_trials_cumulative": n_trials_base,
    "best_sharpe_30": best_overall["sharpe_30"] if best_overall else 0,
    "best_dsr_30": best_overall["dsr_30"] if best_overall else 0,
    "best_max_dd": best_maxdd["max_dd_30"] if best_maxdd else 0,
    "success_gate_met": len(successes) > 0,
    "n_successes": len(successes),
    "baselines": {
        "raw_donchian_30_25": 1.691,
        "meta_unsized": 1.981,
        "buy_and_hold": 1.288,
    },
    "r1_kelly_best": max(r1_results, key=lambda x: x["sharpe_30"]) if r1_results else None,
    "r2_invvol_best": max(r2_results, key=lambda x: x["sharpe_30"]) if r2_results else None,
    "r3_combined_best": max(r3_results, key=lambda x: x["sharpe_30"]) if r3_results else None,
    "successes": successes,
}

with open(out_dir / "session1_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

wb.init(project="sparky-ai", entity="datadex_ai", name="layer4_sizing_session1_final", tags=TAGS, reinit=True)
wb.log(
    {
        "sharpe": best_overall["sharpe_30"] if best_overall else 0,
        "dsr": best_overall["dsr_30"] if best_overall else 0,
        "max_dd": best_maxdd["max_dd_30"] if best_maxdd else 0,
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

print(f"\nDone. Results in {out_dir}/")
