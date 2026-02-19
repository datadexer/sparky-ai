"""Layer 4 sizing session 1 round 3: fine-tune inv-vol around success zone.

Round 2B findings:
- inv-vol is the working mechanism (Kelly fails completely)
- Success gate met: vw=60 tv=0.15 → Sharpe=1.633 MaxDD=-0.248
- vw=45 tv=0.15 → Sharpe=1.650 MaxDD=-0.252 (just outside gate)
- vw=30 tv=0.15 → Sharpe=1.699 MaxDD=-0.267 (outside gate)
- Clear MaxDD vs Sharpe trade-off as tv increases

Round 3 objectives:
- R3A: Explore tv in [0.05, 0.10, 0.12, 0.15, 0.18] with vw in [45, 60, 80, 100]
  Find the Pareto-optimal configs that maximize Sharpe subject to MaxDD > -0.25
- R3B: Full sub-period analysis on top success configs
- R3C: Combined inv-vol + high-confidence filter (thr=0.55)
  Does stricter filtering help MaxDD further?
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

# n_trials: R2 went to 421
n_trials_base = 421

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
# Round 3A: Fine-grained inv-vol sweep around success zone
# Focus: lower tv values to achieve MaxDD > -0.25
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 3A: Fine-grained inv-vol sweep (low target vols)")
print("=" * 60)
print("Success zone: need MaxDD > -0.25 and Sharpe > 1.5")
print("R2B showed vw=60 tv=0.15 just barely achieves MaxDD=-0.248")
print("Exploring lower tvols and longer vol windows")

r3a_results = []

for vw in [45, 60, 80, 100, 120]:
    for tv in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        n_trials_base += 1

        vol_scale = inv_vol_sizing_4h(prices_4h, vw, tv)
        invvol_pos = (filtered_base * vol_scale).clip(0, 1.5)

        gross_iv = invvol_pos.shift(1) * price_ret
        net_iv30 = costs_std.apply(gross_iv, filtered_base)
        net_iv50 = costs_stress.apply(gross_iv, filtered_base)

        m30 = compute_all_metrics(net_iv30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
        m50 = compute_all_metrics(net_iv50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

        avg_pos = float(invvol_pos[invvol_pos > 0].mean()) if (invvol_pos > 0).any() else 0.0

        result = {
            "sizing_family": "inverse_vol",
            "vol_window": vw,
            "target_vol": tv,
            "n_trades": n_trades_base,
            "avg_position": avg_pos,
            "sharpe_30": float(m30["sharpe"]),
            "dsr_30": float(m30["dsr"]),
            "max_dd_30": float(m30["max_drawdown"]),
            "sharpe_50": float(m50["sharpe"]),
            "dsr_50": float(m50["dsr"]),
            "max_dd_50": float(m50["max_drawdown"]),
            "n_trials": n_trials_base,
        }

        gate = "✓" if m30["max_drawdown"] > -0.25 and m30["sharpe"] > 1.5 and m30["dsr"] > 0.95 else " "
        print(
            f"  {gate} vw={vw:3d} tv={tv:.2f}: Sharpe={m30['sharpe']:.4f} DSR={m30['dsr']:.4f} "
            f"MaxDD={m30['max_drawdown']:.4f} avgPos={avg_pos:.3f}"
        )

        if m30["sharpe"] > 1.5:
            sp = subperiod_metrics(net_iv30, prices_4h, filtered_base)
            result["subperiods_30"] = sp
            sp50 = subperiod_metrics(net_iv50, prices_4h, filtered_base)
            result["subperiods_50"] = sp50
            if sp:
                print(
                    f"      2017+: Sharpe={sp.get('2017+', {}).get('sharpe', 'N/A')} "
                    f"MaxDD={sp.get('2017+', {}).get('max_drawdown', 'N/A')}"
                )
                print(
                    f"      2020+: Sharpe={sp.get('2020+', {}).get('sharpe', 'N/A')} "
                    f"MaxDD={sp.get('2020+', {}).get('max_drawdown', 'N/A')}"
                )

        r3a_results.append(result)

successes_3a = [r for r in r3a_results if r["max_dd_30"] > -0.25 and r["sharpe_30"] > 1.5 and r["dsr_30"] > 0.95]
print(f"\nR3A success configs: {len(successes_3a)}")
for r in sorted(successes_3a, key=lambda x: x["sharpe_30"], reverse=True):
    print(
        f"  vw={r['vol_window']} tv={r['target_vol']:.2f}: Sharpe={r['sharpe_30']:.4f} "
        f"MaxDD={r['max_dd_30']:.4f} DSR={r['dsr_30']:.4f} @50bps={r['sharpe_50']:.4f}"
    )

if r3a_results:
    best_success_3a = max(successes_3a, key=lambda x: x["sharpe_30"]) if successes_3a else None
    best_3a = max(r3a_results, key=lambda x: x["sharpe_30"])
    summary_m = best_success_3a if best_success_3a else best_3a
    tracker.log_sweep(
        "layer4_s1_r3a_invvol_fine",
        r3a_results,
        summary_metrics={"sharpe": summary_m["sharpe_30"], "dsr": summary_m["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round3a_invvol_fine.json", "w") as f:
        json.dump(r3a_results, f, indent=2, default=str)
    print(f"R3A done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Round 3B: Inv-vol + High-confidence filter combination
# Test if stricter meta-filter (fewer, better trades) further reduces MaxDD
# while inv-vol handles regime-based sizing
# ─────────────────────────────="0─────────────────────────────
print("\n" + "=" * 60)
print("Round 3B: Inv-vol + High-confidence filter (thr=0.55)")
print("=" * 60)

filtered_hc55 = filter_signals(oof_proba, 0.55)
n_trades_hc = int((filtered_hc55.diff().abs() > 0).sum() // 2)
print(f"High-confidence filtered trades (thr=0.55): {n_trades_hc}")

r3b_results = []

for vw in [45, 60, 80, 100]:
    for tv in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        n_trials_base += 1

        vol_scale = inv_vol_sizing_4h(prices_4h, vw, tv)
        invvol_pos = (filtered_hc55 * vol_scale).clip(0, 1.5)

        gross_iv = invvol_pos.shift(1) * price_ret
        net_iv30 = costs_std.apply(gross_iv, filtered_hc55)
        net_iv50 = costs_stress.apply(gross_iv, filtered_hc55)

        m30 = compute_all_metrics(net_iv30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
        m50 = compute_all_metrics(net_iv50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

        avg_pos = float(invvol_pos[invvol_pos > 0].mean()) if (invvol_pos > 0).any() else 0.0

        result = {
            "sizing_family": "invvol_highconf",
            "vol_window": vw,
            "target_vol": tv,
            "threshold": 0.55,
            "n_trades": n_trades_hc,
            "avg_position": avg_pos,
            "sharpe_30": float(m30["sharpe"]),
            "dsr_30": float(m30["dsr"]),
            "max_dd_30": float(m30["max_drawdown"]),
            "sharpe_50": float(m50["sharpe"]),
            "dsr_50": float(m50["dsr"]),
            "n_trials": n_trials_base,
        }

        gate = "✓" if m30["max_drawdown"] > -0.25 and m30["sharpe"] > 1.5 and m30["dsr"] > 0.95 else " "
        print(
            f"  {gate} vw={vw:3d} tv={tv:.2f}: Sharpe={m30['sharpe']:.4f} "
            f"MaxDD={m30['max_drawdown']:.4f} DSR={m30['dsr']:.4f}"
        )

        if m30["sharpe"] > 1.3:
            sp = subperiod_metrics(net_iv30, prices_4h, filtered_hc55)
            result["subperiods_30"] = sp
            sp50 = subperiod_metrics(net_iv50, prices_4h, filtered_hc55)
            result["subperiods_50"] = sp50
            if sp:
                print(
                    f"      2017+: Sharpe={sp.get('2017+', {}).get('sharpe', 'N/A')} "
                    f"MaxDD={sp.get('2017+', {}).get('max_drawdown', 'N/A')}"
                )
                print(
                    f"      2020+: Sharpe={sp.get('2020+', {}).get('sharpe', 'N/A')} "
                    f"MaxDD={sp.get('2020+', {}).get('max_drawdown', 'N/A')}"
                )

        r3b_results.append(result)

successes_3b = [r for r in r3b_results if r["max_dd_30"] > -0.25 and r["sharpe_30"] > 1.5 and r["dsr_30"] > 0.95]
print(f"\nR3B success configs (thr=0.55 + inv-vol): {len(successes_3b)}")
for r in sorted(successes_3b, key=lambda x: x["sharpe_30"], reverse=True):
    print(f"  vw={r['vol_window']} tv={r['target_vol']:.2f}: Sharpe={r['sharpe_30']:.4f} MaxDD={r['max_dd_30']:.4f}")

if r3b_results:
    best_3b = max(r3b_results, key=lambda x: x["sharpe_30"])
    tracker.log_sweep(
        "layer4_s1_r3b_invvol_hc",
        r3b_results,
        summary_metrics={"sharpe": best_3b["sharpe_30"], "dsr": best_3b["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round3b_invvol_hc.json", "w") as f:
        json.dump(r3b_results, f, indent=2, default=str)
    print(f"R3B done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Round 3C: Regime-conditional inv-vol
# During high-vol regime (HMM bull_state low), reduce position further
# Use regime_proba as a multiplicative scale on inv-vol position
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 3C: Regime-conditional inv-vol sizing")
print("=" * 60)
print("Scale inv-vol position by regime_proba (bull probability)")
print("Low regime confidence → smaller position → reduces drawdown in bear markets")

regime_proba_series = pd.Series(regime_proba_4h_3s.iloc[:, bull_state_3s].values, index=prices_4h.index)
print(f"Regime proba stats: mean={regime_proba_series.mean():.3f} std={regime_proba_series.std():.3f}")

r3c_results = []

for vw in [45, 60, 80]:
    for tv in [0.15, 0.20, 0.25]:
        for regime_power in [0.5, 1.0, 2.0]:  # higher power = more aggressive regime scaling
            n_trials_base += 1

            vol_scale = inv_vol_sizing_4h(prices_4h, vw, tv)
            # Regime scale: regime_proba^power, ranges [0,1]^power
            regime_scale = regime_proba_series**regime_power
            # Combined: inv-vol × regime_confidence
            combined_pos = (filtered_base * vol_scale * regime_scale).clip(0, 1.5)

            gross = combined_pos.shift(1) * price_ret
            net_30 = costs_std.apply(gross, filtered_base)
            net_50 = costs_stress.apply(gross, filtered_base)

            m30 = compute_all_metrics(net_30.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)
            m50 = compute_all_metrics(net_50.dropna(), n_trials=n_trials_base, periods_per_year=PERIODS_PER_YEAR_4H)

            avg_pos = float(combined_pos[combined_pos > 0].mean()) if (combined_pos > 0).any() else 0.0

            result = {
                "sizing_family": "regime_invvol",
                "vol_window": vw,
                "target_vol": tv,
                "regime_power": regime_power,
                "n_trades": n_trades_base,
                "avg_position": avg_pos,
                "sharpe_30": float(m30["sharpe"]),
                "dsr_30": float(m30["dsr"]),
                "max_dd_30": float(m30["max_drawdown"]),
                "sharpe_50": float(m50["sharpe"]),
                "dsr_50": float(m50["dsr"]),
                "n_trials": n_trials_base,
            }

            gate = "✓" if m30["max_drawdown"] > -0.25 and m30["sharpe"] > 1.5 and m30["dsr"] > 0.95 else " "
            print(
                f"  {gate} vw={vw:2d} tv={tv:.2f} rp={regime_power:.1f}: "
                f"Sharpe={m30['sharpe']:.4f} MaxDD={m30['max_drawdown']:.4f} "
                f"avgPos={avg_pos:.3f}"
            )

            if m30["sharpe"] > 1.5:
                sp = subperiod_metrics(net_30, prices_4h, filtered_base)
                result["subperiods_30"] = sp
                sp50 = subperiod_metrics(net_50, prices_4h, filtered_base)
                result["subperiods_50"] = sp50
                if sp:
                    print(
                        f"      2017+: Sharpe={sp.get('2017+', {}).get('sharpe', 'N/A')} "
                        f"MaxDD={sp.get('2017+', {}).get('max_drawdown', 'N/A')}"
                    )
                    print(
                        f"      2020+: Sharpe={sp.get('2020+', {}).get('sharpe', 'N/A')} "
                        f"MaxDD={sp.get('2020+', {}).get('max_drawdown', 'N/A')}"
                    )

            r3c_results.append(result)

successes_3c = [r for r in r3c_results if r["max_dd_30"] > -0.25 and r["sharpe_30"] > 1.5 and r["dsr_30"] > 0.95]
print(f"\nR3C success configs (regime-conditional inv-vol): {len(successes_3c)}")
for r in sorted(successes_3c, key=lambda x: x["sharpe_30"], reverse=True):
    print(
        f"  vw={r['vol_window']} tv={r['target_vol']:.2f} rp={r['regime_power']:.1f}: "
        f"Sharpe={r['sharpe_30']:.4f} MaxDD={r['max_dd_30']:.4f}"
    )

if r3c_results:
    best_3c = max(r3c_results, key=lambda x: x["sharpe_30"])
    tracker.log_sweep(
        "layer4_s1_r3c_regime_invvol",
        r3c_results,
        summary_metrics={"sharpe": best_3c["sharpe_30"], "dsr": best_3c["dsr_30"]},
        tags=TAGS,
    )
    with open(out_dir / "round3c_regime_invvol.json", "w") as f:
        json.dump(r3c_results, f, indent=2, default=str)
    print(f"R3C done. n_trials={n_trials_base}")


# ─────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────
all_results = r3a_results + r3b_results + r3c_results
all_successes = [r for r in all_results if r["max_dd_30"] > -0.25 and r["sharpe_30"] > 1.5 and r["dsr_30"] > 0.95]

print("\n" + "=" * 60)
print("LAYER 4 SESSION 1 ROUND 3 — FINAL SUMMARY")
print("=" * 60)
print(f"n_trials cumulative: {n_trials_base}")
print(f"Configs this round: {n_trials_base - 421}")
print(f"\nSuccess gate configs (MaxDD>-0.25, Sharpe>1.5, DSR>0.95): {len(all_successes)}")

if all_successes:
    best_success = max(all_successes, key=lambda x: x["sharpe_30"])
    print("\nBest success config:")
    print(f"  Family: {best_success['sizing_family']}")
    print(f"  Sharpe@30bps: {best_success['sharpe_30']:.4f}")
    print(f"  Sharpe@50bps: {best_success['sharpe_50']:.4f}")
    print(f"  MaxDD@30bps: {best_success['max_dd_30']:.4f}")
    print(f"  DSR: {best_success['dsr_30']:.4f}")
    if "subperiods_30" in best_success:
        print("  Sub-periods:")
        for k, v in best_success["subperiods_30"].items():
            print(
                f"    {k}: Sharpe={v.get('sharpe', 'N/A')} MaxDD={v.get('max_drawdown', 'N/A')} "
                f"B&H_Sharpe={v.get('bh_sharpe', 'N/A')}"
            )

print("\nTop 5 success configs by Sharpe:")
for r in sorted(all_successes, key=lambda x: x["sharpe_30"], reverse=True)[:5]:
    print(
        f"  {r['sizing_family']:<20} vw={r.get('vol_window', '?')} tv={r.get('target_vol', '?'):.2f} "
        f"Sharpe={r['sharpe_30']:.4f} MaxDD={r['max_dd_30']:.4f} DSR={r['dsr_30']:.4f}"
    )

summary = {
    "n_trials_cumulative": n_trials_base,
    "n_configs_this_round": n_trials_base - 421,
    "success_gate_met": len(all_successes) > 0,
    "n_successes_r3": len(all_successes),
    "best_success_config": max(all_successes, key=lambda x: x["sharpe_30"]) if all_successes else None,
    "all_successes": sorted(all_successes, key=lambda x: x["sharpe_30"], reverse=True),
    "r3a_summary": {
        "n_results": len(r3a_results),
        "n_successes": len(successes_3a),
        "best": max(r3a_results, key=lambda x: x["sharpe_30"]) if r3a_results else None,
    },
    "r3b_summary": {
        "n_results": len(r3b_results),
        "n_successes": len(successes_3b),
        "best": max(r3b_results, key=lambda x: x["sharpe_30"]) if r3b_results else None,
    },
    "r3c_summary": {
        "n_results": len(r3c_results),
        "n_successes": len(successes_3c),
        "best": max(r3c_results, key=lambda x: x["sharpe_30"]) if r3c_results else None,
    },
}

with open(out_dir / "session1_r3_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

best_for_wandb = (
    max(all_successes, key=lambda x: x["sharpe_30"])
    if all_successes
    else max(all_results, key=lambda x: x["sharpe_30"])
)
wb.init(project="sparky-ai", entity="datadex_ai", name="layer4_sizing_session1_r3_final", tags=TAGS, reinit=True)
wb.log(
    {
        "sharpe": best_for_wandb["sharpe_30"],
        "dsr": best_for_wandb["dsr_30"],
        "max_dd": best_for_wandb["max_dd_30"],
        "n_configs_tested": n_trials_base - 421,
        "n_trials_cumulative": n_trials_base,
        "success_gate_met": int(len(all_successes) > 0),
        "n_successes": len(all_successes),
        "best_sharpe_50bps": best_for_wandb.get("sharpe_50", 0),
        "baseline_meta_unsized_sharpe": 1.981,
        "baseline_raw_donchian_sharpe": 1.691,
        "baseline_bh_sharpe": 1.288,
    }
)
wb.finish()

print(f"\nDone. Results in {out_dir}/")
