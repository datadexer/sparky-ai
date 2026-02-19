"""Session 2, round 3: calibration+Kelly sizing + untested Donchian params."""

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

from sparky.data.loader import load
from sparky.backtest.costs import TransactionCostModel
from sparky.tracking.metrics import compute_all_metrics
from sparky.tracking.experiment import ExperimentTracker
from sparky.models.simple_baselines import donchian_channel_strategy
from sparky.models.regime_hmm import train_hmm_regime_model

TAGS = ["meta_labeling", "donchian", "20260218", "session_002"]
out_dir = Path("results/meta_labeling_donchian_20260218")
out_dir.mkdir(parents=True, exist_ok=True)

tracker = ExperimentTracker(experiment_name="meta_labeling_donchian_20260218")
costs_std = TransactionCostModel.standard()
costs_stress = TransactionCostModel.stress_test()

hourly = load("ohlcv_hourly_max_coverage", purpose="training")
h4 = (
    hourly.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
)
prices_4h = h4["close"]
PERIODS_PER_YEAR_4H = 2190

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

    feats["vol_ratio"] = ret.rolling(20).std() / ret.rolling(60).std().clip(lower=1e-8)
    feats["vol_norm"] = ret.rolling(20).std() / ret.rolling(20).std().rolling(100).mean().clip(lower=1e-8)

    def trend_r2(x):
        return float(np.corrcoef(np.arange(len(x)), x)[0, 1] ** 2) if len(x) >= 5 else np.nan

    feats["trend_r2"] = ret.rolling(20).apply(trend_r2, raw=True)

    dm_plus = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    feats["adx_proxy"] = (dm_plus.rolling(14).mean() - dm_minus.rolling(14).mean()).abs() / atr_4h.clip(lower=1e-8)
    atr_1bar = (high - low).abs()
    feats["choppiness"] = np.log10(
        atr_1bar.rolling(14).sum() / (high.rolling(14).max() - low.rolling(14).min()).clip(lower=1e-8)
    ) / np.log10(14)

    feats["vol_momentum"] = volume.rolling(10).mean() / volume.rolling(40).mean().clip(lower=1e-8)
    feats["dist_sma_20"] = (close - close.rolling(20).mean()) / close.rolling(20).mean().clip(lower=1e-8)
    feats["regime_proba_3s"] = regime_proba_4h_3s.iloc[:, bull_state_3s].values

    feats["autocorr_5"] = ret.rolling(20).apply(
        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) > 5 else np.nan, raw=False
    )
    return feats


print("Computing features...")
feats = make_features(h4, prices_4h)

FSETS = {
    "3c": ["trend_r2", "regime_proba_3s", "adx_proxy"],
    "4d": ["trend_r2", "regime_proba_3s", "adx_proxy", "choppiness"],
    "3a": ["vol_ratio", "regime_proba_3s", "trend_r2"],
    "4a": ["trend_r2", "regime_proba_3s", "adx_proxy", "vol_ratio"],
}

# Continue from session 2 round 2 end
n_trials = 223


def get_oof_probas(prices, signals, valid_idx, X_all, y_all, w_all, feat_cols, vert_bars, calibrate=False, C=0.1):
    """Fit purged CV logistic regression, return OOF probas + fold accs."""
    gap_signals = max(1, int(vert_bars * len(valid_idx) / len(prices)))
    max_splits = max(2, len(valid_idx) // max(gap_signals + 2, 5))
    n_splits = min(5, max_splits)

    oof_proba = np.full(len(prices), np.nan)
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

        base_clf = LogisticRegression(C=C, max_iter=500, class_weight="balanced", random_state=42)
        if calibrate:
            # Platt calibration (sigmoid) with cross-val=3
            clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
        else:
            clf = base_clf

        clf.fit(X_tr_s, y_tr, **({"sample_weight": w_tr} if not calibrate else {}))

        proba = clf.predict_proba(X_te_s)[:, 1]
        fold_accs.append(float(((proba > 0.5) == y_te).mean()))
        for i, si in enumerate(test_si_v):
            oof_proba[si] = proba[i]

    return oof_proba, fold_accs


def run_exp_full(
    entry,
    exit_p,
    tp_mult,
    sl_mult,
    vert_bars,
    feat_key,
    confidence_threshold=0.5,
    use_kelly=False,
    kelly_fraction=0.25,
    calibrate=False,
    C=0.1,
):
    global n_trials
    n_trials += 1

    signals = donchian_channel_strategy(prices_4h, entry, exit_p)
    n_entries = int((np.diff(signals.values, prepend=0) > 0).sum())

    labels, weights = triple_barrier_labels(prices_4h, signals, atr_4h, tp_mult, sl_mult, vert_bars)
    valid_idx = np.where(~np.isnan(labels) & ~np.isnan(weights))[0]
    if len(valid_idx) < 20:
        return None

    feat_cols = FSETS[feat_key]
    X_all = feats[feat_cols].values
    y_all = labels
    w_all = weights

    pos_rate = float(np.nanmean(y_all[valid_idx]))
    if pos_rate < 0.05 or pos_rate > 0.95:
        return None

    oof_proba, fold_accs = get_oof_probas(
        prices_4h, signals, valid_idx, X_all, y_all, w_all, feat_cols, vert_bars, calibrate=calibrate, C=C
    )

    if not fold_accs:
        return None

    price_ret = prices_4h.pct_change().fillna(0)

    if use_kelly:
        # Fractional Kelly position sizing based on confidence
        # p = proba of win, b = tp_mult/sl_mult (payoff ratio)
        # Kelly = p - (1-p)/b; use fraction * max(0, kelly_p)
        payoff_ratio = tp_mult / sl_mult
        pos_series = signals.copy().astype(float)
        for t in valid_idx:
            p = oof_proba[t]
            if np.isnan(p):
                continue
            k = p - (1 - p) / payoff_ratio
            if k <= 0:
                # Below Kelly minimum → exit trade
                end = t + 1
                while end < len(pos_series) and pos_series.iloc[end] > 0:
                    end += 1
                pos_series.iloc[t:end] = 0
            else:
                # Scale position by fractional Kelly (capped at 1)
                size = min(1.0, kelly_fraction * k * 4)  # *4 to normalize (typical kelly 0-0.25 → 0-1)
                end = t + 1
                while end < len(pos_series) and pos_series.iloc[end] > 0:
                    end += 1
                pos_series.iloc[t:end] = pos_series.iloc[t:end] * size

        gross = pos_series.shift(1) * price_ret
        # Apply costs using the binary signal (not fractional) for cost calculation
        # — cost model expects {-1, 0, 1} but positions are fractional
        # Approximate: scale costs by average position
        net_30 = costs_std.apply(gross, signals)  # conservative: full costs
        net_50 = costs_stress.apply(gross, signals)
        n_trades = int((signals.diff().abs() > 0).sum() // 2)
    else:
        # Standard binary filter
        filtered_signals = signals.copy()
        for t in valid_idx:
            if np.isnan(oof_proba[t]) or oof_proba[t] >= confidence_threshold:
                continue
            end = t + 1
            while end < len(filtered_signals) and filtered_signals.iloc[end] > 0:
                end += 1
            filtered_signals.iloc[t:end] = 0

        gross = filtered_signals.shift(1) * price_ret
        net_30 = costs_std.apply(gross, filtered_signals)
        net_50 = costs_stress.apply(gross, filtered_signals)
        n_trades = int((filtered_signals.diff().abs() > 0).sum() // 2)

    if n_trades < 5:
        return None

    m30 = compute_all_metrics(net_30.dropna(), n_trials=n_trials, periods_per_year=PERIODS_PER_YEAR_4H)
    m50 = compute_all_metrics(net_50.dropna(), n_trials=n_trials, periods_per_year=PERIODS_PER_YEAR_4H)

    result = {
        "entry": entry,
        "exit": exit_p,
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "vert_bars": vert_bars,
        "feat_key": feat_key,
        "confidence_threshold": confidence_threshold,
        "use_kelly": use_kelly,
        "kelly_fraction": kelly_fraction if use_kelly else None,
        "calibrate": calibrate,
        "C": C,
        "n_entries_primary": n_entries,
        "n_trades_filtered": n_trades,
        "label_pos_rate": pos_rate,
        "meta_accuracy": float(np.mean(fold_accs)),
        "sharpe_30": float(m30["sharpe"]),
        "dsr_30": float(m30["dsr"]),
        "sharpe_50": float(m50["sharpe"]),
        "dsr_50": float(m50["dsr"]),
        "max_dd_30": float(m30["max_drawdown"]),
        "n_trials": n_trials,
    }
    mode = f"kelly(kf={kelly_fraction})" if use_kelly else f"binary(thr={confidence_threshold})"
    calib = "+platt" if calibrate else ""
    print(
        f"  ({entry},{exit_p}) tp={tp_mult} sl={sl_mult} vb={vert_bars} "
        f"feat={feat_key} {mode}{calib}: "
        f"Sharpe={m30['sharpe']:.3f} DSR={m30['dsr']:.3f} "
        f"acc={np.mean(fold_accs):.3f} trades={n_trades}"
    )
    return result


# ─────────────────────────────────────────────────────────────
# Round 3A: Platt calibration on best config
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Round 3A: Platt calibration on best config (30,20) 4d features")
print("=" * 60)

r3a_results = []
# Best: (30,20), tp=3.0, sl=1.5, vb=30, 4d features, threshold=0.5
for feat_key in ["3c", "4d"]:
    for calibrate in [False, True]:
        for thr in [0.5, 0.55]:
            r = run_exp_full(30, 20, 3.0, 1.5, 30, feat_key, confidence_threshold=thr, calibrate=calibrate)
            if r:
                r3a_results.append(r)

if r3a_results:
    best_3a = max(r3a_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR3A best: Sharpe={best_3a['sharpe_30']:.3f} DSR={best_3a['dsr_30']:.3f} "
        f"feat={best_3a['feat_key']} calibrate={best_3a['calibrate']} thr={best_3a['confidence_threshold']}"
    )

tracker.log_sweep(
    "session2_round3a_platt_calibration",
    r3a_results,
    summary_metrics={
        "sharpe": best_3a["sharpe_30"] if r3a_results else 0,
        "dsr": best_3a["dsr_30"] if r3a_results else 0,
    },
    tags=TAGS,
)
with open(out_dir / "session2_round3a_results.json", "w") as f:
    json.dump(r3a_results, f, indent=2, default=str)
print(f"Round 3A done. n_trials={n_trials}")


# ─────────────────────────────────────────────────────────────
# Round 3B: Kelly sizing on best config
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 3B: Fractional Kelly sizing")
print("=" * 60)

r3b_results = []
for feat_key in ["3c", "4d"]:
    for kelly_frac in [0.25, 0.5, 1.0]:
        r = run_exp_full(30, 20, 3.0, 1.5, 30, feat_key, use_kelly=True, kelly_fraction=kelly_frac)
        if r:
            r3b_results.append(r)

if r3b_results:
    best_3b = max(r3b_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR3B best: Sharpe={best_3b['sharpe_30']:.3f} DSR={best_3b['dsr_30']:.3f} "
        f"feat={best_3b['feat_key']} kelly_frac={best_3b['kelly_fraction']}"
    )
    print("  MaxDD:", best_3b["max_dd_30"])

tracker.log_sweep(
    "session2_round3b_kelly_sizing",
    r3b_results,
    summary_metrics={
        "sharpe": best_3b["sharpe_30"] if r3b_results else 0,
        "dsr": best_3b["dsr_30"] if r3b_results else 0,
    },
    tags=TAGS,
)
with open(out_dir / "session2_round3b_results.json", "w") as f:
    json.dump(r3b_results, f, indent=2, default=str)
print(f"Round 3B done. n_trials={n_trials}")


# ─────────────────────────────────────────────────────────────
# Round 3C: Intermediate Donchian params (25/15, 25/20, 30/15)
# that weren't in S1 or R1
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 3C: Intermediate Donchian params (25/15, 25/20, 30/15)")
print("=" * 60)

r3c_results = []
for entry, exit_p in [(25, 15), (25, 20), (30, 15), (35, 20), (30, 25)]:
    for tp, sl, vb in [(3.0, 1.5, 30), (2.0, 1.5, 20)]:
        for fk in ["3c", "4d"]:
            r = run_exp_full(entry, exit_p, tp, sl, vb, fk, confidence_threshold=0.5)
            if r:
                r3c_results.append(r)

if r3c_results:
    best_3c = max(r3c_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR3C best: Sharpe={best_3c['sharpe_30']:.3f} DSR={best_3c['dsr_30']:.3f} "
        f"entry={best_3c['entry']} exit={best_3c['exit']} feat={best_3c['feat_key']}"
    )

    print("\nBy Donchian params:")
    for ep, xp in [(25, 15), (25, 20), (30, 15), (35, 20), (30, 25)]:
        sub = [r for r in r3c_results if r["entry"] == ep and r["exit"] == xp]
        if sub:
            b = max(sub, key=lambda x: x["sharpe_30"])
            print(
                f"  ({ep},{xp}): best Sharpe={b['sharpe_30']:.3f} DSR={b['dsr_30']:.3f} "
                f"signals={b['n_entries_primary']} feat={b['feat_key']}"
            )

tracker.log_sweep(
    "session2_round3c_intermediate_donchian",
    r3c_results,
    summary_metrics={
        "sharpe": best_3c["sharpe_30"] if r3c_results else 0,
        "dsr": best_3c["dsr_30"] if r3c_results else 0,
    },
    tags=TAGS,
)
with open(out_dir / "session2_round3c_results.json", "w") as f:
    json.dump(r3c_results, f, indent=2, default=str)
print(f"Round 3C done. n_trials={n_trials}")


# Summary
all_r3 = r3a_results + r3b_results + r3c_results
if all_r3:
    best_r3 = max(all_r3, key=lambda x: x["sharpe_30"])
    print(f"\n=== Round 3 overall best: Sharpe={best_r3['sharpe_30']:.3f} DSR={best_r3['dsr_30']:.3f} ===")
    print(f"  {best_r3}")
print(f"\nTotal n_trials (cumulative): {n_trials}")
print("Done.")
