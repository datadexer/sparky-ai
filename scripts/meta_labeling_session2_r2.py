"""Meta-labeling session 2, round 2: threshold sweep + XGBoost on best (30,20) config."""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

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

    # Fractional diff of log price (d=0.1)
    d = 0.1
    max_lag = 20
    w = [1.0]
    for k in range(1, max_lag):
        w.append(w[-1] * (d - k + 1) / k)
    w = np.array(w[::-1])
    log_p = np.log(close.values.clip(min=1e-8))
    frac = np.full(len(close), np.nan)
    for i in range(max_lag, len(log_p)):
        frac[i] = float(np.dot(w, log_p[i - max_lag : i]))
    feats["fracdiff"] = frac

    return feats


print("Computing features...")
feats = make_features(h4, prices_4h)

FSETS = {
    "3c": ["trend_r2", "regime_proba_3s", "adx_proxy"],
    "3a": ["vol_ratio", "regime_proba_3s", "trend_r2"],
    "3b": ["vol_norm", "regime_proba_3s", "choppiness"],
    "5best": ["vol_ratio", "vol_norm", "trend_r2", "regime_proba_3s", "choppiness"],
    "3d": ["trend_r2", "regime_proba_3s", "autocorr_5"],
    "4a": ["trend_r2", "regime_proba_3s", "adx_proxy", "vol_ratio"],
    "4c": ["trend_r2", "regime_proba_3s", "adx_proxy", "vol_norm"],
    "4d": ["trend_r2", "regime_proba_3s", "adx_proxy", "choppiness"],
}

# Start from session 2 round 1 end
n_trials = 153


def run_exp(
    entry,
    exit_p,
    tp_mult,
    sl_mult,
    vert_bars,
    feat_key,
    model_type="logreg",
    max_depth=2,
    min_child_weight=10,
    n_estimators=100,
    confidence_threshold=0.5,
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

    gap_signals = max(1, int(vert_bars * len(valid_idx) / len(prices_4h)))
    max_splits = max(2, len(valid_idx) // max(gap_signals + 2, 5))
    n_splits = min(5, max_splits)

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

        if model_type == "logreg":
            clf = LogisticRegression(C=C, max_iter=500, class_weight="balanced", random_state=42)
            clf.fit(X_tr_s, y_tr, sample_weight=w_tr)
        else:  # xgboost
            scale_pos = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
            clf = xgb.XGBClassifier(
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                n_estimators=n_estimators,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos,
                tree_method="hist",
                device="cuda",
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            )
            clf.fit(X_tr_s, y_tr, sample_weight=w_tr)

        proba = clf.predict_proba(X_te_s)[:, 1]
        fold_accs.append(float(((proba > 0.5) == y_te).mean()))
        for i, si in enumerate(test_si_v):
            oof_proba[si] = proba[i]

    if not fold_accs:
        return None

    filtered_signals = signals.copy()
    for t in valid_idx:
        if np.isnan(oof_proba[t]) or oof_proba[t] >= confidence_threshold:
            continue
        end = t + 1
        while end < len(filtered_signals) and filtered_signals.iloc[end] > 0:
            end += 1
        filtered_signals.iloc[t:end] = 0

    price_ret = prices_4h.pct_change().fillna(0)
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
        "model_type": model_type,
        "max_depth": max_depth,
        "confidence_threshold": confidence_threshold,
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
    print(
        f"  ({entry},{exit_p}) tp={tp_mult} sl={sl_mult} vb={vert_bars} "
        f"feat={feat_key} {model_type} C={C} thr={confidence_threshold}: "
        f"Sharpe={m30['sharpe']:.3f} DSR={m30['dsr']:.3f} "
        f"acc={np.mean(fold_accs):.3f} trades={n_trades}"
    )
    return result


# ─────────────────────────────────────────────────────────────
# Round 2A: Confidence threshold sweep on best config
# S1 best: (30,20), tp=3.0, sl=1.5, vb=30, feat=3c, threshold=0.5
# Try thresholds [0.45, 0.5, 0.55, 0.6, 0.65] on top feature sets
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Round 2A: Confidence threshold sweep on (30,20) best barriers")
print("=" * 60)

r2a_results = []
for feat_key in ["3c", "3a", "4a", "4c", "4d"]:
    for thr in [0.45, 0.5, 0.55, 0.6, 0.65]:
        r = run_exp(30, 20, 3.0, 1.5, 30, feat_key, confidence_threshold=thr)
        if r:
            r2a_results.append(r)

if r2a_results:
    best_2a = max(r2a_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR2A best: Sharpe={best_2a['sharpe_30']:.3f} DSR={best_2a['dsr_30']:.3f} "
        f"feat={best_2a['feat_key']} thr={best_2a['confidence_threshold']}"
    )

    print("\nThreshold analysis (3c feature set):")
    for thr in [0.45, 0.5, 0.55, 0.6, 0.65]:
        sub = [r for r in r2a_results if r["feat_key"] == "3c" and abs(r["confidence_threshold"] - thr) < 0.01]
        if sub:
            print(
                f"  thr={thr}: Sharpe={sub[0]['sharpe_30']:.3f} DSR={sub[0]['dsr_30']:.3f} trades={sub[0]['n_trades_filtered']}"
            )

tracker.log_sweep(
    "session2_round2a_threshold_sweep",
    r2a_results,
    summary_metrics={
        "sharpe": best_2a["sharpe_30"] if r2a_results else 0,
        "dsr": best_2a["dsr_30"] if r2a_results else 0,
    },
    tags=TAGS,
)
with open(out_dir / "session2_round2a_results.json", "w") as f:
    json.dump(r2a_results, f, indent=2, default=str)

print(f"Round 2A done. n_trials={n_trials}")


# ─────────────────────────────────────────────────────────────
# Round 2B: XGBoost on (30,20) — now that we have 271 signals
# with right features. Try max_depth 2-3, varying regularization
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 2B: XGBoost on (30,20) best barriers")
print("=" * 60)

r2b_results = []
for feat_key in ["3c", "3a", "4a"]:
    for max_depth in [2, 3]:
        for min_cw in [10, 20]:
            r = run_exp(
                30,
                20,
                3.0,
                1.5,
                30,
                feat_key,
                model_type="xgboost",
                max_depth=max_depth,
                min_child_weight=min_cw,
                n_estimators=100,
                confidence_threshold=0.5,
            )
            if r:
                r2b_results.append(r)

if r2b_results:
    best_2b = max(r2b_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR2B XGB best: Sharpe={best_2b['sharpe_30']:.3f} DSR={best_2b['dsr_30']:.3f} "
        f"feat={best_2b['feat_key']} depth={best_2b['max_depth']} mcw={best_2b['min_child_weight']}"
    )

tracker.log_sweep(
    "session2_round2b_xgboost",
    r2b_results,
    summary_metrics={
        "sharpe": best_2b["sharpe_30"] if r2b_results else 0,
        "dsr": best_2b["dsr_30"] if r2b_results else 0,
    },
    tags=TAGS,
)
with open(out_dir / "session2_round2b_results.json", "w") as f:
    json.dump(r2b_results, f, indent=2, default=str)

print(f"Round 2B done. n_trials={n_trials}")


# ─────────────────────────────────────────────────────────────
# Round 2C: Vary sl_mult more aggressively
# S1 only tested sl=1.0 and sl=1.5. Try sl=2.0 and sl=1.0 again
# with tp=3.0, 4.0 (more asymmetric)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 2C: Asymmetric barrier exploration (wide tp, varying sl)")
print("=" * 60)

r2c_results = []
# Best feature: 3c. Try more barrier combos
for tp, sl, vb in [(3.0, 2.0, 30), (3.0, 1.0, 30), (3.0, 1.5, 60), (2.0, 2.0, 30), (2.0, 1.0, 30)]:
    for feat_key in ["3c", "3a"]:
        r = run_exp(30, 20, tp, sl, vb, feat_key, confidence_threshold=0.5)
        if r:
            r2c_results.append(r)

if r2c_results:
    best_2c = max(r2c_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR2C best: Sharpe={best_2c['sharpe_30']:.3f} DSR={best_2c['dsr_30']:.3f} "
        f"tp={best_2c['tp_mult']} sl={best_2c['sl_mult']} vb={best_2c['vert_bars']} feat={best_2c['feat_key']}"
    )

tracker.log_sweep(
    "session2_round2c_barrier_variants",
    r2c_results,
    summary_metrics={
        "sharpe": best_2c["sharpe_30"] if r2c_results else 0,
        "dsr": best_2c["dsr_30"] if r2c_results else 0,
    },
    tags=TAGS,
)
with open(out_dir / "session2_round2c_results.json", "w") as f:
    json.dump(r2c_results, f, indent=2, default=str)

print(f"Round 2C done. n_trials={n_trials}")

# Overall summary
all_r2 = r2a_results + r2b_results + r2c_results
if all_r2:
    best_r2 = max(all_r2, key=lambda x: x["sharpe_30"])
    print(f"\n=== Round 2 overall best: Sharpe={best_r2['sharpe_30']:.3f} DSR={best_r2['dsr_30']:.3f} ===")
    print(f"  {best_r2}")

print(f"\nTotal n_trials (session cumulative): {n_trials}")
print("Done.")
