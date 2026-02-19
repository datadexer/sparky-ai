"""Session 2, rounds 2B (fix logging) + 2C (barrier variants) + 2D (LogReg C search)."""

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
    "4a": ["trend_r2", "regime_proba_3s", "adx_proxy", "vol_ratio"],
    "4d": ["trend_r2", "regime_proba_3s", "adx_proxy", "choppiness"],
    "4e": ["trend_r2", "regime_proba_3s", "choppiness", "vol_ratio"],
    "5best": ["vol_ratio", "vol_norm", "trend_r2", "regime_proba_3s", "choppiness"],
    # New: include fracdiff as a stationary price proxy
    "3f": ["trend_r2", "regime_proba_3s", "fracdiff"],
    "4f": ["trend_r2", "regime_proba_3s", "adx_proxy", "fracdiff"],
}

# Continue from n_trials=178 (end of 2A) + 12 (2B ran despite crash) = 190
n_trials = 190


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
        else:
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
        "min_child_weight": min_child_weight,
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
# Round 2C: Asymmetric barrier variants (sl=2.0, tp=4.0, vert=60)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Round 2C: Asymmetric barrier exploration")
print("=" * 60)

r2c_results = []
for tp, sl, vb in [(3.0, 2.0, 30), (3.0, 1.0, 30), (3.0, 1.5, 60), (2.0, 2.0, 30), (2.0, 1.0, 30)]:
    for feat_key in ["3c", "4d"]:
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


# ─────────────────────────────────────────────────────────────
# Round 2D: LogReg C (regularization) sweep on best (4d) feature set
# C=0.1 is fixed in S1. Try: 0.01, 0.05, 0.1, 0.5, 1.0
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 2D: LogReg regularization C sweep (4d features)")
print("=" * 60)

r2d_results = []
for feat_key in ["3c", "4d", "4e"]:
    for C in [0.01, 0.05, 0.1, 0.3, 1.0]:
        r = run_exp(30, 20, 3.0, 1.5, 30, feat_key, C=C, confidence_threshold=0.5)
        if r:
            r2d_results.append(r)

if r2d_results:
    best_2d = max(r2d_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR2D best: Sharpe={best_2d['sharpe_30']:.3f} DSR={best_2d['dsr_30']:.3f} "
        f"feat={best_2d['feat_key']} C={best_2d['C']}"
    )

    print("\nBy C value (3c feat):")
    for C in [0.01, 0.05, 0.1, 0.3, 1.0]:
        sub = [r for r in r2d_results if r["feat_key"] == "3c" and abs(r["C"] - C) < 0.001]
        if sub:
            print(f"  C={C}: Sharpe={sub[0]['sharpe_30']:.3f} DSR={sub[0]['dsr_30']:.3f}")

tracker.log_sweep(
    "session2_round2d_logreg_C_sweep",
    r2d_results,
    summary_metrics={
        "sharpe": best_2d["sharpe_30"] if r2d_results else 0,
        "dsr": best_2d["dsr_30"] if r2d_results else 0,
    },
    tags=TAGS,
)
with open(out_dir / "session2_round2d_results.json", "w") as f:
    json.dump(r2d_results, f, indent=2, default=str)
print(f"Round 2D done. n_trials={n_trials}")


# ─────────────────────────────────────────────────────────────
# Round 2E: New feature exploration (fracdiff, dist_sma variations)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 2E: New feature combos (fracdiff, dist_sma)")
print("=" * 60)

r2e_results = []
for feat_key in ["3f", "4f", "3c", "4d"]:
    for tp, sl, vb in [(3.0, 1.5, 30), (2.0, 1.5, 20)]:
        r = run_exp(30, 20, tp, sl, vb, feat_key, confidence_threshold=0.5)
        if r:
            r2e_results.append(r)

if r2e_results:
    best_2e = max(r2e_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR2E best: Sharpe={best_2e['sharpe_30']:.3f} DSR={best_2e['dsr_30']:.3f} "
        f"feat={best_2e['feat_key']} tp={best_2e['tp_mult']}"
    )

tracker.log_sweep(
    "session2_round2e_feature_variants",
    r2e_results,
    summary_metrics={
        "sharpe": best_2e["sharpe_30"] if r2e_results else 0,
        "dsr": best_2e["dsr_30"] if r2e_results else 0,
    },
    tags=TAGS,
)
with open(out_dir / "session2_round2e_results.json", "w") as f:
    json.dump(r2e_results, f, indent=2, default=str)
print(f"Round 2E done. n_trials={n_trials}")

# ─────────────────────────────────────────────────────────────
# Consolidated summary
# ─────────────────────────────────────────────────────────────
all_results = r2c_results + r2d_results + r2e_results
if all_results:
    best_all = max(all_results, key=lambda x: x["sharpe_30"])
    print(f"\n=== Rounds 2C-2E best: Sharpe={best_all['sharpe_30']:.3f} DSR={best_all['dsr_30']:.3f} ===")
    print(
        f"  feat={best_all['feat_key']} tp={best_all['tp_mult']} sl={best_all['sl_mult']} "
        f"vb={best_all['vert_bars']} C={best_all['C']}"
    )

print(f"\nTotal n_trials (cumulative): {n_trials}")
print("Done.")
