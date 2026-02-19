"""Meta-labeling session 1: baseline reproduction + 4h meta-labeling experiments."""

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


TAGS = ["meta_labeling", "donchian", "20260218", "session_001"]
out_dir = Path("results/meta_labeling_donchian_20260218")
out_dir.mkdir(parents=True, exist_ok=True)

tracker = ExperimentTracker(experiment_name="meta_labeling_donchian_20260218")
costs_std = TransactionCostModel.standard()
costs_stress = TransactionCostModel.stress_test()


# ──────────────────────────────────────────────────────────────────────────────
# STEP 0: Reproduce daily baseline Donchian
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 0: Reproduce baseline Donchian (daily)")
print("=" * 60)

daily = load("btc_daily", purpose="training")
print(f"Daily data: {len(daily)} rows")
prices_daily = daily["close"]

_, _, regime_daily, _ = train_hmm_regime_model(prices_daily, n_states=2)

ret_d = prices_daily.pct_change()
regime0_ret_d = ret_d[regime_daily == 0].mean()
regime1_ret_d = ret_d[regime_daily == 1].mean()
bull_state_d = 0 if regime0_ret_d > regime1_ret_d else 1
print(f"Daily bull regime = {bull_state_d}")

signals_daily = donchian_channel_strategy(prices_daily, 30, 20)
signals_daily_reg = signals_daily.copy()
signals_daily_reg[regime_daily != bull_state_d] = 0

n_entries_daily = int((np.diff(signals_daily.values, prepend=0) > 0).sum())
print(f"Daily entry signals (no regime filter): {n_entries_daily}")

price_ret_d = prices_daily.pct_change().fillna(0)
gross_d = signals_daily_reg.shift(1) * price_ret_d
net_d = costs_std.apply(gross_d, signals_daily_reg)

metrics_baseline = compute_all_metrics(net_d.dropna(), n_trials=1, periods_per_year=365)
print(
    f"Baseline Donchian (daily, regime-cond): Sharpe={metrics_baseline['sharpe']:.3f} "
    f"DSR={metrics_baseline['dsr']:.3f} MaxDD={metrics_baseline['max_drawdown']:.3f}"
)

with open(out_dir / "baseline_daily.json", "w") as f:
    json.dump(
        {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics_baseline.items()}, f, indent=2
    )

if abs(metrics_baseline["sharpe"] - 1.777) > 0.5:
    print(f"WARNING: Baseline {metrics_baseline['sharpe']:.3f} vs expected ~1.777 — proceeding anyway")
else:
    print(f"Baseline OK: {metrics_baseline['sharpe']:.3f}")
print(f"Daily signals = {n_entries_daily} — too few for ML. Using 4h data.")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Load hourly BTC, resample to 4h
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Load hourly → 4h")
print("=" * 60)

hourly = load("ohlcv_hourly_max_coverage", purpose="training")
print(f"Hourly data: {len(hourly)} rows, {hourly.index[0]} → {hourly.index[-1]}")

h4 = (
    hourly.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
)
print(f"4h data: {len(h4)} rows, {h4.index[0]} → {h4.index[-1]}")

prices_4h = h4["close"]
PERIODS_PER_YEAR_4H = 2190  # 24/4 * 365

# Check signal counts
for ep, ex in [(10, 5), (15, 10), (20, 10), (20, 15), (30, 15), (30, 20)]:
    s = donchian_channel_strategy(prices_4h, ep, ex)
    n = int((np.diff(s.values, prepend=0) > 0).sum())
    print(f"  Donchian({ep},{ex}): {n} entry signals on 4h")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: HMM regime on 4h
# ──────────────────────────────────────────────────────────────────────────────
print("\nTraining HMM on 4h prices...")
_, _, regime_4h, regime_proba_4h = train_hmm_regime_model(prices_4h, n_states=2)

ret_4h = prices_4h.pct_change()
regime0_ret_4h = ret_4h[regime_4h == 0].mean()
regime1_ret_4h = ret_4h[regime_4h == 1].mean()
bull_state_4h = 0 if regime0_ret_4h > regime1_ret_4h else 1
print(f"4h bull regime = {bull_state_4h}")


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


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

        # Barriers in price units
        tp_price = ep + tp_mult * a
        sl_price = ep - sl_mult * a
        end = min(t + vertical_bars + 1, n)

        label = 0
        for t2 in range(t + 1, end):
            if close[t2] >= tp_price:
                ret = (close[t2] - ep) / ep
                label = 1 if ret >= min_ret else 0
                break
            elif close[t2] <= sl_price:
                label = 0
                break
        else:
            final_ret = (close[end - 1] - ep) / ep
            label = 1 if final_ret >= min_ret else 0

        labels[t] = label
        concurrent[t:end] += 1

    weights = np.full(n, np.nan)
    for t in entry_bars:
        if np.isnan(labels[t]):
            continue
        end = min(t + vertical_bars + 1, n)
        avg_conc = concurrent[t:end].mean()
        weights[t] = 1.0 / max(avg_conc, 1.0)

    return labels, weights


def make_features_4h(h4_df, prices, regime_series, regime_proba):
    close = prices
    high = h4_df["high"]
    low = h4_df["low"]
    volume = h4_df["volume"]
    ret = close.pct_change()

    feats = pd.DataFrame(index=prices.index)
    feats["vol_20"] = ret.rolling(20).std()
    feats["vol_60"] = ret.rolling(60).std()
    feats["vol_ratio"] = feats["vol_20"] / feats["vol_60"].clip(lower=1e-8)

    feats["autocorr_5"] = ret.rolling(20).apply(
        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) > 5 else np.nan, raw=False
    )

    atr = compute_atr(high, low, close)
    dm_plus = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    feats["adx_proxy"] = (dm_plus.rolling(14).mean() - dm_minus.rolling(14).mean()).abs() / atr.clip(lower=1e-8)

    atr_1bar = (high - low).abs()
    h14 = high.rolling(14).max()
    l14 = low.rolling(14).min()
    hl_range = (h14 - l14).clip(lower=1e-8)
    feats["choppiness"] = np.log10(atr_1bar.rolling(14).sum() / hl_range) / np.log10(14)

    feats["vol_momentum"] = volume.rolling(10).mean() / volume.rolling(40).mean().clip(lower=1e-8)
    feats["dist_sma_20"] = (close - close.rolling(20).mean()) / close.rolling(20).mean().clip(lower=1e-8)
    feats["dist_sma_60"] = (close - close.rolling(60).mean()) / close.rolling(60).mean().clip(lower=1e-8)

    # HMM regime probability (bull state)
    bull_proba = regime_proba.iloc[:, bull_state_4h]
    feats["regime_proba"] = bull_proba.values

    # Fractional differentiation d=0.1, fixed-width window
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


FEATURE_SETS = {
    3: ["vol_ratio", "autocorr_5", "adx_proxy"],
    5: ["vol_ratio", "autocorr_5", "adx_proxy", "choppiness", "dist_sma_20"],
    8: [
        "vol_ratio",
        "autocorr_5",
        "adx_proxy",
        "choppiness",
        "dist_sma_20",
        "vol_momentum",
        "fracdiff",
        "regime_proba",
    ],
}


# Precompute features once
print("\nComputing 4h features...")
atr_4h = compute_atr(h4["high"], h4["low"], prices_4h)
feats_4h = make_features_4h(h4, prices_4h, regime_4h, regime_proba_4h)
print(f"Features: {list(feats_4h.columns)}")

all_results = []
n_trials = 1


def run_meta_experiment(
    entry,
    exit_p,
    tp_mult,
    sl_mult,
    vert_bars,
    n_feat,
    model_type="logreg",
    max_depth=2,
    min_child_weight=10,
    confidence_threshold=0.5,
):
    global n_trials
    n_trials += 1

    signals = donchian_channel_strategy(prices_4h, entry, exit_p)
    signals_reg = signals.copy()
    signals_reg[regime_4h != bull_state_4h] = 0

    entry_bars = np.where(np.diff(signals_reg.values, prepend=0) > 0)[0]
    n_entries = len(entry_bars)
    if n_entries < 30:
        print(f"  ({entry},{exit_p}): only {n_entries} signals, skip")
        return None

    labels, weights = triple_barrier_labels(prices_4h, signals_reg, atr_4h, tp_mult, sl_mult, vert_bars)

    valid_idx = np.where(~np.isnan(labels) & ~np.isnan(weights))[0]
    if len(valid_idx) < 20:
        print(f"  ({entry},{exit_p}): only {len(valid_idx)} valid labels, skip")
        return None

    feat_cols = FEATURE_SETS[n_feat]
    X_all = feats_4h[feat_cols].values
    y_all = labels
    w_all = weights

    pos_rate = float(np.nanmean(y_all[valid_idx]))
    if pos_rate < 0.05 or pos_rate > 0.95:
        print(f"  ({entry},{exit_p}) tp={tp_mult} sl={sl_mult}: degenerate labels pos_rate={pos_rate:.3f}, skip")
        return None

    n_splits = min(5, len(valid_idx) // 8)
    if n_splits < 2:
        n_splits = 2

    oof_proba = np.full(len(prices_4h), np.nan)
    fold_accs = []

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=vert_bars)
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
            clf = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=42)
            clf.fit(X_tr_s, y_tr, sample_weight=w_tr)
            proba = clf.predict_proba(X_te_s)[:, 1]
        elif model_type == "xgboost":
            scale_pos = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
            clf = xgb.XGBClassifier(
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                n_estimators=100,
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

    # Apply meta-filter
    filtered_signals = signals_reg.copy()
    for t in valid_idx:
        if np.isnan(oof_proba[t]):
            continue
        if oof_proba[t] < confidence_threshold:
            end = t + 1
            while end < len(filtered_signals) and filtered_signals.iloc[end] > 0:
                end += 1
            filtered_signals.iloc[t:end] = 0

    price_ret = prices_4h.pct_change().fillna(0)
    gross_ret = filtered_signals.shift(1) * price_ret
    net_30 = costs_std.apply(gross_ret, filtered_signals)
    net_50 = costs_stress.apply(gross_ret, filtered_signals)

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
        "n_feat": n_feat,
        "model_type": model_type,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "confidence_threshold": confidence_threshold,
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
        f"  ({entry},{exit_p}) tp={tp_mult} sl={sl_mult} vert={vert_bars} "
        f"n_feat={n_feat} {model_type} thr={confidence_threshold}: "
        f"Sharpe={m30['sharpe']:.3f} DSR={m30['dsr']:.3f} "
        f"acc={np.mean(fold_accs):.3f} trades={n_trades}/{n_entries} pos={pos_rate:.2f}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Round 1: LogReg, barrier sweep, entry=10/5 (most signals)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 1: LogReg, barrier sweep, entry=10/5")
print("=" * 60)

R1 = [
    (10, 5, 1.5, 1.0, 12, 5),
    (10, 5, 2.0, 1.0, 12, 5),
    (10, 5, 2.0, 1.5, 20, 5),
    (10, 5, 3.0, 1.5, 30, 5),
    (10, 5, 1.5, 1.0, 30, 3),
    (10, 5, 2.0, 1.5, 12, 3),
    (15, 10, 2.0, 1.0, 20, 5),
    (15, 10, 1.5, 1.5, 12, 5),
    (20, 10, 2.0, 1.5, 20, 5),
    (20, 10, 3.0, 1.0, 30, 3),
]

r1_results = []
for cfg in R1:
    r = run_meta_experiment(*cfg, model_type="logreg", confidence_threshold=0.5)
    if r:
        r1_results.append(r)
        all_results.append(r)

best_r1 = max(r1_results, key=lambda x: x["sharpe_30"]) if r1_results else None
if best_r1:
    print(
        f"\nR1 best: Sharpe={best_r1['sharpe_30']:.3f} DSR={best_r1['dsr_30']:.3f} "
        f"({best_r1['entry']},{best_r1['exit']}) tp={best_r1['tp_mult']} sl={best_r1['sl_mult']}"
    )

tracker.log_sweep(
    "round1_logreg_barrier_sweep",
    r1_results,
    summary_metrics={
        "sharpe": best_r1["sharpe_30"] if best_r1 else 0,
        "dsr": best_r1["dsr_30"] if best_r1 else 0,
        "n_configs": len(R1),
    },
    tags=TAGS,
)
with open(out_dir / "round1_results.json", "w") as f:
    json.dump(r1_results, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Round 2: Confidence threshold sweep on top R1 barrier configs
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 2: Confidence threshold sweep")
print("=" * 60)

r2_results = []
if r1_results:
    top2_r1 = sorted(r1_results, key=lambda x: x["sharpe_30"], reverse=True)[:2]
    for best in top2_r1:
        for thresh in [0.5, 0.55, 0.6]:
            for nf in [3, 5]:
                r = run_meta_experiment(
                    best["entry"],
                    best["exit"],
                    best["tp_mult"],
                    best["sl_mult"],
                    best["vert_bars"],
                    nf,
                    model_type="logreg",
                    confidence_threshold=thresh,
                )
                if r:
                    r2_results.append(r)
                    all_results.append(r)

best_r2 = max(r2_results, key=lambda x: x["sharpe_30"]) if r2_results else best_r1
if best_r2:
    print(
        f"\nR2 best: Sharpe={best_r2['sharpe_30']:.3f} DSR={best_r2['dsr_30']:.3f} "
        f"thresh={best_r2['confidence_threshold']}"
    )

tracker.log_sweep(
    "round2_threshold_sweep",
    r2_results,
    summary_metrics={"sharpe": best_r2["sharpe_30"] if best_r2 else 0, "dsr": best_r2["dsr_30"] if best_r2 else 0},
    tags=TAGS,
)
with open(out_dir / "round2_results.json", "w") as f:
    json.dump(r2_results, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Round 3: XGBoost meta-labeler
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 3: XGBoost meta-labeler")
print("=" * 60)

R3 = [
    (10, 5, 2.0, 1.0, 20, 5, 2, 10, 0.5),
    (10, 5, 2.0, 1.5, 20, 5, 2, 10, 0.5),
    (10, 5, 2.0, 1.0, 30, 5, 2, 10, 0.55),
    (10, 5, 3.0, 1.5, 30, 5, 3, 10, 0.5),
    (15, 10, 2.0, 1.0, 20, 5, 2, 10, 0.5),
    (15, 10, 2.0, 1.5, 30, 8, 2, 10, 0.5),
    (10, 5, 2.0, 1.0, 20, 8, 2, 20, 0.5),
    (10, 5, 1.5, 1.0, 12, 5, 2, 10, 0.55),
]

r3_results = []
for ep, ex, tp, sl, vb, nf, md, mcw, thresh in R3:
    r = run_meta_experiment(
        ep, ex, tp, sl, vb, nf, model_type="xgboost", max_depth=md, min_child_weight=mcw, confidence_threshold=thresh
    )
    if r:
        r3_results.append(r)
        all_results.append(r)

best_r3 = max(r3_results, key=lambda x: x["sharpe_30"]) if r3_results else None
if best_r3:
    print(f"\nR3 best: Sharpe={best_r3['sharpe_30']:.3f} DSR={best_r3['dsr_30']:.3f}")

tracker.log_sweep(
    "round3_xgboost_sweep",
    r3_results,
    summary_metrics={"sharpe": best_r3["sharpe_30"] if best_r3 else 0, "dsr": best_r3["dsr_30"] if best_r3 else 0},
    tags=TAGS,
)
with open(out_dir / "round3_xgboost_results.json", "w") as f:
    json.dump(r3_results, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Round 4: Broader param sweep — vary primary donchian params + feature sets
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 4: Broader donchian param + feature sweep")
print("=" * 60)

# Find best barrier from all results so far
all_so_far = r1_results + r2_results + r3_results
if all_so_far:
    best_so_far = max(all_so_far, key=lambda x: x["sharpe_30"])
    # Fix best barrier, vary primary params and n_feat
    R4 = []
    for ep, ex in [(10, 5), (15, 10), (20, 10), (30, 15)]:
        for nf in [3, 5, 8]:
            R4.append(
                (
                    ep,
                    ex,
                    best_so_far["tp_mult"],
                    best_so_far["sl_mult"],
                    best_so_far["vert_bars"],
                    nf,
                    best_so_far["confidence_threshold"],
                )
            )

    r4_results = []
    for ep, ex, tp, sl, vb, nf, thresh in R4:
        r = run_meta_experiment(ep, ex, tp, sl, vb, nf, model_type="logreg", confidence_threshold=thresh)
        if r:
            r4_results.append(r)
            all_results.append(r)

    best_r4 = max(r4_results, key=lambda x: x["sharpe_30"]) if r4_results else None
    if best_r4:
        print(
            f"\nR4 best: Sharpe={best_r4['sharpe_30']:.3f} DSR={best_r4['dsr_30']:.3f} "
            f"({best_r4['entry']},{best_r4['exit']}) nf={best_r4['n_feat']}"
        )

    tracker.log_sweep(
        "round4_donchian_param_sweep",
        r4_results,
        summary_metrics={"sharpe": best_r4["sharpe_30"] if best_r4 else 0, "dsr": best_r4["dsr_30"] if best_r4 else 0},
        tags=TAGS,
    )
    with open(out_dir / "round4_results.json", "w") as f:
        json.dump(r4_results, f, indent=2, default=str)
else:
    r4_results = []


# ──────────────────────────────────────────────────────────────────────────────
# Round 5: Primary baseline 4h (no meta) for fair comparison
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 5: 4h primary (no meta) baselines")
print("=" * 60)

primary_4h_results = []
for ep, ex in [(10, 5), (15, 10), (20, 10), (30, 20)]:
    sigs = donchian_channel_strategy(prices_4h, ep, ex)
    sigs_reg = sigs.copy()
    sigs_reg[regime_4h != bull_state_4h] = 0
    price_ret = prices_4h.pct_change().fillna(0)
    gross = sigs_reg.shift(1) * price_ret
    net = costs_std.apply(gross, sigs_reg)
    n_trades = int((sigs_reg.diff().abs() > 0).sum() // 2)
    m = compute_all_metrics(net.dropna(), n_trials=n_trials, periods_per_year=PERIODS_PER_YEAR_4H)
    print(f"  4h Donchian({ep},{ex}) regime-cond: Sharpe={m['sharpe']:.3f} DSR={m['dsr']:.3f} trades={n_trades}")
    primary_4h_results.append(
        {
            "entry": ep,
            "exit": ex,
            "sharpe_30": m["sharpe"],
            "dsr_30": m["dsr"],
            "max_dd": m["max_drawdown"],
            "n_trades": n_trades,
        }
    )

with open(out_dir / "primary_4h_baselines.json", "w") as f:
    json.dump(primary_4h_results, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# FINAL: Log overall best to wandb
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

best_primary_4h = max(primary_4h_results, key=lambda x: x["sharpe_30"]) if primary_4h_results else None
if best_primary_4h:
    print(f"Best 4h primary (no meta): Sharpe={best_primary_4h['sharpe_30']:.3f}")

if all_results:
    best_meta = max(all_results, key=lambda x: x["sharpe_30"])
    print("\nBest meta-labeled result:")
    print(
        f"  ({best_meta['entry']},{best_meta['exit']}) tp={best_meta['tp_mult']} "
        f"sl={best_meta['sl_mult']} vert={best_meta['vert_bars']} "
        f"model={best_meta['model_type']} thresh={best_meta['confidence_threshold']}"
    )
    print(f"  Sharpe@30bps: {best_meta['sharpe_30']:.3f}, DSR: {best_meta['dsr_30']:.3f}")
    print(f"  Sharpe@50bps: {best_meta['sharpe_50']:.3f}, DSR: {best_meta['dsr_50']:.3f}")
    print(f"  Meta accuracy: {best_meta.get('meta_accuracy', 0):.3f}")
    print(f"  n_trials: {n_trials}")

    print(f"\nBaseline daily Sharpe: {metrics_baseline['sharpe']:.3f}")
    if best_primary_4h:
        print(f"Best 4h primary Sharpe: {best_primary_4h['sharpe_30']:.3f}")
    print(f"Best meta-labeled Sharpe: {best_meta['sharpe_30']:.3f}")

    if best_meta["sharpe_30"] > 1.777:
        verdict = "META_IMPROVES_BASELINE"
    elif best_meta["sharpe_30"] > (best_primary_4h["sharpe_30"] if best_primary_4h else 0):
        verdict = "META_IMPROVES_4H_PRIMARY"
    else:
        verdict = "META_DOES_NOT_IMPROVE"
    print(f"\nVerdict: {verdict}")

    import wandb as wb

    wb.init(project="sparky-ai", entity="datadex_ai", name="meta_labeling_session1_final", tags=TAGS, reinit=True)
    wb.log(
        {
            "sharpe": best_meta["sharpe_30"],
            "dsr": best_meta["dsr_30"],
            "sharpe_stress": best_meta["sharpe_50"],
            "dsr_stress": best_meta["dsr_50"],
            "n_configs_tested": n_trials,
            "best_entry": best_meta["entry"],
            "best_exit": best_meta["exit"],
            "best_tp_mult": best_meta["tp_mult"],
            "best_sl_mult": best_meta["sl_mult"],
            "best_vert_bars": best_meta["vert_bars"],
            "best_model_type": best_meta["model_type"],
            "best_n_feat": best_meta["n_feat"],
            "best_confidence_threshold": best_meta["confidence_threshold"],
            "meta_accuracy": best_meta.get("meta_accuracy", 0),
            "baseline_daily_sharpe": metrics_baseline["sharpe"],
            "best_primary_4h_sharpe": best_primary_4h["sharpe_30"] if best_primary_4h else 0,
            "verdict": verdict,
        }
    )
    wb.finish()

    with open(out_dir / "session1_summary.json", "w") as f:
        json.dump(
            {
                "baseline_daily_sharpe": float(metrics_baseline["sharpe"]),
                "n_trials": n_trials,
                "verdict": verdict,
                "best_meta": best_meta,
                "best_primary_4h": best_primary_4h,
                "all_results": all_results,
            },
            f,
            indent=2,
            default=str,
        )

print("\nDone.")
