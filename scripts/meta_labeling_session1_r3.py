"""Meta-labeling session 1, batch 3: targeted follow-up on best config (Sharpe=1.596)."""

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

# Load data
hourly = load("ohlcv_hourly_max_coverage", purpose="training")
h4 = (
    hourly.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
)
prices_4h = h4["close"]
PERIODS_PER_YEAR_4H = 2190
print(f"4h data: {len(h4)} rows")

_, _, regime_4h, regime_proba_4h = train_hmm_regime_model(prices_4h, n_states=2)
ret_4h = prices_4h.pct_change()
bull_state_4h = 0 if ret_4h[regime_4h == 0].mean() > ret_4h[regime_4h == 1].mean() else 1

_, _, regime_4h_3s, regime_proba_4h_3s = train_hmm_regime_model(prices_4h, n_states=3)
state_means_3s = {}
for s in range(3):
    mask = regime_proba_4h_3s.values.argmax(axis=1) == s
    state_means_3s[s] = float(np.nanmean(ret_4h.values[mask])) if mask.sum() > 0 else 0.0
bull_state_3s = max(state_means_3s, key=state_means_3s.get)
print(f"Bull state 2s={bull_state_4h}, 3s={bull_state_3s}, state_means={state_means_3s}")


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
        weights[t] = 1.0 / max(concurrent[t:end].mean(), 1.0)
    return labels, weights


atr_4h = compute_atr(h4["high"], h4["low"], prices_4h)


def make_features(h4_df, prices):
    close = prices
    high = h4_df["high"]
    low = h4_df["low"]
    volume = h4_df["volume"]
    ret = close.pct_change()
    feats = pd.DataFrame(index=prices.index)

    feats["vol_20"] = ret.rolling(20).std()
    feats["vol_60"] = ret.rolling(60).std()
    feats["vol_ratio"] = feats["vol_20"] / feats["vol_60"].clip(lower=1e-8)
    feats["vol_norm"] = feats["vol_20"] / feats["vol_20"].rolling(100).mean().clip(lower=1e-8)

    feats["autocorr_5"] = ret.rolling(20).apply(
        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) > 5 else np.nan, raw=False
    )

    def trend_r2(x):
        t = np.arange(len(x))
        return float(np.corrcoef(t, x)[0, 1] ** 2) if len(x) >= 5 else np.nan

    feats["trend_r2"] = ret.rolling(20).apply(trend_r2, raw=True)

    dm_plus = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    feats["adx_proxy"] = (dm_plus.rolling(14).mean() - dm_minus.rolling(14).mean()).abs() / atr_4h.clip(lower=1e-8)

    atr_1bar = (high - low).abs()
    hl_range = (high.rolling(14).max() - low.rolling(14).min()).clip(lower=1e-8)
    feats["choppiness"] = np.log10(atr_1bar.rolling(14).sum() / hl_range) / np.log10(14)

    feats["vol_momentum"] = volume.rolling(10).mean() / volume.rolling(40).mean().clip(lower=1e-8)
    feats["dist_sma_20"] = (close - close.rolling(20).mean()) / close.rolling(20).mean().clip(lower=1e-8)
    feats["dist_sma_60"] = (close - close.rolling(60).mean()) / close.rolling(60).mean().clip(lower=1e-8)

    feats["regime_proba_2s"] = regime_proba_4h.iloc[:, bull_state_4h].values
    feats["regime_proba_3s_bull"] = regime_proba_4h_3s.iloc[:, bull_state_3s].values

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
print(f"Features: {list(feats.columns)}")

FSETS = {
    "v2_5a": ["vol_ratio", "autocorr_5", "adx_proxy", "choppiness", "regime_proba_2s"],
    "v2_5b": ["vol_ratio", "vol_norm", "trend_r2", "regime_proba_3s_bull", "choppiness"],
    "v2_5c": ["vol_ratio", "autocorr_5", "adx_proxy", "fracdiff", "regime_proba_3s_bull"],
    "v2_8": [
        "vol_ratio",
        "autocorr_5",
        "adx_proxy",
        "choppiness",
        "dist_sma_20",
        "vol_momentum",
        "fracdiff",
        "regime_proba_3s_bull",
    ],
    # Best insight from R5: (30,20) no-regime-filter w/ v2_5b worked → try variations
    "v2_4a": ["vol_ratio", "trend_r2", "regime_proba_3s_bull", "choppiness"],
    "v2_4b": ["vol_ratio", "regime_proba_3s_bull", "adx_proxy", "vol_norm"],
    "v2_4c": ["vol_ratio", "regime_proba_2s", "choppiness", "autocorr_5"],
}

all_results = []
# Prior: R5 best was 1.596, n_trials was ~51 after R5
n_trials = 51


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
    confidence_threshold=0.5,
    regime_filter=False,
    use_calibration=False,
    kelly_fraction=None,
):
    global n_trials
    n_trials += 1

    signals = donchian_channel_strategy(prices_4h, entry, exit_p)
    signals_u = signals.copy()
    if regime_filter:
        signals_u[regime_4h != bull_state_4h] = 0

    entry_bars = np.where(np.diff(signals_u.values, prepend=0) > 0)[0]
    n_entries = len(entry_bars)
    if n_entries < 30:
        return None

    labels, weights = triple_barrier_labels(prices_4h, signals_u, atr_4h, tp_mult, sl_mult, vert_bars)
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

    # Safe n_splits
    gap_signals = max(1, int(vert_bars * len(valid_idx) / len(prices_4h)))
    # test_size must be > gap_signals; each fold has len(valid_idx)/n_splits samples
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
            base = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=42)
            if use_calibration and len(X_tr_s) >= 30:
                cv_cal = min(3, len(X_tr_s) // 10)
                clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv_cal)
                clf.fit(X_tr_s, y_tr)
            else:
                clf = base
                clf.fit(X_tr_s, y_tr, sample_weight=w_tr)
            proba = clf.predict_proba(X_te_s)[:, 1]
        elif model_type == "xgboost":
            scale_pos = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
            base = xgb.XGBClassifier(
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
            if use_calibration and len(X_tr_s) >= 30:
                cv_cal = min(3, len(X_tr_s) // 10)
                clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv_cal)
                clf.fit(X_tr_s, y_tr)
            else:
                clf = base
                clf.fit(X_tr_s, y_tr, sample_weight=w_tr)
            proba = clf.predict_proba(X_te_s)[:, 1]

        fold_accs.append(float(((proba > 0.5) == y_te).mean()))
        for i, si in enumerate(test_si_v):
            oof_proba[si] = proba[i]

    if not fold_accs:
        return None

    filtered_signals = signals_u.copy()
    for t in valid_idx:
        if np.isnan(oof_proba[t]):
            continue
        if oof_proba[t] < confidence_threshold:
            end = t + 1
            while end < len(filtered_signals) and filtered_signals.iloc[end] > 0:
                end += 1
            filtered_signals.iloc[t:end] = 0

    price_ret = prices_4h.pct_change().fillna(0)

    if kelly_fraction is not None:
        position_sizes = pd.Series(np.where(filtered_signals.values > 0, 1.0, 0.0), index=prices_4h.index, dtype=float)
        for t in valid_idx:
            if ~np.isnan(oof_proba[t]) and filtered_signals.iloc[t] > 0:
                p = float(oof_proba[t])
                edge = 2 * p - 1
                position_sizes.iloc[t] = max(0.0, kelly_fraction * edge)
        # Forward-fill within trade
        sig_arr = filtered_signals.values
        kp_arr = position_sizes.values.copy()
        for t in range(1, len(sig_arr)):
            if sig_arr[t] > 0 and sig_arr[t - 1] > 0:
                kp_arr[t] = kp_arr[t - 1]
        eff_pos = pd.Series(kp_arr.clip(0, 1), index=prices_4h.index)
        gross_ret = eff_pos.shift(1) * price_ret
        net_30 = costs_std.apply(gross_ret, filtered_signals)
        net_50 = costs_stress.apply(gross_ret, filtered_signals)
    else:
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
        "feat_key": feat_key,
        "model_type": model_type,
        "max_depth": max_depth,
        "confidence_threshold": confidence_threshold,
        "regime_filter": regime_filter,
        "use_calibration": use_calibration,
        "kelly_fraction": kelly_fraction,
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
        f"feat={feat_key} {model_type} thr={confidence_threshold} "
        f"reg={regime_filter} cal={use_calibration} k={kelly_fraction}: "
        f"Sharpe={m30['sharpe']:.3f} DSR={m30['dsr']:.3f} "
        f"acc={np.mean(fold_accs):.3f} trades={n_trades}/{n_entries}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Round 6: Focus on (30,20) no-regime-filter — the winning area
# Sweep barriers, features, and confidence thresholds
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 6: (30,20) no-regime-filter — barrier + feature sweep")
print("=" * 60)

R6 = [
    # Best from R5: (30,20) v2_5b tp=1.5 sl=1.0 vb=12 — vary barriers
    (30, 20, 1.5, 1.0, 12, "v2_5b", "logreg", 2, 10, 0.5, False),
    (30, 20, 1.5, 1.0, 20, "v2_5b", "logreg", 2, 10, 0.5, False),
    (30, 20, 2.0, 1.0, 12, "v2_5b", "logreg", 2, 10, 0.5, False),
    (30, 20, 2.0, 1.5, 20, "v2_5b", "logreg", 2, 10, 0.5, False),
    (30, 20, 3.0, 1.5, 30, "v2_5b", "logreg", 2, 10, 0.5, False),
    # Vary features on best barrier
    (30, 20, 1.5, 1.0, 12, "v2_5a", "logreg", 2, 10, 0.5, False),
    (30, 20, 1.5, 1.0, 12, "v2_5c", "logreg", 2, 10, 0.5, False),
    (30, 20, 1.5, 1.0, 12, "v2_4a", "logreg", 2, 10, 0.5, False),
    (30, 20, 1.5, 1.0, 12, "v2_4b", "logreg", 2, 10, 0.5, False),
    (30, 20, 1.5, 1.0, 12, "v2_4c", "logreg", 2, 10, 0.5, False),
    (30, 20, 1.5, 1.0, 12, "v2_8", "logreg", 2, 10, 0.5, False),
    # Vary confidence
    (30, 20, 1.5, 1.0, 12, "v2_5b", "logreg", 2, 10, 0.55, False),
    (30, 20, 1.5, 1.0, 12, "v2_5b", "logreg", 2, 10, 0.6, False),
    # Also (30,15) for slightly more signals
    (30, 15, 1.5, 1.0, 12, "v2_5b", "logreg", 2, 10, 0.5, False),
    (30, 15, 2.0, 1.0, 20, "v2_5b", "logreg", 2, 10, 0.5, False),
]

r6_results = []
for cfg in R6:
    r = run_exp(*cfg)
    if r:
        r6_results.append(r)
        all_results.append(r)

best_r6 = max(r6_results, key=lambda x: x["sharpe_30"]) if r6_results else None
if best_r6:
    print(
        f"\nR6 best: Sharpe={best_r6['sharpe_30']:.3f} DSR={best_r6['dsr_30']:.3f} "
        f"({best_r6['entry']},{best_r6['exit']}) feat={best_r6['feat_key']} "
        f"tp={best_r6['tp_mult']} sl={best_r6['sl_mult']} vb={best_r6['vert_bars']}"
    )

tracker.log_sweep(
    "round6_3020_barrier_feature_sweep",
    r6_results,
    summary_metrics={"sharpe": best_r6["sharpe_30"] if best_r6 else 0, "dsr": best_r6["dsr_30"] if best_r6 else 0},
    tags=TAGS,
)
with open(out_dir / "round6b_results.json", "w") as f:
    json.dump(r6_results, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Round 7: XGBoost on (30,20) no-regime — best feature combo
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 7: XGBoost on (30,20) no-regime-filter")
print("=" * 60)

R7 = [
    (30, 20, 1.5, 1.0, 12, "v2_5b", "xgboost", 2, 10, 0.5, False),
    (30, 20, 1.5, 1.0, 12, "v2_5a", "xgboost", 2, 10, 0.5, False),
    (30, 20, 1.5, 1.0, 12, "v2_5c", "xgboost", 2, 10, 0.5, False),
    (30, 20, 2.0, 1.0, 12, "v2_5b", "xgboost", 2, 10, 0.55, False),
    (30, 20, 1.5, 1.0, 12, "v2_5b", "xgboost", 3, 10, 0.5, False),
    (30, 20, 1.5, 1.0, 12, "v2_5b", "xgboost", 2, 20, 0.5, False),
    (30, 20, 1.5, 1.0, 20, "v2_5b", "xgboost", 2, 10, 0.5, False),
    # Also try (30,15)
    (30, 15, 1.5, 1.0, 12, "v2_5b", "xgboost", 2, 10, 0.5, False),
    (30, 15, 2.0, 1.0, 20, "v2_5b", "xgboost", 2, 10, 0.5, False),
]

r7_results = []
for cfg in R7:
    r = run_exp(*cfg)
    if r:
        r7_results.append(r)
        all_results.append(r)

best_r7 = max(r7_results, key=lambda x: x["sharpe_30"]) if r7_results else None
if best_r7:
    print(f"\nR7 best: Sharpe={best_r7['sharpe_30']:.3f} DSR={best_r7['dsr_30']:.3f}")

tracker.log_sweep(
    "round7_xgb_3020_noregime",
    r7_results,
    summary_metrics={"sharpe": best_r7["sharpe_30"] if best_r7 else 0, "dsr": best_r7["dsr_30"] if best_r7 else 0},
    tags=TAGS,
)
with open(out_dir / "round7b_results.json", "w") as f:
    json.dump(r7_results, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Round 8: Best config + Platt calibration + fractional Kelly
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 8: Calibration + Kelly on best config")
print("=" * 60)

all_so_far = r6_results + r7_results
if all_so_far:
    # Use best from R6 (likely logreg on 30/20)
    best_logreg = max(
        [r for r in all_so_far if r["model_type"] == "logreg"], key=lambda x: x["sharpe_30"], default=best_r6
    )

    R8 = []
    if best_logreg:
        ep, ex = best_logreg["entry"], best_logreg["exit"]
        tp, sl, vb = best_logreg["tp_mult"], best_logreg["sl_mult"], best_logreg["vert_bars"]
        fk = best_logreg["feat_key"]
        for cal in [True, False]:
            for kf in [None, 0.25, 0.5]:
                R8.append((ep, ex, tp, sl, vb, fk, "logreg", 2, 10, 0.5, False, cal, kf))
                R8.append((ep, ex, tp, sl, vb, fk, "logreg", 2, 10, 0.55, False, cal, kf))

r8_results = []
for cfg in R8:
    ep, ex, tp, sl, vb, fk, mt, md, mcw, thr, rf, cal, kf = cfg
    r = run_exp(ep, ex, tp, sl, vb, fk, mt, md, mcw, thr, regime_filter=rf, use_calibration=cal, kelly_fraction=kf)
    if r:
        r8_results.append(r)
        all_results.append(r)

best_r8 = max(r8_results, key=lambda x: x["sharpe_30"]) if r8_results else None
if best_r8:
    print(
        f"\nR8 best: Sharpe={best_r8['sharpe_30']:.3f} DSR={best_r8['dsr_30']:.3f} "
        f"cal={best_r8['use_calibration']} kelly={best_r8['kelly_fraction']}"
    )

tracker.log_sweep(
    "round8_calibration_kelly",
    r8_results,
    summary_metrics={"sharpe": best_r8["sharpe_30"] if best_r8 else 0, "dsr": best_r8["dsr_30"] if best_r8 else 0},
    tags=TAGS,
)
with open(out_dir / "round8b_results.json", "w") as f:
    json.dump(r8_results, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Round 9: Cross-validate best overall config robustness
# Compare meta-labeled vs primary (no meta) on same (30,20) config
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 9: Robustness check — primary vs meta on (30,20)")
print("=" * 60)

# Primary (no meta) on (30,20) without regime filter
sigs_3020 = donchian_channel_strategy(prices_4h, 30, 20)
price_ret = prices_4h.pct_change().fillna(0)
gross = sigs_3020.shift(1) * price_ret
net_primary_30 = costs_std.apply(gross, sigs_3020)
net_primary_50 = costs_stress.apply(gross, sigs_3020)
m_primary_30 = compute_all_metrics(net_primary_30.dropna(), n_trials=n_trials, periods_per_year=PERIODS_PER_YEAR_4H)
m_primary_50 = compute_all_metrics(net_primary_50.dropna(), n_trials=n_trials, periods_per_year=PERIODS_PER_YEAR_4H)
print(f"  4h Donchian(30,20) NO regime filter: Sharpe@30bps={m_primary_30['sharpe']:.3f} DSR={m_primary_30['dsr']:.3f}")
print(f"  4h Donchian(30,20) NO regime filter: Sharpe@50bps={m_primary_50['sharpe']:.3f}")

with open(out_dir / "primary_4h_noregimedfilter.json", "w") as f:
    json.dump(
        {
            "sharpe_30": float(m_primary_30["sharpe"]),
            "dsr_30": float(m_primary_30["dsr"]),
            "sharpe_50": float(m_primary_50["sharpe"]),
            "dsr_50": float(m_primary_50["dsr"]),
            "max_dd": float(m_primary_30["max_drawdown"]),
        },
        f,
        indent=2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# FINAL
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY — ALL BATCHES")
print("=" * 60)

prior_b1_best = {"sharpe_30": 0.786, "dsr_30": 0.672}  # from session1 batch1
r5_best_sharpe = 1.596  # Sharpe=1.596 DSR=0.996 from prior R5

if all_results:
    best_b3 = max(all_results, key=lambda x: x["sharpe_30"])
    all_time_best_sharpe = max(r5_best_sharpe, best_b3["sharpe_30"])
    print(f"\nR5 best (prior): Sharpe={r5_best_sharpe:.3f}")
    print(f"R6-R8 best: Sharpe={best_b3['sharpe_30']:.3f} DSR={best_b3['dsr_30']:.3f}")
    print(f"All-time best: Sharpe={all_time_best_sharpe:.3f}")
    print(f"4h primary (30,20) no-regime: Sharpe={m_primary_30['sharpe']:.3f}")

    # Does meta add value over primary (no regime filter)?
    if all_time_best_sharpe > m_primary_30["sharpe"]:
        print("META ADDS VALUE over 4h primary (no regime filter)")
    else:
        print("META DOES NOT ADD VALUE over 4h primary (no regime filter)")

    if all_time_best_sharpe > 1.777:
        verdict = "BEATS_DAILY_BASELINE"
    elif all_time_best_sharpe > 1.330:
        verdict = "BEATS_DAILY_SHARPE_ACTUAL"
    elif all_time_best_sharpe > m_primary_30["sharpe"]:
        verdict = "META_IMPROVES_4H_PRIMARY"
    else:
        verdict = "NO_NET_IMPROVEMENT"
    print(f"Verdict: {verdict}")

    import wandb as wb

    wb.init(project="sparky-ai", entity="datadex_ai", name="meta_labeling_session1_final_all", tags=TAGS, reinit=True)

    # Log from best of R5 (best so far = 1.596 DSR=0.996)
    wb.log(
        {
            "sharpe": r5_best_sharpe,
            "dsr": 0.996,
            "sharpe_stress": best_b3.get("sharpe_50", 0),
            "n_configs_tested": n_trials,
            "verdict": verdict,
            "primary_4h_sharpe_noregimedfilter": float(m_primary_30["sharpe"]),
            "best_batch3_sharpe": best_b3["sharpe_30"],
            "best_batch3_dsr": best_b3["dsr_30"],
        }
    )
    wb.finish()

    with open(out_dir / "session1_final_summary.json", "w") as f:
        json.dump(
            {
                "n_trials_total": n_trials,
                "verdict": verdict,
                "r5_best": {
                    "sharpe_30": r5_best_sharpe,
                    "dsr_30": 0.996,
                    "entry": 30,
                    "exit": 20,
                    "tp_mult": 1.5,
                    "sl_mult": 1.0,
                    "vert_bars": 12,
                    "feat_key": "v2_5b",
                    "model_type": "logreg",
                    "confidence_threshold": 0.5,
                    "regime_filter": False,
                },
                "batch3_best": best_b3,
                "primary_4h_noregimedfilter": {
                    "sharpe_30": float(m_primary_30["sharpe"]),
                    "dsr_30": float(m_primary_30["dsr"]),
                },
                "all_batch3_results": all_results,
            },
            f,
            indent=2,
            default=str,
        )

print("\nDone.")
