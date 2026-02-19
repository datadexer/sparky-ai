"""Meta-labeling session 1, batch 2: targeted experiments based on R1 findings."""

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

# Load prior session results
with open(out_dir / "session1_summary.json") as f:
    prior = json.load(f)
print(f"Prior best meta: Sharpe={prior['best_meta']['sharpe_30']:.3f}")
print(f"Prior n_trials: {prior['n_trials']}")

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
hourly = load("ohlcv_hourly_max_coverage", purpose="training")
h4 = (
    hourly.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
)
prices_4h = h4["close"]
PERIODS_PER_YEAR_4H = 2190

daily = load("btc_daily", purpose="training")
prices_daily = daily["close"]

print(f"4h data: {len(h4)} rows")

_, _, regime_4h, regime_proba_4h = train_hmm_regime_model(prices_4h, n_states=2)
ret_4h = prices_4h.pct_change()
bull_state_4h = 0 if ret_4h[regime_4h == 0].mean() > ret_4h[regime_4h == 1].mean() else 1
print(f"4h bull regime = {bull_state_4h}")

# Also try 3-state HMM
_, _, regime_4h_3s, regime_proba_4h_3s = train_hmm_regime_model(prices_4h, n_states=3)


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


def make_features_v2(h4_df, prices, regime_proba_2s, regime_proba_3s):
    """Enhanced features for batch 2."""
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
    feats["autocorr_3"] = ret.rolling(10).apply(
        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) > 3 else np.nan, raw=False
    )

    # Trend quality: regression R^2
    def trend_r2(x):
        t = np.arange(len(x))
        if len(x) < 5:
            return np.nan
        c = np.corrcoef(t, x)[0, 1]
        return c**2

    feats["trend_r2"] = ret.rolling(20).apply(trend_r2, raw=True)

    # ADX proxy
    dm_plus = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    feats["adx_proxy"] = (dm_plus.rolling(14).mean() - dm_minus.rolling(14).mean()).abs() / atr_4h.clip(lower=1e-8)

    # Choppiness
    atr_1bar = (high - low).abs()
    hl_range = (high.rolling(14).max() - low.rolling(14).min()).clip(lower=1e-8)
    feats["choppiness"] = np.log10(atr_1bar.rolling(14).sum() / hl_range) / np.log10(14)

    feats["vol_momentum"] = volume.rolling(10).mean() / volume.rolling(40).mean().clip(lower=1e-8)
    feats["dist_sma_20"] = (close - close.rolling(20).mean()) / close.rolling(20).mean().clip(lower=1e-8)
    feats["dist_sma_60"] = (close - close.rolling(60).mean()) / close.rolling(60).mean().clip(lower=1e-8)

    # 2-state HMM regime probability
    feats["regime_proba_2s"] = regime_proba_2s.iloc[:, bull_state_4h].values

    # 3-state HMM: identify bull/bear/sideways
    # Bull = highest mean return state
    ret_4h_v = ret.values
    state_means = {}
    for s in range(3):
        mask = regime_proba_3s.values.argmax(axis=1) == s
        if mask.sum() > 0:
            state_means[s] = float(np.nanmean(ret_4h_v[mask]))
        else:
            state_means[s] = 0.0
    bull_state_3s = max(state_means, key=state_means.get)
    feats["regime_proba_3s_bull"] = regime_proba_3s.iloc[:, bull_state_3s].values

    # Fractional differentiation d=0.1
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

    # Rolling primary model hit rate (requires prior labels — computed lazily)
    return feats


print("Computing enhanced features...")
feats_v2 = make_features_v2(h4, prices_4h, regime_proba_4h, regime_proba_4h_3s)
print(f"Features v2: {list(feats_v2.columns)}")

all_results_b2 = []
n_trials = prior["n_trials"]  # continue from prior count


FEATURE_SETS_V2 = {
    "v2_3": ["vol_ratio", "adx_proxy", "regime_proba_2s"],
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
}


def run_experiment_b2(
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
    regime_filter=True,
    use_calibration=False,
    kelly_fraction=None,
):
    global n_trials
    n_trials += 1

    signals = donchian_channel_strategy(prices_4h, entry, exit_p)
    signals_reg = signals.copy()
    if regime_filter:
        signals_reg[regime_4h != bull_state_4h] = 0

    entry_bars = np.where(np.diff(signals_reg.values, prepend=0) > 0)[0]
    n_entries = len(entry_bars)
    if n_entries < 30:
        print(f"  ({entry},{exit_p}): only {n_entries} signals, skip")
        return None

    labels, weights = triple_barrier_labels(prices_4h, signals_reg, atr_4h, tp_mult, sl_mult, vert_bars)
    valid_idx = np.where(~np.isnan(labels) & ~np.isnan(weights))[0]
    if len(valid_idx) < 20:
        return None

    feat_cols = FEATURE_SETS_V2[feat_key]
    X_all = feats_v2[feat_cols].values
    y_all = labels
    w_all = weights

    pos_rate = float(np.nanmean(y_all[valid_idx]))
    if pos_rate < 0.05 or pos_rate > 0.95:
        return None

    # Ensure test_size > gap; gap=vert_bars bars
    min_test = vert_bars + 1
    max_possible_splits = max(2, len(valid_idx) // (min_test + 1))
    n_splits = min(5, len(valid_idx) // 8, max_possible_splits)
    n_splits = max(n_splits, 2)

    oof_proba = np.full(len(prices_4h), np.nan)
    fold_accs = []

    # gap in signal-index space: how many signal-bars does vert_bars span?
    # On average, signals every (len(prices)/n_entries) bars → vert_bars * n_entries / len(prices)
    gap_signals = max(1, int(vert_bars * len(valid_idx) / len(prices_4h)))
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
            if use_calibration:
                clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            else:
                clf = base
            clf.fit(X_tr_s, y_tr, **({"sample_weight": w_tr} if not use_calibration else {}))
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
            if use_calibration:
                clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
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

    if kelly_fraction is not None:
        # Fractional Kelly position sizing using calibrated probabilities
        # p = prob of win, b = avg win/loss ratio, f = (pb - (1-p)) / b * kelly_fraction
        # For simplicity: scale position by kelly_fraction * proba
        position_sizes = pd.Series(np.where(filtered_signals > 0, 1.0, 0.0), index=prices_4h.index)
        kelly_pos = position_sizes.copy().astype(float)
        for t in valid_idx:
            if ~np.isnan(oof_proba[t]) and filtered_signals.iloc[t] > 0:
                p = oof_proba[t]
                kelly_pos.iloc[t] = kelly_fraction * (2 * p - 1)  # edge * kelly
        # Forward-fill kelly position within trade windows
        sig_arr = filtered_signals.values
        kp_arr = kelly_pos.values
        for t in range(1, len(sig_arr)):
            if sig_arr[t] > 0 and sig_arr[t - 1] > 0:
                kp_arr[t] = kp_arr[t - 1]
        kelly_pos = pd.Series(kp_arr.clip(0, 1), index=prices_4h.index)

        gross_ret = kelly_pos.shift(1) * price_ret
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
        f"  ({entry},{exit_p}) tp={tp_mult} sl={sl_mult} vert={vert_bars} "
        f"feat={feat_key} {model_type} thr={confidence_threshold} "
        f"reg={regime_filter} cal={use_calibration} kelly={kelly_fraction}: "
        f"Sharpe={m30['sharpe']:.3f} DSR={m30['dsr']:.3f} "
        f"acc={np.mean(fold_accs):.3f} trades={n_trades}/{n_entries}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Round 5: No regime filter — let meta-model handle regime selection
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 5: No binary regime filter (meta handles regime)")
print("=" * 60)

R5 = [
    (10, 5, 1.5, 1.0, 12, "v2_5a", "logreg", 2, 10, 0.5, False),
    (10, 5, 1.5, 1.0, 12, "v2_5b", "logreg", 2, 10, 0.5, False),
    (10, 5, 2.0, 1.0, 12, "v2_5a", "logreg", 2, 10, 0.55, False),
    (10, 5, 2.0, 1.0, 20, "v2_5c", "logreg", 2, 10, 0.5, False),
    (10, 5, 1.5, 1.0, 12, "v2_8", "logreg", 2, 10, 0.5, False),
    (15, 10, 2.0, 1.0, 20, "v2_5a", "logreg", 2, 10, 0.5, False),
    (30, 20, 1.5, 1.0, 12, "v2_5b", "logreg", 2, 10, 0.5, False),
    (30, 20, 2.0, 1.5, 30, "v2_5c", "logreg", 2, 10, 0.55, False),
]

r5_results = []
for ep, ex, tp, sl, vb, fk, mt, md, mcw, thr, rf in R5:
    r = run_experiment_b2(ep, ex, tp, sl, vb, fk, mt, md, mcw, thr, regime_filter=rf)
    if r:
        r5_results.append(r)
        all_results_b2.append(r)

best_r5 = max(r5_results, key=lambda x: x["sharpe_30"]) if r5_results else None
if best_r5:
    print(
        f"\nR5 best: Sharpe={best_r5['sharpe_30']:.3f} DSR={best_r5['dsr_30']:.3f} "
        f"({best_r5['entry']},{best_r5['exit']}) feat={best_r5['feat_key']}"
    )

tracker.log_sweep(
    "round5_no_regime_filter",
    r5_results,
    summary_metrics={"sharpe": best_r5["sharpe_30"] if best_r5 else 0, "dsr": best_r5["dsr_30"] if best_r5 else 0},
    tags=TAGS,
)
with open(out_dir / "round5_results.json", "w") as f:
    json.dump(r5_results, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Round 6: Enhanced feature sets — 3-state HMM + trend quality
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 6: 3-state HMM features + trend quality")
print("=" * 60)

R6 = [
    (10, 5, 1.5, 1.0, 12, "v2_5b", "logreg", 2, 10, 0.5, True),
    (10, 5, 1.5, 1.0, 12, "v2_5c", "logreg", 2, 10, 0.5, True),
    (10, 5, 2.0, 1.0, 20, "v2_5b", "logreg", 2, 10, 0.55, True),
    (10, 5, 2.0, 1.0, 20, "v2_5c", "logreg", 2, 10, 0.55, True),
    (10, 5, 1.5, 1.0, 12, "v2_8", "logreg", 2, 10, 0.5, True),
    (10, 5, 2.0, 1.0, 12, "v2_8", "logreg", 2, 10, 0.55, True),
    (15, 10, 1.5, 1.0, 12, "v2_5b", "logreg", 2, 10, 0.5, True),
    (15, 10, 2.0, 1.0, 20, "v2_5c", "logreg", 2, 10, 0.5, True),
    (30, 20, 1.5, 1.0, 12, "v2_5b", "logreg", 2, 10, 0.5, True),
    (30, 20, 2.0, 1.5, 30, "v2_5c", "logreg", 2, 10, 0.5, True),
]

r6_results = []
for ep, ex, tp, sl, vb, fk, mt, md, mcw, thr, rf in R6:
    r = run_experiment_b2(ep, ex, tp, sl, vb, fk, mt, md, mcw, thr, regime_filter=rf)
    if r:
        r6_results.append(r)
        all_results_b2.append(r)

best_r6 = max(r6_results, key=lambda x: x["sharpe_30"]) if r6_results else None
if best_r6:
    print(f"\nR6 best: Sharpe={best_r6['sharpe_30']:.3f} DSR={best_r6['dsr_30']:.3f}")

tracker.log_sweep(
    "round6_3state_hmm_features",
    r6_results,
    summary_metrics={"sharpe": best_r6["sharpe_30"] if best_r6 else 0, "dsr": best_r6["dsr_30"] if best_r6 else 0},
    tags=TAGS,
)
with open(out_dir / "round6_results.json", "w") as f:
    json.dump(r6_results, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Round 7: XGBoost with 3-state HMM features
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 7: XGBoost + 3-state HMM")
print("=" * 60)

R7 = [
    (10, 5, 1.5, 1.0, 12, "v2_5b", "xgboost", 2, 10, 0.5, True),
    (10, 5, 1.5, 1.0, 12, "v2_5c", "xgboost", 2, 10, 0.5, True),
    (10, 5, 2.0, 1.0, 12, "v2_5b", "xgboost", 2, 10, 0.55, True),
    (10, 5, 2.0, 1.0, 20, "v2_5c", "xgboost", 3, 10, 0.5, True),
    (10, 5, 1.5, 1.0, 12, "v2_5a", "xgboost", 2, 20, 0.55, True),
    (15, 10, 2.0, 1.0, 20, "v2_5b", "xgboost", 2, 10, 0.5, True),
    (30, 20, 1.5, 1.0, 12, "v2_5c", "xgboost", 2, 10, 0.5, True),
]

r7_results = []
for ep, ex, tp, sl, vb, fk, mt, md, mcw, thr, rf in R7:
    r = run_experiment_b2(ep, ex, tp, sl, vb, fk, mt, md, mcw, thr, regime_filter=rf)
    if r:
        r7_results.append(r)
        all_results_b2.append(r)

best_r7 = max(r7_results, key=lambda x: x["sharpe_30"]) if r7_results else None
if best_r7:
    print(f"\nR7 best: Sharpe={best_r7['sharpe_30']:.3f} DSR={best_r7['dsr_30']:.3f}")

tracker.log_sweep(
    "round7_xgb_3state_hmm",
    r7_results,
    summary_metrics={"sharpe": best_r7["sharpe_30"] if best_r7 else 0, "dsr": best_r7["dsr_30"] if best_r7 else 0},
    tags=TAGS,
)
with open(out_dir / "round7_results.json", "w") as f:
    json.dump(r7_results, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Round 8: Calibrated meta-labeler + Kelly position sizing
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 8: Calibrated meta + Kelly sizing")
print("=" * 60)

# Use best overall feature/barrier config from all rounds
all_b2_so_far = r5_results + r6_results + r7_results
best_b2_so_far = max(all_b2_so_far, key=lambda x: x["sharpe_30"]) if all_b2_so_far else None

R8 = []
for fk in ["v2_5a", "v2_5b", "v2_5c"]:
    for kf in [0.25, 0.5]:
        R8.append((10, 5, 1.5, 1.0, 12, fk, "logreg", 2, 10, 0.5, True, True, kf))
    # No kelly, just calibration
    R8.append((10, 5, 1.5, 1.0, 12, fk, "logreg", 2, 10, 0.5, True, True, None))

# Also try best xgb config with calibration
R8 += [
    (10, 5, 1.5, 1.0, 12, "v2_5b", "xgboost", 2, 10, 0.55, True, True, 0.5),
    (10, 5, 2.0, 1.0, 12, "v2_5c", "xgboost", 2, 10, 0.5, True, True, None),
]

r8_results = []
for cfg in R8:
    ep, ex, tp, sl, vb, fk, mt, md, mcw, thr, rf, cal, kf = cfg
    r = run_experiment_b2(
        ep, ex, tp, sl, vb, fk, mt, md, mcw, thr, regime_filter=rf, use_calibration=cal, kelly_fraction=kf
    )
    if r:
        r8_results.append(r)
        all_results_b2.append(r)

best_r8 = max(r8_results, key=lambda x: x["sharpe_30"]) if r8_results else None
if best_r8:
    print(f"\nR8 best: Sharpe={best_r8['sharpe_30']:.3f} DSR={best_r8['dsr_30']:.3f} kelly={best_r8['kelly_fraction']}")

tracker.log_sweep(
    "round8_calibrated_kelly",
    r8_results,
    summary_metrics={"sharpe": best_r8["sharpe_30"] if best_r8 else 0, "dsr": best_r8["dsr_30"] if best_r8 else 0},
    tags=TAGS,
)
with open(out_dir / "round8_results.json", "w") as f:
    json.dump(r8_results, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BATCH 2 SUMMARY")
print("=" * 60)

prior_best = prior["best_meta"]
print(f"Prior session best: Sharpe={prior_best['sharpe_30']:.3f} DSR={prior_best['dsr_30']:.3f}")

if all_results_b2:
    best_b2 = max(all_results_b2, key=lambda x: x["sharpe_30"])
    print(f"\nBatch 2 best: Sharpe={best_b2['sharpe_30']:.3f} DSR={best_b2['dsr_30']:.3f}")
    print(
        f"  ({best_b2['entry']},{best_b2['exit']}) feat={best_b2['feat_key']} "
        f"model={best_b2['model_type']} thr={best_b2['confidence_threshold']} "
        f"kelly={best_b2['kelly_fraction']} reg={best_b2['regime_filter']}"
    )
    print(f"  Sharpe@50bps: {best_b2['sharpe_50']:.3f} DSR@50bps: {best_b2['dsr_50']:.3f}")
    print(f"  n_trials total: {n_trials}")

    # Compare all-time best
    all_time_candidates = [prior_best, best_b2]
    all_time_best = max(all_time_candidates, key=lambda x: x["sharpe_30"])
    print(f"\nAll-time best Sharpe: {all_time_best['sharpe_30']:.3f} DSR: {all_time_best['dsr_30']:.3f}")

    if all_time_best["sharpe_30"] > 1.777:
        verdict = "BEATS_BASELINE"
    elif all_time_best["sharpe_30"] > prior["best_primary_4h"]["sharpe_30"]:
        verdict = "META_IMPROVES_4H"
    else:
        verdict = "NO_IMPROVEMENT"
    print(f"Verdict: {verdict}")

    import wandb as wb

    wb.init(
        project="sparky-ai", entity="datadex_ai", name="meta_labeling_session1_batch2_final", tags=TAGS, reinit=True
    )
    wb.log(
        {
            "sharpe": all_time_best["sharpe_30"],
            "dsr": all_time_best["dsr_30"],
            "sharpe_stress": all_time_best["sharpe_50"],
            "dsr_stress": all_time_best["dsr_50"],
            "n_configs_tested": n_trials,
            "verdict": verdict,
            "batch2_best_sharpe": best_b2["sharpe_30"],
            "batch2_best_dsr": best_b2["dsr_30"],
        }
    )
    wb.finish()

    with open(out_dir / "session1_batch2_summary.json", "w") as f:
        json.dump(
            {
                "n_trials": n_trials,
                "verdict": verdict,
                "all_time_best": all_time_best,
                "batch2_best": best_b2,
                "batch2_all_results": all_results_b2,
            },
            f,
            indent=2,
            default=str,
        )

print("\nDone.")
