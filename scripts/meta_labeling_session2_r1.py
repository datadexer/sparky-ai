"""Meta-labeling session 2, round 1: shorter Donchian params for more signals."""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

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

print(f"4h bars: {len(h4)}")

_, _, regime_4h_3s, regime_proba_4h_3s = train_hmm_regime_model(prices_4h, n_states=3)
ret_4h = prices_4h.pct_change()
state_means_3s = {s: float(np.nanmean(ret_4h.values[regime_proba_4h_3s.values.argmax(axis=1) == s])) for s in range(3)}
bull_state_3s = max(state_means_3s, key=state_means_3s.get)
print(f"3-state HMM bull state: {bull_state_3s}, means: {state_means_3s}")


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

    # Fractional diff of log price (d=0.1, retains memory)
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

# Session 1 best feature set + variants
FSETS = {
    # Session 1 best: trend_r2, regime_proba_3s, adx_proxy
    "3c": ["trend_r2", "regime_proba_3s", "adx_proxy"],
    # Vol + regime
    "3a": ["vol_ratio", "regime_proba_3s", "trend_r2"],
    "3b": ["vol_norm", "regime_proba_3s", "choppiness"],
    # 5-feature best from S1
    "5best": ["vol_ratio", "vol_norm", "trend_r2", "regime_proba_3s", "choppiness"],
    # New variants with autocorr and fracdiff
    "3d": ["trend_r2", "regime_proba_3s", "autocorr_5"],
    "3e": ["adx_proxy", "regime_proba_3s", "fracdiff"],
    "4a": ["trend_r2", "regime_proba_3s", "adx_proxy", "vol_ratio"],
    "4b": ["trend_r2", "regime_proba_3s", "choppiness", "vol_norm"],
}

# Start from n_trials=123 (session 1 total)
n_trials = 123


def run_exp(entry, exit_p, tp_mult, sl_mult, vert_bars, feat_key, confidence_threshold=0.5):
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

        clf = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=42)
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
        "confidence_threshold": confidence_threshold,
        "n_entries_primary": n_entries,
        "n_signals_labeled": int(len(valid_idx)),
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
        f"feat={feat_key} thr={confidence_threshold}: "
        f"Sharpe={m30['sharpe']:.3f} DSR={m30['dsr']:.3f} "
        f"acc={np.mean(fold_accs):.3f} entries={n_entries} trades={n_trades}"
    )
    return result


# Round 1: Shorter Donchian params for more signals
# Session 1 signal counts: (10,5)=342, (15,10)=193, (20,10)=166, (30,20)=102 (at smaller dataset)
# With full 2013-2023 dataset: expect proportionally more
print("=" * 60)
print("Round 1: Shorter Donchian params sweep (more signals → better ML)")
print("=" * 60)

# Best barriers from S1: tp=3.0, sl=1.5, vert=30 (wide is better)
# Also test: tp=2.0, sl=1.5, vert=20 (second best)
# Donchian configs: (10,5), (15,10), (20,10) — plus (30,20) for reference
donchian_cfgs = [
    (10, 5),
    (15, 10),
    (20, 10),
    (20, 15),
    (30, 20),  # reference
]
barrier_cfgs = [
    (3.0, 1.5, 30),  # S1 best barriers
    (2.0, 1.5, 20),  # S1 second best
]
feat_keys_primary = ["3c", "3a", "5best"]  # S1 best + top variants

r1_results = []
for ep, xp in donchian_cfgs:
    for tp, sl, vb in barrier_cfgs:
        for fk in feat_keys_primary:
            r = run_exp(ep, xp, tp, sl, vb, fk, confidence_threshold=0.5)
            if r:
                r1_results.append(r)

if r1_results:
    best_r1 = max(r1_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR1 best: Sharpe={best_r1['sharpe_30']:.3f} DSR={best_r1['dsr_30']:.3f} "
        f"entry={best_r1['entry']}/exit={best_r1['exit']} "
        f"feat={best_r1['feat_key']} tp={best_r1['tp_mult']} sl={best_r1['sl_mult']} vb={best_r1['vert_bars']}"
    )

    # Summary by Donchian params
    print("\nBy Donchian params:")
    for ep, xp in donchian_cfgs:
        sub = [r for r in r1_results if r["entry"] == ep and r["exit"] == xp]
        if sub:
            b = max(sub, key=lambda x: x["sharpe_30"])
            print(
                f"  ({ep},{xp}): best Sharpe={b['sharpe_30']:.3f} DSR={b['dsr_30']:.3f} "
                f"signals={b['n_entries_primary']} feat={b['feat_key']}"
            )

tracker.log_sweep(
    "session2_round1_shorter_donchian",
    r1_results,
    summary_metrics={
        "sharpe": best_r1["sharpe_30"] if r1_results else 0,
        "dsr": best_r1["dsr_30"] if r1_results else 0,
    },
    tags=TAGS,
)
with open(out_dir / "session2_round1_results.json", "w") as f:
    json.dump(r1_results, f, indent=2, default=str)

print(f"\nRound 1 done. n_trials={n_trials}")
print("Done.")
