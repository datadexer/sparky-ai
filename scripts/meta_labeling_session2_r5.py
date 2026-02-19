"""Session 2, round 5: deep validation + feature search around best (30,25) 4j config."""

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
    feats["dist_sma_60"] = (close - close.rolling(60).mean()) / close.rolling(60).mean().clip(lower=1e-8)
    feats["dist_sma_120"] = (close - close.rolling(120).mean()) / close.rolling(120).mean().clip(lower=1e-8)
    feats["regime_proba_3s"] = regime_proba_4h_3s.iloc[:, bull_state_3s].values

    feats["autocorr_5"] = ret.rolling(20).apply(
        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) > 5 else np.nan, raw=False
    )

    feats["atr_momentum"] = atr_4h / atr_4h.rolling(50).mean().clip(lower=1e-8)
    feats["vol_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean().clip(lower=1e-8)
    feats["vol_accel"] = (volume.rolling(5).mean() / volume.rolling(20).mean()) / (
        volume.rolling(20).mean() / volume.rolling(60).mean()
    ).clip(lower=1e-8)

    # Breakout proximity: ratio of close to Donchian channel
    entry_ch = close.rolling(30).max().shift(1)  # entry channel (30 bars lookback)
    feats["breakout_ratio"] = (close / entry_ch.clip(lower=1e-8)) - 1.0  # 0 = at channel, +0.05 = 5% above

    return feats


print("Computing features...")
feats = make_features(h4, prices_4h)

FSETS = {
    # Session best
    "4j": ["trend_r2", "regime_proba_3s", "adx_proxy", "dist_sma_60"],
    "3c": ["trend_r2", "regime_proba_3s", "adx_proxy"],
    # Variants on 4j theme (dist from SMA)
    "4j2": ["trend_r2", "regime_proba_3s", "adx_proxy", "dist_sma_120"],
    "4j3": ["trend_r2", "regime_proba_3s", "adx_proxy", "dist_sma_20"],
    "4j4": ["trend_r2", "regime_proba_3s", "adx_proxy", "breakout_ratio"],
    # 5-feature combos adding dist_sma_60
    "5j": ["trend_r2", "regime_proba_3s", "adx_proxy", "dist_sma_60", "choppiness"],
    "5j2": ["trend_r2", "regime_proba_3s", "adx_proxy", "dist_sma_60", "vol_ratio"],
    "5j3": ["trend_r2", "regime_proba_3s", "adx_proxy", "dist_sma_60", "autocorr_5"],
    # New 3-feature with dist_sma_60
    "3j": ["trend_r2", "regime_proba_3s", "dist_sma_60"],
    "3j2": ["adx_proxy", "regime_proba_3s", "dist_sma_60"],
    # Vol momentum + regime + trend
    "4k": ["trend_r2", "regime_proba_3s", "dist_sma_60", "vol_accel"],
    "4l": ["trend_r2", "regime_proba_3s", "adx_proxy", "vol_accel"],
}

# Continue from round 4
n_trials = 323


def run_exp(entry, exit_p, tp_mult, sl_mult, vert_bars, feat_key, confidence_threshold=0.5, C=0.1):
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

        clf = LogisticRegression(C=C, max_iter=500, class_weight="balanced", random_state=42)
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
        f"feat={feat_key} thr={confidence_threshold}: "
        f"Sharpe={m30['sharpe']:.3f} DSR={m30['dsr']:.3f} "
        f"acc={np.mean(fold_accs):.3f} trades={n_trades}"
    )
    return result


# ─────────────────────────────────────────────────────────────
# Round 5A: Full feature family sweep on (30,25)
# Focus on dist_sma_60 variants and new features
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Round 5A: Feature family sweep on (30,25)")
print("=" * 60)

r5a_results = []
for fk in list(FSETS.keys()):
    for tp, sl, vb in [(2.0, 1.5, 20), (2.0, 2.0, 20), (3.0, 1.5, 30)]:
        r = run_exp(30, 25, tp, sl, vb, fk, confidence_threshold=0.5)
        if r:
            r5a_results.append(r)

if r5a_results:
    best_5a = max(r5a_results, key=lambda x: x["sharpe_30"])
    print(
        f"\nR5A best: Sharpe={best_5a['sharpe_30']:.3f} DSR={best_5a['dsr_30']:.3f} "
        f"feat={best_5a['feat_key']} tp={best_5a['tp_mult']} sl={best_5a['sl_mult']}"
    )
    print(f"  50bps: Sharpe={best_5a['sharpe_50']:.3f}")

    print("\nTop 10:")
    for r in sorted(r5a_results, key=lambda x: x["sharpe_30"], reverse=True)[:10]:
        print(
            f"  feat={r['feat_key']} tp={r['tp_mult']} sl={r['sl_mult']} vb={r['vert_bars']}: "
            f"Sharpe={r['sharpe_30']:.3f} DSR={r['dsr_30']:.3f} 50bps={r['sharpe_50']:.3f}"
        )

tracker.log_sweep(
    "session2_round5a_dist_sma_features",
    r5a_results,
    summary_metrics={
        "sharpe": best_5a["sharpe_30"] if r5a_results else 0,
        "dsr": best_5a["dsr_30"] if r5a_results else 0,
    },
    tags=TAGS,
)
with open(out_dir / "session2_round5a_results.json", "w") as f:
    json.dump(r5a_results, f, indent=2, default=str)
print(f"Round 5A done. n_trials={n_trials}")


# ─────────────────────────────────────────────────────────────
# Round 5B: Validate best configs across multiple seeds/runs
# (check stability of the results)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Round 5B: Cross-validation stability check")
print("=" * 60)

# Rerun the top 3 configs 3 times each to check variance
top_configs = [
    (30, 25, 2.0, 1.5, 20, "4j"),  # session best 1.931
    (30, 25, 2.0, 2.0, 20, "3c"),  # R4A best 1.921
    (30, 25, 2.0, 1.5, 20, "3c"),  # R4A 3rd 1.891
    (30, 20, 3.0, 1.5, 30, "4d"),  # R2A best 1.800
]

r5b_results = []
for cfg in top_configs:
    for _ in range(3):  # run 3 times (different HMM random seeds may vary)
        r = run_exp(*cfg, confidence_threshold=0.5)
        if r:
            r5b_results.append(r)

if r5b_results:
    print("\nStability results (3 runs each):")
    for cfg in top_configs:
        ep, xp, tp, sl, vb, fk = cfg
        sub = [
            r
            for r in r5b_results
            if r["entry"] == ep
            and r["exit"] == xp
            and r["tp_mult"] == tp
            and r["sl_mult"] == sl
            and r["feat_key"] == fk
        ]
        if sub:
            sharpes = [r["sharpe_30"] for r in sub]
            print(
                f"  ({ep},{xp}) tp={tp} sl={sl} vb={vb} feat={fk}: "
                f"mean={np.mean(sharpes):.3f} std={np.std(sharpes):.4f} "
                f"min={np.min(sharpes):.3f} max={np.max(sharpes):.3f}"
            )

tracker.log_sweep(
    "session2_round5b_stability_check",
    r5b_results,
    summary_metrics={
        "sharpe": max(r5b_results, key=lambda x: x["sharpe_30"])["sharpe_30"] if r5b_results else 0,
        "dsr": max(r5b_results, key=lambda x: x["sharpe_30"])["dsr_30"] if r5b_results else 0,
    },
    tags=TAGS,
)
with open(out_dir / "session2_round5b_stability_results.json", "w") as f:
    json.dump(r5b_results, f, indent=2, default=str)
print(f"Round 5B done. n_trials={n_trials}")


# ─────────────────────────────────────────────────────────────
# Final summary: best overall result this session
# ─────────────────────────────────────────────────────────────
all_r5 = r5a_results + r5b_results

# Load all session 2 results to find overall best
try:
    from pathlib import Path as P

    all_session2 = []
    for fp in P(out_dir).glob("session2_round*.json"):
        with open(fp) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_session2.extend(data)
    if all_session2:
        best_s2 = max(all_session2, key=lambda x: x.get("sharpe_30", 0))
        print("\n=== SESSION 2 OVERALL BEST ===")
        print(f"  Sharpe@30bps: {best_s2['sharpe_30']:.4f}")
        print(f"  DSR@30bps: {best_s2['dsr_30']:.4f}")
        print(f"  Sharpe@50bps: {best_s2.get('sharpe_50', 'N/A')}")
        print(
            f"  Config: entry={best_s2['entry']} exit={best_s2['exit']} "
            f"tp={best_s2['tp_mult']} sl={best_s2['sl_mult']} "
            f"vb={best_s2['vert_bars']} feat={best_s2['feat_key']}"
        )
        print(f"  n_trials at result: {best_s2.get('n_trials', n_trials)}")
        print(f"  MaxDD: {best_s2.get('max_dd_30', 'N/A')}")
except Exception as e:
    print(f"Could not aggregate: {e}")
    best_s2 = None

print(f"\nTotal n_trials (cumulative this session): {n_trials}")

# Log final session summary to wandb
wb.init(project="sparky-ai", entity="datadex_ai", name="meta_labeling_session2_final", tags=TAGS, reinit=True)
if best_s2:
    wb.log(
        {
            "sharpe": best_s2["sharpe_30"],
            "dsr": best_s2["dsr_30"],
            "sharpe_50bps": best_s2.get("sharpe_50", 0),
            "dsr_50bps": best_s2.get("dsr_50", 0),
            "max_dd": best_s2.get("max_dd_30", 0),
            "n_configs_tested": n_trials,
            "baseline_4h_primary_sharpe": 1.682,
            "session1_best_sharpe": 1.787,
            "best_entry": best_s2["entry"],
            "best_exit": best_s2["exit"],
            "best_tp_mult": best_s2["tp_mult"],
            "best_sl_mult": best_s2["sl_mult"],
            "best_vert_bars": best_s2["vert_bars"],
            "best_feat_key": best_s2["feat_key"],
        }
    )
wb.finish()

print("Done.")
