#!/usr/bin/env python3
"""Funding rate carry monitor.

Runs every 8h via systemd timer. Syncs OKX funding rates using the existing
data pipeline, computes carry viability indicators, and alerts on regime changes.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sparky.data.funding_rate import sync_funding_rates
from sparky.data.storage import DataStore

# Paths — funding rates use existing pipeline, prices use monitor-specific store
FUNDING_PARQUET = PROJECT_ROOT / "data" / "raw" / "funding_rates" / "btc_okx.parquet"
PRICE_PARQUET = PROJECT_ROOT / "data" / "monitors" / "btc_daily_close.parquet"
STATUS_FILE = PROJECT_ROOT / "coordination" / "data" / "funding_monitor.json"
ALERT_SCRIPT = PROJECT_ROOT / "bin" / "alert.sh"
COOLDOWN_HOURS = 48

REGIME_ORDER = {"INVERTED": 0, "COMPRESSED": 1, "NORMAL": 2, "BULL_CARRY": 3}
REGIME_ICONS = {
    "BULL_CARRY": "\U0001f7e2",
    "NORMAL": "\U0001f7e1",
    "COMPRESSED": "\U0001f7e0",
    "INVERTED": "\U0001f534",
}


def classify_regime(avg_30d: float) -> str:
    if avg_30d > 0.01:
        return "BULL_CARRY"
    if avg_30d > 0.003:
        return "NORMAL"
    if avg_30d > 0.0:
        return "COMPRESSED"
    return "INVERTED"


def sync_funding() -> pd.DataFrame:
    """Sync OKX funding rates using existing pipeline, return full history."""
    sync_funding_rates(asset="BTC", exchanges=["okx"])
    store = DataStore()
    if FUNDING_PARQUET.exists():
        df, _ = store.load(str(FUNDING_PARQUET))
        return df
    return pd.DataFrame()


def fetch_btc_prices() -> pd.DataFrame:
    """Fetch recent BTC daily close from OKX for momentum calculation."""
    store = DataStore()
    exchange = ccxt.okx({"enableRateLimit": True})

    last_ts = store.get_last_timestamp(str(PRICE_PARQUET))
    since_ms = (
        int((last_ts.timestamp() + 1) * 1000)
        if last_ts
        else int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    )

    try:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", since=since_ms, limit=365)
    except (ccxt.NetworkError, ccxt.ExchangeError):
        ohlcv = []

    if ohlcv:
        rows = [{"timestamp": pd.Timestamp(r[0], unit="ms", tz="UTC"), "close": float(r[4])} for r in ohlcv]
        new_df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        new_df = new_df[~new_df.index.duplicated(keep="last")]
        store.append(new_df, str(PRICE_PARQUET), metadata={"source": "okx", "timeframe": "1d"})

    if PRICE_PARQUET.exists():
        df, _ = store.load(str(PRICE_PARQUET))
        return df
    return pd.DataFrame()


def compute_indicators(funding_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    recent_90 = funding_df.tail(90)  # 30 days of 8h periods
    recent_21 = funding_df.tail(21)  # 7 days

    avg_30d = float(recent_90["funding_rate"].mean()) if len(recent_90) > 0 else 0.0
    avg_7d = float(recent_21["funding_rate"].mean()) if len(recent_21) > 0 else 0.0

    regime = classify_regime(avg_30d)
    ann_30d = avg_30d * 3 * 365 * 100  # per-8h * 3/day * 365 days * 100%
    ann_7d = avg_7d * 3 * 365 * 100

    btc_mom = 0.0
    if len(price_df) >= 30:
        btc_mom = float((price_df["close"].iloc[-1] / price_df["close"].iloc[-30] - 1) * 100)

    funding_momentum = "RISING" if avg_7d > avg_30d else "FALLING"
    activation = bool(btc_mom > 0 and funding_momentum == "RISING")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "avg_30d_per_8h": round(avg_30d, 8),
        "avg_7d_per_8h": round(avg_7d, 8),
        "annualized_30d_pct": round(ann_30d, 2),
        "annualized_7d_pct": round(ann_7d, 2),
        "btc_momentum_30d_pct": round(btc_mom, 1),
        "funding_momentum": funding_momentum,
        "activation_signal": activation,
        "data_freshness": funding_df.index[-1].isoformat() if len(funding_df) > 0 else None,
        "n_funding_records": len(funding_df),
    }


def load_previous_status() -> dict | None:
    if not STATUS_FILE.exists():
        return None
    try:
        return json.loads(STATUS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def check_alerts(indicators: dict, prev_status: dict | None) -> list[dict]:
    if prev_status is None:
        return []  # no transitions to detect on first run

    alerts = []
    now = datetime.now(timezone.utc)
    prev_alerts = prev_status.get("last_alert_times", {})

    def cooled_down(alert_type: str) -> bool:
        last = prev_alerts.get(alert_type)
        if not last:
            return True
        return (now - datetime.fromisoformat(last)).total_seconds() > COOLDOWN_HOURS * 3600

    prev_regime = prev_status.get("regime", "INVERTED")
    curr_regime = indicators["regime"]

    if (
        REGIME_ORDER.get(curr_regime, 0) > REGIME_ORDER.get(prev_regime, 0)
        and curr_regime in ("NORMAL", "BULL_CARRY")
        and cooled_down("REGIME_UPGRADE")
    ):
        alerts.append(
            {
                "type": "REGIME_UPGRADE",
                "severity": "INFO",
                "message": f"Funding regime: {prev_regime} -> {curr_regime} ({indicators['annualized_30d_pct']:.1f}% ann)",
            }
        )

    prev_ann = prev_status.get("annualized_30d_pct", 0) if prev_status else 0
    if prev_ann < 5.5 and indicators["annualized_30d_pct"] >= 5.5 and cooled_down("COMFORTABLE_CROSSING"):
        alerts.append(
            {
                "type": "COMFORTABLE_CROSSING",
                "severity": "INFO",
                "message": f"Funding carry above comfortable threshold: {indicators['annualized_30d_pct']:.1f}% ann (>5.5%)",
            }
        )

    prev_activation = prev_status.get("activation_signal", False) if prev_status else False
    if indicators["activation_signal"] and not prev_activation:
        if cooled_down("ACTIVATION_SIGNAL"):
            alerts.append(
                {
                    "type": "ACTIVATION_SIGNAL",
                    "severity": "WARN",
                    "message": (
                        f"CARRY ACTIVATION: BTC momentum +{indicators['btc_momentum_30d_pct']:.1f}% "
                        f"+ funding {indicators['funding_momentum']}. Consider scoping carry research."
                    ),
                }
            )

    if curr_regime == "BULL_CARRY" and prev_regime != "BULL_CARRY" and cooled_down("BULL_CARRY_ENTRY"):
        alerts.append(
            {
                "type": "BULL_CARRY_ENTRY",
                "severity": "WARN",
                "message": f"Entered BULL_CARRY regime: {indicators['annualized_30d_pct']:.1f}% ann",
            }
        )

    return alerts


def update_cooldowns(prev_status: dict | None, alerts: list[dict]) -> dict:
    cooldowns = prev_status.get("last_alert_times", {}).copy() if prev_status else {}
    now_iso = datetime.now(timezone.utc).isoformat()
    for alert in alerts:
        cooldowns[alert["type"]] = now_iso
    return cooldowns


def save_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2, default=str) + "\n")


def send_alerts(alerts: list[dict]) -> None:
    for alert in alerts:
        try:
            subprocess.run(
                [str(ALERT_SCRIPT), alert["severity"], alert["message"]],
                timeout=10,
                capture_output=True,
            )
        except Exception:
            pass


def print_summary(status: dict, cached: bool = False) -> None:
    ts = status.get("timestamp", "unknown")
    tag = " [cached]" if cached else ""
    regime = status.get("regime", "UNKNOWN")
    icon = REGIME_ICONS.get(regime, "\u26aa")
    ann_30d = status.get("annualized_30d_pct", 0)
    ann_7d = status.get("annualized_7d_pct", 0)
    avg_30d = status.get("avg_30d_per_8h", 0)
    avg_7d = status.get("avg_7d_per_8h", 0)
    btc_mom = status.get("btc_momentum_30d_pct", 0)
    fm = status.get("funding_momentum", "UNKNOWN")
    activation = status.get("activation_signal", False)
    n_records = status.get("n_funding_records", 0)
    freshness = status.get("data_freshness", "unknown")
    error = status.get("error")
    active_alerts = status.get("active_alerts", [])

    sign = "+" if ann_30d >= 0 else ""
    print(f"Funding Rate Monitor \u2014 {ts[:19]}{tag}")
    print("\u2501" * 45)
    print(f"Regime:     {icon} {regime} ({sign}{ann_30d:.1f}% ann)")
    print(f"30d avg:    {avg_30d:+.6f} per 8h ({sign}{ann_30d:.2f}% ann)")
    print(f"7d avg:     {avg_7d:+.6f} per 8h ({'+' if ann_7d >= 0 else ''}{ann_7d:.2f}% ann)")
    print(f"Momentum:   BTC 30d: {btc_mom:+.1f}% | Funding: {fm}")
    act_str = "\u2705 ACTIVE" if activation else "\u274c Inactive"
    print(f"Activation: {act_str}")
    print(f"Data:       {n_records:,} funding records, last: {freshness}")
    if error:
        print(f"\u26a0\ufe0f  Error: {error}")
    if active_alerts:
        print(f"\n\U0001f6a8 Alerts ({len(active_alerts)}):")
        for a in active_alerts:
            print(f"  \u26a0\ufe0f  {a}")


def check_mode() -> int:
    if not STATUS_FILE.exists():
        print("No status file found. Run without --check first.")
        return 1
    try:
        status = json.loads(STATUS_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: Status file unreadable: {e}")
        return 1
    print_summary(status, cached=True)
    return 0


def normal_mode() -> int:
    try:
        funding_df = sync_funding()
        price_df = fetch_btc_prices()
        indicators = compute_indicators(funding_df, price_df)
        prev_status = load_previous_status()
        alerts = check_alerts(indicators, prev_status)
        indicators["active_alerts"] = [a["type"] for a in alerts]
        indicators["last_alert_times"] = update_cooldowns(prev_status, alerts)
        indicators["error"] = None
        save_status(indicators)
        send_alerts(alerts)
        print_summary(indicators, cached=False)
        return 0
    except Exception as e:
        error_status = {"timestamp": datetime.now(timezone.utc).isoformat(), "error": str(e)}
        if STATUS_FILE.exists():
            try:
                prev = json.loads(STATUS_FILE.read_text())
                prev["error"] = str(e)
                prev["timestamp"] = error_status["timestamp"]
                save_status(prev)
            except (json.JSONDecodeError, OSError):
                save_status(error_status)
        else:
            save_status(error_status)
        print(f"ERROR: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Funding rate carry monitor")
    parser.add_argument("--check", action="store_true", help="Read cached status only")
    args = parser.parse_args()
    return check_mode() if args.check else normal_mode()


if __name__ == "__main__":
    sys.exit(main())
