"""
Sparky AI CEO Dashboard — Streamlit app for monitoring workflow progress,
telemetry sessions, W&B runs, alerts, and git commits.
"""

import glob
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.stats import norm

from sparky.tracking.metrics import expected_max_sharpe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/akamath/sparky-ai")
TELEMETRY_DIR = PROJECT_ROOT / "logs" / "telemetry"
STATE_DIR = PROJECT_ROOT / "workflows" / "state"
ALERTS_LOG = PROJECT_ROOT / "logs" / "alerts.log"
GITHUB_URL = "https://github.com/datadexer/sparky-ai"
WANDB_URL = "https://wandb.ai/datadex_ai/sparky-ai"
# Thresholds keyed by wandb step tag (not workflow step name)
DONE_THRESHOLDS = {
    "feature_analysis": 0,  # done_when checks file, not wandb
    "sweep": 20,
    "regime": 8,
    "ensemble": 30,
    "novel": 15,
}
STATUS_EMOJI = {
    "completed": "green",
    "running": "orange",
    "pending": "gray",
    "failed": "red",
    "skipped": "blue",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sparky AI", layout="wide", page_icon="⚡")

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30)
def load_telemetry() -> list[dict]:
    """Load all telemetry JSON files sorted by session_id (ascending)."""
    sessions = []
    pattern = str(TELEMETRY_DIR / "*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                data = json.load(f)
            sessions.append(data)
        except Exception:
            pass
    return sessions


@st.cache_data(ttl=30)
def load_workflow_state() -> dict:
    """Load contract-004 workflow state JSON."""
    state_file = STATE_DIR / "contract-004.json"
    try:
        with open(state_file) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=30)
def load_alerts() -> list[dict]:
    """Parse alerts.log into structured records."""
    alerts = []
    try:
        with open(ALERTS_LOG) as f:
            lines = f.readlines()
        pattern = re.compile(r"\[(?P<ts>[^\]]+)\]\s+\[(?P<level>[^\]]+)\]\s+(?P<msg>.+)")
        for line in lines:
            m = pattern.match(line.strip())
            if m:
                alerts.append(
                    {
                        "timestamp": m.group("ts"),
                        "level": m.group("level"),
                        "message": m.group("msg"),
                    }
                )
    except Exception:
        pass
    return alerts


@st.cache_data(ttl=300)
def load_wandb_session_runs() -> dict[str, str]:
    """Load W&B session runs (job_type=session, group=ceo_sessions) via GraphQL.

    Returns a dict mapping session_id -> wandb run URL for fast lookup.
    """
    try:
        import requests as _req

        import wandb

        api = wandb.Api(timeout=30)
        api_key = api.api_key

        query_str = """
        query Runs($project: String!, $entity: String!, $filters: JSONString,
                    $order: String, $first: Int) {
          project(name: $project, entityName: $entity) {
            runs(filters: $filters, order: $order, first: $first) {
              edges {
                node {
                  id
                  name
                  displayName
                  createdAt
                  config
                  jobType
                  group
                }
              }
            }
          }
        }
        """

        resp = _req.post(
            "https://api.wandb.ai/graphql",
            json={
                "query": query_str,
                "variables": {
                    "project": "sparky-ai",
                    "entity": "datadex_ai",
                    "filters": json.dumps({"group": "ceo_sessions", "jobType": "session"}),
                    "order": "-created_at",
                    "first": 500,
                },
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        edges = resp.json()["data"]["project"]["runs"]["edges"]

        session_map: dict[str, str] = {}
        for edge in edges:
            node = edge["node"]
            run_id = node.get("name") or node.get("id") or ""
            cfg_raw = node.get("config") or "{}"
            try:
                cfg = json.loads(cfg_raw)
            except (json.JSONDecodeError, TypeError):
                cfg = {}
            # wandb config values are wrapped: {"session_id": {"value": "...", "desc": null}}
            sid_entry = cfg.get("session_id", {})
            sid = sid_entry.get("value") if isinstance(sid_entry, dict) else sid_entry
            if sid and run_id:
                session_map[str(sid)] = f"{WANDB_URL}/runs/{run_id}"
        return session_map
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_wandb_runs() -> list[dict]:
    """Load W&B runs tagged contract_004 via direct GraphQL (fast bulk fetch).

    Uses a single GraphQL request that returns all runs with summaryMetrics
    included. This avoids the wandb SDK's lazy per-run API calls for .summary
    and .config, which take ~40s for 200 runs. Direct GraphQL: ~0.5s.
    """
    try:
        import requests as _req

        import wandb

        api = wandb.Api(timeout=30)
        api_key = api.api_key

        query_str = """
        query Runs($project: String!, $entity: String!, $filters: JSONString,
                    $order: String, $first: Int) {
          project(name: $project, entityName: $entity) {
            runs(filters: $filters, order: $order, first: $first) {
              edges {
                node {
                  id
                  name
                  displayName
                  state
                  group
                  jobType
                  tags
                  commit
                  createdAt
                  summaryMetrics
                }
              }
            }
          }
        }
        """

        resp = _req.post(
            "https://api.wandb.ai/graphql",
            json={
                "query": query_str,
                "variables": {
                    "project": "sparky-ai",
                    "entity": "datadex_ai",
                    "filters": json.dumps({"tags": {"$in": ["contract_004"]}}),
                    "order": "-created_at",
                    "first": 500,
                },
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        edges = resp.json()["data"]["project"]["runs"]["edges"]

        records = []
        for edge in edges:
            node = edge["node"]
            run_name = node.get("displayName") or node.get("name") or ""
            run_id = node.get("name") or node.get("id") or ""

            # Sharpe: prefer summaryMetrics, fall back to name regex
            summary = {}
            sm = node.get("summaryMetrics")
            if sm:
                try:
                    summary = json.loads(sm)
                except (json.JSONDecodeError, TypeError):
                    pass
            sharpe = summary.get("sharpe_ratio") or summary.get("best_sharpe") or summary.get("mean_sharpe")
            if sharpe is None:
                m = re.search(r"_S([\d.]+)", run_name)
                if m:
                    try:
                        sharpe = float(m.group(1))
                    except ValueError:
                        pass

            # Step tag from tags list
            step_tag = "other"
            for tag in node.get("tags") or []:
                if tag in ("sweep", "regime", "ensemble", "novel", "feature_analysis"):
                    step_tag = tag
                    break

            records.append(
                {
                    "id": run_id,
                    "name": run_name,
                    "step_tag": step_tag,
                    "tags": node.get("tags") or [],
                    "group": node.get("group") or "",
                    "created_at": node.get("createdAt") or "",
                    "sharpe": sharpe,
                    "git_hash": node.get("commit") or None,
                    "url": f"{WANDB_URL}/runs/{run_id}",
                    # Extended significance and risk metrics
                    "dsr": summary.get("dsr"),
                    "psr": summary.get("psr"),
                    "sortino": summary.get("sortino"),
                    "max_drawdown": summary.get("max_drawdown"),
                    "calmar": summary.get("calmar"),
                    "worst_year_sharpe": summary.get("worst_year_sharpe"),
                    "profit_factor": summary.get("profit_factor"),
                }
            )
        return records
    except Exception:
        return []


def get_service_status() -> str:
    """Return systemd status for sparky-ceo service."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "sparky-ceo"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def get_git_log(n: int = 10) -> list[dict]:
    """Return recent git commits as list of dicts."""
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "log", f"-{n}", "--oneline", "--no-decorate"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        commits = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2:
                commits.append({"hash": parts[0], "message": parts[1]})
        return commits
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_dt(raw: str | None) -> str:
    if not raw:
        return "—"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        return raw[:16] if raw else "—"


def _service_color(status: str) -> str:
    colors = {"active": "green", "inactive": "gray", "failed": "red"}
    return colors.get(status, "orange")


def _status_badge(status: str) -> str:
    badges = {
        "completed": "✅ completed",
        "running": "🔄 running",
        "pending": "⏳ pending",
        "failed": "❌ failed",
        "skipped": "⏭ skipped",
    }
    return badges.get(status, status)


def _level_color(level: str) -> str:
    return {"ERROR": "red", "WARN": "orange", "INFO": "blue"}.get(level.upper(), "gray")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main():
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Last updated: {_now_str()}")
        st.markdown("---")
        st.markdown(f"[W&B Project]({WANDB_URL})")
        st.markdown(f"[GitHub Repo]({GITHUB_URL})")

    # Load all data
    telemetry = load_telemetry()
    state = load_workflow_state()
    alerts = load_alerts()
    wandb_runs = load_wandb_runs()
    session_wandb_map = load_wandb_session_runs()

    # -----------------------------------------------------------------------
    # 1. Header
    # -----------------------------------------------------------------------
    st.title("⚡ Sparky AI — CEO Dashboard")

    svc_status = get_service_status()
    steps_info = state.get("steps", {})
    step_list = list(steps_info.keys())
    current_idx = state.get("current_step_index", 0)
    n_steps = len(step_list)
    current_step_name = step_list[current_idx - 1] if 0 < current_idx <= n_steps else "—"

    # Best Sharpe from telemetry done_when or wandb
    best_sharpe = None
    if wandb_runs:
        sharpes = [r["sharpe"] for r in wandb_runs if r["sharpe"] is not None]
        if sharpes:
            best_sharpe = max(sharpes)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = _service_color(svc_status)
        st.metric("CEO Service", svc_status.upper())
        st.markdown(
            f"<span style='color:{color};font-size:12px'>● sparky-ceo.service</span>",
            unsafe_allow_html=True,
        )
    with col2:
        step_label = f"{current_idx}/{n_steps}: {current_step_name}" if state else "—"
        st.metric("Current Step", step_label)
    with col3:
        budget = state.get("budget", {})
        hours_used = budget.get("hours_used", 0.0)
        max_hours = budget.get("max_hours", 12.0)
        st.metric("Budget Used", f"{hours_used:.2f} / {max_hours:.1f} h")
    with col4:
        sharpe_str = f"{best_sharpe:.3f}" if best_sharpe is not None else "N/A"
        n_total_runs = len(wandb_runs)
        T_hourly = 8760  # 5 years of hourly data

        if best_sharpe is not None and n_total_runs > 0:
            sr0 = expected_max_sharpe(n_total_runs, T=T_hourly)
            se = 1.0 / (T_hourly**0.5)
            approx_dsr = float(norm.cdf((best_sharpe - sr0) / se)) if se > 0 else 0.0
            if approx_dsr > 0.95:
                verdict = "SIGNIFICANT"
                verdict_color = "green"
            elif approx_dsr >= 0.80:
                verdict = "MARGINAL"
                verdict_color = "orange"
            else:
                verdict = "LIKELY FLUKE"
                verdict_color = "red"
            st.metric("Best Sharpe (W&B)", sharpe_str)
            st.markdown(
                f"<span style='font-size:12px'>DSR ≈ {approx_dsr:.3f} | "
                f"<span style='color:{verdict_color};font-weight:bold'>{verdict}</span></span>",
                unsafe_allow_html=True,
            )
        else:
            st.metric("Best Sharpe (W&B)", sharpe_str)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Statistical Health panel
    # -----------------------------------------------------------------------
    st.subheader("Statistical Health")

    n_total_runs = len(wandb_runs)
    T_hourly = 8760  # 5 years of hourly data

    if n_total_runs > 0 and best_sharpe is not None:
        ems = expected_max_sharpe(n_total_runs, T=T_hourly)
        se = 1.0 / (T_hourly**0.5)
        approx_dsr = float(norm.cdf((best_sharpe - ems) / se)) if se > 0 else 0.0

        # Minimum track record: observations still needed
        # Using PSR-style formula: need T such that norm.cdf((sr - 0) / (1/sqrt(T))) > 0.95
        # => sr * sqrt(T) > 1.645 => T > (1.645 / sr)^2
        if best_sharpe > 0:
            import math as _math

            z_95 = norm.ppf(0.95)
            T_needed = (z_95 / best_sharpe) ** 2
            extra_obs = max(0, int(_math.ceil(T_needed - T_hourly)))
            track_str = f"{extra_obs:,} more observations needed" if extra_obs > 0 else "Sufficient track record"
        else:
            track_str = "Negative Sharpe — no track record is sufficient"

        sh1, sh2, sh3, sh4, sh5 = st.columns(5)
        sh1.metric("Total Trials", f"{n_total_runs}")
        sh2.metric("Expected Max SR (Noise)", f"{ems:.3f}")
        sh3.metric("Observed Best SR", f"{best_sharpe:.3f}")
        sh4.metric("Approx DSR", f"{approx_dsr:.3f}")
        sh5.metric("Min Track Record", track_str)
    else:
        st.info("No W&B runs available — cannot compute statistical health metrics.")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 2. Workflow Progress
    # -----------------------------------------------------------------------
    st.subheader("Workflow Progress")

    if not state:
        st.warning("No workflow state found at workflows/state/contract-004.json")
    else:
        # Count wandb runs per step tag
        runs_per_step: dict[str, int] = {}
        for r in wandb_runs:
            tag = r["step_tag"]
            runs_per_step[tag] = runs_per_step.get(tag, 0) + 1

        # Step tag mapping (workflow step name -> wandb tag)
        step_to_tag = {
            "feature_analysis": "feature_analysis",
            "two_stage_sweep": "sweep",
            "regime_aware_hybrid": "regime",
            "ensemble": "ensemble",
            "novel_exploration": "novel",
        }

        rows = []
        for step_name, step_data in steps_info.items():
            tag = step_to_tag.get(step_name, step_name)
            threshold = DONE_THRESHOLDS.get(tag, 1)
            run_count = runs_per_step.get(tag, 0)
            # feature_analysis uses file check, not wandb
            uses_file_check = tag == "feature_analysis"
            if uses_file_check:
                file_exists = (PROJECT_ROOT / "results" / "feature_importance.json").exists()
                progress_val = 1.0 if file_exists else 0.0
            else:
                progress_val = min(run_count / threshold, 1.0) if threshold > 0 else 1.0
            rows.append(
                {
                    "step_name": step_name,
                    "status": step_data.get("status", "unknown"),
                    "attempts": step_data.get("attempts", 0),
                    "completed_at": _fmt_dt(step_data.get("completed_at")),
                    "last_attempt_at": _fmt_dt(step_data.get("last_attempt_at")),
                    "runs": run_count,
                    "threshold": threshold,
                    "progress": progress_val,
                    "uses_file_check": uses_file_check,
                }
            )

        for row in rows:
            c1, c2, c3, c4, c5 = st.columns([3, 2, 1, 2, 4])
            with c1:
                st.markdown(f"**{row['step_name']}**")
            with c2:
                st.markdown(_status_badge(row["status"]))
            with c3:
                st.markdown(f"×{row['attempts']}")
            with c4:
                st.markdown(f"done: {row['completed_at']}")
            with c5:
                if row["uses_file_check"]:
                    label = "file check (feature_importance.json)"
                    st.progress(row["progress"], text=label)
                else:
                    label = f"{row['runs']} / {row['threshold']} W&B runs"
                    st.progress(row["progress"], text=label)

        st.markdown("---")
        # Budget row
        budget = state.get("budget", {})
        # Prefer CLI ground-truth cost_usd when available, fall back to estimated
        total_cost_telem = sum(s.get("cost_usd") or s.get("estimated_cost_usd", 0) for s in telemetry)
        has_cli_cost = any(s.get("cost_usd") for s in telemetry)
        cost_label = "Cost (telemetry, CLI)" if has_cli_cost else "Cost (telemetry, est.)"
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Hours Used (budget)", f"{budget.get('hours_used', 0):.3f} h")
        b2.metric("Max Hours", f"{budget.get('max_hours', 12):.1f} h")
        b3.metric("Cost (budget)", f"${budget.get('estimated_cost_usd', 0):.4f}")
        b4.metric(cost_label, f"${total_cost_telem:.4f}")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 3. Session Timeline
    # -----------------------------------------------------------------------
    st.subheader("Session Timeline")

    if not telemetry:
        st.info("No telemetry sessions found in logs/telemetry/")
    else:
        rows_telem = []
        for s in reversed(telemetry):  # most recent first
            flags = s.get("behavioral_flags", [])
            # Prefer CLI ground-truth cost_usd; fall back to estimated
            cost = s.get("cost_usd") or s.get("estimated_cost_usd", 0)
            sid = s.get("session_id", "")
            wandb_url = session_wandb_map.get(sid, "")
            wandb_link = f"[W&B]({wandb_url})" if wandb_url else ""
            rows_telem.append(
                {
                    "Session": sid,
                    "Step": s.get("step", ""),
                    "Attempt": s.get("attempt", 0),
                    "Started": _fmt_dt(s.get("started_at")),
                    "Duration (m)": round(s.get("duration_minutes", 0), 2),
                    "Tools": s.get("tool_calls", 0),
                    "Cost ($)": round(cost, 4),
                    "CLI Cost ($)": round(s.get("cost_usd", 0), 4),
                    "Cache Read (K)": round(s.get("tokens_cache_read", 0) / 1000, 1),
                    "Turns": s.get("num_turns", 0),
                    "Exit": s.get("exit_reason", ""),
                    "Flags": ", ".join(flags) if flags else "",
                    "W&B": wandb_link,
                }
            )

        df_telem = pd.DataFrame(rows_telem)

        def _color_exit(val):
            if val == "completed":
                return "color: green"
            elif val in ("rate_limited", "timeout"):
                return "color: orange"
            elif val in ("failed", "error"):
                return "color: red"
            return ""

        def _color_flags(val):
            if val:
                return "color: orange"
            return ""

        styled = df_telem.style.applymap(_color_exit, subset=["Exit"]).applymap(_color_flags, subset=["Flags"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 4. Rate Limits
    # -----------------------------------------------------------------------
    st.subheader("Rate Limit Events")

    rate_alerts = [a for a in alerts if "rate limit" in a["message"].lower() or "rate_limit" in a["message"].lower()]
    if not rate_alerts:
        st.success("No rate limit events recorded.")
    else:
        # Group consecutive events on same step
        groups = []
        current_group = None
        for alert in rate_alerts:
            step_m = re.search(r"step '([^']+)'", alert["message"])
            step = step_m.group(1) if step_m else "unknown"
            if current_group and current_group["step"] == step:
                current_group["retries"] += 1
                current_group["last_ts"] = alert["timestamp"]
                if "persist" in alert["message"].lower() or "failed" in alert["level"].lower():
                    current_group["recovered"] = False
            else:
                if current_group:
                    groups.append(current_group)
                current_group = {
                    "step": step,
                    "first_ts": alert["timestamp"],
                    "last_ts": alert["timestamp"],
                    "retries": 1,
                    "recovered": True,
                }
        if current_group:
            groups.append(current_group)

        rl_rows = []
        for g in groups:
            rl_rows.append(
                {
                    "Step": g["step"],
                    "First Event": g["first_ts"],
                    "Last Event": g["last_ts"],
                    "Retries": g["retries"],
                    "Recovered": "Yes" if g["recovered"] else "No",
                }
            )

        st.dataframe(pd.DataFrame(rl_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 5. W&B Runs
    # -----------------------------------------------------------------------
    st.subheader("W&B Experiment Runs")

    if not wandb_runs:
        st.info(f"No W&B runs loaded (offline or no runs tagged contract_004). [Open W&B project]({WANDB_URL})")
    else:
        # Metric cards per step tag
        tag_counts: dict[str, int] = {}
        for r in wandb_runs:
            tag = r["step_tag"]
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        display_tags = {t: c for t, c in tag_counts.items() if t != "feature_analysis"}
        cols = st.columns(len(display_tags) if display_tags else 1)
        for i, (tag, count) in enumerate(sorted(display_tags.items())):
            threshold = DONE_THRESHOLDS.get(tag, 1)
            cols[i].metric(f"{tag}", f"{count} / {threshold} runs")

        st.markdown("")

        # --- 2d. Per-step summary table ---
        st.markdown("**Step Summary**")
        step_summary_rows = []
        step_tags_seen = sorted(set(r["step_tag"] for r in wandb_runs if r["step_tag"] != "feature_analysis"))
        for tag in step_tags_seen:
            tag_runs = [r for r in wandb_runs if r["step_tag"] == tag]
            run_count = len(tag_runs)
            sharpes_tag = [r["sharpe"] for r in tag_runs if r["sharpe"] is not None]
            dsrs_tag = [r["dsr"] for r in tag_runs if r.get("dsr") is not None]
            sortinos_tag = [r["sortino"] for r in tag_runs if r.get("sortino") is not None]
            dds_tag = [r["max_drawdown"] for r in tag_runs if r.get("max_drawdown") is not None]

            best_sharpe_tag = f"{max(sharpes_tag):.3f}" if sharpes_tag else "—"
            best_dsr_tag = f"{max(dsrs_tag):.3f}" if dsrs_tag else "—"
            best_sortino_tag = f"{max(sortinos_tag):.3f}" if sortinos_tag else "—"
            worst_dd_tag = f"{min(dds_tag):.3f}" if dds_tag else "—"

            step_summary_rows.append(
                {
                    "Step": tag,
                    "Runs": run_count,
                    "Best Sharpe": best_sharpe_tag,
                    "Best DSR": best_dsr_tag,
                    "Best Sortino": best_sortino_tag,
                    "Worst DD": worst_dd_tag,
                }
            )

        if step_summary_rows:
            st.dataframe(pd.DataFrame(step_summary_rows), use_container_width=True, hide_index=True)

        st.markdown("")

        # --- 2b. Top Strategies table (enhanced) ---
        st.markdown("**Top Strategies**")

        sorted_runs = sorted(
            [r for r in wandb_runs if r["sharpe"] is not None],
            key=lambda r: r["sharpe"],
            reverse=True,
        )[:10]

        if not sorted_runs:
            # Show all runs without Sharpe filter
            sorted_runs = wandb_runs[:10]

        wb_rows = []
        for r in sorted_runs:
            sharpe_str = f"{r['sharpe']:.4f}" if r["sharpe"] is not None else "N/A"
            dsr_val = r.get("dsr")
            sortino_val = r.get("sortino")
            dd_val = r.get("max_drawdown")

            dsr_str = f"{dsr_val:.3f}" if dsr_val is not None else "N/A"
            sortino_str = f"{sortino_val:.3f}" if sortino_val is not None else "N/A"
            dd_str = f"{dd_val:.3f}" if dd_val is not None else "N/A"

            # Verdict based on DSR (direct from wandb) or "N/A" if unavailable
            if dsr_val is not None:
                if dsr_val > 0.95:
                    verdict = "Significant"
                elif dsr_val >= 0.80:
                    verdict = "Marginal"
                else:
                    verdict = "Likely fluke"
            else:
                verdict = "N/A"

            wb_rows.append(
                {
                    "Run": f"[{r['name']}]({r['url']})",
                    "Step": r["step_tag"],
                    "Sharpe": sharpe_str,
                    "DSR": dsr_str,
                    "Sortino": sortino_str,
                    "Max DD": dd_str,
                    "Verdict": verdict,
                }
            )

        if wb_rows:
            df_wb = pd.DataFrame(wb_rows)
            st.dataframe(df_wb, use_container_width=True, hide_index=True)
        else:
            st.info("No runs with Sharpe data found.")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 6. Git & Alerts
    # -----------------------------------------------------------------------
    st.subheader("Git & Alerts")
    col_git, col_alerts = st.columns(2)

    with col_git:
        st.markdown("**Recent Commits**")
        commits = get_git_log(10)
        if not commits:
            st.info("Could not load git log.")
        else:
            git_rows = []
            for c in commits:
                link = f"[`{c['hash']}`]({GITHUB_URL}/commit/{c['hash']})"
                git_rows.append({"Hash": link, "Message": c["message"]})
            st.dataframe(pd.DataFrame(git_rows), use_container_width=True, hide_index=True)

    with col_alerts:
        st.markdown("**Recent Alerts (last 20)**")
        recent_alerts = alerts[-20:] if alerts else []
        if not recent_alerts:
            st.info("No alerts found in logs/alerts.log")
        else:
            alert_rows = []
            for a in reversed(recent_alerts):
                alert_rows.append(
                    {
                        "Time": a["timestamp"],
                        "Level": a["level"],
                        "Message": a["message"],
                    }
                )

            def _color_level(val):
                colors = {"ERROR": "color: red", "WARN": "color: orange", "INFO": "color: steelblue"}
                return colors.get(val.upper(), "")

            df_alerts = pd.DataFrame(alert_rows)
            styled_alerts = df_alerts.style.applymap(_color_level, subset=["Level"])
            st.dataframe(styled_alerts, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    st.caption(
        f"Sparky AI CEO Dashboard | Auto-refresh every 30s via sidebar | [W&B]({WANDB_URL}) | [GitHub]({GITHUB_URL})"
    )


if __name__ == "__main__":
    main()
