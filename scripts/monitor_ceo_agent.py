#!/usr/bin/env python3
"""Real-time CEO agent monitoring dashboard.

Reads agent activity logs and MLflow experiments to show:
- Current task status
- Recent experiments and results
- Time since last activity
- Progress against Phase 3 checklist
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_activity_log(log_path: Path) -> list[dict]:
    """Parse JSONL activity log into list of events."""
    if not log_path.exists():
        return []

    events = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def format_timedelta(td):
    """Format timedelta as human-readable string."""
    seconds = int(td.total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def main():
    print("=" * 80)
    print("CEO AGENT MONITORING DASHBOARD")
    print("=" * 80)

    # 1. Check for activity logs
    log_dir = Path("logs/agent_activity")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = log_dir / f"ceo_{today}.jsonl"

    if not log_path.exists():
        print(f"\n❌ No activity log found: {log_path}")
        print("   CEO agent may not be running or hasn't initialized logging.")
        return

    events = parse_activity_log(log_path)

    if not events:
        print(f"\n⚠️  Activity log is empty: {log_path}")
        return

    print(f"\n✅ Found {len(events)} activity log entries")

    # 2. Recent activity timeline
    now = datetime.now(timezone.utc)
    last_event = events[-1]
    last_timestamp = datetime.fromisoformat(last_event["timestamp"])
    idle_time = now - last_timestamp

    print(f"\n{'─' * 80}")
    print("RECENT ACTIVITY")
    print(f"{'─' * 80}")
    print(f"Last activity: {format_timedelta(idle_time)} ago")
    print(f"Last action: {last_event['action_type']}")
    if "task" in last_event:
        print(f"Last task: {last_event['task']}")
    if "description" in last_event:
        print(f"Description: {last_event['description']}")

    # Show last 5 events
    print(f"\n{'─' * 80}")
    print("LAST 5 EVENTS")
    print(f"{'─' * 80}")
    for event in events[-5:]:
        ts = datetime.fromisoformat(event["timestamp"])
        age = format_timedelta(now - ts)
        action = event["action_type"].replace("_", " ").title()
        task = event.get("task", "N/A")
        print(f"{age:>8} ago | {action:20} | {task}")

    # 3. Experiment summary
    experiments = [e for e in events if e["action_type"] == "experiment_completed"]

    print(f"\n{'─' * 80}")
    print(f"EXPERIMENTS COMPLETED: {len(experiments)}")
    print(f"{'─' * 80}")

    if experiments:
        for exp in experiments[-10:]:  # Last 10 experiments
            task = exp.get("task", "unknown")
            result = exp.get("result", {})
            sharpe = result.get("sharpe", "N/A")
            mlflow_id = exp.get("mlflow_run_id", "")

            if isinstance(sharpe, (int, float)):
                sharpe_str = f"Sharpe={sharpe:+.4f}"
            else:
                sharpe_str = "Sharpe=N/A"

            print(f"  {task:30} | {sharpe_str:20} | MLflow: {mlflow_id[:8]}...")

    # 4. Task completion status
    tasks_started = [e for e in events if e["action_type"] == "task_started"]
    tasks_completed = [e for e in events if e["action_type"] == "task_completed"]

    print(f"\n{'─' * 80}")
    print("TASK STATUS")
    print(f"{'─' * 80}")
    print(f"Tasks started: {len(tasks_started)}")
    print(f"Tasks completed: {len(tasks_completed)}")

    # Find currently in-progress tasks
    started_tasks = {e["task"]: e for e in tasks_started}
    completed_tasks = {e["task"] for e in tasks_completed}
    in_progress = set(started_tasks.keys()) - completed_tasks

    if in_progress:
        print(f"\n⏳ IN PROGRESS:")
        for task in in_progress:
            started_at = datetime.fromisoformat(started_tasks[task]["timestamp"])
            duration = format_timedelta(now - started_at)
            print(f"   - {task} (running for {duration})")
    else:
        print(f"\n✅ No tasks currently in progress")

    # 5. Data quality check
    print(f"\n{'─' * 80}")
    print("DATA STATUS")
    print(f"{'─' * 80}")

    feature_path = Path("data/processed/feature_matrix_btc.parquet")
    if feature_path.exists():
        df = pd.read_parquet(feature_path)
        modified = datetime.fromtimestamp(feature_path.stat().st_mtime, tz=timezone.utc)
        age = format_timedelta(now - modified)
        print(f"✅ Feature matrix: {df.shape} (updated {age} ago)")
        print(f"   Features: {', '.join(df.columns.tolist())}")
    else:
        print("❌ No feature matrix found")

    # 6. Health check
    print(f"\n{'─' * 80}")
    print("HEALTH CHECK")
    print(f"{'─' * 80}")

    if idle_time.total_seconds() > 600:  # 10 minutes
        print(f"⚠️  AGENT IDLE: {format_timedelta(idle_time)} since last activity")
        print("   Agent may be blocked, waiting for input, or paused.")
    elif idle_time.total_seconds() > 300:  # 5 minutes
        print(f"⚠️  Low activity: {format_timedelta(idle_time)} since last log")
        print("   Agent may be running long computation or debugging.")
    else:
        print(f"✅ Agent active: last log {format_timedelta(idle_time)} ago")

    print(f"\n{'─' * 80}")
    print(f"Dashboard generated at: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'─' * 80}")


if __name__ == "__main__":
    main()
