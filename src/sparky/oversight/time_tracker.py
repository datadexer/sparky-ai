"""Wall-clock time tracker for agent sessions.

Provides external verification of time claims.
Logs start/end timestamps that cannot be retroactively modified.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TIME_LOG_PATH = Path("logs/time_tracking.jsonl")


class TaskTimer:
    """Track wall-clock time for agent tasks.

    Usage:
        timer = TaskTimer(agent_id="research")
        timer.start("regime_detection_experiments")
        # ... do work ...
        timer.end(claimed_duration_minutes=45)
        # Logs actual duration and flags if claimed differs >2x
    """

    def __init__(self, agent_id: str, log_path: Optional[Path] = None):
        self.agent_id = agent_id
        self.log_path = log_path or TIME_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._current_task: Optional[str] = None
        self._start_time: Optional[datetime] = None

    def start(self, task_name: str) -> None:
        """Start timing a task."""
        if self._current_task is not None:
            logger.warning(
                f"[TIME] Starting '{task_name}' but '{self._current_task}' "
                f"was never ended. Auto-ending previous task."
            )
            self.end(claimed_duration_minutes=0)

        self._current_task = task_name
        self._start_time = datetime.now(timezone.utc)

        entry = {
            "event": "task_start",
            "timestamp": self._start_time.isoformat(),
            "agent_id": self.agent_id,
            "task": task_name,
        }
        self._write(entry)

    def end(self, claimed_duration_minutes: float = 0) -> dict:
        """End timing and log results."""
        end_time = datetime.now(timezone.utc)
        actual_seconds = (end_time - self._start_time).total_seconds() if self._start_time else 0
        actual_minutes = actual_seconds / 60

        discrepancy_flag = False
        if claimed_duration_minutes > 0 and actual_minutes > 0:
            ratio = max(claimed_duration_minutes, actual_minutes) / max(min(claimed_duration_minutes, actual_minutes), 0.1)
            discrepancy_flag = ratio > 2.0

        entry = {
            "event": "task_end",
            "timestamp": end_time.isoformat(),
            "agent_id": self.agent_id,
            "task": self._current_task,
            "actual_minutes": round(actual_minutes, 1),
            "claimed_minutes": claimed_duration_minutes,
            "discrepancy_flag": discrepancy_flag,
        }
        self._write(entry)

        if discrepancy_flag:
            logger.warning(
                f"[TIME] DISCREPANCY: Task '{self._current_task}' — "
                f"actual {actual_minutes:.1f}min vs claimed {claimed_duration_minutes}min"
            )

        self._current_task = None
        self._start_time = None
        return entry

    def _write(self, entry: dict) -> None:
        """Append entry to time log."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning(f"[TIME] Failed to write time log: {e}")
