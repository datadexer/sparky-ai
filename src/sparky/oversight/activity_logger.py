"""Structured JSONL agent activity logger.

Every agent session MUST initialize this logger and log all
task_started, task_completed, decision_made, and error_encountered events.
These logs are the Research Strategy Analyst's primary data source.

Logs are append-only, crash-safe (flush after every write), and
written to logs/agent_activity/{agent_id}_{date}.jsonl.

If the logger itself fails, the agent MUST continue working —
oversight failure never blocks research.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

LOG_DIR = Path("logs/agent_activity")


class AgentActivityLogger:
    """Structured JSONL logger for agent activity tracking.

    Usage:
        logger = AgentActivityLogger(agent_id="research", session_id="phase-0-validation")
        logger.log_task_started("phase_0", "returns_calculations", "Implementing returns")
    """

    def __init__(self, agent_id: str, session_id: str, log_dir: Optional[Path] = None):
        self.agent_id = agent_id
        self.session_id = session_id
        self.log_dir = log_dir or LOG_DIR
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Create log directory if it doesn't exist."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"[OVERSIGHT] Failed to create log dir: {e}")

    def _log_file_path(self) -> Path:
        """Get today's log file path: {agent_id}_{date}.jsonl"""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"{self.agent_id}_{date_str}.jsonl"

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Write a single JSON entry to the log file. Flush immediately."""
        try:
            path = self._log_file_path()
            with open(path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            # Oversight failure must never block research
            logger.warning(f"[OVERSIGHT] Failed to write log entry: {e}")

    def _base_entry(self, action_type: str) -> dict[str, Any]:
        """Create base entry with common fields."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "action_type": action_type,
        }

    def log_task_started(self, phase: str, task: str, description: str) -> None:
        """Log the start of a STATE.yaml task."""
        entry = self._base_entry("task_started")
        entry.update({"phase": phase, "task": task, "description": description})
        self._write_entry(entry)

    def log_task_completed(
        self,
        phase: str,
        task: str,
        description: str,
        files_changed: list[str],
        git_commit: str,
    ) -> None:
        """Log completion of a STATE.yaml task."""
        entry = self._base_entry("task_completed")
        entry.update(
            {
                "phase": phase,
                "task": task,
                "description": description,
                "files_changed": files_changed,
                "git_commit": git_commit,
            }
        )
        self._write_entry(entry)

    def log_experiment(
        self,
        task: str,
        hypothesis: str,
        strategic_goal: str,
        result: dict[str, Any],
        conclusion: str,
        mlflow_run_id: str,
    ) -> None:
        """Log a completed experiment with results."""
        entry = self._base_entry("experiment_completed")
        entry.update(
            {
                "task": task,
                "hypothesis": hypothesis,
                "strategic_goal": strategic_goal,
                "result": result,
                "conclusion": conclusion,
                "mlflow_run_id": mlflow_run_id,
            }
        )
        self._write_entry(entry)

    def log_validation(self, mlflow_run_id: str, new_status: str, reason: str) -> None:
        """Log a validation status change (preliminary -> validated -> proven)."""
        entry = self._base_entry("validation_check")
        entry.update(
            {
                "mlflow_run_id": mlflow_run_id,
                "new_status": new_status,
                "reason": reason,
            }
        )
        self._write_entry(entry)

    def log_decision(
        self,
        description: str,
        options: list[str],
        chosen: str,
        reasoning: str,
    ) -> None:
        """Log a decision made by the agent."""
        entry = self._base_entry("decision_made")
        entry.update(
            {
                "description": description,
                "options": options,
                "chosen": chosen,
                "reasoning": reasoning,
            }
        )
        self._write_entry(entry)

    def log_error(self, description: str, recovery: str) -> None:
        """Log an error and recovery action."""
        entry = self._base_entry("error_encountered")
        entry.update({"description": description, "recovery": recovery})
        self._write_entry(entry)

    def log_direction_change(self, old_direction: str, new_direction: str, reasoning: str) -> None:
        """Log a change in research direction."""
        entry = self._base_entry("direction_change")
        entry.update(
            {
                "old_direction": old_direction,
                "new_direction": new_direction,
                "reasoning": reasoning,
            }
        )
        self._write_entry(entry)

    def log_hypothesis_proposed(self, hypothesis: str, strategic_goal: str, expected_outcome: str) -> None:
        """Log a new research hypothesis."""
        entry = self._base_entry("hypothesis_proposed")
        entry.update(
            {
                "hypothesis": hypothesis,
                "strategic_goal": strategic_goal,
                "expected_outcome": expected_outcome,
            }
        )
        self._write_entry(entry)
