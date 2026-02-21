"""Append-only trial counter for DSR multiple testing correction.

Records every strategy config tested, tracks cumulative trial count
per family and globally. File-locked for concurrent safety.
"""

import fcntl
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

__all__ = ["TrialCounter"]

logger = logging.getLogger(__name__)


class TrialCounter:
    """Append-only JSONL trial log with file locking."""

    def __init__(self, log_path: str | Path, project_id: str):
        self.log_path = Path(log_path)
        self.project_id = project_id
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_trial(
        self,
        strategy_family: str,
        config: dict,
        sharpe: float,
        passed: bool,
    ) -> None:
        """Append a trial record (file-locked)."""
        entry = {
            "project_id": self.project_id,
            "strategy_family": strategy_family,
            "config": config,
            "sharpe": sharpe,
            "passed": passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.log_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry, default=str) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _read_entries(self, strategy_family: Optional[str] = None) -> list[dict]:
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("project_id") != self.project_id:
                        continue
                    if strategy_family and entry.get("strategy_family") != strategy_family:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries

    def count(self, strategy_family: Optional[str] = None) -> int:
        """Count trials, optionally filtered by family."""
        return len(self._read_entries(strategy_family))

    def best_sharpe(self, strategy_family: Optional[str] = None) -> float:
        """Best Sharpe ratio across trials."""
        entries = self._read_entries(strategy_family)
        if not entries:
            return float("-inf")
        return max(e.get("sharpe", float("-inf")) for e in entries)

    def get_dsr_n(self) -> int:
        """Total trial count for DSR computation (all families)."""
        return self.count()
