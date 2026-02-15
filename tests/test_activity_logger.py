"""Tests for AgentActivityLogger.

Validates:
- Schema compliance for all log entry types
- Concurrent writes don't corrupt the file
- Missing log directory is auto-created
- Logger failures don't raise exceptions
"""

import json
import tempfile
import threading
from pathlib import Path

import pytest

from sparky.oversight.activity_logger import AgentActivityLogger


@pytest.fixture
def tmp_log_dir(tmp_path):
    """Provide a temporary log directory."""
    return tmp_path / "agent_activity"


@pytest.fixture
def logger(tmp_log_dir):
    """Create a logger with a temporary directory."""
    return AgentActivityLogger(
        agent_id="test_ceo",
        session_id="test-session-001",
        log_dir=tmp_log_dir,
    )


def _read_entries(log_dir: Path) -> list[dict]:
    """Read all JSONL entries from the log directory."""
    entries = []
    for path in log_dir.glob("*.jsonl"):
        with open(path) as f:
            for line in f:
                entries.append(json.loads(line))
    return entries


class TestActivityLoggerBasic:
    """Test basic logger functionality."""

    def test_log_dir_auto_created(self, tmp_log_dir):
        """Missing log directory should be auto-created."""
        assert not tmp_log_dir.exists()
        AgentActivityLogger("ceo", "test", log_dir=tmp_log_dir)
        assert tmp_log_dir.exists()

    def test_log_task_started(self, logger, tmp_log_dir):
        """task_started entries have required fields."""
        logger.log_task_started("phase_0", "returns_calculations", "Implementing returns")
        entries = _read_entries(tmp_log_dir)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["action_type"] == "task_started"
        assert entry["agent_id"] == "test_ceo"
        assert entry["session_id"] == "test-session-001"
        assert entry["phase"] == "phase_0"
        assert entry["task"] == "returns_calculations"
        assert "timestamp" in entry

    def test_log_task_completed(self, logger, tmp_log_dir):
        """task_completed entries have files_changed and git_commit."""
        logger.log_task_completed(
            "phase_0", "returns_calculations", "Done",
            files_changed=["src/sparky/features/returns.py"],
            git_commit="abc123",
        )
        entries = _read_entries(tmp_log_dir)
        assert len(entries) == 1
        assert entries[0]["files_changed"] == ["src/sparky/features/returns.py"]
        assert entries[0]["git_commit"] == "abc123"

    def test_log_experiment(self, logger, tmp_log_dir):
        """experiment_completed entries have result and conclusion."""
        logger.log_experiment(
            task="feature_ablation",
            hypothesis="On-chain features improve Sharpe",
            strategic_goal="validate_onchain_alpha",
            result={"sharpe": 0.72, "p_value": 0.023},
            conclusion="On-chain adds +0.15 Sharpe",
            mlflow_run_id="run_abc123",
        )
        entries = _read_entries(tmp_log_dir)
        assert len(entries) == 1
        assert entries[0]["action_type"] == "experiment_completed"
        assert entries[0]["result"]["sharpe"] == 0.72

    def test_log_validation(self, logger, tmp_log_dir):
        """validation_check entries have status and reason."""
        logger.log_validation("run_abc123", "validated", "Multi-seed check passed")
        entries = _read_entries(tmp_log_dir)
        assert entries[0]["new_status"] == "validated"

    def test_log_decision(self, logger, tmp_log_dir):
        """decision_made entries have options and reasoning."""
        logger.log_decision(
            "Choose data source for ETH",
            options=["CoinMetrics", "BGeometrics"],
            chosen="CoinMetrics",
            reasoning="BGeometrics has no ETH coverage",
        )
        entries = _read_entries(tmp_log_dir)
        assert entries[0]["chosen"] == "CoinMetrics"

    def test_log_error(self, logger, tmp_log_dir):
        """error_encountered entries have description and recovery."""
        logger.log_error("API timeout", "Retrying in 30s")
        entries = _read_entries(tmp_log_dir)
        assert entries[0]["action_type"] == "error_encountered"
        assert entries[0]["recovery"] == "Retrying in 30s"

    def test_log_direction_change(self, logger, tmp_log_dir):
        """direction_change entries capture old and new directions."""
        logger.log_direction_change("BTC features", "ETH features", "BTC saturated")
        entries = _read_entries(tmp_log_dir)
        assert entries[0]["old_direction"] == "BTC features"

    def test_log_hypothesis_proposed(self, logger, tmp_log_dir):
        """hypothesis_proposed entries have strategic_goal."""
        logger.log_hypothesis_proposed(
            "SOPR < 1 signals capitulation",
            "validate_onchain_alpha",
            "Sharpe improvement >0.1",
        )
        entries = _read_entries(tmp_log_dir)
        assert entries[0]["action_type"] == "hypothesis_proposed"


class TestActivityLoggerSchema:
    """Test schema compliance across all entry types."""

    def test_all_entries_have_required_fields(self, logger, tmp_log_dir):
        """Every entry type must have timestamp, agent_id, session_id, action_type."""
        logger.log_task_started("p0", "t1", "desc")
        logger.log_task_completed("p0", "t1", "done", ["f.py"], "abc")
        logger.log_experiment("t1", "h1", "g1", {"s": 1}, "c1", "r1")
        logger.log_validation("r1", "validated", "ok")
        logger.log_decision("d1", ["a", "b"], "a", "because")
        logger.log_error("err", "fix")
        logger.log_direction_change("old", "new", "why")
        logger.log_hypothesis_proposed("hyp", "goal", "expected")

        entries = _read_entries(tmp_log_dir)
        assert len(entries) == 8
        required = {"timestamp", "agent_id", "session_id", "action_type"}
        for entry in entries:
            assert required.issubset(entry.keys()), f"Missing fields in {entry['action_type']}"

    def test_timestamps_are_utc_iso(self, logger, tmp_log_dir):
        """Timestamps should be UTC ISO format."""
        logger.log_task_started("p0", "t1", "desc")
        entries = _read_entries(tmp_log_dir)
        ts = entries[0]["timestamp"]
        assert "T" in ts  # ISO format
        assert ts.endswith("+00:00") or ts.endswith("Z")  # UTC


class TestActivityLoggerConcurrency:
    """Test concurrent write safety."""

    def test_concurrent_writes(self, tmp_log_dir):
        """Multiple loggers writing concurrently should not corrupt the file."""
        n_entries_per_logger = 20
        n_loggers = 3

        def write_entries(agent_id: str):
            log = AgentActivityLogger(agent_id, "concurrent-test", log_dir=tmp_log_dir)
            for i in range(n_entries_per_logger):
                log.log_task_started("p0", f"task_{i}", f"desc_{i}")

        threads = [
            threading.Thread(target=write_entries, args=(f"agent_{i}",))
            for i in range(n_loggers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Read all entries — should be valid JSON and correct count
        entries = _read_entries(tmp_log_dir)
        assert len(entries) == n_entries_per_logger * n_loggers

        # Each entry should be valid
        for entry in entries:
            assert "timestamp" in entry
            assert "agent_id" in entry
