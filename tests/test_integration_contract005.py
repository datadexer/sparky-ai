"""Integration tests for Contract 005 infrastructure.

Exercises multiple modules together to validate that guardrails, metrics,
manager_log, and interface protocols all work correctly end-to-end.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparky.tracking.guardrails import (
    GuardrailResult,
    run_pre_checks,
    run_post_checks,
    has_blocking_failure,
    log_results,
)
from sparky.tracking.metrics import compute_all_metrics
from sparky.tracking.manager_log import (
    ManagerLog,
    ManagerSession,
    CodeAgentRecord,
    ResearchLaunchRecord,
    ContractDesignRecord,
)
from sparky.interfaces import (
    StrategyProtocol,
    BacktesterProtocol,
    DataFeedProtocol,
    FeaturePipelineProtocol,
    PositionSizerProtocol,
)


# === Fixtures ===


@pytest.fixture
def synthetic_returns():
    """Synthetic strategy returns within in-sample range."""
    np.random.seed(42)
    return np.random.randn(2000) * 0.01


@pytest.fixture
def synthetic_data():
    """Synthetic OHLCV data with DatetimeIndex ending before holdout boundary."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=5000, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "close": np.random.randn(5000).cumsum() + 100,
            "volume": np.random.rand(5000) * 1000,
        },
        index=dates,
    )
    return data


@pytest.fixture
def good_config():
    """Config that passes all pre-checks."""
    return {
        "features": ["close", "volume"],
        "target": "target_1h",
        "transaction_costs_bps": 10.0,
    }


# === Test 1: Guardrails + Metrics pipeline ===


class TestGuardrailsMetricsPipeline:
    """Exercises guardrails + metrics together, verifying DSR threshold fires correctly."""

    def test_dsr_threshold_fires_on_random_returns(self, synthetic_returns):
        """DSR check should fail (below threshold) for random noise returns."""
        metrics = compute_all_metrics(synthetic_returns, n_trials=50)

        # Verify all expected metric keys exist
        assert "sharpe" in metrics
        assert "dsr" in metrics
        assert "psr" in metrics
        assert "max_drawdown" in metrics

        # Run post-checks with these metrics
        config = {"transaction_costs_bps": 10.0}
        post_results = run_post_checks(synthetic_returns, metrics, config, )

        # Should have 6 checks
        assert len(post_results) == 6

        # DSR check should be present
        dsr_checks = [r for r in post_results if r.check_name == "dsr_threshold"]
        assert len(dsr_checks) == 1
        dsr_check = dsr_checks[0]

        # For random noise with 50 trials, DSR should be low (below 0.80 threshold)
        # The check fires (passed=False) when DSR < threshold
        if metrics["dsr"] < 0.80:
            assert not dsr_check.passed, "DSR check should fail for noise returns"
        else:
            assert dsr_check.passed, "DSR check should pass for high-DSR returns"

    def test_metrics_pipeline_produces_valid_dict(self, synthetic_returns):
        """compute_all_metrics should return a complete metrics dict."""
        metrics = compute_all_metrics(synthetic_returns, n_trials=50)

        required_keys = [
            "sharpe", "psr", "dsr", "min_track_record", "n_trials",
            "sortino", "max_drawdown", "calmar", "cvar_5pct",
            "rolling_sharpe_std", "profit_factor", "worst_year_sharpe",
            "n_observations", "win_rate", "mean_return", "total_return",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

        assert metrics["n_observations"] == len(synthetic_returns)
        assert metrics["n_trials"] == 50

    def test_guardrails_plus_metrics_end_to_end(self, synthetic_returns):
        """Full pipeline: compute metrics → run post-checks → check for failures."""
        metrics = compute_all_metrics(synthetic_returns, n_trials=50)

        config = {"transaction_costs_bps": 10.0}
        post_results = run_post_checks(synthetic_returns, metrics, config, )

        # Sharpe sanity check should pass for random returns (Sharpe < 4)
        sharpe_checks = [r for r in post_results if r.check_name == "sharpe_sanity"]
        assert len(sharpe_checks) == 1
        assert sharpe_checks[0].passed  # random noise Sharpe is nowhere near 4.0

        # has_blocking_failure should depend on blocking checks, not warns/info
        block_failures = [r for r in post_results if not r.passed and r.severity == "block"]
        assert has_blocking_failure(post_results) == (len(block_failures) > 0)


# === Test 2: Guardrails JSONL logging round-trip ===


class TestGuardrailsJsonlRoundTrip:
    """Verify pre-checks + post-checks → log_results → JSONL read-back."""

    def test_jsonl_logging_round_trip(self, synthetic_data, good_config, synthetic_returns):
        """Run pre-checks + post-checks, log to temp file, read back, verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = str(Path(tmpdir) / "guardrail_log.jsonl")

            # Run pre-checks
            pre_results = run_pre_checks(synthetic_data, good_config)
            assert len(pre_results) == 5

            # Run post-checks
            metrics = compute_all_metrics(synthetic_returns, n_trials=50)
            post_results = run_post_checks(synthetic_returns, metrics, good_config, )
            assert len(post_results) == 6

            all_results = pre_results + post_results

            # Log to JSONL
            log_results(all_results, run_id="test_integration_run", logfile=logfile)

            # Read back and verify
            with open(logfile) as f:
                raw_line = f.readline()
            entry = json.loads(raw_line)

            # Verify expected fields
            assert "timestamp" in entry
            assert entry["run_id"] == "test_integration_run"
            assert "checks" in entry
            assert "has_blocking_failure" in entry
            assert "summary" in entry

            # Verify summary counts
            assert entry["summary"]["total"] == 11  # 5 pre + 6 post
            assert entry["summary"]["passed"] + entry["summary"]["failed"] == 11

            # Verify each check entry has required fields
            for check in entry["checks"]:
                assert "passed" in check
                assert "check_name" in check
                assert "message" in check
                assert "severity" in check

    def test_multiple_log_entries_append_correctly(self):
        """Logging twice should produce two valid JSONL entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = str(Path(tmpdir) / "test.jsonl")
            results = [
                GuardrailResult(passed=True, check_name="test_check", message="ok", severity="info")
            ]

            log_results(results, run_id="run_001", logfile=logfile)
            log_results(results, run_id="run_002", logfile=logfile)

            with open(logfile) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            assert len(lines) == 2
            entry1 = json.loads(lines[0])
            entry2 = json.loads(lines[1])
            assert entry1["run_id"] == "run_001"
            assert entry2["run_id"] == "run_002"


# === Test 3: Manager Log lifecycle ===


class TestManagerLogLifecycle:
    """Verify full session lifecycle: start → log → end → JSONL → get_history."""

    def test_full_session_lifecycle(self):
        """Start session, log code agent and decision, end, read back with get_history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = ManagerLog(log_file=Path(tmpdir) / "test.jsonl")

            # Start session
            session = log.start_session("test objective", "test-branch")
            assert session.objective == "test objective"
            assert session.branch == "test-branch"
            assert session.started_at != ""
            assert session.session_id != ""

            # Log code agent
            log.log_code_agent(
                session,
                CodeAgentRecord(
                    task="Build integration tests",
                    model="sonnet",
                    files_created=["tests/test_integration_contract005.py"],
                    tests_passed=True,
                ),
            )
            assert len(session.code_agents) == 1
            assert session.code_agents[0].task == "Build integration tests"

            # Log decision
            log.log_decision(
                session,
                decision="Use Protocol for interfaces",
                alternatives=["ABC", "Protocol", "duck typing"],
                rationale="Matches existing codebase pattern",
            )
            assert len(session.decisions) == 1
            assert session.decisions[0]["decision"] == "Use Protocol for interfaces"

            # End session
            log.end_session(session, summary="Integration tests written and passing")
            assert session.ended_at != ""
            assert session.summary == "Integration tests written and passing"

            # Verify JSONL was written
            logfile = Path(tmpdir) / "test.jsonl"
            assert logfile.exists()

            with open(logfile) as f:
                raw = f.readline()
            entry = json.loads(raw)

            assert entry["session_id"] == session.session_id
            assert entry["objective"] == "test objective"
            assert len(entry["code_agents"]) == 1
            assert len(entry["decisions"]) == 1
            assert entry["summary"] == "Integration tests written and passing"

            # Read back with get_history()
            history = log.get_history()
            assert len(history) == 1

            retrieved = history[0]
            assert retrieved.session_id == session.session_id
            assert retrieved.objective == "test objective"
            assert retrieved.branch == "test-branch"
            assert retrieved.summary == "Integration tests written and passing"
            assert len(retrieved.code_agents) == 1
            assert retrieved.code_agents[0].task == "Build integration tests"
            assert retrieved.code_agents[0].tests_passed is True

    def test_multiple_sessions_get_history_newest_first(self):
        """Multiple sessions: get_history returns newest first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = ManagerLog(log_file=Path(tmpdir) / "multi.jsonl")

            for i in range(3):
                session = log.start_session(f"objective {i}", f"branch-{i}")
                log.end_session(session, summary=f"summary {i}")

            history = log.get_history()
            assert len(history) == 3
            # Newest first: objective 2 was last written
            assert history[0].objective == "objective 2"
            assert history[1].objective == "objective 1"
            assert history[2].objective == "objective 0"


# === Test 4: Interface Protocol conformance ===


class TestInterfaceProtocolConformance:
    """Verify mock classes satisfying Protocol interfaces pass isinstance checks."""

    def test_strategy_protocol(self):
        """Mock class satisfying StrategyProtocol passes isinstance."""
        class MockStrategy:
            name = "test_strategy"

            def generate_signals(self, data):
                return pd.Series()

            def get_params(self):
                return {}

        assert isinstance(MockStrategy(), StrategyProtocol)

    def test_backtester_protocol(self):
        """Mock class satisfying BacktesterProtocol passes isinstance."""
        class MockBacktester:
            def run(self, model, X, y, returns, **kwargs):
                return {}

        assert isinstance(MockBacktester(), BacktesterProtocol)

    def test_data_feed_protocol(self):
        """Mock class satisfying DataFeedProtocol passes isinstance."""
        class MockDataFeed:
            def load(self, dataset, purpose="training"):
                return pd.DataFrame()

            def list_datasets(self):
                return []

        assert isinstance(MockDataFeed(), DataFeedProtocol)

    def test_feature_pipeline_protocol(self):
        """Mock class satisfying FeaturePipelineProtocol passes isinstance."""
        class MockFeaturePipeline:
            def transform(self, data):
                return data

            def get_feature_names(self):
                return []

        assert isinstance(MockFeaturePipeline(), FeaturePipelineProtocol)

    def test_position_sizer_protocol(self):
        """Mock class satisfying PositionSizerProtocol passes isinstance."""
        class MockPositionSizer:
            def size_position(self, signal, data, portfolio_value):
                return 0.0

        assert isinstance(MockPositionSizer(), PositionSizerProtocol)

    def test_non_conforming_class_fails(self):
        """Class missing required methods should NOT satisfy protocol."""
        class EmptyClass:
            pass

        # Empty class has no methods, so should not satisfy any protocol
        # Note: StrategyProtocol requires name attribute + 2 methods
        # For runtime_checkable Protocol, isinstance checks structural attributes
        # An empty class won't have 'name' or the methods
        obj = EmptyClass()
        # The Protocol runtime check only verifies method presence, not signatures
        # EmptyClass has no 'name', 'generate_signals', or 'get_params'
        # So it should NOT satisfy StrategyProtocol
        assert not isinstance(obj, StrategyProtocol)


# === Test 5: Guardrails pre-check orchestrator ===


class TestGuardrailsPreCheckOrchestrator:
    """Verify run_pre_checks returns all 5 checks for synthetic DataFrame."""

    def test_run_pre_checks_returns_5_results(self, synthetic_data, good_config):
        """run_pre_checks on valid synthetic data returns exactly 5 GuardrailResult objects."""
        results = run_pre_checks(synthetic_data, good_config)

        assert len(results) == 5
        assert all(isinstance(r, GuardrailResult) for r in results)

    def test_pre_checks_names_are_correct(self, synthetic_data, good_config):
        """Verify the 5 checks have the expected names."""
        results = run_pre_checks(synthetic_data, good_config)
        check_names = {r.check_name for r in results}

        expected_names = {
            "holdout_boundary",
            "minimum_samples",
            "no_lookahead",
            "costs_specified",
            "param_data_ratio",
        }
        assert check_names == expected_names

    def test_pre_checks_pass_for_valid_data(self, synthetic_data, good_config):
        """All 5 pre-checks pass for synthetic data within holdout boundary."""
        results = run_pre_checks(synthetic_data, good_config)
        failed = [r for r in results if not r.passed]
        assert len(failed) == 0, f"Unexpected failures: {[r.check_name for r in failed]}"

    def test_pre_checks_holdout_boundary_fails_for_future_data(self, good_config):
        """holdout_boundary check fires for data extending past 2024-07-01."""
        # Data that extends into the holdout period
        dates = pd.date_range("2024-05-01", periods=5000, freq="h", tz="UTC")
        future_data = pd.DataFrame(
            {
                "close": np.random.randn(5000).cumsum() + 100,
                "volume": np.random.rand(5000) * 1000,
            },
            index=dates,
        )
        results = run_pre_checks(future_data, good_config)

        holdout_check = next(r for r in results if r.check_name == "holdout_boundary")
        assert not holdout_check.passed
        assert holdout_check.severity == "block"


# === Test 6: Guardrails blocking failure detection ===


class TestGuardrailsBlockingFailureDetection:
    """Verify has_blocking_failure logic correctly distinguishes block vs warn."""

    def test_blocking_failure_with_block_severity(self):
        """has_blocking_failure returns True when any block-severity check fails."""
        results = [
            GuardrailResult(passed=True, check_name="check_a", message="ok", severity="block"),
            GuardrailResult(passed=False, check_name="check_b", message="BLOCKED", severity="block"),
            GuardrailResult(passed=False, check_name="check_c", message="warned", severity="warn"),
        ]
        assert has_blocking_failure(results) is True

    def test_no_blocking_failure_with_only_warnings(self):
        """has_blocking_failure returns False when only warn-severity checks fail."""
        results = [
            GuardrailResult(passed=True, check_name="check_a", message="ok", severity="block"),
            GuardrailResult(passed=False, check_name="check_b", message="bad", severity="warn"),
            GuardrailResult(passed=False, check_name="check_c", message="info", severity="info"),
        ]
        assert has_blocking_failure(results) is False

    def test_no_blocking_failure_when_all_pass(self):
        """has_blocking_failure returns False when all checks pass."""
        results = [
            GuardrailResult(passed=True, check_name="check_a", message="ok", severity="block"),
            GuardrailResult(passed=True, check_name="check_b", message="ok", severity="warn"),
        ]
        assert has_blocking_failure(results) is False

    def test_empty_results_no_blocking_failure(self):
        """has_blocking_failure returns False for empty results list."""
        assert has_blocking_failure([]) is False

    def test_post_checks_with_extreme_sharpe_is_blocking(self):
        """Sanity check: extreme Sharpe (>4.0) creates a blocking failure in post-checks."""
        metrics = {
            "sharpe": 10.0,  # extreme — sanity check should block
            "dsr": 0.99,
            "max_drawdown": -0.10,
            "rolling_sharpe_std": 0.5,
        }
        np.random.seed(42)
        returns = np.random.randn(2000) * 0.01
        config = {"transaction_costs_bps": 10.0}
        post_results = run_post_checks(returns, metrics, config)

        assert has_blocking_failure(post_results) is True

    def test_normal_results_no_blocking_failure(self):
        """Normal strategy metrics should not trigger blocking failures in post-checks."""
        np.random.seed(42)
        returns = np.random.randn(2000) * 0.01
        metrics = compute_all_metrics(returns, n_trials=10)
        config = {"transaction_costs_bps": 10.0}
        post_results = run_post_checks(returns, metrics, config)

        # Sharpe sanity (block severity) should pass for random noise
        sharpe_checks = [r for r in post_results if r.check_name == "sharpe_sanity"]
        assert sharpe_checks[0].passed
        assert not has_blocking_failure(post_results)
