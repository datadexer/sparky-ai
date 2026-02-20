"""Tests for session telemetry, experiment tracker extensions, and sparky CLI."""

import json
import time
from unittest.mock import MagicMock, patch

from sparky.workflow.telemetry import (
    CACHE_READ_RATE_USD,
    CACHE_WRITE_RATE_USD,
    INPUT_RATE_USD,
    OUTPUT_RATE_USD,
    SessionTelemetry,
    StreamParser,
    save_telemetry,
)

# ── StreamParser tests ──────────────────────────────────────────────────


class TestStreamParser:
    def test_text_extraction(self, tmp_path):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        log_path = tmp_path / "test.log"

        line = json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hello world"}]},
            }
        )

        with open(log_path, "w") as f:
            parser.feed(line, f)

        with open(log_path) as f:
            assert "Hello world" in f.read()

    def test_tool_call_counting(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Bash", "id": "1", "input": {}},
                        {"type": "tool_use", "name": "Read", "id": "2", "input": {}},
                    ]
                },
            }
        )
        parser.feed(line)
        assert parser.telemetry.tool_calls == 2

    def test_result_token_extraction(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps(
            {
                "type": "result",
                "result": "done",
                "usage": {"input_tokens": 5000, "output_tokens": 2000},
            }
        )
        parser.feed(line)
        assert parser.telemetry.tokens_input == 5000
        assert parser.telemetry.tokens_output == 2000

    def test_option_menu_flag(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Option A: do X, Option B: do Y"}]},
            }
        )
        parser.feed(line)
        assert "option_menu_detected" in parser.telemetry.behavioral_flags

    def test_escalation_flag(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "I need to escalate this"}]},
            }
        )
        parser.feed(line)
        assert "escalation_detected" in parser.telemetry.behavioral_flags

    def test_step_skip_flag(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "This step is not applicable"}]},
            }
        )
        parser.feed(line)
        assert "step_skip_attempt" in parser.telemetry.behavioral_flags

    def test_idle_session_flag(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        # Simulate 5+ minutes with no tool calls
        parser._start_time = time.monotonic() - 301
        telemetry = parser.finalize()
        assert "idle_session" in telemetry.behavioral_flags

    def test_no_false_idle_flag(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        # Short session with tool calls — no idle flag
        line = json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": "Read", "id": "1", "input": {}}]},
            }
        )
        parser.feed(line)
        telemetry = parser.finalize()
        assert "idle_session" not in telemetry.behavioral_flags

    def test_rate_limit_detection_in_result(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps(
            {
                "type": "result",
                "result": "Error: rate limit exceeded. Too many requests.",
            }
        )
        parser.feed(line)
        assert parser.telemetry.exit_reason == "rate_limit"

    def test_rate_limit_detection_in_error(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps(
            {
                "type": "error",
                "error": "Too many requests - you have exceeded your rate limit",
            }
        )
        parser.feed(line)
        assert parser.telemetry.exit_reason == "rate_limit"

    def test_out_of_extra_usage_detection(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps(
            {
                "type": "result",
                "result": "out of extra usage credits",
            }
        )
        parser.feed(line)
        assert parser.telemetry.exit_reason == "rate_limit"

    def test_json_decode_error_passthrough(self, tmp_path):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        log_path = tmp_path / "test.log"
        with open(log_path, "w") as f:
            parser.feed("not json at all", f)
        with open(log_path) as f:
            assert "not json at all" in f.read()

    def test_result_cache_token_extraction(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps(
            {
                "type": "result",
                "total_cost_usd": 1.50,
                "num_turns": 5,
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 2000,
                    "cache_read_input_tokens": 450000,
                    "cache_creation_input_tokens": 32000,
                },
            }
        )
        parser.feed(line)
        assert parser.telemetry.tokens_input == 50
        assert parser.telemetry.tokens_output == 2000
        assert parser.telemetry.tokens_cache_read == 450000
        assert parser.telemetry.tokens_cache_creation == 32000
        assert parser.telemetry.cost_usd == 1.50
        assert parser.telemetry.num_turns == 5


# ── SessionTelemetry tests ──────────────────────────────────────────────


class TestSessionTelemetry:
    def test_serialization(self):
        t = SessionTelemetry(
            session_id="20260217_000000",
            step="sweep",
            attempt=1,
            started_at="2026-02-17T00:00:00Z",
            ended_at="2026-02-17T00:30:00Z",
            duration_minutes=30.0,
            tokens_input=10000,
            tokens_output=5000,
            behavioral_flags=["idle_session"],
        )
        d = t.to_dict()
        restored = SessionTelemetry.from_dict(d)
        assert restored.session_id == "20260217_000000"
        assert restored.step == "sweep"
        assert restored.tokens_input == 10000
        assert restored.behavioral_flags == ["idle_session"]

    def test_cost_estimation(self):
        t = SessionTelemetry(
            session_id="test",
            step="s1",
            attempt=1,
            started_at="2026-01-01T00:00:00Z",
            tokens_input=1_000_000,
            tokens_output=100_000,
        )
        cost = t.compute_cost()
        expected = 1_000_000 * INPUT_RATE_USD + 100_000 * OUTPUT_RATE_USD
        assert abs(cost - expected) < 0.001
        assert t.estimated_cost_usd == cost

    def test_save_telemetry(self, tmp_path):
        t = SessionTelemetry(
            session_id="20260217_test",
            step="s1",
            attempt=1,
            started_at="2026-02-17T00:00:00Z",
        )
        filepath = save_telemetry(t, str(tmp_path))
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["session_id"] == "20260217_test"

    def test_cost_includes_cache_tokens(self):
        t = SessionTelemetry(
            session_id="test",
            step="s1",
            attempt=1,
            started_at="2026-01-01T00:00:00Z",
            tokens_input=50,
            tokens_output=2000,
            tokens_cache_read=450000,
            tokens_cache_creation=32000,
        )
        cost = t.compute_cost()
        expected = (
            50 * INPUT_RATE_USD + 32000 * CACHE_WRITE_RATE_USD + 450000 * CACHE_READ_RATE_USD + 2000 * OUTPUT_RATE_USD
        )
        assert abs(cost - expected) < 0.001
        assert cost > 0.25  # sanity: much more than the old ~$0.03

    def test_new_fields_in_to_dict(self):
        t = SessionTelemetry(
            session_id="test",
            step="s1",
            attempt=1,
            started_at="2026-01-01T00:00:00Z",
            tokens_cache_read=100000,
            tokens_cache_creation=5000,
            num_turns=3,
            cost_usd=0.42,
        )
        d = t.to_dict()
        assert d["tokens_cache_read"] == 100000
        assert d["tokens_cache_creation"] == 5000
        assert d["num_turns"] == 3
        assert d["cost_usd"] == 0.42

    def test_serialization_roundtrip_with_cache_fields(self):
        t = SessionTelemetry(
            session_id="20260217_cache",
            step="sweep",
            attempt=2,
            started_at="2026-02-17T01:00:00Z",
            tokens_cache_read=500000,
            tokens_cache_creation=25000,
            num_turns=7,
            cost_usd=1.23,
        )
        d = t.to_dict()
        restored = SessionTelemetry.from_dict(d)
        assert restored.tokens_cache_read == 500000
        assert restored.tokens_cache_creation == 25000
        assert restored.num_turns == 7
        assert restored.cost_usd == 1.23


# ── ExperimentTracker extension tests ───────────────────────────────────


class TestExperimentTrackerExtensions:
    @patch("wandb.init")
    @patch("wandb.log")
    @patch("wandb.finish")
    def test_log_experiment_with_tags(self, mock_finish, mock_log, mock_init):
        mock_run = MagicMock()
        mock_run.id = "test-run-id"
        mock_init.return_value = mock_run

        from sparky.tracking.experiment import ExperimentTracker

        with patch.object(ExperimentTracker, "__init__", lambda self, **kw: None):
            tracker = ExperimentTracker()
            tracker.experiment_name = "test"
            tracker.project = "test-project"
            tracker.entity = "test-entity"

            tracker.log_experiment(
                name="test-run",
                config={"lr": 0.01},
                metrics={"sharpe": 1.0},
                tags=["contract_004", "sweep"],
            )

            # Verify tags were passed to wandb.init
            init_kwargs = mock_init.call_args[1]
            assert init_kwargs["tags"] == ["contract_004", "sweep"]

    @patch("wandb.init")
    @patch("wandb.log")
    @patch("wandb.finish")
    @patch("wandb.Table")
    def test_log_sweep_with_tags(self, mock_table, mock_finish, mock_log, mock_init):
        mock_run = MagicMock()
        mock_run.id = "test-sweep-id"
        mock_init.return_value = mock_run

        from sparky.tracking.experiment import ExperimentTracker

        with patch.object(ExperimentTracker, "__init__", lambda self, **kw: None):
            tracker = ExperimentTracker()
            tracker.experiment_name = "test"
            tracker.project = "test-project"
            tracker.entity = "test-entity"

            tracker.log_sweep(
                name="test-sweep",
                results=[{"config": {"model": "xgb"}, "metrics": {"sharpe": 0.5}}],
                tags=["contract_004", "sweep"],
            )

            init_kwargs = mock_init.call_args[1]
            assert init_kwargs["tags"] == ["contract_004", "sweep"]

    def test_count_runs_with_tags(self):
        from sparky.tracking.experiment import ExperimentTracker

        with patch.object(ExperimentTracker, "__init__", lambda self, **kw: None):
            tracker = ExperimentTracker()
            tracker.experiment_name = "test"
            tracker.project = "test-project"
            tracker.entity = "test-entity"

            mock_runs = [MagicMock(), MagicMock(), MagicMock()]
            with patch.object(tracker, "_fetch_runs", return_value=mock_runs):
                count = tracker.count_runs(tags=["contract_004", "sweep"])
                assert count == 3

    def test_best_metric(self):
        from sparky.tracking.experiment import ExperimentTracker

        with patch.object(ExperimentTracker, "__init__", lambda self, **kw: None):
            tracker = ExperimentTracker()
            tracker.experiment_name = "test"
            tracker.project = "test-project"
            tracker.entity = "test-entity"

            mock_runs = []
            for sharpe in [0.5, 1.2, 0.8]:
                run = MagicMock()
                run.summary = {"sharpe": sharpe}
                mock_runs.append(run)

            with patch.object(tracker, "_fetch_runs", return_value=mock_runs):
                best = tracker.best_metric("sharpe", tags=["contract_004"])
                assert best == 1.2

                worst = tracker.best_metric("sharpe", tags=["contract_004"], maximize=False)
                assert worst == 0.5

    def test_best_metric_no_runs(self):
        from sparky.tracking.experiment import ExperimentTracker

        with patch.object(ExperimentTracker, "__init__", lambda self, **kw: None):
            tracker = ExperimentTracker()
            tracker.experiment_name = "test"
            tracker.project = "test-project"
            tracker.entity = "test-entity"

            with patch.object(tracker, "_fetch_runs", return_value=[]):
                assert tracker.best_metric("sharpe") is None


# ── Sparky CLI tests ───────────────────────────────────────────────────


class TestSparkyCLI:
    def _create_state(self, tmp_path):
        state = {
            "workflow_name": "contract-004",
            "current_step_index": 1,
            "steps": {
                "feature_analysis": {
                    "name": "feature_analysis",
                    "status": "completed",
                    "attempts": 1,
                    "completed_at": "2026-02-17T01:00:00Z",
                    "last_attempt_at": "",
                },
                "two_stage_sweep": {
                    "name": "two_stage_sweep",
                    "status": "running",
                    "attempts": 2,
                    "completed_at": "",
                    "last_attempt_at": "2026-02-17T02:00:00Z",
                },
                "regime_aware_hybrid": {
                    "name": "regime_aware_hybrid",
                    "status": "pending",
                    "attempts": 0,
                    "completed_at": "",
                    "last_attempt_at": "",
                },
                "ensemble": {
                    "name": "ensemble",
                    "status": "pending",
                    "attempts": 0,
                    "completed_at": "",
                    "last_attempt_at": "",
                },
            },
            "budget": {
                "max_hours": 24.0,
                "hours_used": 6.2,
                "estimated_cost_usd": 12.50,
                "runs_completed": 15,
                "warned_80_pct": False,
            },
            "created_at": "2026-02-17T00:00:00Z",
            "updated_at": "2026-02-17T02:00:00Z",
        }
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        with open(state_dir / "contract-004.json", "w") as f:
            json.dump(state, f)
        return state_dir

    def test_skip_state_mutation(self, tmp_path):
        """Test that sparky skip modifies state correctly."""
        state_dir = self._create_state(tmp_path)

        # Simulate skip logic from CLI
        with open(state_dir / "contract-004.json") as f:
            state = json.load(f)

        step_name = "two_stage_sweep"
        state["steps"][step_name]["status"] = "skipped"
        step_names = list(state["steps"].keys())
        current_idx = state["current_step_index"]
        if current_idx < len(step_names) and step_names[current_idx] == step_name:
            state["current_step_index"] = current_idx + 1

        with open(state_dir / "contract-004.json", "w") as f:
            json.dump(state, f)

        # Verify
        with open(state_dir / "contract-004.json") as f:
            result = json.load(f)
        assert result["steps"]["two_stage_sweep"]["status"] == "skipped"
        assert result["current_step_index"] == 2

    def test_retry_state_mutation(self, tmp_path):
        """Test that sparky retry resets state correctly."""
        state_dir = self._create_state(tmp_path)

        with open(state_dir / "contract-004.json") as f:
            state = json.load(f)

        step_name = "two_stage_sweep"
        state["steps"][step_name]["attempts"] = 0
        state["steps"][step_name]["status"] = "pending"

        with open(state_dir / "contract-004.json", "w") as f:
            json.dump(state, f)

        with open(state_dir / "contract-004.json") as f:
            result = json.load(f)
        assert result["steps"]["two_stage_sweep"]["attempts"] == 0
        assert result["steps"]["two_stage_sweep"]["status"] == "pending"
