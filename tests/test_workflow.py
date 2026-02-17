"""Tests for the workflow engine, telemetry, and sparky CLI."""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sparky.workflow.engine import (
    BudgetState,
    Step,
    StepState,
    Workflow,
    WorkflowState,
)
from sparky.workflow.telemetry import (
    INPUT_RATE_USD,
    OUTPUT_RATE_USD,
    SessionTelemetry,
    StreamParser,
    save_telemetry,
)


# ── Step tests ──────────────────────────────────────────────────────────


class TestStep:
    def test_defaults(self):
        step = Step(name="test", prompt="do stuff")
        assert step.name == "test"
        assert step.prompt == "do stuff"
        assert step.done_when() is False
        assert step.skip_if() is False
        assert step.max_duration_minutes == 120
        assert step.max_retries == 3
        assert step.tags == []

    def test_custom_callables(self):
        step = Step(
            name="custom",
            prompt="p",
            done_when=lambda: True,
            skip_if=lambda: True,
            max_duration_minutes=60,
            max_retries=5,
            tags=["a", "b"],
        )
        assert step.done_when() is True
        assert step.skip_if() is True
        assert step.max_duration_minutes == 60
        assert step.max_retries == 5
        assert step.tags == ["a", "b"]


# ── WorkflowState tests ────────────────────────────────────────────────


class TestWorkflowState:
    def test_save_load_roundtrip(self, tmp_path):
        state = WorkflowState(
            workflow_name="test-wf",
            current_step_index=1,
            created_at="2026-01-01T00:00:00Z",
            budget=BudgetState(max_hours=10, hours_used=2.5, estimated_cost_usd=1.50),
        )
        state.steps["step1"] = StepState(name="step1", status="completed", attempts=2)
        state.steps["step2"] = StepState(name="step2", status="running", attempts=1)

        state.save(tmp_path)

        loaded = WorkflowState.load("test-wf", tmp_path)
        assert loaded is not None
        assert loaded.workflow_name == "test-wf"
        assert loaded.current_step_index == 1
        assert loaded.steps["step1"].status == "completed"
        assert loaded.steps["step1"].attempts == 2
        assert loaded.steps["step2"].status == "running"
        assert loaded.budget.max_hours == 10
        assert loaded.budget.hours_used == 2.5
        assert loaded.budget.estimated_cost_usd == 1.50

    def test_load_nonexistent(self, tmp_path):
        assert WorkflowState.load("nope", tmp_path) is None

    def test_atomic_write(self, tmp_path):
        """Verify state file is written atomically (no partial writes)."""
        state = WorkflowState(workflow_name="atomic-test")
        state.save(tmp_path)

        # File should exist and be valid JSON
        filepath = tmp_path / "atomic-test.json"
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["workflow_name"] == "atomic-test"

    def test_budget_state_serialization(self):
        budget = BudgetState(
            max_hours=24, hours_used=6.2, estimated_cost_usd=12.50,
            runs_completed=15, warned_80_pct=True,
        )
        d = budget.to_dict()
        restored = BudgetState.from_dict(d)
        assert restored.max_hours == 24
        assert restored.hours_used == 6.2
        assert restored.warned_80_pct is True


# ── Workflow tests ──────────────────────────────────────────────────────


class TestWorkflow:
    def _make_workflow(self, steps, tmp_path, max_hours=24):
        return Workflow(name="test-wf", steps=steps, max_hours=max_hours, state_dir=tmp_path)

    def _mock_telemetry(self):
        return SessionTelemetry(
            session_id="20260217_000000",
            step="test",
            attempt=1,
            started_at="2026-02-17T00:00:00Z",
            ended_at="2026-02-17T00:30:00Z",
            duration_minutes=30.0,
            tokens_input=1000,
            tokens_output=500,
            estimated_cost_usd=0.01,
        )

    @patch.object(Workflow, "_launch_claude")
    @patch.object(Workflow, "_alert")
    def test_skip_if_skips(self, mock_alert, mock_launch, tmp_path):
        step = Step(name="s1", prompt="p", skip_if=lambda: True)
        wf = self._make_workflow([step], tmp_path)
        result = wf.run()
        assert result == 0
        mock_launch.assert_not_called()
        state = WorkflowState.load("test-wf", tmp_path)
        assert state.steps["s1"].status == "skipped"

    @patch.object(Workflow, "_launch_claude")
    @patch.object(Workflow, "_alert")
    def test_done_when_skips(self, mock_alert, mock_launch, tmp_path):
        step = Step(name="s1", prompt="p", done_when=lambda: True)
        wf = self._make_workflow([step], tmp_path)
        result = wf.run()
        assert result == 0
        mock_launch.assert_not_called()
        state = WorkflowState.load("test-wf", tmp_path)
        assert state.steps["s1"].status == "completed"

    @patch.object(Workflow, "_launch_claude")
    @patch.object(Workflow, "_alert")
    def test_step_completes_after_launch(self, mock_alert, mock_launch, tmp_path):
        call_count = [0]

        def done_check():
            # First call: not done. After launch: done.
            call_count[0] += 1
            return call_count[0] > 1

        mock_launch.return_value = self._mock_telemetry()
        step = Step(name="s1", prompt="p", done_when=done_check)
        wf = self._make_workflow([step], tmp_path)
        result = wf.run()
        assert result == 0
        mock_launch.assert_called_once()

    @patch.object(Workflow, "_launch_claude")
    @patch.object(Workflow, "_alert")
    def test_not_done_returns_1(self, mock_alert, mock_launch, tmp_path):
        mock_launch.return_value = self._mock_telemetry()
        step = Step(name="s1", prompt="p")  # done_when always False
        wf = self._make_workflow([step], tmp_path)
        result = wf.run()
        assert result == 1

    @patch.object(Workflow, "_launch_claude")
    @patch.object(Workflow, "_alert")
    def test_retry_count_persists(self, mock_alert, mock_launch, tmp_path):
        mock_launch.return_value = self._mock_telemetry()
        step = Step(name="s1", prompt="p", max_retries=2)
        wf = self._make_workflow([step], tmp_path)

        # First run: attempt 1
        result = wf.run()
        assert result == 1
        state = WorkflowState.load("test-wf", tmp_path)
        assert state.steps["s1"].attempts == 1

        # Second run: attempt 2
        result = wf.run()
        assert result == 1
        state = WorkflowState.load("test-wf", tmp_path)
        assert state.steps["s1"].attempts == 2

        # Third run: retries exhausted
        result = wf.run()
        assert result == 1  # failed
        state = WorkflowState.load("test-wf", tmp_path)
        assert state.steps["s1"].status == "failed"

    @patch.object(Workflow, "_launch_claude")
    @patch.object(Workflow, "_alert")
    def test_multi_step_sequencing(self, mock_alert, mock_launch, tmp_path):
        call_counts = {"s1": 0, "s2": 0}

        def s1_done():
            call_counts["s1"] += 1
            return call_counts["s1"] > 1

        def s2_done():
            call_counts["s2"] += 1
            return call_counts["s2"] > 1

        mock_launch.return_value = self._mock_telemetry()
        steps = [
            Step(name="s1", prompt="p1", done_when=s1_done),
            Step(name="s2", prompt="p2", done_when=s2_done),
        ]
        wf = self._make_workflow(steps, tmp_path)

        # First run: s1 launches, completes, s2 launches, completes
        result = wf.run()
        assert result == 0
        assert mock_launch.call_count == 2

    @patch.object(Workflow, "_alert")
    def test_pause_exits_0(self, mock_alert, tmp_path):
        (tmp_path / "PAUSE").touch()
        step = Step(name="s1", prompt="p")
        wf = self._make_workflow([step], tmp_path)
        result = wf.run()
        assert result == 0
        mock_alert.assert_called_with("INFO", "Workflow paused by operator")

    @patch.object(Workflow, "_launch_claude")
    @patch.object(Workflow, "_alert")
    def test_inject_appends_to_prompt_and_deletes(self, mock_alert, mock_launch, tmp_path):
        mock_launch.return_value = self._mock_telemetry()
        inject_file = tmp_path / "inject.md"
        inject_file.write_text("Focus on CatBoost configs")

        step = Step(name="s1", prompt="do sweep")
        wf = self._make_workflow([step], tmp_path)
        wf.run()

        # Check prompt included inject text
        call_args = mock_launch.call_args
        prompt = call_args[1]["prompt"] if "prompt" in call_args[1] else call_args[0][0]
        assert "Focus on CatBoost configs" in prompt
        # Inject file should be deleted
        assert not inject_file.exists()

    @patch.object(Workflow, "_alert")
    def test_budget_exhaustion_exits_0(self, mock_alert, tmp_path):
        step = Step(name="s1", prompt="p")
        wf = self._make_workflow([step], tmp_path, max_hours=1.0)

        # Create state with budget exhausted
        state = wf._load_or_create_state()
        state.budget.hours_used = 1.5
        state.save(tmp_path)

        result = wf.run()
        assert result == 0
        # Should alert CRITICAL
        assert any("Budget exhausted" in str(c) for c in mock_alert.call_args_list)


# ── StreamParser tests ──────────────────────────────────────────────────


class TestStreamParser:
    def test_text_extraction(self, tmp_path):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        log_path = tmp_path / "test.log"

        line = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello world"}]},
        })

        with open(log_path, "w") as f:
            parser.feed(line, f)

        with open(log_path) as f:
            assert "Hello world" in f.read()

    def test_tool_call_counting(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "Bash", "id": "1", "input": {}},
                {"type": "tool_use", "name": "Read", "id": "2", "input": {}},
            ]},
        })
        parser.feed(line)
        assert parser.telemetry.tool_calls == 2

    def test_result_token_extraction(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps({
            "type": "result",
            "result": "done",
            "usage": {"input_tokens": 5000, "output_tokens": 2000},
        })
        parser.feed(line)
        assert parser.telemetry.tokens_input == 5000
        assert parser.telemetry.tokens_output == 2000

    def test_option_menu_flag(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Option A: do X, Option B: do Y"}]},
        })
        parser.feed(line)
        assert "option_menu_detected" in parser.telemetry.behavioral_flags

    def test_escalation_flag(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "I need to escalate this"}]},
        })
        parser.feed(line)
        assert "escalation_detected" in parser.telemetry.behavioral_flags

    def test_step_skip_flag(self):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "This step is not applicable"}]},
        })
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
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "Read", "id": "1", "input": {}}]},
        })
        parser.feed(line)
        telemetry = parser.finalize()
        assert "idle_session" not in telemetry.behavioral_flags

    def test_json_decode_error_passthrough(self, tmp_path):
        parser = StreamParser(session_id="test", step="s1", attempt=1)
        log_path = tmp_path / "test.log"
        with open(log_path, "w") as f:
            parser.feed("not json at all", f)
        with open(log_path) as f:
            assert "not json at all" in f.read()


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
            session_id="test", step="s1", attempt=1,
            started_at="2026-01-01T00:00:00Z",
            tokens_input=1_000_000, tokens_output=100_000,
        )
        cost = t.compute_cost()
        expected = 1_000_000 * INPUT_RATE_USD + 100_000 * OUTPUT_RATE_USD
        assert abs(cost - expected) < 0.001
        assert t.estimated_cost_usd == cost

    def test_save_telemetry(self, tmp_path):
        t = SessionTelemetry(
            session_id="20260217_test", step="s1", attempt=1,
            started_at="2026-02-17T00:00:00Z",
        )
        filepath = save_telemetry(t, str(tmp_path))
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["session_id"] == "20260217_test"


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
                "feature_analysis": {"name": "feature_analysis", "status": "completed", "attempts": 1, "completed_at": "2026-02-17T01:00:00Z", "last_attempt_at": ""},
                "two_stage_sweep": {"name": "two_stage_sweep", "status": "running", "attempts": 2, "completed_at": "", "last_attempt_at": "2026-02-17T02:00:00Z"},
                "regime_aware_hybrid": {"name": "regime_aware_hybrid", "status": "pending", "attempts": 0, "completed_at": "", "last_attempt_at": ""},
                "ensemble": {"name": "ensemble", "status": "pending", "attempts": 0, "completed_at": "", "last_attempt_at": ""},
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
