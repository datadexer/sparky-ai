"""Tests for the research orchestrator."""

import json
import os
from unittest.mock import patch

import pytest

from sparky.workflow.orchestrator import (
    ContextBuilder,
    OrchestratorState,
    ResearchDirective,
    ResearchOrchestrator,
    SessionLimits,
    SessionRecord,
    StoppingCriteria,
    StrategySpec,
    jaccard_similarity,
    mean_pairwise_jaccard,
)
from sparky.workflow.telemetry import SessionTelemetry


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_directive(**overrides) -> ResearchDirective:
    """Create a minimal directive for testing."""
    defaults = {
        "name": "test_directive",
        "objective": "Test objective",
        "constraints": {"asset": "btc"},
        "strategy_space": [],
        "stopping_criteria": StoppingCriteria(max_sessions=5, max_hours=2.0, max_cost_usd=50.0),
        "session_limits": SessionLimits(min_session_minutes=1.0, max_consecutive_crashes=3),
        "wandb_tags": ["test"],
        "gates": [],
        "exclusions": [],
    }
    defaults.update(overrides)
    return ResearchDirective(**defaults)


def _make_telemetry(duration_minutes: float = 30.0, exit_reason: str = "completed") -> SessionTelemetry:
    """Create a mock telemetry result."""
    return SessionTelemetry(
        session_id="20260218_000000",
        step="session_001",
        attempt=1,
        started_at="2026-02-18T00:00:00Z",
        ended_at="2026-02-18T00:30:00Z",
        duration_minutes=duration_minutes,
        tokens_input=1000,
        tokens_output=500,
        estimated_cost_usd=0.50,
        exit_reason=exit_reason,
    )


# ── ResearchDirective tests ──────────────────────────────────────────────


class TestResearchDirective:
    def test_from_yaml_loads_example(self):
        d = ResearchDirective.from_yaml("directives/archive/example.yaml")
        assert d.name == "btc_momentum_deep_dive"
        assert "momentum" in d.objective.lower()
        assert d.constraints["asset"] == "btc"
        assert len(d.strategy_space) == 1
        assert d.strategy_space[0].family == "momentum_regime"
        assert d.stopping_criteria.success_min_sharpe == 1.0
        assert d.stopping_criteria.max_sessions == 20
        assert d.session_limits.max_session_minutes == 120
        assert d.session_limits.min_session_minutes == 5
        assert "directive_001" in d.wandb_tags
        assert len(d.gates) == 2
        assert len(d.exclusions) == 2

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ResearchDirective.from_yaml("nonexistent.yaml")

    def test_from_yaml_missing_name(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("objective: test\n")
        with pytest.raises(ValueError, match="name"):
            ResearchDirective.from_yaml(p)

    def test_from_yaml_missing_objective(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("name: test\n")
        with pytest.raises(ValueError, match="objective"):
            ResearchDirective.from_yaml(p)

    def test_from_yaml_not_a_mapping(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- just a list\n")
        with pytest.raises(ValueError, match="mapping"):
            ResearchDirective.from_yaml(p)

    def test_from_yaml_minimal(self, tmp_path):
        p = tmp_path / "minimal.yaml"
        p.write_text("name: minimal\nobjective: do stuff\n")
        d = ResearchDirective.from_yaml(p)
        assert d.name == "minimal"
        assert d.objective == "do stuff"
        # Defaults
        assert d.stopping_criteria.max_sessions == 20
        assert d.session_limits.max_session_minutes == 120

    def test_parameter_ranges_parsed(self):
        d = ResearchDirective.from_yaml("directives/archive/example.yaml")
        ranges = d.strategy_space[0].parameter_ranges
        assert "momentum_lookback" in ranges
        assert ranges["momentum_lookback"] == [10, 20, 30, 40, 60]

    def test_stop_on_success_defaults_true(self, tmp_path):
        p = tmp_path / "minimal.yaml"
        p.write_text("name: test\nobjective: test\n")
        d = ResearchDirective.from_yaml(p)
        assert d.stopping_criteria.stop_on_success is True

    def test_stop_on_success_parsed_from_yaml(self, tmp_path):
        p = tmp_path / "no_stop.yaml"
        p.write_text("name: test\nobjective: test\nstopping_criteria:\n  stop_on_success: false\n")
        d = ResearchDirective.from_yaml(p)
        assert d.stopping_criteria.stop_on_success is False


# ── OrchestratorState tests ──────────────────────────────────────────────


class TestOrchestratorState:
    def test_save_load_roundtrip(self, tmp_path):
        state = OrchestratorState(
            name="test",
            status="running",
            session_count=3,
            best_result={"sharpe": 0.95, "dsr": 0.88},
            stall_counter=2,
            crash_counter=1,
            crash_backoff_seconds=240,
            total_cost_usd=5.50,
            total_hours=1.5,
        )
        state.sessions.append(
            SessionRecord(
                session_id="20260218_001",
                session_number=1,
                start_ts="2026-02-18T00:00:00Z",
                end_ts="2026-02-18T00:30:00Z",
                duration_minutes=30.0,
                best_sharpe=0.95,
                best_dsr=0.88,
                wandb_run_ids=["abc123"],
                wandb_run_configs=[{"lr": 0.01}],
            )
        )
        state.save(tmp_path)

        loaded = OrchestratorState.load("test", tmp_path)
        assert loaded is not None
        assert loaded.name == "test"
        assert loaded.session_count == 3
        assert loaded.best_result["sharpe"] == 0.95
        assert loaded.stall_counter == 2
        assert loaded.crash_counter == 1
        assert loaded.crash_backoff_seconds == 240
        assert len(loaded.sessions) == 1
        assert loaded.sessions[0].best_sharpe == 0.95
        assert loaded.sessions[0].wandb_run_ids == ["abc123"]

    def test_load_nonexistent(self, tmp_path):
        assert OrchestratorState.load("nope", tmp_path) is None

    def test_atomic_save(self, tmp_path):
        state = OrchestratorState(name="atomic_test")
        state.save(tmp_path)
        filepath = tmp_path / "orchestrator_atomic_test.json"
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["name"] == "atomic_test"

    def test_gate_message_persists(self, tmp_path):
        state = OrchestratorState(name="gate_test", gate_message="Need human review")
        state.save(tmp_path)
        loaded = OrchestratorState.load("gate_test", tmp_path)
        assert loaded.gate_message == "Need human review"


# ── SessionRecord tests ──────────────────────────────────────────────────


class TestSessionRecord:
    def test_serialization_roundtrip(self):
        record = SessionRecord(
            session_id="20260218_001",
            session_number=1,
            start_ts="2026-02-18T00:00:00Z",
            end_ts="2026-02-18T00:30:00Z",
            duration_minutes=30.0,
            best_sharpe=1.05,
            best_dsr=0.97,
            wandb_run_ids=["abc", "def"],
            wandb_run_configs=[{"lr": 0.01}, {"lr": 0.05}],
            estimated_cost_usd=0.50,
        )
        d = record.to_dict()
        restored = SessionRecord.from_dict(d)
        assert restored.session_id == "20260218_001"
        assert restored.best_sharpe == 1.05
        assert restored.wandb_run_configs == [{"lr": 0.01}, {"lr": 0.05}]

    def test_new_fields_roundtrip(self):
        record = SessionRecord(
            session_id="20260218_002",
            session_number=2,
            start_ts="2026-02-18T00:00:00Z",
            best_max_drawdown=-0.25,
            best_annual_return=0.42,
        )
        d = record.to_dict()
        assert d["best_max_drawdown"] == -0.25
        assert d["best_annual_return"] == 0.42
        restored = SessionRecord.from_dict(d)
        assert restored.best_max_drawdown == -0.25
        assert restored.best_annual_return == 0.42

    def test_new_fields_default_none(self):
        record = SessionRecord(session_id="s1", session_number=1, start_ts="t")
        assert record.best_max_drawdown is None
        assert record.best_annual_return is None

    def test_from_dict_ignores_unknown_fields(self):
        d = {"session_id": "s1", "session_number": 1, "start_ts": "t", "unknown_field": 42}
        record = SessionRecord.from_dict(d)
        assert record.session_id == "s1"


# ── Jaccard tests ─────────────────────────────────────────────────────────


class TestJaccardSimilarity:
    def test_identical_configs(self):
        cfg = {"a": 1, "b": 2}
        assert jaccard_similarity(cfg, cfg) == 1.0

    def test_disjoint_configs(self):
        assert jaccard_similarity({"a": 1}, {"b": 2}) == 0.0

    def test_partial_overlap(self):
        cfg_a = {"a": 1, "b": 2, "c": 3}
        cfg_b = {"a": 1, "b": 2, "d": 4}
        # intersection: {(a,1), (b,2)} = 2, union = 4
        assert jaccard_similarity(cfg_a, cfg_b) == 2.0 / 4.0

    def test_empty_configs(self):
        assert jaccard_similarity({}, {}) == 0.0

    def test_mean_pairwise_single(self):
        assert mean_pairwise_jaccard([{"a": 1}]) == 0.0

    def test_mean_pairwise_identical(self):
        configs = [{"a": 1, "b": 2}] * 3
        assert mean_pairwise_jaccard(configs) == 1.0

    def test_mean_pairwise_mixed(self):
        configs = [{"a": 1}, {"a": 1, "b": 2}, {"c": 3}]
        # (0,1): {(a,1)} / {(a,1),(b,2)} = 1/2
        # (0,2): {} / {(a,1),(c,3)} = 0
        # (1,2): {} / {(a,1),(b,2),(c,3)} = 0
        expected = (0.5 + 0 + 0) / 3
        assert abs(mean_pairwise_jaccard(configs) - expected) < 1e-9


# ── Stall Detection tests ────────────────────────────────────────────────


class TestStallDetection:
    def _make_orchestrator(self, tmp_path) -> ResearchOrchestrator:
        directive = _make_directive(
            stopping_criteria=StoppingCriteria(
                improvement_threshold=0.05,
                diversity_threshold=0.80,
                sessions_without_improvement=3,
            )
        )
        return ResearchOrchestrator(directive, state_dir=tmp_path)

    def test_sharpe_stall_increments(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        state = OrchestratorState(name="test", best_result={"sharpe": 1.0})

        record = SessionRecord(
            session_id="s1",
            session_number=1,
            start_ts="t",
            best_sharpe=1.02,  # only 0.02 improvement, below 0.05 threshold
        )
        orch._update_stall(state, record)
        assert state.stall_counter == 1

    def test_sharpe_improvement_resets(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        state = OrchestratorState(name="test", best_result={"sharpe": 1.0}, stall_counter=2)

        record = SessionRecord(
            session_id="s1",
            session_number=1,
            start_ts="t",
            best_sharpe=1.10,  # 0.10 improvement, above threshold
        )
        orch._update_stall(state, record)
        assert state.stall_counter == 0

    def test_diversity_stall(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        # Last 3 sessions all have identical configs → high Jaccard
        identical_cfg = {"lr": 0.01, "depth": 6, "n_estimators": 200}
        state = OrchestratorState(
            name="test",
            best_result={"sharpe": 1.0},
            sessions=[
                SessionRecord(
                    session_id=f"s{i}",
                    session_number=i,
                    start_ts="t",
                    wandb_run_configs=[identical_cfg],
                )
                for i in range(1, 4)
            ],
        )
        # This session also has similar configs but no sharpe improvement
        record = SessionRecord(
            session_id="s4",
            session_number=4,
            start_ts="t",
            best_sharpe=1.02,
            wandb_run_configs=[identical_cfg],
        )
        # Append the new session first so _update_stall sees it in state.sessions
        state.sessions.append(record)
        orch._update_stall(state, record)
        assert state.stall_counter == 1  # Both sharpe stall and diversity stall


# ── Update Best tests ────────────────────────────────────────────────────


class TestUpdateBest:
    def test_stores_max_drawdown_and_annual_return(self, tmp_path):
        directive = _make_directive()
        orch = ResearchOrchestrator(directive, state_dir=tmp_path)
        state = OrchestratorState(name="test")

        record = SessionRecord(
            session_id="s1",
            session_number=1,
            start_ts="t",
            best_sharpe=1.5,
            best_dsr=0.98,
            best_max_drawdown=-0.25,
            best_annual_return=0.42,
            wandb_run_ids=["r1"],
        )
        orch._update_best(state, record)
        assert state.best_result["sharpe"] == 1.5
        assert state.best_result["max_drawdown"] == -0.25
        assert state.best_result["annual_return"] == 0.42

    def test_none_fields_when_not_provided(self, tmp_path):
        directive = _make_directive()
        orch = ResearchOrchestrator(directive, state_dir=tmp_path)
        state = OrchestratorState(name="test")

        record = SessionRecord(
            session_id="s1",
            session_number=1,
            start_ts="t",
            best_sharpe=1.0,
            wandb_run_ids=["r1"],
        )
        orch._update_best(state, record)
        assert state.best_result["max_drawdown"] is None
        assert state.best_result["annual_return"] is None


# ── Crash Loop Protection tests ──────────────────────────────────────────


class TestCrashLoopProtection:
    @patch("sparky.workflow.orchestrator.launch_claude_session")
    @patch("sparky.workflow.orchestrator.send_alert")
    @patch("time.sleep")
    def test_fast_exit_increments_counter(self, mock_sleep, mock_alert, mock_launch, tmp_path):
        directive = _make_directive(
            session_limits=SessionLimits(
                min_session_minutes=5.0,
                max_consecutive_crashes=3,
            ),
            stopping_criteria=StoppingCriteria(max_sessions=10),
        )
        orch = ResearchOrchestrator(directive, state_dir=tmp_path, log_dir=tmp_path / "logs")

        # First session: fast exit (1 min < 5 min threshold)
        # Second session: also fast exit
        # Third session: crash limit reached → pause
        fast_telemetry = _make_telemetry(duration_minutes=1.0)
        mock_launch.return_value = fast_telemetry

        with patch.object(
            orch,
            "_query_session_results",
            return_value={"best_sharpe": None, "best_dsr": None, "run_ids": [], "configs": []},
        ):
            orch.run()

        state = OrchestratorState.load("test_directive", tmp_path)
        assert state.status == "paused"
        assert state.crash_counter >= 3

    @patch("sparky.workflow.orchestrator.launch_claude_session")
    @patch("sparky.workflow.orchestrator.send_alert")
    @patch("time.sleep")
    def test_successful_session_resets_crash_counter(self, mock_sleep, mock_alert, mock_launch, tmp_path):
        directive = _make_directive(
            session_limits=SessionLimits(
                min_session_minutes=5.0,
                max_consecutive_crashes=5,
            ),
            stopping_criteria=StoppingCriteria(max_sessions=2),
        )
        orch = ResearchOrchestrator(directive, state_dir=tmp_path, log_dir=tmp_path / "logs")

        # First session: fast exit, second: normal → resets counter, then stops (max_sessions=2)
        fast = _make_telemetry(duration_minutes=1.0)
        normal = _make_telemetry(duration_minutes=30.0)
        mock_launch.side_effect = [fast, normal]

        with patch.object(
            orch,
            "_query_session_results",
            return_value={"best_sharpe": None, "best_dsr": None, "run_ids": [], "configs": []},
        ):
            orch.run()

        state = OrchestratorState.load("test_directive", tmp_path)
        assert state.crash_counter == 0
        assert state.session_count == 2

    @patch("sparky.workflow.orchestrator.launch_claude_session")
    @patch("sparky.workflow.orchestrator.send_alert")
    @patch("time.sleep")
    def test_backoff_doubles(self, mock_sleep, mock_alert, mock_launch, tmp_path):
        directive = _make_directive(
            session_limits=SessionLimits(
                min_session_minutes=5.0,
                max_consecutive_crashes=5,
            ),
            stopping_criteria=StoppingCriteria(max_sessions=3),
        )
        orch = ResearchOrchestrator(directive, state_dir=tmp_path, log_dir=tmp_path / "logs")

        fast = _make_telemetry(duration_minutes=1.0)
        mock_launch.return_value = fast

        with patch.object(
            orch,
            "_query_session_results",
            return_value={"best_sharpe": None, "best_dsr": None, "run_ids": [], "configs": []},
        ):
            orch.run()

        state = OrchestratorState.load("test_directive", tmp_path)
        # After 3 crashes: 120→240→480→960
        assert state.crash_backoff_seconds > 120


# ── Cost Tracking tests ──────────────────────────────────────────────────


class TestCostTracking:
    @patch("sparky.workflow.orchestrator.launch_claude_session")
    @patch("sparky.workflow.orchestrator.send_alert")
    @patch("time.sleep")
    def test_cost_accumulates(self, mock_sleep, mock_alert, mock_launch, tmp_path):
        directive = _make_directive(
            stopping_criteria=StoppingCriteria(max_sessions=2, max_cost_usd=100.0),
        )
        orch = ResearchOrchestrator(directive, state_dir=tmp_path, log_dir=tmp_path / "logs")

        t = _make_telemetry(duration_minutes=30.0)
        t.estimated_cost_usd = 2.50
        mock_launch.return_value = t

        with patch.object(
            orch,
            "_query_session_results",
            return_value={"best_sharpe": None, "best_dsr": None, "run_ids": [], "configs": []},
        ):
            orch.run()

        state = OrchestratorState.load("test_directive", tmp_path)
        assert state.total_cost_usd == pytest.approx(5.0, abs=0.01)  # 2 sessions × $2.50
        assert state.total_hours == pytest.approx(1.0, abs=0.01)  # 2 × 30min


# ── Lockfile tests ────────────────────────────────────────────────────────


class TestLockfile:
    def test_acquire_creates_file(self, tmp_path):
        directive = _make_directive()
        orch = ResearchOrchestrator(directive, state_dir=tmp_path)
        orch._acquire_lock()
        assert orch._lockfile.exists()
        assert int(orch._lockfile.read_text().strip()) == os.getpid()
        orch._release_lock()

    def test_stale_pid_overwritten(self, tmp_path):
        directive = _make_directive()
        orch = ResearchOrchestrator(directive, state_dir=tmp_path)
        # Write a PID that definitely doesn't exist
        tmp_path.mkdir(parents=True, exist_ok=True)
        orch._lockfile.write_text("99999999")
        orch._acquire_lock()  # should succeed (stale lock)
        assert int(orch._lockfile.read_text().strip()) == os.getpid()
        orch._release_lock()

    def test_active_pid_raises(self, tmp_path):
        directive = _make_directive()
        orch = ResearchOrchestrator(directive, state_dir=tmp_path)
        # Write our own PID — simulates active process
        tmp_path.mkdir(parents=True, exist_ok=True)
        orch._lockfile.write_text(str(os.getpid()))
        with pytest.raises(RuntimeError, match="already running"):
            orch._acquire_lock()

    def test_release_cleans_up(self, tmp_path):
        directive = _make_directive()
        orch = ResearchOrchestrator(directive, state_dir=tmp_path)
        orch._acquire_lock()
        orch._release_lock()
        assert not orch._lockfile.exists()


# ── Session Prompt tests ──────────────────────────────────────────────────


class TestSessionPrompt:
    def test_exit_protocol_and_duration_present(self):
        directive = _make_directive()
        orch = ResearchOrchestrator(directive)
        prompt = orch._build_session_prompt(1, "")
        assert "GATE_REQUEST.md" in prompt
        assert "When to Exit" in prompt
        assert "Session Duration" in prompt
        assert "Do NOT exit after running one sweep" in prompt
        assert "RESEARCH_AGENT.md" in prompt

    def test_session_tag_injected(self):
        directive = _make_directive(wandb_tags=["my_tag"])
        orch = ResearchOrchestrator(directive)
        prompt = orch._build_session_prompt(3, "")
        assert "session_003" in prompt
        assert "my_tag" in prompt

    def test_context_injected(self):
        directive = _make_directive()
        orch = ResearchOrchestrator(directive)
        prompt = orch._build_session_prompt(2, "## Previous Results\nSharpe 0.85")
        assert "Previous Results" in prompt
        assert "Sharpe 0.85" in prompt

    def test_exclusions_present(self):
        directive = _make_directive(exclusions=["Do not use OOS data", "No paper trading"])
        orch = ResearchOrchestrator(directive)
        prompt = orch._build_session_prompt(1, "")
        assert "Do not use OOS data" in prompt
        assert "No paper trading" in prompt

    def test_constraints_present(self):
        directive = _make_directive(constraints={"asset": "btc", "timeframe": "1h"})
        orch = ResearchOrchestrator(directive)
        prompt = orch._build_session_prompt(1, "")
        assert "asset: btc" in prompt
        assert "timeframe: 1h" in prompt

    def test_strategy_space_in_prompt(self):
        directive = _make_directive(
            strategy_space=[
                StrategySpec(family="momentum_regime", description="Momentum with filter", priority=1),
            ]
        )
        orch = ResearchOrchestrator(directive)
        prompt = orch._build_session_prompt(1, "")
        assert "momentum_regime" in prompt


# ── ContextBuilder tests ──────────────────────────────────────────────────


class TestContextBuilder:
    def test_empty_wandb_no_crash(self):
        directive = _make_directive()
        state = OrchestratorState(name="test")
        builder = ContextBuilder(directive, state)
        context = builder.build()
        assert "No results yet" in context

    def test_with_sessions_shows_last(self):
        directive = _make_directive()
        state = OrchestratorState(name="test")
        state.sessions.append(
            SessionRecord(
                session_id="s1",
                session_number=1,
                start_ts="t",
                best_sharpe=0.85,
                duration_minutes=25.0,
            )
        )
        builder = ContextBuilder(directive, state)
        context = builder.build()
        assert "Last session" in context
        assert "0.850" in context

    def test_stall_warning(self):
        directive = _make_directive()
        state = OrchestratorState(name="test", stall_counter=3)
        builder = ContextBuilder(directive, state)
        context = builder.build()
        assert "Stall warning" in context

    def test_overall_best_shown(self):
        directive = _make_directive()
        state = OrchestratorState(name="test", best_result={"sharpe": 1.05, "dsr": 0.97})
        builder = ContextBuilder(directive, state)
        context = builder.build()
        assert "1.050" in context
        assert "0.970" in context

    def test_coverage_hint_with_ranges(self):
        directive = _make_directive(
            strategy_space=[
                StrategySpec(
                    family="test",
                    parameter_ranges={"lr": [0.01, 0.05, 0.1]},
                )
            ]
        )
        state = OrchestratorState(name="test")
        # No sessions yet, so 0% coverage
        builder = ContextBuilder(directive, state)
        context = builder.build()
        assert "lr:" in context
        assert "0%" in context

    def test_coverage_partial(self):
        directive = _make_directive(
            strategy_space=[
                StrategySpec(
                    family="test",
                    parameter_ranges={"lr": [0.01, 0.05, 0.1]},
                )
            ]
        )
        state = OrchestratorState(name="test")
        state.sessions.append(
            SessionRecord(
                session_id="s1",
                session_number=1,
                start_ts="t",
                wandb_run_configs=[{"lr": 0.01}],
            )
        )
        builder = ContextBuilder(directive, state)
        context = builder.build()
        # 1 of 3 covered = 33%
        assert "33%" in context


# ── ResearchOrchestrator integration tests ────────────────────────────────


class TestResearchOrchestrator:
    @patch("sparky.workflow.orchestrator.launch_claude_session")
    @patch("sparky.workflow.orchestrator.send_alert")
    @patch("time.sleep")
    def test_success_halt(self, mock_sleep, mock_alert, mock_launch, tmp_path):
        directive = _make_directive(
            stopping_criteria=StoppingCriteria(
                success_min_sharpe=1.0,
                success_min_dsr=0.95,
                max_sessions=10,
            ),
        )
        orch = ResearchOrchestrator(directive, state_dir=tmp_path, log_dir=tmp_path / "logs")

        t = _make_telemetry(duration_minutes=30.0)
        mock_launch.return_value = t

        # Session returns Sharpe=1.1, DSR=0.97 → success
        with patch.object(
            orch,
            "_query_session_results",
            return_value={"best_sharpe": 1.1, "best_dsr": 0.97, "run_ids": ["r1"], "configs": [{"lr": 0.01}]},
        ):
            result = orch.run()

        assert result == 0
        state = OrchestratorState.load("test_directive", tmp_path)
        assert state.status == "done"
        # Should have sent alert about success
        alert_msgs = [str(c) for c in mock_alert.call_args_list]
        assert any("SUCCESS" in m for m in alert_msgs)

    @patch("sparky.workflow.orchestrator.launch_claude_session")
    @patch("sparky.workflow.orchestrator.send_alert")
    @patch("time.sleep")
    def test_stop_on_success_false_continues(self, mock_sleep, mock_alert, mock_launch, tmp_path):
        """With stop_on_success=False, success criteria don't halt the loop."""
        directive = _make_directive(
            stopping_criteria=StoppingCriteria(
                stop_on_success=False,
                success_min_sharpe=1.0,
                success_min_dsr=0.95,
                max_sessions=3,
            ),
        )
        orch = ResearchOrchestrator(directive, state_dir=tmp_path, log_dir=tmp_path / "logs")

        t = _make_telemetry(duration_minutes=10.0)
        mock_launch.return_value = t

        # Every session returns metrics above success thresholds
        with patch.object(
            orch,
            "_query_session_results",
            return_value={"best_sharpe": 1.5, "best_dsr": 0.99, "run_ids": ["r1"], "configs": [{"lr": 0.01}]},
        ):
            result = orch.run()

        assert result == 0
        state = OrchestratorState.load("test_directive", tmp_path)
        # Should have run all 3 sessions (budget halt), not stopped at 1 (success)
        assert state.session_count == 3
        alert_msgs = [str(c) for c in mock_alert.call_args_list]
        assert not any("SUCCESS" in m for m in alert_msgs)

    @patch("sparky.workflow.orchestrator.launch_claude_session")
    @patch("sparky.workflow.orchestrator.send_alert")
    @patch("time.sleep")
    def test_budget_halt(self, mock_sleep, mock_alert, mock_launch, tmp_path):
        directive = _make_directive(
            stopping_criteria=StoppingCriteria(max_sessions=2),
        )
        orch = ResearchOrchestrator(directive, state_dir=tmp_path, log_dir=tmp_path / "logs")

        t = _make_telemetry(duration_minutes=30.0)
        mock_launch.return_value = t

        with patch.object(
            orch,
            "_query_session_results",
            return_value={"best_sharpe": 0.5, "best_dsr": 0.3, "run_ids": [], "configs": []},
        ):
            result = orch.run()

        assert result == 0
        state = OrchestratorState.load("test_directive", tmp_path)
        assert state.status == "done"
        assert state.session_count == 2

    @patch("sparky.workflow.orchestrator.launch_claude_session")
    @patch("sparky.workflow.orchestrator.send_alert")
    @patch("time.sleep")
    def test_stall_halt(self, mock_sleep, mock_alert, mock_launch, tmp_path):
        directive = _make_directive(
            stopping_criteria=StoppingCriteria(
                max_sessions=20,
                sessions_without_improvement=2,
                improvement_threshold=0.05,
            ),
        )
        orch = ResearchOrchestrator(directive, state_dir=tmp_path, log_dir=tmp_path / "logs")

        t = _make_telemetry(duration_minutes=30.0)
        mock_launch.return_value = t

        # All sessions return same Sharpe → stall after 2
        with patch.object(
            orch,
            "_query_session_results",
            return_value={"best_sharpe": 0.5, "best_dsr": 0.3, "run_ids": [], "configs": []},
        ):
            result = orch.run()

        assert result == 0
        state = OrchestratorState.load("test_directive", tmp_path)
        assert state.status == "done"
        assert state.stall_counter >= 2

    @patch("sparky.workflow.orchestrator.launch_claude_session")
    @patch("sparky.workflow.orchestrator.send_alert")
    @patch("time.sleep")
    def test_gate_request_pauses(self, mock_sleep, mock_alert, mock_launch, tmp_path):
        directive = _make_directive(
            stopping_criteria=StoppingCriteria(max_sessions=10),
        )
        orch = ResearchOrchestrator(directive, state_dir=tmp_path, log_dir=tmp_path / "logs")

        t = _make_telemetry(duration_minutes=30.0)
        mock_launch.return_value = t

        call_count = [0]

        def _mock_query(*a, **kw):
            call_count[0] += 1
            # After first session, create GATE_REQUEST.md
            if call_count[0] == 1:
                from sparky.workflow.orchestrator import GATE_REQUEST_PATH

                GATE_REQUEST_PATH.write_text("Need human input on next strategy direction")
            return {"best_sharpe": 0.5, "best_dsr": 0.3, "run_ids": [], "configs": []}

        with patch.object(orch, "_query_session_results", side_effect=_mock_query):
            try:
                result = orch.run()
            finally:
                # Clean up gate file
                from sparky.workflow.orchestrator import GATE_REQUEST_PATH

                GATE_REQUEST_PATH.unlink(missing_ok=True)

        assert result == 0
        state = OrchestratorState.load("test_directive", tmp_path)
        assert state.status == "gate_triggered"
        assert "Need human input" in (state.gate_message or "")

    @patch("sparky.workflow.orchestrator.launch_claude_session")
    @patch("sparky.workflow.orchestrator.send_alert")
    @patch("time.sleep")
    def test_digest_alert_fires(self, mock_sleep, mock_alert, mock_launch, tmp_path):
        directive = _make_directive(
            stopping_criteria=StoppingCriteria(max_sessions=6, digest_every=3),
        )
        orch = ResearchOrchestrator(directive, state_dir=tmp_path, log_dir=tmp_path / "logs")

        t = _make_telemetry(duration_minutes=30.0)
        mock_launch.return_value = t

        with patch.object(
            orch,
            "_query_session_results",
            return_value={"best_sharpe": None, "best_dsr": None, "run_ids": [], "configs": []},
        ):
            orch.run()

        # Should have digest alerts at session 3 and 6
        alert_msgs = [str(c) for c in mock_alert.call_args_list]
        assert any("Digest" in m for m in alert_msgs)

    @patch("sparky.workflow.orchestrator.launch_claude_session")
    @patch("sparky.workflow.orchestrator.send_alert")
    @patch("time.sleep")
    def test_concurrent_run_blocked(self, mock_sleep, mock_alert, mock_launch, tmp_path):
        directive = _make_directive()
        orch = ResearchOrchestrator(directive, state_dir=tmp_path, log_dir=tmp_path / "logs")

        # Simulate an active lock with our own PID
        tmp_path.mkdir(parents=True, exist_ok=True)
        orch._lockfile.write_text(str(os.getpid()))

        result = orch.run()
        assert result == 1  # should fail to acquire lock


class TestQuerySessionResults:
    """Tests for _query_session_results tag filtering."""

    def test_filters_by_directive_tags(self, tmp_path):
        """_query_session_results must filter by directive wandb_tags to prevent cross-experiment contamination."""
        from unittest.mock import MagicMock

        directive = _make_directive(wandb_tags=["broad_exploration", "20260218"])
        orch = ResearchOrchestrator(directive, state_dir=tmp_path)

        mock_run = MagicMock()
        mock_run.id = "run123"
        mock_run.config = {"lr": 0.01}
        mock_run.summary = {"sharpe": 1.5, "dsr": 0.9}

        with patch("sparky.tracking.experiment.ExperimentTracker._fetch_runs", return_value=[mock_run]) as mock_fetch:
            orch._query_session_results("session_001")
            call_filters = mock_fetch.call_args[1].get("filters") or mock_fetch.call_args[0][0]
            required = call_filters["tags"]["$all"]
            assert "session_001" in required
            assert "broad_exploration" in required
            assert "20260218" in required


class TestReconstructFromWandb:
    """Tests for OrchestratorState.reconstruct_from_wandb."""

    def _make_mock_run(self, session_num: int, sharpe: float, dsr: float):
        """Create a minimal mock wandb run."""
        from unittest.mock import MagicMock

        run = MagicMock()
        run.id = f"run_{session_num:03d}"
        run.tags = [f"session_{session_num:03d}", "test_directive"]
        run.config = {"transaction_costs_bps": 50, "model": "xgb"}
        run.summary = {"sharpe": sharpe, "dsr": dsr}
        run.created_at = f"2024-0{session_num}-01T00:00:00Z"
        return run

    def test_returns_none_when_no_runs(self):
        """Reconstruction returns None when no matching runs exist."""
        with patch("sparky.tracking.experiment.ExperimentTracker._fetch_runs", return_value=[]):
            state = OrchestratorState.reconstruct_from_wandb("test_directive", ["test_directive"])
        assert state is None

    def test_returns_none_on_wandb_error(self):
        """Reconstruction returns None gracefully when wandb is unreachable."""
        with patch(
            "sparky.tracking.experiment.ExperimentTracker._fetch_runs",
            side_effect=Exception("wandb connection refused"),
        ):
            state = OrchestratorState.reconstruct_from_wandb("test_directive", ["test_directive"])
        assert state is None

    def test_reconstructs_sessions_from_runs(self):
        """State is reconstructed with correct session count and best metrics."""
        runs = [
            self._make_mock_run(1, sharpe=0.8, dsr=0.72),
            self._make_mock_run(2, sharpe=1.2, dsr=0.91),
            self._make_mock_run(3, sharpe=1.1, dsr=0.88),
        ]
        with patch("sparky.tracking.experiment.ExperimentTracker._fetch_runs", return_value=runs):
            state = OrchestratorState.reconstruct_from_wandb("test_directive", ["test_directive"])

        assert state is not None
        assert state.session_count == 3
        assert len(state.sessions) == 3
        assert state.best_result["sharpe"] == pytest.approx(1.2)
        assert state.best_result["dsr"] == pytest.approx(0.91)

    def test_falls_back_to_best_sharpe_key(self):
        """Reconstruction also reads best_sharpe/best_dsr wandb keys."""
        from unittest.mock import MagicMock

        run = MagicMock()
        run.id = "run_001"
        run.tags = ["session_001", "test_directive"]
        run.config = {}
        run.summary = {"best_sharpe": 1.5, "best_dsr": 0.96}  # no "sharpe"/"dsr"
        run.created_at = "2024-01-01T00:00:00Z"

        with patch("sparky.tracking.experiment.ExperimentTracker._fetch_runs", return_value=[run]):
            state = OrchestratorState.reconstruct_from_wandb("test_directive", ["test_directive"])

        assert state is not None
        assert state.best_result["sharpe"] == pytest.approx(1.5)
