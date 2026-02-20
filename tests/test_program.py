"""Tests for the research program module."""

import json

import pytest
import yaml

from sparky.workflow.program import (
    PhaseConfig,
    PhaseState,
    ResearchProgram,
    _safe_float,
    _safe_get,
    _safe_int,
    check_stall,
    evaluate_coverage,
    evaluate_phase_transition,
    extract_coverage,
    format_coverage_gaps,
    read_core_memory,
)


def _make_program_yaml(tmp_path, overrides=None):
    """Write a minimal program YAML and return the path."""
    base = {
        "program": {
            "name": "test_program",
            "version": 1,
            "budget_cap_usd": 50.0,
            "max_calendar_days": 7,
            "success_definition": "Test",
            "phases": {
                "explore": {
                    "objective": "Explore strategies",
                    "min_sessions": 2,
                    "max_sessions": 10,
                    "coverage_requirements": {
                        "min_families_screened": 5,
                        "min_total_configs": 100,
                    },
                    "exit_criteria": {
                        "coverage_requirements_met": True,
                        "min_tier1_candidates": 3,
                    },
                    "next_phase": "validate",
                    "session_minutes": 120,
                    "stall_policy": {
                        "sessions_without_new_tier1": 4,
                        "on_stall": "proceed_with_available",
                    },
                },
                "validate": {
                    "objective": "Validate candidates",
                    "min_sessions": 1,
                    "max_sessions": 5,
                    "human_review": "required",
                    "next_phase": None,
                },
            },
        },
    }
    if overrides:
        base["program"].update(overrides)

    path = tmp_path / "test_program.yaml"
    with open(path, "w") as f:
        yaml.dump(base, f)
    return path


# ── Parsing ──────────────────────────────────────────────────────────────


class TestProgramParsing:
    def test_parse_full_program_yaml(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        assert prog.name == "test_program"
        assert prog.version == 1
        assert "explore" in prog.phases
        assert "validate" in prog.phases
        assert isinstance(prog.phases["explore"], PhaseConfig)

    def test_phase_order_preserved(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        assert prog.phase_order == ["explore", "validate"]

    def test_missing_program_key_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump({"not_program": {}}, f)
        with pytest.raises(ValueError, match="program"):
            ResearchProgram.from_yaml(path)

    def test_missing_name_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump({"program": {"phases": {"a": {}}}}, f)
        with pytest.raises(ValueError, match="name"):
            ResearchProgram.from_yaml(path)

    def test_missing_phases_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump({"program": {"name": "x"}}, f)
        with pytest.raises(ValueError, match="phases"):
            ResearchProgram.from_yaml(path)

    def test_phase_defaults(self, tmp_path):
        path = tmp_path / "minimal.yaml"
        with open(path, "w") as f:
            yaml.dump(
                {"program": {"name": "x", "phases": {"only": {"objective": "test"}}}},
                f,
            )
        prog = ResearchProgram.from_yaml(path)
        pc = prog.phases["only"]
        assert pc.human_review == "none"
        assert pc.agents == 1
        assert pc.min_sessions == 1
        assert pc.max_sessions == 20
        assert pc.session_minutes == 360
        assert pc.next_phase is None  # last phase

    def test_invalid_next_phase_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(
                {
                    "program": {
                        "name": "x",
                        "phases": {
                            "a": {"next_phase": "nonexistent"},
                        },
                    }
                },
                f,
            )
        with pytest.raises(ValueError, match="nonexistent"):
            ResearchProgram.from_yaml(path)

    def test_to_directive(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        directive = prog.to_directive("explore")
        assert directive.name == "test_program"
        assert directive.objective == "Explore strategies"
        assert directive.stopping_criteria.max_cost_usd == 50.0
        assert directive.session_limits.max_session_minutes == 120
        assert "program_test_program" in directive.wandb_tags


# ── PhaseState ───────────────────────────────────────────────────────────


class TestPhaseState:
    def test_roundtrip(self):
        ps = PhaseState(
            current_phase="explore",
            phase_session_count=3,
            coverage_status={"a": 1},
            phase_history=[{"phase": "init"}],
            pending_human_review=True,
        )
        d = ps.to_dict()
        ps2 = PhaseState.from_dict(d)
        assert ps2.current_phase == "explore"
        assert ps2.phase_session_count == 3
        assert ps2.coverage_status == {"a": 1}
        assert ps2.phase_history == [{"phase": "init"}]
        assert ps2.pending_human_review is True

    def test_from_dict_defaults(self):
        ps = PhaseState.from_dict({})
        assert ps.current_phase == ""
        assert ps.phase_session_count == 0
        assert ps.coverage_status == {}
        assert ps.phase_history == []
        assert ps.pending_human_review is False

    def test_from_dict_non_dict_raises(self):
        with pytest.raises(ValueError, match="dict"):
            PhaseState.from_dict("not a dict")


# ── Phase Transitions ────────────────────────────────────────────────────


class TestPhaseTransitions:
    def test_no_transition_before_min_sessions(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        ps = PhaseState(current_phase="explore", phase_session_count=1)
        result = evaluate_phase_transition(prog, ps, {})
        assert result is None

    def test_force_transition_at_max_sessions(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        ps = PhaseState(current_phase="explore", phase_session_count=10)
        result = evaluate_phase_transition(prog, ps, {})
        assert result == "validate"

    def test_transition_when_coverage_met(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        core = {
            "coverage": {
                "families_screened": {f"f{i}": {"status": "done"} for i in range(5)},
                "total_configs_tested": 150,
            },
            "top_candidates": [{"id": f"c{i}", "sharpe": 1.0} for i in range(3)],
        }
        ps = PhaseState(current_phase="explore", phase_session_count=3)
        result = evaluate_phase_transition(prog, ps, core)
        assert result == "validate"

    def test_no_transition_coverage_not_met(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        core = {
            "coverage": {
                "families_screened": {"f0": {"status": "done"}},
                "total_configs_tested": 10,
            },
        }
        ps = PhaseState(current_phase="explore", phase_session_count=3)
        result = evaluate_phase_transition(prog, ps, core)
        assert result is None

    def test_human_review_blocks_transition(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        # validate phase has human_review="required" and no coverage_requirements
        ps = PhaseState(current_phase="validate", phase_session_count=2)
        result = evaluate_phase_transition(prog, ps, {})
        assert result is None
        assert ps.pending_human_review is True

    def test_terminal_phase_max_sessions(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        # validate has next_phase=None, at max_sessions=5
        ps = PhaseState(current_phase="validate", phase_session_count=5)
        result = evaluate_phase_transition(prog, ps, {})
        assert result is None  # next_phase is None


# ── Coverage Tracking ────────────────────────────────────────────────────


class TestCoverageTracking:
    def test_extract_explore_coverage(self):
        core = {
            "coverage": {
                "families_screened": {
                    "donchian": {"status": "done", "round": 2},
                    "momentum": {"status": "null_result", "round": 1},
                    "mean_revert": {"status": "done", "round": 3},
                },
                "total_configs_tested": 42,
            },
            "top_candidates": [{"id": "btc_don", "sharpe": 1.2}],
        }
        cov = extract_coverage(core, "explore")
        assert cov["min_families_screened"] == 3
        assert cov["min_families_with_null_result"] == 1
        assert cov["min_families_deep_explored"] == 2  # round >= 2
        assert cov["min_total_configs"] == 42
        assert cov["min_tier1_candidates"] == 1

    def test_extract_explore_coverage_empty(self):
        cov = extract_coverage({}, "explore")
        assert cov["min_families_screened"] == 0
        assert cov["min_total_configs"] == 0
        assert cov["min_tier1_candidates"] == 0

    def test_extract_investigate_coverage(self):
        core = {
            "top_candidates": [
                {"id": "c1", "investigation_complete": True},
                {"id": "c2", "investigation_complete": False},
            ],
        }
        cov = extract_coverage(core, "investigate")
        assert cov["investigated_count"] == 1
        assert cov["total_candidates"] == 2
        assert cov["all_tier1_candidates_investigated"] is False

    def test_extract_validate_coverage(self):
        core = {
            "candidates": {
                "c1": {"status": "validated_pass"},
                "c2": {"status": "validated_fail"},
                "c3": {"status": "pending"},
            },
        }
        cov = extract_coverage(core, "validate")
        assert cov["all_candidates_tested"] is False  # 2 tested out of 3
        assert cov["min_passing_candidates"] == 1

    def test_evaluate_coverage_all_met(self):
        coverage = {"min_families_screened": 5, "min_total_configs": 100}
        requirements = {"min_families_screened": 5, "min_total_configs": 100}
        met, status = evaluate_coverage(coverage, requirements)
        assert met is True
        assert all(v["met"] for v in status.values())

    def test_evaluate_coverage_partial(self):
        coverage = {"min_families_screened": 3, "min_total_configs": 200}
        requirements = {"min_families_screened": 5, "min_total_configs": 100}
        met, status = evaluate_coverage(coverage, requirements)
        assert met is False
        assert status["min_families_screened"]["met"] is False
        assert status["min_total_configs"]["met"] is True

    def test_format_coverage_gaps_all_met(self):
        status = {
            "a": {"required": 5, "current": 5, "met": True},
            "b": {"required": 10, "current": 20, "met": True},
        }
        result = format_coverage_gaps(status)
        assert result == "All coverage requirements met."

    def test_format_coverage_gaps_has_gaps(self):
        status = {
            "min_families_screened": {"required": 5, "current": 2, "met": False},
            "min_total_configs": {"required": 100, "current": 100, "met": True},
        }
        result = format_coverage_gaps(status)
        assert "Coverage Gaps" in result
        assert "2 / 5" in result
        assert "total_configs" not in result  # met, shouldn't appear


# ── Core Memory Resilience ───────────────────────────────────────────────


class TestCoreMemoryResilience:
    def test_read_missing_file(self):
        result = read_core_memory("/nonexistent/path/core.json")
        assert result == {}

    def test_read_corrupt_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        assert read_core_memory(path) == {}

    def test_read_non_dict_json(self, tmp_path):
        path = tmp_path / "array.json"
        path.write_text("[1, 2, 3]")
        assert read_core_memory(path) == {}

    def test_read_valid_json(self, tmp_path):
        path = tmp_path / "good.json"
        path.write_text(json.dumps({"key": "value"}))
        assert read_core_memory(path) == {"key": "value"}

    def test_extract_coverage_wrong_types(self):
        core = {"coverage": {"families_screened": "not_a_dict", "total_configs_tested": "xyz"}}
        cov = extract_coverage(core, "explore")
        assert cov["min_families_screened"] == 0
        assert cov["min_total_configs"] == 0

    def test_extract_coverage_extra_fields(self):
        core = {
            "coverage": {
                "families_screened": {},
                "total_configs_tested": 5,
                "unexpected_key": "ignored",
            },
            "some_random_field": True,
        }
        cov = extract_coverage(core, "explore")
        assert cov["min_families_screened"] == 0
        assert cov["min_total_configs"] == 5


# ── Stall Policy ─────────────────────────────────────────────────────────


class TestStallPolicy:
    def test_stall_triggers_action(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        ps = PhaseState(current_phase="explore")
        result = check_stall(prog, ps, sessions_without_new_result=4)
        assert result == "proceed_with_available"

    def test_no_stall_below_threshold(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        ps = PhaseState(current_phase="explore")
        result = check_stall(prog, ps, sessions_without_new_result=2)
        assert result is None

    def test_stall_no_policy(self, tmp_path):
        path = _make_program_yaml(tmp_path)
        prog = ResearchProgram.from_yaml(path)
        ps = PhaseState(current_phase="validate")
        result = check_stall(prog, ps, sessions_without_new_result=100)
        assert result is None


# ── Backwards Compatibility ──────────────────────────────────────────────


class TestBackwardsCompatibility:
    def test_legacy_directive_still_works(self):
        from sparky.workflow.orchestrator import ResearchDirective

        path = "directives/archive/example.yaml"
        d = ResearchDirective.from_yaml(path)
        assert d.name == "btc_momentum_deep_dive"
        assert d.objective

    def test_orchestrator_state_without_program_state(self, tmp_path):
        from sparky.workflow.orchestrator import OrchestratorState

        state = OrchestratorState(name="test")
        d = state.to_dict()
        assert d["program_state"] is None
        restored = OrchestratorState.from_dict(d)
        assert restored.program_state is None


# ── Safe Helpers ─────────────────────────────────────────────────────────


class TestSafeHelpers:
    def test_safe_get_nested(self):
        assert _safe_get({"a": {"b": 1}}, "a", "b") == 1

    def test_safe_get_missing(self):
        assert _safe_get({"a": 1}, "b") is None

    def test_safe_get_deep_missing(self):
        assert _safe_get({"a": 1}, "a", "b") is None

    def test_safe_int_string(self):
        assert _safe_int("12") == 12

    def test_safe_int_none(self):
        assert _safe_int(None) == 0

    def test_safe_int_garbage(self):
        assert _safe_int("abc") == 0

    def test_safe_float_string(self):
        assert _safe_float("1.5") == 1.5
