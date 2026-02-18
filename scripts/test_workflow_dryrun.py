#!/usr/bin/env python3
"""Dry-run Contract 005 workflow to validate engine + state management.

Exercises the full workflow machinery without spending API tokens by mocking
the Claude subprocess launch. Validates:
1. State file is created/updated correctly
2. done_when evaluation works
3. Step advancement logic works
4. Budget tracking works
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparky.workflow.engine import WorkflowState
from sparky.workflow.telemetry import SessionTelemetry


def load_contract_005_module():
    """Load the real contract 005 workflow module (for patching _RESULTS_DIR)."""
    spec = importlib.util.spec_from_file_location("c005", "workflows/contract_005_statistical_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_contract_005():
    """Load the real contract 005 workflow definition."""
    mod = load_contract_005_module()
    return mod.build_workflow()


def make_mock_telemetry(step_name, attempt, duration_minutes=5.0):
    """Create a fake telemetry result for a mocked session."""
    return SessionTelemetry(
        session_id=f"dryrun_{step_name}_{attempt}",
        step=step_name,
        attempt=attempt,
        started_at="2026-02-17T00:00:00+00:00",
        ended_at="2026-02-17T00:05:00+00:00",
        duration_minutes=duration_minutes,
        tokens_input=1000,
        tokens_output=500,
        tool_calls=5,
        estimated_cost_usd=0.05,
        exit_reason="completed",
    )


def test_workflow_loads():
    """Test that the workflow definition loads and has correct structure."""
    wf = load_contract_005()
    assert wf.name == "contract-005-statistical-audit", f"Bad name: {wf.name}"
    assert len(wf.steps) == 3, f"Expected 3 steps, got {len(wf.steps)}"
    assert wf.max_hours == 6.0, f"Expected 6h budget, got {wf.max_hours}"

    step_names = [s.name for s in wf.steps]
    assert step_names == ["audit_existing_runs", "validate_metrics_integration", "summary_report"]

    # Verify max durations
    assert wf.steps[0].max_duration_minutes == 60
    assert wf.steps[1].max_duration_minutes == 90
    assert wf.steps[2].max_duration_minutes == 30

    print("PASS: workflow loads correctly with 3 steps")


def test_state_creation():
    """Test that state is created correctly when workflow first runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = load_contract_005()
        wf.state_dir = Path(tmpdir)

        state = wf._load_or_create_state()

        assert state.workflow_name == "contract-005-statistical-audit"
        assert state.current_step_index == 0
        assert len(state.steps) == 3
        assert state.budget.max_hours == 6.0
        assert state.budget.hours_used == 0.0

        # Verify state file was written
        state_file = Path(tmpdir) / "contract-005-statistical-audit.json"
        assert state_file.exists(), "State file not created"

        # Verify JSON is valid
        with open(state_file) as f:
            data = json.load(f)
        assert data["workflow_name"] == "contract-005-statistical-audit"
        assert len(data["steps"]) == 3

        print("PASS: state creation and persistence works")


def test_done_when_evaluation():
    """Test that done_when functions work correctly via _RESULTS_DIR monkeypatch."""
    c005_mod = load_contract_005_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        results_dir.mkdir()

        # Monkeypatch _RESULTS_DIR instead of os.chdir
        c005_mod._RESULTS_DIR = results_dir
        try:
            wf = c005_mod.build_workflow()

            # Step 1 done_when checks for results/contract_005_audit.md
            assert not wf.steps[0].done_when(), "Step 1 done_when should be False initially"

            # Create the file
            (results_dir / "contract_005_audit.md").write_text("# Audit Report\n")
            assert wf.steps[0].done_when(), "Step 1 done_when should be True after file created"

            # Step 2
            assert not wf.steps[1].done_when(), "Step 2 done_when should be False"
            (results_dir / "contract_005_validation.md").write_text("# Validation\n")
            assert wf.steps[1].done_when(), "Step 2 done_when should be True after file created"

            # Step 3
            assert not wf.steps[2].done_when(), "Step 3 done_when should be False"
            (results_dir / "contract_005_summary.md").write_text("# Summary\n")
            assert wf.steps[2].done_when(), "Step 3 done_when should be True after file created"

            print("PASS: done_when evaluation works for all 3 steps")
        finally:
            # Restore default
            c005_mod._RESULTS_DIR = Path("results")


def test_step_advancement():
    """Test that completed steps advance the workflow correctly."""
    c005_mod = load_contract_005_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        results_dir.mkdir()

        # Monkeypatch _RESULTS_DIR instead of os.chdir
        c005_mod._RESULTS_DIR = results_dir
        try:
            wf = c005_mod.build_workflow()
            wf.state_dir = Path(tmpdir)

            # Create all done files so workflow completes instantly
            (results_dir / "contract_005_audit.md").write_text("# Audit\n")
            (results_dir / "contract_005_validation.md").write_text("# Validation\n")
            (results_dir / "contract_005_summary.md").write_text("# Summary\n")

            # Mock _alert and _get_best_result_summary to avoid side effects
            with patch.object(wf, "_alert"), patch.object(wf, "_get_best_result_summary", return_value="Best: N/A"):
                exit_code = wf.run()

            assert exit_code == 0, f"Workflow should complete with exit 0, got {exit_code}"

            # Verify all steps completed
            state = WorkflowState.load("contract-005-statistical-audit", Path(tmpdir))
            assert state is not None, "State should be loadable"
            assert state.current_step_index == 3, f"Should be at step 3, got {state.current_step_index}"

            for step_name, step_state in state.steps.items():
                assert step_state.status == "completed", (
                    f"Step {step_name} should be completed, got {step_state.status}"
                )

            print("PASS: step advancement with pre-satisfied done_when")
        finally:
            c005_mod._RESULTS_DIR = Path("results")


def test_budget_tracking():
    """Test that budget is tracked correctly during mock execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = load_contract_005()
        wf.state_dir = Path(tmpdir)

        state = wf._load_or_create_state()

        assert state.budget.max_hours == 6.0
        assert state.budget.hours_used == 0.0
        assert state.budget.estimated_cost_usd == 0.0
        assert state.budget.runs_completed == 0

        # Simulate some usage
        state.budget.hours_used = 2.5
        state.budget.estimated_cost_usd = 1.25
        state.budget.runs_completed = 3
        state.save(Path(tmpdir))

        # Reload and verify
        reloaded = WorkflowState.load("contract-005-statistical-audit", Path(tmpdir))
        assert reloaded.budget.hours_used == 2.5
        assert reloaded.budget.estimated_cost_usd == 1.25
        assert reloaded.budget.runs_completed == 3

        print("PASS: budget tracking and persistence works")


def main():
    """Run all dry-run tests."""
    print("=== Contract 005 Workflow Dry-Run ===\n")

    tests = [
        test_workflow_loads,
        test_state_creation,
        test_done_when_evaluation,
        test_step_advancement,
        test_budget_tracking,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} — {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")

    if failed > 0:
        sys.exit(1)
    else:
        print("Dry-run complete: workflow engine + state management validated")


if __name__ == "__main__":
    main()
