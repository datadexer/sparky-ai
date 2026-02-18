"""Tests for manager session tracker.

Tests JSONL write/read, serialization of all record types,
get_history ordering, and session lifecycle.
"""

import json

import pytest

from sparky.tracking.manager_log import (
    CodeAgentRecord,
    ContractDesignRecord,
    ManagerLog,
    ResearchLaunchRecord,
)


@pytest.fixture
def log_file(tmp_path):
    """Provide a temporary JSONL log file path."""
    return tmp_path / "manager_sessions" / "session_log.jsonl"


@pytest.fixture
def manager_log(log_file):
    """Create a ManagerLog with a temporary log file."""
    return ManagerLog(log_file=log_file)


class TestManagerSessionLifecycle:
    """Test the full session lifecycle: start → log events → end → read back."""

    def test_start_session(self, manager_log):
        session = manager_log.start_session("Test objective", "test-branch")
        assert session.objective == "Test objective"
        assert session.branch == "test-branch"
        assert session.session_id != ""
        assert session.started_at != ""

    def test_end_session_writes_jsonl(self, manager_log, log_file):
        session = manager_log.start_session("Test", "branch")
        manager_log.end_session(session, summary="Done")

        assert log_file.exists()
        with open(log_file) as f:
            entry = json.loads(f.readline())
        assert entry["objective"] == "Test"
        assert entry["summary"] == "Done"
        assert entry["ended_at"] != ""

    def test_full_lifecycle(self, manager_log, log_file):
        session = manager_log.start_session("Build infrastructure", "manager/test")

        # Log a code agent
        manager_log.log_code_agent(
            session,
            CodeAgentRecord(
                task="Build guardrails",
                model="sonnet",
                files_created=["guardrails.py"],
                tests_passed=True,
            ),
        )

        # Log a decision
        manager_log.log_decision(
            session,
            decision="Use Protocol",
            alternatives=["ABC", "Protocol"],
            rationale="Matches existing pattern",
        )

        # Log infrastructure
        manager_log.log_infrastructure(
            session,
            module="guardrails",
            purpose="Experiment quality control",
            files=["guardrails.py", "test_guardrails.py"],
            rationale="Prevent common mistakes",
        )

        # Log research launch
        manager_log.log_research_launch(
            session,
            ResearchLaunchRecord(
                workflow="contract_005",
                contract="005",
                branch="manager/test",
            ),
        )

        # Log contract design
        manager_log.log_contract_design(
            session,
            ContractDesignRecord(
                contract_name="Contract 005",
                objective="Statistical audit",
                steps=["audit", "validate", "report"],
                budget_hours=6.0,
                success_criteria="DSR computed for all runs",
            ),
        )

        manager_log.end_session(session, summary="All infrastructure built")

        # Read back
        with open(log_file) as f:
            entry = json.loads(f.readline())

        assert len(entry["code_agents"]) == 1
        assert entry["code_agents"][0]["task"] == "Build guardrails"
        assert len(entry["decisions"]) == 1
        assert entry["decisions"][0]["decision"] == "Use Protocol"
        assert len(entry["infrastructure"]) == 1
        assert len(entry["research_launches"]) == 1
        assert len(entry["contract_designs"]) == 1


class TestCodeAgentRecord:
    def test_defaults(self):
        record = CodeAgentRecord(task="Test task")
        assert record.model == "sonnet"
        assert record.files_created == []
        assert record.tests_passed is None

    def test_full_record(self):
        record = CodeAgentRecord(
            task="Build module",
            model="opus",
            files_created=["a.py", "b.py"],
            files_modified=["c.py"],
            tests_passed=True,
            duration_seconds=120.5,
            notes="All good",
        )
        assert record.task == "Build module"
        assert record.duration_seconds == 120.5


class TestResearchLaunchRecord:
    def test_defaults(self):
        record = ResearchLaunchRecord(
            workflow="contract_005",
            contract="005",
            branch="main",
        )
        assert record.service_name == "sparky-research"

    def test_custom_service(self):
        record = ResearchLaunchRecord(
            workflow="contract_005",
            contract="005",
            branch="main",
            service_name="custom-service",
        )
        assert record.service_name == "custom-service"


class TestContractDesignRecord:
    def test_full_record(self):
        record = ContractDesignRecord(
            contract_name="Contract 006",
            objective="Deep momentum research",
            steps=["step1", "step2"],
            budget_hours=12.0,
            success_criteria="Sharpe > 1.5",
            rationale="Momentum shows promise",
        )
        assert record.budget_hours == 12.0
        assert len(record.steps) == 2


class TestGetHistory:
    def test_empty_history(self, manager_log):
        history = manager_log.get_history()
        assert history == []

    def test_single_session(self, manager_log):
        session = manager_log.start_session("Test", "branch")
        manager_log.end_session(session, summary="Done")

        history = manager_log.get_history()
        assert len(history) == 1
        assert history[0].objective == "Test"
        assert history[0].summary == "Done"

    def test_multiple_sessions_newest_first(self, manager_log):
        for i in range(5):
            session = manager_log.start_session(f"Objective {i}", f"branch-{i}")
            manager_log.end_session(session, summary=f"Summary {i}")

        history = manager_log.get_history()
        assert len(history) == 5
        # Newest first
        assert history[0].summary == "Summary 4"
        assert history[4].summary == "Summary 0"

    def test_n_sessions_limit(self, manager_log):
        for i in range(10):
            session = manager_log.start_session(f"Obj {i}", f"b-{i}")
            manager_log.end_session(session, summary=f"Sum {i}")

        history = manager_log.get_history(n_sessions=3)
        assert len(history) == 3
        # Most recent 3
        assert history[0].summary == "Sum 9"

    def test_roundtrip_with_nested_records(self, manager_log):
        session = manager_log.start_session("Roundtrip test", "branch")
        manager_log.log_code_agent(
            session,
            CodeAgentRecord(
                task="Test task",
                files_created=["test.py"],
                tests_passed=True,
            ),
        )
        manager_log.log_decision(
            session,
            decision="Test decision",
            alternatives=["A", "B"],
            rationale="Because A is better",
        )
        manager_log.end_session(session, summary="Roundtrip complete")

        history = manager_log.get_history()
        assert len(history) == 1
        restored = history[0]
        assert len(restored.code_agents) == 1
        assert restored.code_agents[0].task == "Test task"
        assert restored.code_agents[0].tests_passed is True
        assert len(restored.decisions) == 1
        assert restored.decisions[0]["decision"] == "Test decision"


class TestJSONLFormat:
    def test_each_line_is_valid_json(self, manager_log, log_file):
        for i in range(3):
            session = manager_log.start_session(f"Test {i}", "branch")
            manager_log.end_session(session, summary=f"Done {i}")

        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            data = json.loads(line)  # Should not raise
            assert "session_id" in data
            assert "objective" in data

    def test_log_file_auto_created(self, tmp_path):
        log_file = tmp_path / "deep" / "nested" / "dir" / "log.jsonl"
        log = ManagerLog(log_file=log_file)
        session = log.start_session("Test", "branch")
        log.end_session(session, "Done")
        assert log_file.exists()


class TestSessionIdCollision:
    """M-8: Two sessions started in same second get different IDs (microsecond resolution)."""

    def test_same_second_different_ids(self, manager_log):
        """Sessions started rapidly should get unique IDs due to microseconds."""
        session1 = manager_log.start_session("Objective 1", "branch-1")
        session2 = manager_log.start_session("Objective 2", "branch-2")
        assert session1.session_id != session2.session_id, (
            f"Session IDs should differ: {session1.session_id} vs {session2.session_id}"
        )
