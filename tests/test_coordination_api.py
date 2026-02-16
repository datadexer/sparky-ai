"""Tests for the coordination API."""

import json
import tempfile
from pathlib import Path

import pytest

from coordination import (
    CoordinationAPI,
    AgentRole,
    TaskPriority,
    MessagePriority,
    TaskStatus,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def api(temp_dir):
    """Create coordination API instance."""
    return CoordinationAPI(temp_dir)


class TestAgentLifecycle:
    """Test agent registration and lifecycle."""

    def test_register_ceo(self, api):
        """Test CEO registration."""
        agent = api.register_agent("ceo", AgentRole.CEO)
        assert agent.agent_id == "ceo"
        assert agent.role == AgentRole.CEO

    def test_only_one_ceo_allowed(self, api):
        """Test that only one CEO can be registered."""
        api.register_agent("ceo", AgentRole.CEO)

        with pytest.raises(ValueError, match="Only one CEO agent allowed"):
            api.register_agent("ceo2", AgentRole.CEO)

    def test_register_subagent(self, api):
        """Test sub-agent registration."""
        agent = api.register_agent("validation-001", AgentRole.VALIDATION)
        assert agent.agent_id == "validation-001"
        assert agent.role == AgentRole.VALIDATION

    def test_update_activity(self, api):
        """Test updating agent activity."""
        api.register_agent("ceo", AgentRole.CEO)
        api.update_agent_activity("ceo", "Testing coordination")

        ceo = api.get_ceo()
        assert ceo.current_task == "Testing coordination"

    def test_terminate_agent(self, api):
        """Test agent termination."""
        api.register_agent("validation-001", AgentRole.VALIDATION)
        api.terminate_agent("validation-001")

        # Should be able to register new CEO after termination
        api.register_agent("ceo", AgentRole.CEO)  # Should work


class TestTaskManagement:
    """Test task management operations."""

    def test_create_task(self, api):
        """Test task creation."""
        api.register_agent("ceo", AgentRole.CEO)
        task = api.create_task(
            task_id="data-expansion",
            description="Expand dataset to 10K samples",
            assigned_to="ceo",
            priority=TaskPriority.HIGH,
        )

        assert task.task_id == "data-expansion"
        assert task.description == "Expand dataset to 10K samples"
        assert task.assigned_to == "ceo"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.QUEUED

    def test_task_lifecycle(self, api):
        """Test task status transitions."""
        api.create_task("test-task", "Test description", "ceo")

        # Start task
        api.start_task("test-task")
        task = api.task_manager.get_task("test-task")
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None

        # Complete task
        api.complete_task("test-task")
        task = api.task_manager.get_task("test-task")
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None

    def test_get_my_tasks(self, api):
        """Test retrieving agent's tasks."""
        api.create_task("task1", "Task 1", "ceo")
        api.create_task("task2", "Task 2", "ceo")
        api.create_task("task3", "Task 3", "validation")

        ceo_tasks = api.get_my_tasks("ceo")
        assert len(ceo_tasks) == 2
        assert all(t.assigned_to == "ceo" for t in ceo_tasks)

    def test_check_duplicate_work(self, api):
        """Test duplicate work detection."""
        api.create_task("task1", "Data expansion with hourly BTC", "ceo")
        api.create_task("task2", "Feature engineering", "validation")

        duplicates = api.check_duplicate_work("data expansion")
        assert len(duplicates) == 1
        assert duplicates[0].task_id == "task1"

    def test_export_task_markdown(self, api, temp_dir):
        """Test exporting tasks to markdown."""
        api.create_task("task1", "High priority task", "ceo", TaskPriority.HIGH)
        api.create_task("task2", "Low priority task", "ceo", TaskPriority.LOW)

        output_file = temp_dir / "tasks.md"
        api.export_task_markdown(output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "High priority task" in content
        assert "Low priority task" in content


class TestMessaging:
    """Test messaging operations."""

    def test_send_message(self, api):
        """Test sending a message."""
        message = api.send_message(
            from_agent="validation-001",
            to_agent="ceo",
            subject="Audit Complete",
            body="Found 10 issues in validation summary",
            priority=MessagePriority.HIGH,
        )

        assert message.from_agent == "validation-001"
        assert message.to_agent == "ceo"
        assert message.subject == "Audit Complete"
        assert not message.read

    def test_get_unread_messages(self, api):
        """Test retrieving unread messages."""
        api.send_message("agent1", "ceo", "Message 1", "Body 1")
        api.send_message("agent2", "ceo", "Message 2", "Body 2")

        unread = api.get_unread_messages("ceo")
        assert len(unread) == 2
        assert all(not m.read for m in unread)

    def test_mark_message_read(self, api):
        """Test marking message as read."""
        msg = api.send_message("agent1", "ceo", "Test", "Body")

        api.mark_message_read(msg.message_id)

        msg_updated = api.inbox_manager.get_message(msg.message_id)
        assert msg_updated.read

    def test_message_priority_sorting(self, api):
        """Test messages are sorted by priority."""
        api.send_message("a", "ceo", "Low", "Body", MessagePriority.LOW)
        api.send_message("a", "ceo", "Critical", "Body", MessagePriority.CRITICAL)
        api.send_message("a", "ceo", "High", "Body", MessagePriority.HIGH)

        unread = api.get_unread_messages("ceo")
        assert unread[0].subject == "Critical"
        assert unread[1].subject == "High"
        assert unread[2].subject == "Low"

    def test_export_inbox_markdown(self, api, temp_dir):
        """Test exporting inbox to markdown."""
        api.send_message("validation", "ceo", "Test Subject", "Test Body")

        output_file = temp_dir / "inbox.md"
        api.export_inbox_markdown("ceo", output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Test Subject" in content
        assert "validation" in content


class TestHighLevelWorkflows:
    """Test high-level workflow methods."""

    def test_ceo_startup_checklist(self, api):
        """Test CEO startup checklist."""
        api.register_agent("ceo", AgentRole.CEO)
        api.create_task("task1", "Test task", "ceo")
        api.send_message("validation", "ceo", "Test", "Body")

        summary = api.ceo_startup_checklist("ceo")

        assert summary["unread_messages"] == 1
        assert summary["active_tasks"] == 1
        assert len(summary["messages"]) == 1
        assert len(summary["tasks"]) == 1

    def test_subagent_report_and_terminate(self, api):
        """Test sub-agent report workflow."""
        api.register_agent("ceo", AgentRole.CEO)
        api.register_agent("validation-001", AgentRole.VALIDATION)

        api.subagent_report_and_terminate(
            agent_id="validation-001",
            report_subject="Audit Complete",
            report_body="Found issues in validation",
        )

        # Check message was sent
        unread = api.get_unread_messages("ceo")
        assert len(unread) == 1
        assert unread[0].subject == "Audit Complete"

        # Check agent terminated
        agent = api.agent_registry.agents["validation-001"]
        assert agent.status.value == "terminated"


class TestPersistence:
    """Test data persistence across API instances."""

    def test_tasks_persist(self, temp_dir):
        """Test tasks persist across instances."""
        api1 = CoordinationAPI(temp_dir)
        api1.create_task("task1", "Test task", "ceo")

        api2 = CoordinationAPI(temp_dir)
        task = api2.task_manager.get_task("task1")
        assert task is not None
        assert task.description == "Test task"

    def test_messages_persist(self, temp_dir):
        """Test messages persist across instances."""
        api1 = CoordinationAPI(temp_dir)
        api1.send_message("agent1", "ceo", "Test", "Body")

        api2 = CoordinationAPI(temp_dir)
        unread = api2.get_unread_messages("ceo")
        assert len(unread) == 1

    def test_agents_persist(self, temp_dir):
        """Test agent registry persists across instances."""
        api1 = CoordinationAPI(temp_dir)
        api1.register_agent("ceo", AgentRole.CEO)

        api2 = CoordinationAPI(temp_dir)
        ceo = api2.get_ceo()
        assert ceo is not None
        assert ceo.agent_id == "ceo"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_start_nonexistent_task(self, api):
        """Test starting non-existent task raises error."""
        with pytest.raises(KeyError):
            api.start_task("nonexistent")

    def test_update_activity_nonexistent_agent(self, api):
        """Test updating non-existent agent raises error."""
        with pytest.raises(KeyError):
            api.update_agent_activity("nonexistent")

    def test_mark_nonexistent_message_read(self, api):
        """Test marking non-existent message raises error."""
        with pytest.raises(KeyError):
            api.mark_message_read("nonexistent")

    def test_duplicate_task_id(self, api):
        """Test creating duplicate task ID raises error."""
        api.create_task("task1", "Description", "ceo")

        with pytest.raises(ValueError):
            api.create_task("task1", "Different description", "ceo")
