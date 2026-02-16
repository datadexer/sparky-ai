"""Main coordination API - single interface for all coordination operations."""

from pathlib import Path
from typing import Optional

from .agent_registry import AgentRegistry, AgentRole, AgentStatus
from .inbox_manager import InboxManager, MessagePriority
from .task_manager import TaskManager, TaskStatus, TaskPriority


class CoordinationAPI:
    """Main API for multi-agent coordination.

    This is the primary interface agents should use for all coordination operations.
    """

    def __init__(self, base_dir: Path):
        """Initialize coordination API.

        Args:
            base_dir: Base directory for coordination files (usually project root)
        """
        self.base_dir = Path(base_dir)
        coord_dir = self.base_dir / "coordination" / "data"
        coord_dir.mkdir(parents=True, exist_ok=True)

        self.task_manager = TaskManager(coord_dir / "tasks.json")
        self.inbox_manager = InboxManager(coord_dir / "inbox.json")
        self.agent_registry = AgentRegistry(coord_dir / "agents.json")

    # ========== Agent Lifecycle ==========

    def register_agent(
        self, agent_id: str, role: AgentRole, metadata: Optional[dict] = None
    ):
        """Register a new agent at session start.

        Args:
            agent_id: Unique agent identifier (e.g., "ceo", "validation-001")
            role: Agent role
            metadata: Optional metadata

        Raises:
            ValueError: If trying to register second CEO
        """
        return self.agent_registry.register_agent(agent_id, role, metadata)

    def update_agent_activity(self, agent_id: str, current_task: Optional[str] = None):
        """Update agent activity timestamp (call periodically).

        Args:
            agent_id: Agent ID
            current_task: Current task description
        """
        self.agent_registry.update_activity(agent_id, current_task)

    def terminate_agent(self, agent_id: str):
        """Mark agent as terminated (call when agent finishes).

        Args:
            agent_id: Agent ID to terminate
        """
        self.agent_registry.terminate_agent(agent_id)

    def get_ceo(self):
        """Get the active CEO agent info."""
        return self.agent_registry.get_ceo()

    # ========== Task Management ==========

    def create_task(
        self,
        task_id: str,
        description: str,
        assigned_to: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Optional[list[str]] = None,
    ):
        """Create a new task.

        Args:
            task_id: Unique task ID
            description: Task description
            assigned_to: Agent to assign task to
            priority: Task priority
            dependencies: List of task IDs this depends on

        Returns:
            Created task
        """
        return self.task_manager.create_task(
            task_id=task_id,
            description=description,
            assigned_to=assigned_to,
            priority=priority,
            dependencies=dependencies,
        )

    def start_task(self, task_id: str):
        """Mark task as started (IN_PROGRESS).

        Args:
            task_id: Task ID to start
        """
        self.task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS)

    def complete_task(self, task_id: str):
        """Mark task as completed.

        Args:
            task_id: Task ID to complete
        """
        self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)

    def get_my_tasks(self, agent_id: str):
        """Get all active tasks for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of active tasks
        """
        return self.task_manager.get_active_tasks(agent=agent_id)

    def check_duplicate_work(self, description_pattern: str):
        """Check if similar work is already in progress.

        Args:
            description_pattern: Pattern to search for

        Returns:
            List of potentially duplicate tasks
        """
        return self.task_manager.check_duplicate_work(description_pattern)

    # ========== Messaging ==========

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        subject: str,
        body: str,
        priority: MessagePriority = MessagePriority.MEDIUM,
    ):
        """Send a message from one agent to another.

        Args:
            from_agent: Sending agent ID
            to_agent: Receiving agent ID
            subject: Message subject
            body: Message body
            priority: Message priority

        Returns:
            Created message
        """
        return self.inbox_manager.send_message(
            from_agent=from_agent,
            to_agent=to_agent,
            subject=subject,
            body=body,
            priority=priority,
        )

    def get_unread_messages(self, agent_id: str):
        """Get unread messages for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of unread messages (sorted by priority)
        """
        return self.inbox_manager.get_unread_messages(agent_id)

    def mark_message_read(self, message_id: str):
        """Mark a message as read.

        Args:
            message_id: Message ID
        """
        self.inbox_manager.mark_as_read(message_id)

    def mark_all_messages_read(self, agent_id: str):
        """Mark all messages as read for an agent.

        Args:
            agent_id: Agent ID
        """
        self.inbox_manager.mark_all_as_read(agent_id)

    # ========== Export for Human Reading ==========

    def export_task_markdown(self, output_file: Path):
        """Export tasks to markdown file for human reading.

        Args:
            output_file: Path to write markdown file
        """
        markdown = self.task_manager.export_markdown()
        with open(output_file, "w") as f:
            f.write(markdown)

    def export_inbox_markdown(self, agent_id: str, output_file: Path):
        """Export inbox to markdown file for human reading.

        Args:
            agent_id: Agent whose inbox to export
            output_file: Path to write markdown file
        """
        markdown = self.inbox_manager.export_markdown(agent_id)
        with open(output_file, "w") as f:
            f.write(markdown)

    # ========== High-Level Workflows ==========

    def ceo_startup_checklist(self, ceo_id: str = "ceo") -> dict:
        """Execute CEO startup checklist and return summary.

        This is what the CEO should call at the start of every session.

        Args:
            ceo_id: CEO agent ID

        Returns:
            Dictionary with startup summary
        """
        summary = {
            "unread_messages": len(self.inbox_manager.get_unread_messages(ceo_id)),
            "active_tasks": len(self.task_manager.get_active_tasks(agent=ceo_id)),
            "messages": self.inbox_manager.get_unread_messages(ceo_id),
            "tasks": self.task_manager.get_active_tasks(agent=ceo_id),
        }

        # Update activity
        self.agent_registry.update_activity(ceo_id)

        return summary

    def subagent_report_and_terminate(
        self, agent_id: str, report_subject: str, report_body: str
    ):
        """Sub-agent workflow: send report to CEO and terminate.

        Args:
            agent_id: Sub-agent ID
            report_subject: Report subject
            report_body: Report body
        """
        # Send report to CEO
        self.send_message(
            from_agent=agent_id,
            to_agent="ceo",
            subject=report_subject,
            body=report_body,
            priority=MessagePriority.HIGH,
        )

        # Terminate agent
        self.terminate_agent(agent_id)
