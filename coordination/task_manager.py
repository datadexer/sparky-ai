"""Task management for multi-agent coordination."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class TaskStatus(Enum):
    """Task status states."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Task:
    """Represents a task in the coordination system."""

    task_id: str
    description: str
    assigned_to: str
    status: TaskStatus
    priority: TaskPriority
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    dependencies: list[str] = None
    metadata: dict = None

    def __post_init__(self):
        """Initialize default values."""
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

        # Convert enums to strings if needed
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)
        if isinstance(self.priority, str):
            self.priority = TaskPriority(self.priority)

    def to_dict(self) -> dict:
        """Convert task to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Create task from dictionary."""
        return cls(**data)


class TaskManager:
    """Manages task assignments and prevents duplicate work."""

    def __init__(self, data_file: Path):
        """Initialize task manager.

        Args:
            data_file: Path to the task assignments JSON file
        """
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_tasks()

    def _load_tasks(self):
        """Load tasks from file."""
        if self.data_file.exists():
            with open(self.data_file) as f:
                data = json.load(f)
                self.tasks = {
                    task_id: Task.from_dict(task_data)
                    for task_id, task_data in data.get("tasks", {}).items()
                }
        else:
            self.tasks = {}
            self._save_tasks()

    def _save_tasks(self):
        """Save tasks to file."""
        data = {
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_task(
        self,
        task_id: str,
        description: str,
        assigned_to: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Task:
        """Create a new task.

        Args:
            task_id: Unique task identifier
            description: Task description
            assigned_to: Agent assigned to this task
            priority: Task priority level
            dependencies: List of task IDs this task depends on
            metadata: Additional task metadata

        Returns:
            Created task

        Raises:
            ValueError: If task_id already exists
        """
        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")

        task = Task(
            task_id=task_id,
            description=description,
            assigned_to=assigned_to,
            status=TaskStatus.QUEUED,
            priority=priority,
            created_at=datetime.now(timezone.utc).isoformat(),
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        self.tasks[task_id] = task
        self._save_tasks()
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def update_task_status(
        self, task_id: str, status: TaskStatus, timestamp: Optional[str] = None
    ):
        """Update task status.

        Args:
            task_id: Task ID to update
            status: New status
            timestamp: Optional timestamp (defaults to now)

        Raises:
            KeyError: If task_id doesn't exist
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task.status = status

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        if status == TaskStatus.IN_PROGRESS and task.started_at is None:
            task.started_at = timestamp
        elif status == TaskStatus.COMPLETED:
            task.completed_at = timestamp

        self._save_tasks()

    def get_active_tasks(self, agent: Optional[str] = None) -> list[Task]:
        """Get all active (queued or in progress) tasks.

        Args:
            agent: Optional filter by assigned agent

        Returns:
            List of active tasks
        """
        active_statuses = {TaskStatus.QUEUED, TaskStatus.IN_PROGRESS}
        tasks = [t for t in self.tasks.values() if t.status in active_statuses]

        if agent:
            tasks = [t for t in tasks if t.assigned_to == agent]

        return sorted(tasks, key=lambda t: (t.priority.value, t.created_at))

    def get_completed_tasks(self, agent: Optional[str] = None) -> list[Task]:
        """Get completed tasks.

        Args:
            agent: Optional filter by assigned agent

        Returns:
            List of completed tasks
        """
        tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]

        if agent:
            tasks = [t for t in tasks if t.assigned_to == agent]

        return sorted(tasks, key=lambda t: t.completed_at, reverse=True)

    def check_duplicate_work(self, description_pattern: str) -> list[Task]:
        """Check if similar work is already assigned.

        Args:
            description_pattern: Pattern to match in task descriptions

        Returns:
            List of potentially duplicate tasks
        """
        pattern_lower = description_pattern.lower()
        duplicates = []

        for task in self.tasks.values():
            if task.status in {TaskStatus.QUEUED, TaskStatus.IN_PROGRESS}:
                if pattern_lower in task.description.lower():
                    duplicates.append(task)

        return duplicates

    def export_markdown(self) -> str:
        """Export tasks to markdown format.

        Returns:
            Markdown-formatted task list
        """
        lines = ["# Active Task Assignments", "", "**Last Updated**: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), ""]

        # Active tasks
        active = self.get_active_tasks()
        if active:
            lines.append("## Currently Active")
            lines.append("")
            lines.append("| Task ID | Description | Assigned To | Status | Priority | Started |")
            lines.append("|---------|-------------|-------------|--------|----------|---------|")

            for task in active:
                status_emoji = "🔄" if task.status == TaskStatus.IN_PROGRESS else "⏳"
                started = task.started_at[:10] if task.started_at else "—"
                lines.append(
                    f"| {task.task_id} | {task.description} | {task.assigned_to} | "
                    f"{status_emoji} {task.status.value.upper()} | {task.priority.value.upper()} | {started} |"
                )
            lines.append("")

        # Completed tasks (recent 5)
        completed = self.get_completed_tasks()[:5]
        if completed:
            lines.append("## Recently Completed")
            lines.append("")
            lines.append("| Task ID | Description | Assigned To | Completed |")
            lines.append("|---------|-------------|-------------|-----------|")

            for task in completed:
                completed_date = task.completed_at[:10] if task.completed_at else "—"
                lines.append(
                    f"| {task.task_id} | {task.description} | {task.assigned_to} | {completed_date} |"
                )
            lines.append("")

        return "\n".join(lines)
