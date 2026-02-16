"""Multi-Agent Coordination System API.

This module provides a structured, testable API for coordinating multiple AI agents
working on the Sparky AI cryptocurrency trading ML project.

Key Components:
- TaskManager: Manages task assignments and prevents duplicate work
- InboxManager: Handles agent-to-agent messaging
- AgentRegistry: Tracks active agents and their roles
- CoordinationAPI: Main interface for agent coordination
"""

from .task_manager import TaskManager, Task, TaskStatus, TaskPriority
from .inbox_manager import InboxManager, Message, MessagePriority
from .agent_registry import AgentRegistry, AgentRole, AgentStatus
from .coordination_api import CoordinationAPI

__all__ = [
    "TaskManager",
    "Task",
    "TaskStatus",
    "TaskPriority",
    "InboxManager",
    "Message",
    "MessagePriority",
    "AgentRegistry",
    "AgentRole",
    "AgentStatus",
    "CoordinationAPI",
]

__version__ = "1.0.0"
