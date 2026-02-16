"""Inbox management for agent-to-agent messaging."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class MessagePriority(Enum):
    """Message priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Message:
    """Represents a message in the inbox."""

    message_id: str
    from_agent: str
    to_agent: str
    subject: str
    body: str
    priority: MessagePriority
    timestamp: str
    read: bool = False
    metadata: dict = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}

        # Convert enum to string if needed
        if isinstance(self.priority, str):
            self.priority = MessagePriority(self.priority)

    def to_dict(self) -> dict:
        """Convert message to dictionary."""
        data = asdict(self)
        data["priority"] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create message from dictionary."""
        return cls(**data)


class InboxManager:
    """Manages agent-to-agent messaging."""

    def __init__(self, data_file: Path):
        """Initialize inbox manager.

        Args:
            data_file: Path to the inbox JSON file
        """
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_messages()

    def _load_messages(self):
        """Load messages from file."""
        if self.data_file.exists():
            with open(self.data_file) as f:
                data = json.load(f)
                self.messages = {
                    msg_id: Message.from_dict(msg_data)
                    for msg_id, msg_data in data.get("messages", {}).items()
                }
        else:
            self.messages = {}
            self._save_messages()

    def _save_messages(self):
        """Save messages to file."""
        data = {
            "messages": {
                msg_id: msg.to_dict() for msg_id, msg in self.messages.items()
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        subject: str,
        body: str,
        priority: MessagePriority = MessagePriority.MEDIUM,
        metadata: Optional[dict] = None,
    ) -> Message:
        """Send a message from one agent to another.

        Args:
            from_agent: Sending agent name
            to_agent: Receiving agent name
            subject: Message subject
            body: Message body
            priority: Message priority
            metadata: Additional metadata

        Returns:
            Created message
        """
        timestamp = datetime.now(timezone.utc)
        message_id = f"{from_agent}_to_{to_agent}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

        message = Message(
            message_id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            subject=subject,
            body=body,
            priority=priority,
            timestamp=timestamp.isoformat(),
            read=False,
            metadata=metadata or {},
        )

        self.messages[message_id] = message
        self._save_messages()
        return message

    def get_unread_messages(self, agent: str) -> list[Message]:
        """Get unread messages for an agent.

        Args:
            agent: Agent name

        Returns:
            List of unread messages sorted by priority then timestamp
        """
        unread = [
            msg
            for msg in self.messages.values()
            if msg.to_agent == agent and not msg.read
        ]

        # Sort by priority (critical first) then timestamp (newest first)
        priority_order = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.MEDIUM: 2,
            MessagePriority.LOW: 3,
        }

        return sorted(
            unread, key=lambda m: (priority_order[m.priority], m.timestamp)
        )

    def mark_as_read(self, message_id: str):
        """Mark a message as read.

        Args:
            message_id: Message ID to mark as read

        Raises:
            KeyError: If message_id doesn't exist
        """
        if message_id not in self.messages:
            raise KeyError(f"Message {message_id} not found")

        self.messages[message_id].read = True
        self._save_messages()

    def mark_all_as_read(self, agent: str):
        """Mark all messages for an agent as read.

        Args:
            agent: Agent name
        """
        for msg in self.messages.values():
            if msg.to_agent == agent and not msg.read:
                msg.read = True
        self._save_messages()

    def get_message(self, message_id: str) -> Optional[Message]:
        """Get message by ID."""
        return self.messages.get(message_id)

    def export_markdown(self, agent: str) -> str:
        """Export inbox to markdown format.

        Args:
            agent: Agent name whose inbox to export

        Returns:
            Markdown-formatted inbox
        """
        lines = [
            f"# {agent.upper()} Inbox",
            "",
            "## ⚠️ CRITICAL: Read this file at START of EVERY session",
            "",
        ]

        unread = self.get_unread_messages(agent)
        if unread:
            lines.append("## Unread Messages")
            lines.append("")

            for msg in unread:
                priority_emoji = {
                    MessagePriority.CRITICAL: "🔴",
                    MessagePriority.HIGH: "🟠",
                    MessagePriority.MEDIUM: "🟡",
                    MessagePriority.LOW: "⚪",
                }[msg.priority]

                lines.append(f"### {priority_emoji} [{msg.timestamp[:10]}] From: {msg.from_agent}")
                lines.append(f"**Subject**: {msg.subject}")
                lines.append(f"**Priority**: {msg.priority.value.upper()}")
                lines.append("")
                lines.append(msg.body)
                lines.append("")
                lines.append("---")
                lines.append("")
        else:
            lines.append("## Unread Messages")
            lines.append("")
            lines.append("✅ No unread messages")
            lines.append("")

        # Show recent read messages
        read = [
            msg
            for msg in self.messages.values()
            if msg.to_agent == agent and msg.read
        ]
        read = sorted(read, key=lambda m: m.timestamp, reverse=True)[:5]

        if read:
            lines.append("## Recently Read")
            lines.append("")
            for msg in read:
                lines.append(
                    f"- [{msg.timestamp[:10]}] **{msg.subject}** (from {msg.from_agent})"
                )
            lines.append("")

        return "\n".join(lines)
