"""Agent registry for tracking active agents."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class AgentRole(Enum):
    """Agent role types."""

    CEO = "ceo"
    VALIDATION = "validation"
    DATA_ENGINEER = "data_engineer"
    RESEARCH = "research"


class AgentStatus(Enum):
    """Agent status states."""

    ACTIVE = "active"
    IDLE = "idle"
    TERMINATED = "terminated"


@dataclass
class AgentInfo:
    """Information about an agent."""

    agent_id: str
    role: AgentRole
    status: AgentStatus
    started_at: str
    last_activity: str
    current_task: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}

        # Convert enums if needed
        if isinstance(self.role, str):
            self.role = AgentRole(self.role)
        if isinstance(self.status, str):
            self.status = AgentStatus(self.status)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["role"] = self.role.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "AgentInfo":
        """Create from dictionary."""
        return cls(**data)


class AgentRegistry:
    """Registry for tracking active agents."""

    def __init__(self, data_file: Path):
        """Initialize agent registry.

        Args:
            data_file: Path to registry JSON file
        """
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def _load_registry(self):
        """Load registry from file."""
        if self.data_file.exists():
            with open(self.data_file) as f:
                data = json.load(f)
                self.agents = {
                    agent_id: AgentInfo.from_dict(agent_data)
                    for agent_id, agent_data in data.get("agents", {}).items()
                }
        else:
            self.agents = {}
            self._save_registry()

    def _save_registry(self):
        """Save registry to file."""
        data = {
            "agents": {
                agent_id: agent.to_dict() for agent_id, agent in self.agents.items()
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_agent(
        self, agent_id: str, role: AgentRole, metadata: Optional[dict] = None
    ) -> AgentInfo:
        """Register a new agent.

        Args:
            agent_id: Unique agent identifier
            role: Agent role
            metadata: Additional metadata

        Returns:
            Created agent info

        Raises:
            ValueError: If CEO role already exists (only one CEO allowed)
        """
        # Enforce single CEO constraint
        if role == AgentRole.CEO:
            existing_ceo = [
                a for a in self.agents.values() if a.role == AgentRole.CEO and a.status != AgentStatus.TERMINATED
            ]
            if existing_ceo:
                raise ValueError(
                    f"Only one CEO agent allowed. Existing: {existing_ceo[0].agent_id}"
                )

        timestamp = datetime.now(timezone.utc).isoformat()
        agent = AgentInfo(
            agent_id=agent_id,
            role=role,
            status=AgentStatus.ACTIVE,
            started_at=timestamp,
            last_activity=timestamp,
            metadata=metadata or {},
        )

        self.agents[agent_id] = agent
        self._save_registry()
        return agent

    def update_activity(self, agent_id: str, current_task: Optional[str] = None):
        """Update agent's last activity timestamp.

        Args:
            agent_id: Agent ID
            current_task: Optional current task description
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]
        agent.last_activity = datetime.now(timezone.utc).isoformat()
        if current_task is not None:
            agent.current_task = current_task

        self._save_registry()

    def terminate_agent(self, agent_id: str):
        """Mark an agent as terminated.

        Args:
            agent_id: Agent ID to terminate
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        self.agents[agent_id].status = AgentStatus.TERMINATED
        self._save_registry()

    def get_active_agents(self, role: Optional[AgentRole] = None) -> list[AgentInfo]:
        """Get all active agents.

        Args:
            role: Optional filter by role

        Returns:
            List of active agents
        """
        agents = [a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]

        if role:
            agents = [a for a in agents if a.role == role]

        return agents

    def get_ceo(self) -> Optional[AgentInfo]:
        """Get the active CEO agent.

        Returns:
            CEO agent info or None
        """
        ceos = [
            a
            for a in self.agents.values()
            if a.role == AgentRole.CEO and a.status != AgentStatus.TERMINATED
        ]
        return ceos[0] if ceos else None
