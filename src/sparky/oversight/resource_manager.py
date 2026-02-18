"""Resource manager for preventing system overload.

Monitors system resources (CPU, memory, disk) and enforces limits on
concurrent agent spawning, model training, and data fetching to prevent
crashes and ensure system stability.

Usage:
    from sparky.oversight.resource_manager import ResourceManager

    manager = ResourceManager()

    # Before spawning agents
    if manager.can_spawn_agent(agent_type="model_training"):
        # spawn agent
        agent_id = "training-xyz"
        manager.register_agent(agent_id, "model_training")

    # After agent completes
    manager.unregister_agent(agent_id)

    # Check system health
    status = manager.get_system_status()
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import yaml

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Information about a running agent."""

    agent_id: str
    agent_type: str  # "general", "model_training", "data_fetch", etc.
    started_at: datetime
    memory_mb: float = 0.0


@dataclass
class SystemStatus:
    """Current system resource status."""

    cpu_percent: float
    memory_percent: float
    disk_free_gb: float
    active_agents: int
    agents_by_type: Dict[str, int]
    under_pressure: bool
    circuit_breaker_open: bool
    warnings: List[str]


class ResourceManagerError(Exception):
    """Raised when resource limits are exceeded."""

    pass


class CircuitBreakerOpen(ResourceManagerError):
    """Raised when circuit breaker is open."""

    pass


class ResourceManager:
    """Manages system resources and enforces limits.

    This class prevents system crashes by:
    1. Tracking concurrent agents and enforcing concurrency limits
    2. Monitoring CPU, memory, and disk usage
    3. Implementing circuit breaker pattern for failures
    4. Gracefully degrading under pressure
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize resource manager.

        Args:
            config_path: Path to resource_limits.yaml. If None, uses default.
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "configs" / "resource_limits.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Active agents tracking
        self.active_agents: Dict[str, AgentInfo] = {}

        # Circuit breaker state
        self.circuit_breaker_open = False
        self.circuit_breaker_opened_at: Optional[datetime] = None
        self.consecutive_errors = 0

        # Concurrency adjustments (for graceful degradation)
        self.current_max_concurrent = self.config["concurrency"]["max_concurrent_agents"]

        logger.info(f"ResourceManager initialized. Max concurrent agents: {self.current_max_concurrent}")

    def can_spawn_agent(self, agent_type: str = "general") -> bool:
        """Check if a new agent can be spawned.

        Args:
            agent_type: Type of agent ("general", "model_training", "data_fetch")

        Returns:
            True if agent can be spawned, False otherwise

        Raises:
            CircuitBreakerOpen: If circuit breaker is open
            ResourceManagerError: If resource limits exceeded
        """
        # Check circuit breaker
        self._check_circuit_breaker()

        # Check concurrency limits
        self._check_concurrency_limits(agent_type)

        # Check system resources
        self._check_system_resources()

        return True

    def register_agent(self, agent_id: str, agent_type: str = "general") -> None:
        """Register a newly spawned agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent
        """
        if agent_id in self.active_agents:
            logger.warning(f"Agent {agent_id} already registered, updating")

        self.active_agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            agent_type=agent_type,
            started_at=datetime.now(timezone.utc),
        )

        logger.info(
            f"Agent registered: {agent_id} ({agent_type}). "
            f"Active agents: {len(self.active_agents)}/{self.current_max_concurrent}"
        )

        # Reset consecutive errors on successful registration
        self.consecutive_errors = 0

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister a completed agent.

        Args:
            agent_id: Unique identifier for the agent
        """
        if agent_id not in self.active_agents:
            logger.warning(f"Agent {agent_id} not found in registry")
            return

        agent_info = self.active_agents.pop(agent_id)
        duration = (datetime.now(timezone.utc) - agent_info.started_at).total_seconds()

        logger.info(
            f"Agent unregistered: {agent_id} ({agent_info.agent_type}). "
            f"Duration: {duration:.1f}s. Active agents: {len(self.active_agents)}"
        )

    def get_system_status(self) -> SystemStatus:
        """Get current system resource status.

        Returns:
            SystemStatus with current resource usage and warnings
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Count agents by type
        agents_by_type: Dict[str, int] = {}
        for agent_info in self.active_agents.values():
            agents_by_type[agent_info.agent_type] = agents_by_type.get(agent_info.agent_type, 0) + 1

        # Check for warnings
        warnings = []
        alert = self.config["monitoring"]["alert_thresholds"]

        if cpu_percent > alert["cpu_percent"]:
            warnings.append(f"High CPU usage: {cpu_percent:.1f}%")

        if memory.percent > alert["memory_percent"]:
            warnings.append(f"High memory usage: {memory.percent:.1f}%")

        disk_free_gb = disk.free / (1024**3)
        if disk_free_gb < alert["disk_free_gb"]:
            warnings.append(f"Low disk space: {disk_free_gb:.1f} GB free")

        # Check for long-running agents
        timeout_warning = self.config["concurrency"]["agent_timeout_warning"]
        now = datetime.now(timezone.utc)
        for agent_info in self.active_agents.values():
            duration = (now - agent_info.started_at).total_seconds()
            if duration > timeout_warning:
                warnings.append(f"Agent {agent_info.agent_id} running for {duration / 60:.1f} minutes")

        # Determine if under pressure
        pressure = self.config["degradation"]
        under_pressure = (
            cpu_percent > pressure["pressure_cpu_threshold"] or memory.percent > pressure["pressure_memory_threshold"]
        )

        return SystemStatus(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_free_gb=disk_free_gb,
            active_agents=len(self.active_agents),
            agents_by_type=agents_by_type,
            under_pressure=under_pressure,
            circuit_breaker_open=self.circuit_breaker_open,
            warnings=warnings,
        )

    def _check_circuit_breaker(self) -> None:
        """Check circuit breaker state and handle recovery.

        Raises:
            CircuitBreakerOpen: If circuit breaker is open and cooldown not expired
        """
        if not self.config["circuit_breaker"]["enabled"]:
            return

        if not self.circuit_breaker_open:
            return

        # Check if cooldown period has passed
        cooldown = self.config["circuit_breaker"]["cooldown_seconds"]
        elapsed = (datetime.now(timezone.utc) - self.circuit_breaker_opened_at).total_seconds()

        if elapsed > cooldown:
            if self.config["circuit_breaker"]["auto_recovery"]:
                logger.info("Circuit breaker cooldown expired, attempting recovery")
                self.circuit_breaker_open = False
                self.consecutive_errors = 0
                return

        raise CircuitBreakerOpen(
            f"Circuit breaker open. Cooldown: {cooldown - elapsed:.0f}s remaining. "
            f"Reason: {self.consecutive_errors} consecutive resource errors."
        )

    def _check_concurrency_limits(self, agent_type: str) -> None:
        """Check if concurrency limits allow spawning new agent.

        Args:
            agent_type: Type of agent to spawn

        Raises:
            ResourceManagerError: If concurrency limits exceeded
        """
        # Check global concurrent agent limit
        if len(self.active_agents) >= self.current_max_concurrent:
            self._handle_resource_error(
                f"Cannot spawn agent: at max concurrency ({len(self.active_agents)}/{self.current_max_concurrent})"
            )

        # Check type-specific limits
        if agent_type == "model_training":
            training_agents = sum(1 for a in self.active_agents.values() if a.agent_type == "model_training")
            max_training = self.config["model_training"]["max_concurrent_training"]
            if training_agents >= max_training:
                self._handle_resource_error(
                    f"Cannot spawn training agent: at max training concurrency ({training_agents}/{max_training})"
                )

        if agent_type == "data_fetch":
            fetch_agents = sum(1 for a in self.active_agents.values() if a.agent_type == "data_fetch")
            max_fetch = self.config["data_fetching"]["max_concurrent_fetches"]
            if fetch_agents >= max_fetch:
                self._handle_resource_error(
                    f"Cannot spawn data fetch agent: at max fetch concurrency ({fetch_agents}/{max_fetch})"
                )

    def _check_system_resources(self) -> None:
        """Check system resource availability.

        Raises:
            ResourceManagerError: If system resources critical
        """
        if not self.config["monitoring"]["enabled"]:
            return

        status = self.get_system_status()
        halt = self.config["monitoring"]["halt_thresholds"]

        # Check halt thresholds
        if status.cpu_percent > halt["cpu_percent"]:
            self._handle_resource_error(f"CPU usage critical: {status.cpu_percent:.1f}% > {halt['cpu_percent']}%")

        if status.memory_percent > halt["memory_percent"]:
            self._handle_resource_error(
                f"Memory usage critical: {status.memory_percent:.1f}% > {halt['memory_percent']}%"
            )

        if status.disk_free_gb < halt["disk_free_gb"]:
            self._handle_resource_error(
                f"Disk space critical: {status.disk_free_gb:.1f} GB < {halt['disk_free_gb']} GB"
            )

        # Handle graceful degradation
        if status.under_pressure:
            self._apply_degradation()

    def _apply_degradation(self) -> None:
        """Apply graceful degradation under resource pressure."""
        if not self.config["degradation"]["auto_reduce_concurrency"]:
            return

        min_concurrent = self.config["degradation"]["min_concurrent_agents"]
        if self.current_max_concurrent > min_concurrent:
            old_max = self.current_max_concurrent
            self.current_max_concurrent = max(min_concurrent, self.current_max_concurrent - 1)
            logger.warning(
                f"System under pressure. Reducing max concurrent agents: {old_max} -> {self.current_max_concurrent}"
            )

    def _handle_resource_error(self, error_msg: str) -> None:
        """Handle resource limit violation.

        Args:
            error_msg: Error message describing the violation

        Raises:
            ResourceManagerError: Always raised with error_msg
        """
        logger.error(error_msg)

        # Increment consecutive errors
        self.consecutive_errors += 1

        # Check if circuit breaker should open
        if self.config["circuit_breaker"]["enabled"]:
            max_errors = self.config["circuit_breaker"]["max_consecutive_errors"]
            if self.consecutive_errors >= max_errors:
                self.circuit_breaker_open = True
                self.circuit_breaker_opened_at = datetime.now(timezone.utc)
                logger.critical(f"Circuit breaker opened after {self.consecutive_errors} consecutive errors")

        raise ResourceManagerError(error_msg)

    def cleanup_stale_agents(self, force_kill_timeout: Optional[int] = None) -> List[str]:
        """Clean up agents that have exceeded timeout.

        Args:
            force_kill_timeout: Override timeout (seconds). If None, uses config.

        Returns:
            List of agent IDs that were cleaned up
        """
        if force_kill_timeout is None:
            force_kill_timeout = self.config["concurrency"]["agent_timeout_kill"]

        now = datetime.now(timezone.utc)
        stale_agents = []

        for agent_id, agent_info in list(self.active_agents.items()):
            duration = (now - agent_info.started_at).total_seconds()
            if duration > force_kill_timeout:
                logger.warning(f"Force-killing stale agent: {agent_id} (running {duration / 60:.1f} minutes)")
                stale_agents.append(agent_id)
                self.unregister_agent(agent_id)

        return stale_agents


# Global singleton instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get global ResourceManager singleton.

    Returns:
        ResourceManager instance
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager
