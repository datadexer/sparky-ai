"""Tests for resource manager."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from sparky.oversight.resource_manager import (
    ResourceManager,
    ResourceManagerError,
    CircuitBreakerOpen,
    SystemStatus,
)


@pytest.fixture
def test_config(tmp_path):
    """Create a test resource_limits.yaml config."""
    config = {
        "concurrency": {
            "max_concurrent_agents": 3,
            "max_parallel_spawn": 2,
            "max_tasks_per_agent": 10,
            "agent_poll_interval": 5,
            "agent_timeout_warning": 600,
            "agent_timeout_kill": 1800,
        },
        "compute": {
            "max_cpu_percent": 80,
            "max_memory_percent": 75,
            "max_memory_per_agent_gb": 16,
            "max_disk_io_mbps": 500,
            "min_free_disk_gb": 50,
        },
        "model_training": {
            "max_concurrent_training": 1,
            "training_timeout_warning": 3600,
            "training_timeout_kill": 7200,
            "max_training_rows": 10000000,
            "max_hyperparameter_combos": 50,
        },
        "data_fetching": {
            "max_concurrent_fetches": 2,
            "max_requests_per_second": 5,
            "max_retries": 3,
            "retry_backoff_multiplier": 2,
            "max_fetch_size_mb": 100,
        },
        "monitoring": {
            "enabled": True,
            "check_interval_seconds": 10,
            "alert_thresholds": {
                "cpu_percent": 70,
                "memory_percent": 65,
                "disk_free_gb": 100,
            },
            "halt_thresholds": {
                "cpu_percent": 85,
                "memory_percent": 80,
                "disk_free_gb": 50,
            },
        },
        "degradation": {
            "auto_reduce_concurrency": True,
            "pressure_cpu_threshold": 75,
            "pressure_memory_threshold": 70,
            "min_concurrent_agents": 1,
        },
        "circuit_breaker": {
            "enabled": True,
            "max_consecutive_errors": 3,
            "cooldown_seconds": 300,
            "auto_recovery": True,
        },
    }

    config_path = tmp_path / "resource_limits.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


def test_resource_manager_initialization(test_config):
    """Test resource manager initialization."""
    manager = ResourceManager(config_path=test_config)

    assert manager.current_max_concurrent == 3
    assert manager.circuit_breaker_open is False
    assert manager.consecutive_errors == 0
    assert len(manager.active_agents) == 0


def test_register_and_unregister_agents(test_config):
    """Test agent registration and unregistration."""
    manager = ResourceManager(config_path=test_config)

    # Register agents
    manager.register_agent("agent-1", "general")
    assert len(manager.active_agents) == 1
    assert "agent-1" in manager.active_agents

    manager.register_agent("agent-2", "model_training")
    assert len(manager.active_agents) == 2

    # Unregister agent
    manager.unregister_agent("agent-1")
    assert len(manager.active_agents) == 1
    assert "agent-1" not in manager.active_agents

    # Unregister non-existent agent (should log warning, not error)
    manager.unregister_agent("non-existent")
    assert len(manager.active_agents) == 1


@patch("sparky.oversight.resource_manager.psutil")
def test_can_spawn_agent_success(mock_psutil, test_config):
    """Test successful agent spawn check."""
    # Mock system resources (healthy)
    mock_psutil.cpu_percent.return_value = 50.0
    mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
    mock_psutil.disk_usage.return_value = Mock(free=200 * 1024**3)  # 200 GB

    manager = ResourceManager(config_path=test_config)

    # Should succeed
    assert manager.can_spawn_agent("general") is True


@patch("sparky.oversight.resource_manager.psutil")
def test_can_spawn_agent_max_concurrency(mock_psutil, test_config):
    """Test agent spawn blocked by max concurrency."""
    # Mock system resources (healthy)
    mock_psutil.cpu_percent.return_value = 50.0
    mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
    mock_psutil.disk_usage.return_value = Mock(free=200 * 1024**3)

    manager = ResourceManager(config_path=test_config)

    # Register 3 agents (max)
    manager.register_agent("agent-1", "general")
    manager.register_agent("agent-2", "general")
    manager.register_agent("agent-3", "general")

    # Should fail on 4th agent
    with pytest.raises(ResourceManagerError, match="at max concurrency"):
        manager.can_spawn_agent("general")


@patch("sparky.oversight.resource_manager.psutil")
def test_can_spawn_agent_training_limit(mock_psutil, test_config):
    """Test model training concurrency limit."""
    # Mock system resources (healthy)
    mock_psutil.cpu_percent.return_value = 50.0
    mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
    mock_psutil.disk_usage.return_value = Mock(free=200 * 1024**3)

    manager = ResourceManager(config_path=test_config)

    # Register 1 training agent (max for training)
    manager.register_agent("training-1", "model_training")

    # Should fail on 2nd training agent
    with pytest.raises(ResourceManagerError, match="at max training concurrency"):
        manager.can_spawn_agent("model_training")

    # But general agents should still work
    assert manager.can_spawn_agent("general") is True


@patch("sparky.oversight.resource_manager.psutil")
def test_can_spawn_agent_high_cpu(mock_psutil, test_config):
    """Test agent spawn blocked by high CPU."""
    # Mock high CPU usage
    mock_psutil.cpu_percent.return_value = 90.0  # Above 85% threshold
    mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
    mock_psutil.disk_usage.return_value = Mock(free=200 * 1024**3)

    manager = ResourceManager(config_path=test_config)

    # Should fail due to high CPU
    with pytest.raises(ResourceManagerError, match="CPU usage critical"):
        manager.can_spawn_agent("general")


@patch("sparky.oversight.resource_manager.psutil")
def test_can_spawn_agent_high_memory(mock_psutil, test_config):
    """Test agent spawn blocked by high memory."""
    # Mock high memory usage
    mock_psutil.cpu_percent.return_value = 50.0
    mock_psutil.virtual_memory.return_value = Mock(percent=85.0)  # Above 80% threshold
    mock_psutil.disk_usage.return_value = Mock(free=200 * 1024**3)

    manager = ResourceManager(config_path=test_config)

    # Should fail due to high memory
    with pytest.raises(ResourceManagerError, match="Memory usage critical"):
        manager.can_spawn_agent("general")


@patch("sparky.oversight.resource_manager.psutil")
def test_can_spawn_agent_low_disk(mock_psutil, test_config):
    """Test agent spawn blocked by low disk space."""
    # Mock low disk space
    mock_psutil.cpu_percent.return_value = 50.0
    mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
    mock_psutil.disk_usage.return_value = Mock(free=40 * 1024**3)  # 40 GB, below 50 GB threshold

    manager = ResourceManager(config_path=test_config)

    # Should fail due to low disk
    with pytest.raises(ResourceManagerError, match="Disk space critical"):
        manager.can_spawn_agent("general")


@patch("sparky.oversight.resource_manager.psutil")
def test_circuit_breaker_opens(mock_psutil, test_config):
    """Test circuit breaker opens after consecutive errors."""
    # Mock high CPU to trigger errors
    mock_psutil.cpu_percent.return_value = 90.0
    mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
    mock_psutil.disk_usage.return_value = Mock(free=200 * 1024**3)

    manager = ResourceManager(config_path=test_config)

    # Trigger 3 consecutive errors
    for i in range(3):
        with pytest.raises(ResourceManagerError):
            manager.can_spawn_agent("general")

    # Circuit breaker should now be open
    assert manager.circuit_breaker_open is True

    # Even with good resources, should fail
    mock_psutil.cpu_percent.return_value = 50.0
    with pytest.raises(CircuitBreakerOpen):
        manager.can_spawn_agent("general")


@patch("sparky.oversight.resource_manager.psutil")
def test_get_system_status(mock_psutil, test_config):
    """Test system status reporting."""
    # Mock system resources
    mock_psutil.cpu_percent.return_value = 65.0
    mock_psutil.virtual_memory.return_value = Mock(percent=70.0)
    mock_psutil.disk_usage.return_value = Mock(free=150 * 1024**3)

    manager = ResourceManager(config_path=test_config)
    manager.register_agent("agent-1", "general")
    manager.register_agent("agent-2", "model_training")

    status = manager.get_system_status()

    assert isinstance(status, SystemStatus)
    assert status.cpu_percent == 65.0
    assert status.memory_percent == 70.0
    assert status.disk_free_gb == pytest.approx(150.0, rel=0.1)
    assert status.active_agents == 2
    assert status.agents_by_type["general"] == 1
    assert status.agents_by_type["model_training"] == 1
    assert status.under_pressure is False  # Below pressure thresholds


@patch("sparky.oversight.resource_manager.psutil")
def test_system_status_warnings(mock_psutil, test_config):
    """Test system status includes warnings."""
    # Mock resources above alert thresholds
    mock_psutil.cpu_percent.return_value = 72.0  # Above 70% alert
    mock_psutil.virtual_memory.return_value = Mock(percent=68.0)  # Above 65% alert
    mock_psutil.disk_usage.return_value = Mock(free=80 * 1024**3)  # Below 100 GB alert

    manager = ResourceManager(config_path=test_config)
    status = manager.get_system_status()

    assert len(status.warnings) == 3
    assert any("High CPU usage" in w for w in status.warnings)
    assert any("High memory usage" in w for w in status.warnings)
    assert any("Low disk space" in w for w in status.warnings)


@patch("sparky.oversight.resource_manager.psutil")
def test_cleanup_stale_agents(mock_psutil, test_config):
    """Test cleanup of stale agents."""
    # Mock system resources
    mock_psutil.cpu_percent.return_value = 50.0
    mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
    mock_psutil.disk_usage.return_value = Mock(free=200 * 1024**3)

    manager = ResourceManager(config_path=test_config)

    # Register agents
    manager.register_agent("agent-1", "general")
    time.sleep(0.1)
    manager.register_agent("agent-2", "general")

    # Manually set start time for agent-1 to be stale
    from datetime import datetime, timedelta, timezone
    manager.active_agents["agent-1"].started_at = datetime.now(timezone.utc) - timedelta(hours=2)

    # Cleanup with 1 hour timeout
    stale = manager.cleanup_stale_agents(force_kill_timeout=3600)

    assert len(stale) == 1
    assert "agent-1" in stale
    assert len(manager.active_agents) == 1
    assert "agent-2" in manager.active_agents


def test_data_fetch_concurrency_limit(test_config):
    """Test data fetch concurrency limit."""
    # Use real psutil for this test
    manager = ResourceManager(config_path=test_config)

    # Register 2 data fetch agents (max)
    manager.register_agent("fetch-1", "data_fetch")
    manager.register_agent("fetch-2", "data_fetch")

    # Should fail on 3rd fetch agent
    with pytest.raises(ResourceManagerError, match="at max fetch concurrency"):
        manager.can_spawn_agent("data_fetch")

    # Unregister one
    manager.unregister_agent("fetch-1")

    # Now should succeed
    # We need to mock psutil for this check
    with patch("sparky.oversight.resource_manager.psutil") as mock_psutil:
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
        mock_psutil.disk_usage.return_value = Mock(free=200 * 1024**3)
        assert manager.can_spawn_agent("data_fetch") is True
