"""Tests for W&B experiment tracker."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from sparky.tracking.experiment import (
    ExperimentTracker,
    config_hash,
    clear_current_session,
    get_current_session,
    set_current_session,
)


@pytest.fixture
def tracker():
    """Create a tracker instance with mocked wandb."""
    with patch("sparky.tracking.experiment._ensure_wandb_login"):
        tracker = ExperimentTracker(
            experiment_name="test_exp",
            project="test-project",
            entity="test-entity",
        )
        return tracker


class TestConfigHash:
    def test_deterministic(self):
        cfg = {"model": "xgboost", "lr": 0.05}
        assert config_hash(cfg) == config_hash(cfg)

    def test_key_order_independent(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_different_configs_different_hashes(self):
        assert config_hash({"model": "xgb"}) != config_hash({"model": "catboost"})

    def test_returns_16_char_hex(self):
        h = config_hash({"test": True})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestExperimentTracker:
    def test_init(self):
        with patch("sparky.tracking.experiment._ensure_wandb_login") as mock_login:
            tracker = ExperimentTracker(
                experiment_name="my_exp",
                project="my-project",
                entity="my-entity",
            )
            mock_login.assert_called_once()
            assert tracker.experiment_name == "my_exp"
            assert tracker.project == "my-project"
            assert tracker.entity == "my-entity"

    def test_get_git_hash_success(self, tracker):
        with patch("sparky.tracking.experiment.subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="abc123\n", returncode=0)
            assert tracker._get_git_hash() == "abc123"

    def test_get_git_hash_failure(self, tracker):
        with patch(
            "sparky.tracking.experiment.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert tracker._get_git_hash() == "unknown"

    def test_is_duplicate_true(self, tracker):
        mock_run = MagicMock()
        with patch.object(tracker, "_fetch_runs", return_value=[mock_run]):
            assert tracker.is_duplicate("abc123") is True

    def test_is_duplicate_false(self, tracker):
        with patch.object(tracker, "_fetch_runs", return_value=[]):
            assert tracker.is_duplicate("abc123") is False

    def test_is_duplicate_handles_error(self, tracker):
        with patch.object(tracker, "_fetch_runs", side_effect=Exception("network")):
            assert tracker.is_duplicate("abc123") is False

    def test_log_experiment(self, tracker):
        mock_run = MagicMock()
        mock_run.id = "run_abc123"
        with patch("sparky.tracking.experiment.wandb") as mock_wandb:
            mock_wandb.init.return_value = mock_run
            with patch.object(tracker, "_get_git_hash", return_value="def456"):
                with patch.object(tracker, "_get_data_manifest_hash", return_value="hash789"):
                    run_id = tracker.log_experiment(
                        name="test_run",
                        config={"learning_rate": 0.01, "model_type": "xgboost"},
                        metrics={"accuracy": 0.95, "sharpe": 1.2},
                    )

                    assert run_id == "run_abc123"
                    mock_wandb.init.assert_called_once()
                    call_kwargs = mock_wandb.init.call_args[1]
                    assert call_kwargs["project"] == "test-project"
                    assert call_kwargs["entity"] == "test-entity"
                    assert call_kwargs["name"] == "test_run"
                    assert call_kwargs["group"] == "test_exp"
                    assert call_kwargs["config"]["learning_rate"] == 0.01
                    assert call_kwargs["config"]["config_hash"] is not None
                    assert call_kwargs["config"]["git_hash"] == "def456"
                    mock_wandb.log.assert_called_once_with({"accuracy": 0.95, "sharpe": 1.2})
                    mock_wandb.finish.assert_called_once()

    def test_log_experiment_with_features(self, tracker):
        mock_run = MagicMock()
        mock_run.id = "run_1"
        with patch("sparky.tracking.experiment.wandb") as mock_wandb:
            mock_wandb.init.return_value = mock_run
            with patch.object(tracker, "_get_git_hash", return_value="abc"):
                with patch.object(tracker, "_get_data_manifest_hash", return_value="def"):
                    tracker.log_experiment(
                        name="test",
                        config={"lr": 0.01},
                        metrics={"auc": 0.6},
                        features_used=["rsi", "macd"],
                    )
                    call_kwargs = mock_wandb.init.call_args[1]
                    assert call_kwargs["config"]["features_used"] == ["rsi", "macd"]

    def test_log_experiment_with_date_range(self, tracker):
        mock_run = MagicMock()
        mock_run.id = "run_1"
        with patch("sparky.tracking.experiment.wandb") as mock_wandb:
            mock_wandb.init.return_value = mock_run
            with patch.object(tracker, "_get_git_hash", return_value="abc"):
                with patch.object(tracker, "_get_data_manifest_hash", return_value="def"):
                    tracker.log_experiment(
                        name="test",
                        config={"lr": 0.01},
                        metrics={"auc": 0.6},
                        date_range=("2020-01-01", "2024-06-01"),
                    )
                    call_kwargs = mock_wandb.init.call_args[1]
                    assert call_kwargs["config"]["date_range_start"] == "2020-01-01"
                    assert call_kwargs["config"]["date_range_end"] == "2024-06-01"

    def test_get_best_run(self, tracker):
        mock_run = MagicMock()
        mock_run.id = "run_best"
        mock_run.name = "best_catboost"
        mock_run.summary = {"sharpe": 1.5, "auc": 0.65}
        mock_run.config = {"model_type": "catboost", "lr": 0.05}

        mock_api = MagicMock()
        mock_api.runs.return_value = [mock_run]

        with patch.object(tracker, "_get_api", return_value=mock_api):
            result = tracker.get_best_run("sharpe", maximize=True)
            assert result["run_id"] == "run_best"
            assert result["metrics"]["sharpe"] == 1.5
            assert result["params"]["model_type"] == "catboost"

    def test_get_best_run_no_runs(self, tracker):
        mock_api = MagicMock()
        mock_api.runs.return_value = []

        with patch.object(tracker, "_get_api", return_value=mock_api):
            with pytest.raises(ValueError, match="No runs found"):
                tracker.get_best_run("sharpe")

    def test_get_summary_empty(self, tracker):
        with patch.object(tracker, "_fetch_runs", return_value=[]):
            summary = tracker.get_summary()
            assert summary["total_runs"] == 0
            assert summary["best"] is None

    def test_get_summary_with_runs(self, tracker):
        run1 = MagicMock()
        run1.id = "run_1"
        run1.name = "catboost_v1"
        run1.config = {"model_type": "catboost"}
        run1.summary = {"sharpe": 1.2}

        run2 = MagicMock()
        run2.id = "run_2"
        run2.name = "xgboost_v1"
        run2.config = {"model_type": "xgboost"}
        run2.summary = {"sharpe": 0.8}

        with patch.object(tracker, "_fetch_runs", return_value=[run1, run2]):
            summary = tracker.get_summary()
            assert summary["total_runs"] == 2
            assert summary["best"]["sharpe"] == 1.2
            assert summary["best"]["run_id"] == "run_1"
            assert summary["by_model_type"]["catboost"] == 1
            assert summary["by_model_type"]["xgboost"] == 1
            assert len(summary["recent"]) == 2

    def test_get_summary_handles_error(self, tracker):
        with patch.object(tracker, "_fetch_runs", side_effect=Exception("network")):
            summary = tracker.get_summary()
            assert summary["total_runs"] == 0

    def test_list_runs(self, tracker):
        mock_run = MagicMock()
        mock_run.id = "run_1"
        mock_run.name = "test_run"
        mock_run.summary = {"sharpe": 1.0}
        mock_run.config = {"model_type": "xgboost"}
        mock_run.state = "finished"

        with patch.object(tracker, "_fetch_runs", return_value=[mock_run]):
            runs = tracker.list_runs()
            assert len(runs) == 1
            assert runs[0]["run_id"] == "run_1"
            assert runs[0]["state"] == "finished"

    def test_list_runs_empty(self, tracker):
        with patch.object(tracker, "_fetch_runs", return_value=[]):
            assert tracker.list_runs() == []

    def test_static_config_hash(self, tracker):
        h = tracker.config_hash({"a": 1})
        assert h == config_hash({"a": 1})


class TestSessionTracking:
    def setup_method(self):
        """Ensure clean session state before each test."""
        clear_current_session()

    def teardown_method(self):
        """Clean up session state after each test."""
        clear_current_session()

    def test_initial_session_is_none(self):
        assert get_current_session() is None

    def test_set_and_get_session(self):
        set_current_session("20260217_031500")
        assert get_current_session() == "20260217_031500"

    def test_clear_session(self):
        set_current_session("20260217_031500")
        clear_current_session()
        assert get_current_session() is None

    def test_set_session_overwrites_previous(self):
        set_current_session("session_a")
        set_current_session("session_b")
        assert get_current_session() == "session_b"

    def test_log_experiment_includes_session_id(self, tracker):
        """When a session is active, log_experiment should inject session_id into run config."""
        mock_run = MagicMock()
        mock_run.id = "run_sid_test"
        set_current_session("test_session_abc")
        try:
            with patch("sparky.tracking.experiment.wandb") as mock_wandb:
                mock_wandb.init.return_value = mock_run
                with patch.object(tracker, "_get_git_hash", return_value="abc"):
                    with patch.object(tracker, "_get_data_manifest_hash", return_value="def"):
                        tracker.log_experiment(
                            name="sid_test",
                            config={"model_type": "xgboost"},
                            metrics={"sharpe": 1.0},
                        )
                        call_kwargs = mock_wandb.init.call_args[1]
                        assert call_kwargs["config"]["session_id"] == "test_session_abc"
        finally:
            clear_current_session()

    def test_log_experiment_no_session_id_when_cleared(self, tracker):
        """When no session is active, session_id should not appear in run config."""
        mock_run = MagicMock()
        mock_run.id = "run_no_sid"
        clear_current_session()
        with patch("sparky.tracking.experiment.wandb") as mock_wandb:
            mock_wandb.init.return_value = mock_run
            with patch.object(tracker, "_get_git_hash", return_value="abc"):
                with patch.object(tracker, "_get_data_manifest_hash", return_value="def"):
                    tracker.log_experiment(
                        name="no_sid_test",
                        config={"model_type": "xgboost"},
                        metrics={"sharpe": 1.0},
                    )
                    call_kwargs = mock_wandb.init.call_args[1]
                    assert "session_id" not in call_kwargs["config"]

    def test_log_sweep_includes_session_id(self, tracker):
        """When a session is active, log_sweep should inject session_id into sweep config."""
        mock_run = MagicMock()
        mock_run.id = "run_sweep_sid"
        set_current_session("sweep_session_xyz")
        try:
            with patch("sparky.tracking.experiment.wandb") as mock_wandb:
                mock_wandb.init.return_value = mock_run
                with patch.object(tracker, "_get_git_hash", return_value="abc"):
                    with patch.object(tracker, "_get_data_manifest_hash", return_value="def"):
                        tracker.log_sweep(
                            name="test_sweep",
                            results=[{"config": {"model_type": "xgboost"}, "metrics": {"sharpe": 0.9}}],
                        )
                        call_kwargs = mock_wandb.init.call_args[1]
                        assert call_kwargs["config"]["session_id"] == "sweep_session_xyz"
        finally:
            clear_current_session()
