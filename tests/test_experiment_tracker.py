"""Tests for MLflow experiment tracker."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pandas as pd
import pytest

from sparky.tracking.experiment import ExperimentTracker


@pytest.fixture
def tracker():
    """Create a tracker instance with mocked MLflow."""
    with patch("sparky.tracking.experiment.mlflow"):
        tracker = ExperimentTracker(experiment_name="test_exp", tracking_uri="test_uri")
        return tracker


class TestExperimentTracker:
    """Test suite for ExperimentTracker."""

    def test_init_sets_tracking_uri(self):
        """Test that initialization sets MLflow tracking URI."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            tracker = ExperimentTracker(
                experiment_name="my_exp", tracking_uri="my_uri"
            )

            mock_mlflow.set_tracking_uri.assert_called_once_with("my_uri")
            mock_mlflow.set_experiment.assert_called_once_with("my_exp")
            assert tracker.experiment_name == "my_exp"
            assert tracker.tracking_uri == "my_uri"

    def test_get_git_hash_success(self, tracker):
        """Test that git hash is captured successfully."""
        with patch("sparky.tracking.experiment.subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="abc123\n", returncode=0)

            git_hash = tracker._get_git_hash()

            assert git_hash == "abc123"
            mock_run.assert_called_once_with(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )

    def test_get_git_hash_failure(self, tracker):
        """Test that git hash returns 'unknown' on failure."""
        with patch(
            "sparky.tracking.experiment.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            git_hash = tracker._get_git_hash()
            assert git_hash == "unknown"

    def test_get_data_manifest_hash_success(self, tracker):
        """Test that data manifest hash is calculated correctly."""
        manifest_content = '{"file1": {"sha256": "abc"}}'
        expected_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        with patch("sparky.tracking.experiment.Path.exists", return_value=True):
            with patch(
                "builtins.open", mock_open(read_data=manifest_content)
            ) as mock_file:
                with patch(
                    "sparky.tracking.experiment.hashlib.sha256"
                ) as mock_sha:
                    mock_sha.return_value.hexdigest.return_value = expected_hash

                    manifest_hash = tracker._get_data_manifest_hash()

                    assert manifest_hash == expected_hash
                    mock_file.assert_called_once_with(
                        Path("data/data_manifest.json"), "r"
                    )

    def test_get_data_manifest_hash_not_found(self, tracker):
        """Test that data manifest hash returns 'not_found' when file doesn't exist."""
        with patch("sparky.tracking.experiment.Path.exists", return_value=False):
            manifest_hash = tracker._get_data_manifest_hash()
            assert manifest_hash == "not_found"

    def test_log_experiment_basic(self, tracker):
        """Test logging an experiment with basic params and metrics."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            # Mock the context manager and run
            mock_run = MagicMock()
            mock_run.info.run_id = "run_123"
            mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

            # Mock helper methods
            with patch.object(tracker, "_get_git_hash", return_value="abc123"):
                with patch.object(
                    tracker, "_get_data_manifest_hash", return_value="def456"
                ):
                    config = {"learning_rate": 0.01, "batch_size": 32}
                    metrics = {"accuracy": 0.95, "loss": 0.05}

                    run_id = tracker.log_experiment(
                        name="test_run", config=config, metrics=metrics
                    )

                    assert run_id == "run_123"

                    # Verify git hash was logged
                    mock_mlflow.log_param.assert_any_call("git_hash", "abc123")

                    # Verify data manifest hash was logged
                    mock_mlflow.log_param.assert_any_call(
                        "data_manifest_hash", "def456"
                    )

                    # Verify config params were logged
                    mock_mlflow.log_param.assert_any_call("learning_rate", 0.01)
                    mock_mlflow.log_param.assert_any_call("batch_size", 32)

                    # Verify metrics were logged
                    mock_mlflow.log_metric.assert_any_call("accuracy", 0.95)
                    mock_mlflow.log_metric.assert_any_call("loss", 0.05)

    def test_log_experiment_with_random_seed(self, tracker):
        """Test that random seed is logged when present in config."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run_123"
            mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

            with patch.object(tracker, "_get_git_hash", return_value="abc123"):
                with patch.object(
                    tracker, "_get_data_manifest_hash", return_value="def456"
                ):
                    config = {"random_seed": 42, "learning_rate": 0.01}
                    metrics = {"accuracy": 0.95}

                    tracker.log_experiment(
                        name="test_run", config=config, metrics=metrics
                    )

                    # Verify random seed was logged
                    mock_mlflow.log_param.assert_any_call("random_seed", 42)

    def test_log_experiment_with_features_used(self, tracker):
        """Test that features are logged when provided."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run_123"
            mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

            with patch.object(tracker, "_get_git_hash", return_value="abc123"):
                with patch.object(
                    tracker, "_get_data_manifest_hash", return_value="def456"
                ):
                    config = {"learning_rate": 0.01}
                    metrics = {"accuracy": 0.95}
                    features = ["feature1", "feature2", "feature3"]

                    tracker.log_experiment(
                        name="test_run",
                        config=config,
                        metrics=metrics,
                        features_used=features,
                    )

                    # Verify features were logged as JSON
                    mock_mlflow.log_param.assert_any_call(
                        "features_used", json.dumps(features)
                    )

    def test_log_experiment_with_date_range(self, tracker):
        """Test that date range is logged when provided."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run_123"
            mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

            with patch.object(tracker, "_get_git_hash", return_value="abc123"):
                with patch.object(
                    tracker, "_get_data_manifest_hash", return_value="def456"
                ):
                    config = {"learning_rate": 0.01}
                    metrics = {"accuracy": 0.95}
                    date_range = ("2026-01-01", "2026-01-31")

                    tracker.log_experiment(
                        name="test_run",
                        config=config,
                        metrics=metrics,
                        date_range=date_range,
                    )

                    # Verify date range was logged
                    mock_mlflow.log_param.assert_any_call(
                        "date_range_start", "2026-01-01"
                    )
                    mock_mlflow.log_param.assert_any_call(
                        "date_range_end", "2026-01-31"
                    )

    def test_log_experiment_with_artifacts(self, tracker):
        """Test that artifacts are logged when provided."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run_123"
            mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

            with patch.object(tracker, "_get_git_hash", return_value="abc123"):
                with patch.object(
                    tracker, "_get_data_manifest_hash", return_value="def456"
                ):
                    with patch("sparky.tracking.experiment.Path.exists", return_value=True):
                        config = {"learning_rate": 0.01}
                        metrics = {"accuracy": 0.95}
                        artifacts = {
                            "model": "/path/to/model.pkl",
                            "plot": "/path/to/plot.png",
                        }

                        tracker.log_experiment(
                            name="test_run",
                            config=config,
                            metrics=metrics,
                            artifacts=artifacts,
                        )

                        # Verify artifacts were logged
                        mock_mlflow.log_artifact.assert_any_call(
                            "/path/to/model.pkl", "model"
                        )
                        mock_mlflow.log_artifact.assert_any_call(
                            "/path/to/plot.png", "plot"
                        )

    def test_log_experiment_with_complex_config(self, tracker):
        """Test that complex config values (lists, dicts) are JSON-serialized."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run_123"
            mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

            with patch.object(tracker, "_get_git_hash", return_value="abc123"):
                with patch.object(
                    tracker, "_get_data_manifest_hash", return_value="def456"
                ):
                    config = {
                        "layers": [64, 128, 256],
                        "optimizer": {"name": "adam", "lr": 0.001},
                    }
                    metrics = {"accuracy": 0.95}

                    tracker.log_experiment(
                        name="test_run", config=config, metrics=metrics
                    )

                    # Verify complex values were JSON-serialized
                    mock_mlflow.log_param.assert_any_call(
                        "layers", json.dumps([64, 128, 256])
                    )
                    mock_mlflow.log_param.assert_any_call(
                        "optimizer", json.dumps({"name": "adam", "lr": 0.001})
                    )

    def test_get_best_run_maximize(self, tracker):
        """Test getting the best run when maximizing a metric."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            # Mock experiment
            mock_experiment = Mock()
            mock_experiment.experiment_id = "exp_123"
            mock_mlflow.get_experiment_by_name.return_value = mock_experiment

            # Mock search results
            mock_runs_df = pd.DataFrame(
                {
                    "run_id": ["run_1"],
                    "metrics.accuracy": [0.95],
                    "metrics.loss": [0.05],
                    "params.learning_rate": ["0.01"],
                    "params.git_hash": ["abc123"],
                    "start_time": [pd.Timestamp("2026-01-01")],
                    "end_time": [pd.Timestamp("2026-01-01 01:00:00")],
                }
            )
            mock_mlflow.search_runs.return_value = mock_runs_df

            result = tracker.get_best_run("accuracy", maximize=True)

            assert result["run_id"] == "run_1"
            assert result["metrics"]["accuracy"] == 0.95
            assert result["metrics"]["loss"] == 0.05
            assert result["params"]["learning_rate"] == "0.01"
            assert result["params"]["git_hash"] == "abc123"

            # Verify search was called with correct order
            mock_mlflow.search_runs.assert_called_once_with(
                experiment_ids=["exp_123"],
                order_by=["metrics.accuracy DESC"],
                max_results=1,
            )

    def test_get_best_run_minimize(self, tracker):
        """Test getting the best run when minimizing a metric."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            mock_experiment = Mock()
            mock_experiment.experiment_id = "exp_123"
            mock_mlflow.get_experiment_by_name.return_value = mock_experiment

            mock_runs_df = pd.DataFrame(
                {
                    "run_id": ["run_1"],
                    "metrics.loss": [0.05],
                    "params.learning_rate": ["0.01"],
                    "start_time": [pd.Timestamp("2026-01-01")],
                    "end_time": [pd.Timestamp("2026-01-01 01:00:00")],
                }
            )
            mock_mlflow.search_runs.return_value = mock_runs_df

            result = tracker.get_best_run("loss", maximize=False)

            assert result["run_id"] == "run_1"

            # Verify search was called with ASC order
            mock_mlflow.search_runs.assert_called_once_with(
                experiment_ids=["exp_123"],
                order_by=["metrics.loss ASC"],
                max_results=1,
            )

    def test_get_best_run_experiment_not_found(self, tracker):
        """Test that error is raised when experiment is not found."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None

            with pytest.raises(ValueError, match="Experiment 'test_exp' not found"):
                tracker.get_best_run("accuracy")

    def test_get_best_run_no_runs(self, tracker):
        """Test that error is raised when no runs are found."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            mock_experiment = Mock()
            mock_experiment.experiment_id = "exp_123"
            mock_mlflow.get_experiment_by_name.return_value = mock_experiment
            mock_mlflow.search_runs.return_value = pd.DataFrame()

            with pytest.raises(ValueError, match="No runs found"):
                tracker.get_best_run("accuracy")

    def test_list_runs(self, tracker):
        """Test listing all runs in an experiment."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            mock_experiment = Mock()
            mock_experiment.experiment_id = "exp_123"
            mock_mlflow.get_experiment_by_name.return_value = mock_experiment

            mock_runs_df = pd.DataFrame(
                {
                    "run_id": ["run_1", "run_2"],
                    "metrics.accuracy": [0.95, 0.93],
                    "metrics.loss": [0.05, 0.07],
                    "params.learning_rate": ["0.01", "0.001"],
                    "start_time": [
                        pd.Timestamp("2026-01-01"),
                        pd.Timestamp("2026-01-02"),
                    ],
                    "end_time": [
                        pd.Timestamp("2026-01-01 01:00:00"),
                        pd.Timestamp("2026-01-02 01:00:00"),
                    ],
                }
            )
            mock_mlflow.search_runs.return_value = mock_runs_df

            results = tracker.list_runs()

            assert len(results) == 2
            assert results[0]["run_id"] == "run_1"
            assert results[0]["metrics"]["accuracy"] == 0.95
            assert results[0]["params"]["learning_rate"] == "0.01"
            assert results[1]["run_id"] == "run_2"
            assert results[1]["metrics"]["accuracy"] == 0.93

    def test_list_runs_empty(self, tracker):
        """Test listing runs when experiment has no runs."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            mock_experiment = Mock()
            mock_experiment.experiment_id = "exp_123"
            mock_mlflow.get_experiment_by_name.return_value = mock_experiment
            mock_mlflow.search_runs.return_value = pd.DataFrame()

            results = tracker.list_runs()

            assert results == []

    def test_list_runs_experiment_not_found(self, tracker):
        """Test that error is raised when experiment is not found."""
        with patch("sparky.tracking.experiment.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None

            with pytest.raises(ValueError, match="Experiment 'test_exp' not found"):
                tracker.list_runs()
