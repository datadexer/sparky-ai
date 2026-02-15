"""MLflow experiment tracking for Sparky AI."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Optional

import mlflow
import pandas as pd


class ExperimentTracker:
    """Track experiments using MLflow."""

    def __init__(self, experiment_name: str = "sparky", tracking_uri: str = "mlruns"):
        """Initialize experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking (local directory or remote server)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        mlflow.set_experiment(experiment_name)

    def _get_git_hash(self) -> str:
        """Get current git commit hash.

        Returns:
            Git commit hash (short form) or 'unknown' if not in a git repo
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def _get_data_manifest_hash(self) -> str:
        """Get hash of data manifest file for reproducibility.

        Returns:
            SHA256 hash of data_manifest.json or 'not_found' if file doesn't exist
        """
        manifest_path = Path("data/data_manifest.json")
        if not manifest_path.exists():
            return "not_found"

        try:
            with open(manifest_path, "r") as f:
                content = f.read()
            return hashlib.sha256(content.encode()).hexdigest()
        except Exception:
            return "error_reading"

    def log_experiment(
        self,
        name: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        artifacts: Optional[dict[str, str]] = None,
        features_used: Optional[list[str]] = None,
        date_range: Optional[tuple[str, str]] = None,
    ) -> str:
        """Log an experiment run to MLflow.

        Args:
            name: Run name
            config: Configuration parameters
            metrics: Metrics to log
            artifacts: Optional dict of {artifact_name: file_path} to log
            features_used: Optional list of feature names used
            date_range: Optional tuple of (start_date, end_date)

        Returns:
            MLflow run ID
        """
        with mlflow.start_run(run_name=name) as run:
            # Log git hash
            git_hash = self._get_git_hash()
            mlflow.log_param("git_hash", git_hash)

            # Log data manifest hash
            manifest_hash = self._get_data_manifest_hash()
            mlflow.log_param("data_manifest_hash", manifest_hash)

            # Log random seed if present in config
            if "random_seed" in config:
                mlflow.log_param("random_seed", config["random_seed"])

            # Log all config parameters
            for key, value in config.items():
                # MLflow has limitations on param types, convert complex types to strings
                if isinstance(value, (list, dict)):
                    mlflow.log_param(key, json.dumps(value))
                else:
                    mlflow.log_param(key, value)

            # Log features used
            if features_used:
                mlflow.log_param("features_used", json.dumps(features_used))

            # Log date range
            if date_range:
                mlflow.log_param("date_range_start", date_range[0])
                mlflow.log_param("date_range_end", date_range[1])

            # Log all metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    if Path(artifact_path).exists():
                        mlflow.log_artifact(artifact_path, artifact_name)

            return run.info.run_id

    def get_best_run(
        self,
        metric_name: str,
        maximize: bool = True,
        experiment_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get the best run based on a metric.

        Args:
            metric_name: Name of the metric to optimize
            maximize: If True, get run with highest metric value. If False, lowest.
            experiment_name: Optional experiment name (defaults to self.experiment_name)

        Returns:
            Dictionary with run information including run_id, metrics, and params
        """
        exp_name = experiment_name or self.experiment_name
        experiment = mlflow.get_experiment_by_name(exp_name)

        if experiment is None:
            raise ValueError(f"Experiment '{exp_name}' not found")

        # Search for runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"],
            max_results=1,
        )

        if runs.empty:
            raise ValueError(f"No runs found in experiment '{exp_name}'")

        best_run = runs.iloc[0]

        # Extract relevant information
        result = {
            "run_id": best_run["run_id"],
            "metrics": {
                col.replace("metrics.", ""): best_run[col]
                for col in best_run.index
                if col.startswith("metrics.")
            },
            "params": {
                col.replace("params.", ""): best_run[col]
                for col in best_run.index
                if col.startswith("params.")
            },
            "start_time": best_run.get("start_time"),
            "end_time": best_run.get("end_time"),
        }

        return result

    def list_runs(self, experiment_name: Optional[str] = None) -> list[dict[str, Any]]:
        """List all runs in an experiment.

        Args:
            experiment_name: Optional experiment name (defaults to self.experiment_name)

        Returns:
            List of run summaries with run_id, metrics, and params
        """
        exp_name = experiment_name or self.experiment_name
        experiment = mlflow.get_experiment_by_name(exp_name)

        if experiment is None:
            raise ValueError(f"Experiment '{exp_name}' not found")

        # Search for all runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            return []

        # Convert to list of dicts
        result = []
        for _, run in runs.iterrows():
            run_info = {
                "run_id": run["run_id"],
                "metrics": {
                    col.replace("metrics.", ""): run[col]
                    for col in run.index
                    if col.startswith("metrics.") and not pd.isna(run[col])
                },
                "params": {
                    col.replace("params.", ""): run[col]
                    for col in run.index
                    if col.startswith("params.") and not pd.isna(run[col])
                },
                "start_time": run.get("start_time"),
                "end_time": run.get("end_time"),
            }
            result.append(run_info)

        return result
