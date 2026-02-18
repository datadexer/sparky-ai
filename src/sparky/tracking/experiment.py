"""Weights & Biases experiment tracking for Sparky AI.

Provides experiment logging, dedup via config hashing, and querying.

Usage:
    tracker = ExperimentTracker()

    # Check for duplicates before running
    h = tracker.config_hash({"model": "xgboost", "lr": 0.05})
    if tracker.is_duplicate(h):
        print("SKIP — already ran")
    else:
        run_id = tracker.log_experiment("xgb_v1", config={...}, metrics={...})

    # Query results
    best = tracker.get_best_run("sharpe")
    summary = tracker.get_summary()
"""

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

import wandb
import yaml

logger = logging.getLogger(__name__)

WANDB_PROJECT = "sparky-ai"
WANDB_ENTITY = "datadex_ai"

_current_session_id: str | None = None


def set_current_session(session_id: str) -> None:
    """Set the active CEO session ID for experiment tracking."""
    global _current_session_id
    _current_session_id = session_id


def clear_current_session() -> None:
    """Clear the active CEO session ID."""
    global _current_session_id
    _current_session_id = None


def get_current_session() -> str | None:
    """Return the currently active CEO session ID, or None."""
    return _current_session_id


def _load_wandb_key() -> Optional[str]:
    """Load W&B API key from secrets.yaml or environment."""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    secrets_path = Path("configs/secrets.yaml")
    if secrets_path.exists():
        with open(secrets_path) as f:
            secrets = yaml.safe_load(f)
        if secrets and "wandb" in secrets:
            return secrets["wandb"].get("api_key")
    return None


def _ensure_wandb_login() -> None:
    """Log in to W&B if not already authenticated."""
    key = _load_wandb_key()
    if key:
        wandb.login(key=key, relogin=False)


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic hash of an experiment config.

    Sorts keys recursively to ensure stable hashing regardless of
    insertion order.

    Args:
        config: Dictionary of model config (hyperparams, features, etc.)

    Returns:
        SHA-256 hex digest (first 16 chars).
    """
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _build_run_name(config: dict[str, Any], metrics: dict[str, float]) -> str:
    """Build a descriptive run name from config and metrics.

    Generates names like: xgb_lr0.05_d6_n200_S0.83

    Args:
        config: Model configuration dict.
        metrics: Metrics dict (uses 'sharpe' if present).

    Returns:
        Descriptive run name string.
    """
    # Model type prefix
    model = config.get("model_type", config.get("model", "unknown"))
    prefix_map = {
        "xgboost": "xgb",
        "lightgbm": "lgbm",
        "catboost": "cat",
    }
    prefix = prefix_map.get(model.lower(), model[:4].lower())

    # Key hyperparameters (abbreviated)
    param_abbrevs = [
        (["learning_rate", "lr", "eta"], "lr"),
        (["max_depth", "depth"], "d"),
        (["n_estimators", "num_boost_round", "iterations"], "n"),
        (["num_leaves"], "nl"),
        (["reg_alpha", "l1_regularization"], "L1"),
        (["reg_lambda", "l2_regularization", "l2_leaf_reg"], "L2"),
    ]
    parts = [prefix]
    for keys, abbrev in param_abbrevs:
        for key in keys:
            if key in config:
                val = config[key]
                if isinstance(val, float):
                    parts.append(f"{abbrev}{val:g}")
                else:
                    parts.append(f"{abbrev}{val}")
                break

    # Sharpe from metrics
    sharpe = metrics.get("sharpe")
    if sharpe is not None:
        parts.append(f"S{sharpe:.2f}")

    return "_".join(parts)


class ExperimentTracker:
    """Track experiments using Weights & Biases with config-hash dedup.

    Wraps wandb to provide:
    - Experiment logging with automatic git/data provenance
    - Config hashing for dedup (is_duplicate check before running)
    - Best-run and summary queries via the wandb API
    """

    def __init__(
        self,
        experiment_name: str = "sparky",
        project: str = WANDB_PROJECT,
        entity: str = WANDB_ENTITY,
    ):
        """Initialize experiment tracker.

        Args:
            experiment_name: Used as run group in W&B.
            project: W&B project name.
            entity: W&B entity (team or user).
        """
        self.experiment_name = experiment_name
        self.project = project
        self.entity = entity
        _ensure_wandb_login()

    def _get_git_hash(self) -> str:
        """Get current git commit hash."""
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
        """Get hash of data manifest file for reproducibility."""
        manifest_path = Path("data/data_manifest.json")
        if not manifest_path.exists():
            return "not_found"
        try:
            with open(manifest_path, "r") as f:
                content = f.read()
            return hashlib.sha256(content.encode()).hexdigest()
        except Exception:
            return "error_reading"

    @staticmethod
    def config_hash(config: dict[str, Any]) -> str:
        """Deterministic hash of an experiment config. Delegates to module-level function."""
        return config_hash(config)

    def _get_api(self) -> wandb.Api:
        """Get a wandb API instance."""
        return wandb.Api()

    def _fetch_runs(self, filters: Optional[dict] = None) -> list:
        """Fetch runs from W&B API with optional filters.

        Args:
            filters: MongoDB-style filter dict for wandb API.

        Returns:
            List of wandb Run objects.
        """
        api = self._get_api()
        path = f"{self.entity}/{self.project}"
        return list(api.runs(path, filters=filters or {}))

    def is_duplicate(self, cfg_hash: str) -> bool:
        """Check if an experiment config has already been run.

        Searches W&B for any run with a matching config_hash in config.

        Args:
            cfg_hash: Hash from config_hash().

        Returns:
            True if this config was already logged.
        """
        try:
            runs = self._fetch_runs(
                filters={"config.config_hash": cfg_hash}
            )
            return len(runs) > 0
        except Exception as e:
            logger.warning(f"[TRACKER] is_duplicate check failed: {e}")
            return False

    def log_experiment(
        self,
        name: Optional[str],
        config: dict[str, Any],
        metrics: dict[str, float],
        artifacts: Optional[dict[str, str]] = None,
        features_used: Optional[list[str]] = None,
        date_range: Optional[tuple[str, str]] = None,
        tags: Optional[list[str]] = None,
        job_type: Optional[str] = None,
        group: Optional[str] = None,
    ) -> str:
        """Log an experiment run to W&B.

        Automatically computes and stores a config_hash for dedup.

        Args:
            name: Run name. If None, auto-generates a descriptive name from
                config and metrics (e.g. 'xgb_lr0.05_d6_S0.83').
            config: Configuration parameters.
            metrics: Metrics to log.
            artifacts: Optional dict of {artifact_name: file_path} to log.
            features_used: Optional list of feature names used.
            date_range: Optional tuple of (start_date, end_date).
            tags: Optional list of tags for the run.
            job_type: Optional W&B job type for step-level grouping
                (e.g. 'sweep', 'regime', 'ensemble', 'novel').
            group: Optional W&B group for collapsible parent/child runs
                (e.g. 'regime_vol', 'regime_adx'). Defaults to experiment_name.

        Returns:
            W&B run ID.
        """
        if name is None:
            name = _build_run_name(config, metrics)

        cfg_hash = config_hash(config)
        git_hash = self._get_git_hash()
        manifest_hash = self._get_data_manifest_hash()

        run_config = {
            **config,
            "config_hash": cfg_hash,
            "git_hash": git_hash,
            "data_manifest_hash": manifest_hash,
        }
        if features_used:
            run_config["features_used"] = features_used
        if date_range:
            run_config["date_range_start"] = date_range[0]
            run_config["date_range_end"] = date_range[1]
        sid = get_current_session()
        if sid:
            run_config["session_id"] = sid

        init_kwargs: dict[str, Any] = {
            "project": self.project,
            "entity": self.entity,
            "name": name,
            "group": group or self.experiment_name,
            "config": run_config,
            "tags": tags or [],
            "reinit": True,
        }
        if job_type:
            init_kwargs["job_type"] = job_type

        run = wandb.init(**init_kwargs)

        wandb.log(metrics)

        if artifacts:
            for artifact_name, artifact_path in artifacts.items():
                if Path(artifact_path).exists():
                    art = wandb.Artifact(artifact_name, type="result")
                    art.add_file(artifact_path)
                    run.log_artifact(art)

        run_id = run.id
        wandb.finish()

        logger.info(f"[TRACKER] Logged run {name} ({cfg_hash}) to W&B: {run_id}")
        return run_id

    def log_sweep(
        self,
        name: str,
        results: list[dict[str, Any]],
        summary_metrics: Optional[dict[str, float]] = None,
        tags: Optional[list[str]] = None,
        job_type: Optional[str] = None,
        group: Optional[str] = None,
    ) -> str:
        """Log a complete sweep as a single W&B run with a results table.

        Use this instead of calling log_experiment() per config. Creates one
        run with a wandb.Table containing all configs and their metrics.

        Args:
            name: Sweep name (e.g. "stage1_screening_27configs").
            results: List of dicts, each with 'config' and 'metrics' keys.
            summary_metrics: Optional top-level metrics (e.g. best_auc, best_model).
            tags: Optional list of tags for the run.
            job_type: Optional W&B job type for step-level grouping
                (e.g. 'sweep', 'regime', 'ensemble', 'novel').
            group: Optional W&B group for collapsible parent/child runs.
                Defaults to experiment_name.

        Returns:
            W&B run ID.
        """
        git_hash = self._get_git_hash()
        manifest_hash = self._get_data_manifest_hash()

        sweep_config: dict[str, Any] = {
            "sweep_type": "two_stage",
            "total_configs": len(results),
            "git_hash": git_hash,
            "data_manifest_hash": manifest_hash,
        }
        sid = get_current_session()
        if sid:
            sweep_config["session_id"] = sid

        init_kwargs: dict[str, Any] = {
            "project": self.project,
            "entity": self.entity,
            "name": name,
            "group": group or self.experiment_name,
            "config": sweep_config,
            "tags": tags or [],
            "reinit": True,
        }
        if job_type:
            init_kwargs["job_type"] = job_type

        run = wandb.init(**init_kwargs)

        # Build results table
        columns = ["model", "config_hash", "auc", "accuracy", "sharpe", "elapsed_s"]
        table = wandb.Table(columns=columns)
        for r in results:
            cfg = r.get("config", {})
            metrics = r.get("metrics", {})
            table.add_data(
                cfg.get("model_type", cfg.get("model", "unknown")),
                config_hash(cfg),
                metrics.get("auc", None),
                metrics.get("accuracy", None),
                metrics.get("sharpe", None),
                metrics.get("elapsed_seconds", None),
            )

        wandb.log({"sweep_results": table})

        # Log summary metrics
        if summary_metrics:
            wandb.log(summary_metrics)

        run_id = run.id
        wandb.finish()
        logger.info(f"[TRACKER] Logged sweep {name} ({len(results)} configs) to W&B: {run_id}")
        return run_id

    def count_runs(self, tags: Optional[list[str]] = None) -> int:
        """Count runs, optionally filtered by tags.

        Args:
            tags: If provided, only count runs that have ALL of these tags.

        Returns:
            Number of matching runs.
        """
        try:
            filters = {}
            if tags:
                filters["tags"] = {"$all": tags}
            runs = self._fetch_runs(filters=filters)
            return len(runs)
        except Exception as e:
            logger.warning(f"[TRACKER] count_runs failed: {e}")
            return 0

    def best_metric(
        self,
        metric: str,
        tags: Optional[list[str]] = None,
        maximize: bool = True,
    ) -> Optional[float]:
        """Get the best value for a metric, optionally filtered by tags.

        Args:
            metric: Metric name (e.g. "sharpe").
            tags: If provided, only consider runs with ALL of these tags.
            maximize: If True, return max; otherwise return min.

        Returns:
            Best metric value, or None if no runs found.
        """
        try:
            filters = {}
            if tags:
                filters["tags"] = {"$all": tags}
            runs = self._fetch_runs(filters=filters)
            values = []
            for r in runs:
                v = r.summary.get(metric)
                if v is not None:
                    values.append(float(v))
            if not values:
                return None
            return max(values) if maximize else min(values)
        except Exception as e:
            logger.warning(f"[TRACKER] best_metric failed: {e}")
            return None

    def get_best_run(
        self,
        metric_name: str,
        maximize: bool = True,
    ) -> dict[str, Any]:
        """Get the best run based on a metric.

        Args:
            metric_name: Name of the metric to optimize (e.g. "sharpe").
            maximize: If True, get run with highest metric value.

        Returns:
            Dict with run_id, metrics, params, and summary.
        """
        order = f"-summary_metrics.{metric_name}" if maximize else f"+summary_metrics.{metric_name}"
        api = self._get_api()
        path = f"{self.entity}/{self.project}"
        runs = api.runs(path, filters={"group": self.experiment_name}, order=order, per_page=1)
        runs_list = list(runs)

        if not runs_list:
            raise ValueError(f"No runs found for group '{self.experiment_name}'")

        best = runs_list[0]
        return {
            "run_id": best.id,
            "name": best.name,
            "metrics": dict(best.summary),
            "params": dict(best.config),
        }

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all experiments in this group.

        Returns:
            Dict with total_runs, by_model_type counts, best run, and recent runs.
        """
        try:
            runs = self._fetch_runs()
        except Exception as e:
            logger.warning(f"[TRACKER] get_summary failed: {e}")
            return {"total_runs": 0, "by_model_type": {}, "best": None, "recent": []}

        if not runs:
            return {"total_runs": 0, "by_model_type": {}, "best": None, "recent": []}

        total = len(runs)

        # Count by model type
        by_model: dict[str, int] = {}
        for r in runs:
            mt = r.config.get("model_type", "unknown")
            by_model[mt] = by_model.get(mt, 0) + 1

        # Best by Sharpe
        best = None
        best_sharpe = None
        for r in runs:
            s = r.summary.get("sharpe")
            if s is not None and (best_sharpe is None or s > best_sharpe):
                best_sharpe = s
                best = {
                    "run_id": r.id,
                    "name": r.name,
                    "sharpe": s,
                    "model_type": r.config.get("model_type", "unknown"),
                }

        # Recent 5 (runs are returned newest first by default)
        recent = []
        for r in runs[:5]:
            recent.append({
                "run_id": r.id,
                "name": r.name,
                "model_type": r.config.get("model_type", "unknown"),
                "sharpe": r.summary.get("sharpe"),
            })

        return {
            "total_runs": total,
            "by_model_type": by_model,
            "best": best,
            "recent": recent,
        }

    def list_runs(self) -> list[dict[str, Any]]:
        """List all runs in this experiment group.

        Returns:
            List of run summaries with run_id, metrics, and params.
        """
        runs = self._fetch_runs()
        result = []
        for r in runs:
            result.append({
                "run_id": r.id,
                "name": r.name,
                "metrics": dict(r.summary),
                "params": dict(r.config),
                "state": r.state,
            })
        return result
