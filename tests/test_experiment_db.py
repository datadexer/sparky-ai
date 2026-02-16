"""Tests for sparky.tracking.experiment_db — SQLite experiment tracking."""

import pytest
from pathlib import Path

from sparky.tracking.experiment_db import (
    get_db,
    config_hash,
    is_duplicate,
    log_experiment,
    get_best,
    get_summary,
)


@pytest.fixture
def db(tmp_path):
    """Create a temporary experiment database."""
    return get_db(db_path=tmp_path / "test_experiments.db")


class TestConfigHash:
    def test_deterministic(self):
        cfg = {"model": "xgboost", "lr": 0.05, "depth": 3}
        assert config_hash(cfg) == config_hash(cfg)

    def test_key_order_independent(self):
        cfg1 = {"a": 1, "b": 2}
        cfg2 = {"b": 2, "a": 1}
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_different_configs_different_hashes(self):
        cfg1 = {"model": "xgboost", "lr": 0.05}
        cfg2 = {"model": "catboost", "lr": 0.05}
        assert config_hash(cfg1) != config_hash(cfg2)

    def test_returns_hex_string(self):
        h = config_hash({"test": True})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestIsDuplicate:
    def test_not_duplicate_initially(self, db):
        assert not is_duplicate(db, "abc123")

    def test_duplicate_after_logging(self, db):
        h = config_hash({"model": "xgb"})
        log_experiment(db, config_hash=h, model_type="xgboost", sharpe=1.0)
        assert is_duplicate(db, h)


class TestLogExperiment:
    def test_log_and_retrieve(self, db):
        h = config_hash({"model": "catboost", "lr": 0.01})
        row_id = log_experiment(
            db,
            config_hash=h,
            model_type="catboost",
            approach_family="tree_ensemble",
            features=["rsi", "macd", "bb_width"],
            hyperparams={"lr": 0.01, "depth": 5},
            sharpe=1.24,
            wf_mean_sharpe=1.1,
            tier="TIER 1",
            wall_clock_seconds=120.5,
            notes="test run",
        )
        assert row_id is not None

        best = get_best(db, n=1)
        assert len(best) == 1
        assert best[0]["sharpe"] == 1.24
        assert best[0]["model_type"] == "catboost"

    def test_replace_on_duplicate_hash(self, db):
        h = config_hash({"model": "test"})
        log_experiment(db, config_hash=h, model_type="xgboost", sharpe=0.5)
        log_experiment(db, config_hash=h, model_type="xgboost", sharpe=0.8)
        # Should have replaced, not duplicated
        total = db.execute("SELECT COUNT(*) FROM experiments WHERE config_hash = ?", (h,)).fetchone()[0]
        assert total == 1
        best = get_best(db, n=1)
        assert best[0]["sharpe"] == 0.8


class TestGetBest:
    def test_returns_sorted(self, db):
        for i, sharpe in enumerate([0.3, 1.5, 0.8, 1.2]):
            log_experiment(
                db,
                config_hash=f"hash_{i}",
                model_type="xgboost",
                sharpe=sharpe,
            )
        best = get_best(db, n=3)
        assert len(best) == 3
        assert best[0]["sharpe"] == 1.5
        assert best[1]["sharpe"] == 1.2
        assert best[2]["sharpe"] == 0.8

    def test_skips_null_sharpe(self, db):
        log_experiment(db, config_hash="a", model_type="xgboost", sharpe=None)
        log_experiment(db, config_hash="b", model_type="xgboost", sharpe=1.0)
        best = get_best(db, n=5)
        assert len(best) == 1


class TestGetSummary:
    def test_empty_db(self, db):
        summary = get_summary(db)
        assert summary["total_experiments"] == 0
        assert summary["best"] is None

    def test_populated_db(self, db):
        log_experiment(db, config_hash="h1", model_type="xgboost", sharpe=0.8, tier="TIER 3")
        log_experiment(db, config_hash="h2", model_type="catboost", sharpe=1.2, tier="TIER 1")
        log_experiment(db, config_hash="h3", model_type="xgboost", sharpe=0.5, tier="TIER 4")

        summary = get_summary(db)
        assert summary["total_experiments"] == 3
        assert summary["by_model_type"]["xgboost"] == 2
        assert summary["by_model_type"]["catboost"] == 1
        assert summary["best"]["sharpe"] == 1.2
        assert len(summary["recent"]) == 3
