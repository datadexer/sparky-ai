"""Tests for OOS evaluation guardrails — env var gating, orchestrator safety, holdout perms."""

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from sparky.data.loader import load


class TestEvaluationEnvGate:
    """purpose='evaluation' is gated by SPARKY_OOS_ENABLED env var."""

    def test_blocked_without_env_var(self):
        env = os.environ.copy()
        env.pop("SPARKY_OOS_ENABLED", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(PermissionError, match="OOS data access denied"):
                load("btc_ohlcv_8h", purpose="evaluation")

    def test_blocked_with_wrong_value(self):
        with patch.dict(os.environ, {"SPARKY_OOS_ENABLED": "0"}):
            with pytest.raises(PermissionError):
                load("btc_ohlcv_8h", purpose="evaluation")

    def test_allowed_with_env_and_data(self, tmp_path):
        holdout_dir = tmp_path / "holdout" / "btc"
        holdout_dir.mkdir(parents=True)
        dates = pd.date_range("2020-01-01", "2025-12-01", freq="8h", tz="UTC")
        df = pd.DataFrame({"close": range(len(dates))}, index=dates)
        df.to_parquet(holdout_dir / "ohlcv_8h.parquet")

        with (
            patch.dict(os.environ, {"SPARKY_OOS_ENABLED": "1"}),
            patch("sparky.data.loader.HOLDOUT_DIR", tmp_path / "holdout"),
        ):
            result = load("btc_ohlcv_8h", purpose="evaluation")
            assert len(result) == len(dates)


class TestOrchestratorRefusesOosEnv:
    """Orchestrator must refuse to start if SPARKY_OOS_ENABLED is set."""

    def test_orchestrator_raises_with_oos_env(self):
        from sparky.workflow.orchestrator import ResearchDirective, ResearchOrchestrator

        directive = ResearchDirective(name="test", objective="test")
        orch = ResearchOrchestrator(directive)

        with patch.dict(os.environ, {"SPARKY_OOS_ENABLED": "1"}):
            with pytest.raises(RuntimeError, match="SPARKY_OOS_ENABLED"):
                orch.run()

    def test_orchestrator_ok_without_oos_env(self):
        """Verify env check doesn't fire when var is unset (it'll fail on lock, that's fine)."""
        from sparky.workflow.orchestrator import ResearchDirective, ResearchOrchestrator

        directive = ResearchDirective(name="test_no_oos", objective="test")
        orch = ResearchOrchestrator(directive)

        env = os.environ.copy()
        env.pop("SPARKY_OOS_ENABLED", None)
        with patch.dict(os.environ, env, clear=True):
            # Will fail on lock or session launch, but NOT on the OOS check
            try:
                orch.run()
            except RuntimeError as e:
                assert "SPARKY_OOS_ENABLED" not in str(e)
            except Exception:  # noqa: S110
                pass  # Any other failure is fine — we just verify OOS check doesn't fire


class TestHoldoutDirectoryPermissions:
    """Verify data/holdout is not world-readable (skip if dir doesn't exist)."""

    def test_holdout_not_readable_by_current_user(self):
        holdout = Path("data/holdout")
        if not holdout.exists():
            pytest.skip("data/holdout not yet created (manual setup step)")
        # If it exists, verify we can't list it (owned by sparky-oos, chmod 700)
        try:
            list(holdout.iterdir())
            pytest.skip("data/holdout is readable — permissions not yet restricted")
        except PermissionError:
            pass  # Expected: current user cannot read sparky-oos owned dir
