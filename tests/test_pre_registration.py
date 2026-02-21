"""Tests for strategy pre-registration."""

import pytest

from sparky.tracking.pre_registration import (
    PreRegistration,
    StrategyFamily,
    check_trial_budget,
    load_pre_registration,
    save_pre_registration,
)


@pytest.fixture
def sample_reg():
    return PreRegistration(
        project_id="p002",
        strategy_families=[
            StrategyFamily(name="garch_vol", parameter_ranges={"window": [126, 252]}, max_configs=20),
            StrategyFamily(name="funding_carry", parameter_ranges={"threshold": [0.05, 0.1]}, max_configs=15),
        ],
        kill_criteria={"max_drawdown": -0.5, "min_sharpe_after_10": 0.4},
        advance_criteria={"min_sharpe": 1.0, "min_dsr": 0.95},
        max_total_trials=50,
        holdout_date="2024-01-01",
        notes="P002 tier 1 registration",
    )


class TestRoundTrip:
    def test_save_and_load(self, tmp_path, sample_reg):
        path = tmp_path / "reg.yaml"
        save_pre_registration(sample_reg, path)
        loaded = load_pre_registration(path)
        assert loaded.project_id == "p002"
        assert len(loaded.strategy_families) == 2
        assert loaded.max_total_trials == 50
        assert loaded.holdout_date == "2024-01-01"

    def test_preserves_all_fields(self, tmp_path, sample_reg):
        path = tmp_path / "reg.yaml"
        save_pre_registration(sample_reg, path)
        loaded = load_pre_registration(path)
        assert loaded.kill_criteria == {"max_drawdown": -0.5, "min_sharpe_after_10": 0.4}
        assert loaded.advance_criteria == {"min_sharpe": 1.0, "min_dsr": 0.95}
        assert loaded.notes == "P002 tier 1 registration"
        assert loaded.strategy_families[0].name == "garch_vol"
        assert loaded.strategy_families[1].max_configs == 15


class TestHashTamperDetection:
    def test_tampered_content_raises(self, tmp_path, sample_reg):
        path = tmp_path / "reg.yaml"
        save_pre_registration(sample_reg, path)

        content = path.read_text()
        tampered = content.replace("max_total_trials: 50", "max_total_trials: 999")
        path.write_text(tampered)

        with pytest.raises(ValueError, match="Hash mismatch"):
            load_pre_registration(path)

    def test_missing_hash_raises(self, tmp_path):
        path = tmp_path / "reg.yaml"
        path.write_text("project_id: p002\n")

        with pytest.raises(ValueError, match="No SHA-256 hash"):
            load_pre_registration(path)


class TestTrialBudget:
    def test_within_budget(self, sample_reg):
        assert check_trial_budget(sample_reg, 10) is True

    def test_at_budget_limit(self, sample_reg):
        assert check_trial_budget(sample_reg, 50) is False

    def test_over_budget(self, sample_reg):
        assert check_trial_budget(sample_reg, 51) is False

    def test_family_budget(self, sample_reg):
        assert check_trial_budget(sample_reg, 19, family="garch_vol") is True
        assert check_trial_budget(sample_reg, 20, family="garch_vol") is False

    def test_unknown_family_uses_global(self, sample_reg):
        assert check_trial_budget(sample_reg, 10, family="unknown") is True
