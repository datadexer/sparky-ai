"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def holdout_dates():
    """Holdout boundary dates from configs/holdout_policy.yaml.

    All tests that need holdout dates should use this fixture instead of
    hardcoding dates. When the policy changes, tests update automatically.
    """
    from sparky.oversight.holdout_guard import HoldoutGuard

    guard = HoldoutGuard()
    oos_start, embargo_days = guard.get_oos_boundary("btc")
    max_train = guard.get_max_training_date("btc")
    oos_ts = pd.Timestamp(oos_start, tz="UTC")
    return {
        "oos_start": oos_start,
        "embargo_days": embargo_days,
        "max_training_ts": max_train,
        "oos_start_ts": oos_ts,
    }
