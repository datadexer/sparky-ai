"""Tests for Kelly Criterion position sizing."""

import numpy as np
import pandas as pd

from sparky.portfolio.kelly_criterion import (
    apply_fixed_sizing,
    apply_kelly_sizing,
    calculate_kelly_parameters,
)


def test_calculate_kelly_parameters_basic():
    """Test Kelly parameter calculation with simple data."""
    # Create simple test data: 60% win rate, 2:1 win/loss ratio
    # Note: signals.shift(1) drops first element, so 8 returns -> 7 position returns
    returns = pd.Series([0.0, 0.02, -0.01, 0.02, -0.01, 0.02, 0.02, -0.01, 0.02])
    signals = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1])

    win_rate, win_loss_ratio, kelly_frac = calculate_kelly_parameters(returns, signals)

    # Expected: ~62.5% win rate (5/8), b~2.0, f*~0.4
    assert 0.55 <= win_rate <= 0.7, f"Win rate {win_rate} not in expected range"
    assert 1.5 <= win_loss_ratio <= 2.5, f"Win/loss ratio {win_loss_ratio} not in expected range"
    assert 0.3 <= kelly_frac <= 0.6, f"Kelly fraction {kelly_frac} not in expected range"


def test_calculate_kelly_parameters_negative_edge():
    """Test Kelly with negative edge (losing strategy)."""
    # Losing strategy: 40% win rate
    returns = pd.Series([0.01, -0.01, -0.01, -0.01, 0.01, -0.01])
    signals = pd.Series([1, 1, 1, 1, 1, 1])

    win_rate, win_loss_ratio, kelly_frac = calculate_kelly_parameters(returns, signals)

    # Negative edge should give negative Kelly fraction
    assert kelly_frac < 0, f"Expected negative Kelly for losing strategy, got {kelly_frac}"


def test_calculate_kelly_parameters_no_trades():
    """Test Kelly with no trades."""
    returns = pd.Series([0.01, 0.02, -0.01])
    signals = pd.Series([0, 0, 0])  # No positions

    win_rate, win_loss_ratio, kelly_frac = calculate_kelly_parameters(returns, signals)

    assert win_rate == 0.0
    assert win_loss_ratio == 0.0
    assert kelly_frac == 0.0


def test_apply_fixed_sizing():
    """Test fixed position sizing."""
    signals = pd.Series([1, 0, 1, 1, 0])
    position_sizes = apply_fixed_sizing(signals, fixed_size=1.0)

    assert (position_sizes == signals).all()


def test_apply_kelly_sizing_basic():
    """Test Kelly sizing with synthetic data."""
    # Create winning strategy data
    np.random.seed(42)
    n = 500

    # Simulate price returns with positive trend
    returns = pd.Series(np.random.normal(0.001, 0.02, n))
    signals = pd.Series([1] * n)  # Always LONG

    position_sizes = apply_kelly_sizing(
        signals=signals,
        returns=returns,
        fraction=0.25,
        max_leverage=2.0,
        lookback=252,
    )

    # Check properties
    assert len(position_sizes) == len(signals)
    assert (position_sizes >= 0).all(), "Position sizes should be non-negative"
    assert (position_sizes <= 2.0).all(), "Position sizes should respect max leverage"

    # First lookback period should be zero (not enough data)
    assert (position_sizes[:252] == 0).all(), "First lookback period should have zero positions"


def test_apply_kelly_sizing_respects_signals():
    """Test that Kelly sizing respects zero signals."""
    np.random.seed(42)
    n = 300

    returns = pd.Series(np.random.normal(0.001, 0.02, n))
    signals = pd.Series([1, 0, 1, 0] * (n // 4))

    position_sizes = apply_kelly_sizing(
        signals=signals,
        returns=returns,
        fraction=0.25,
        max_leverage=2.0,
        lookback=252,
    )

    # Where signal is 0, position size should be 0
    zero_signal_idx = signals[signals == 0].index
    assert (position_sizes.loc[zero_signal_idx] == 0).all(), "Zero signals should have zero positions"


def test_apply_kelly_sizing_max_leverage():
    """Test that Kelly sizing respects max leverage cap."""
    np.random.seed(42)
    n = 300

    # Create strong winning data that would suggest high Kelly
    returns = pd.Series(np.random.normal(0.01, 0.01, n))  # High positive returns, low vol
    signals = pd.Series([1] * n)

    position_sizes = apply_kelly_sizing(
        signals=signals,
        returns=returns,
        fraction=1.0,  # Full Kelly (aggressive)
        max_leverage=1.5,  # But cap at 1.5
        lookback=252,
    )

    # No position should exceed max leverage
    assert (position_sizes <= 1.5).all(), f"Max position size {position_sizes.max()} exceeds max_leverage 1.5"


def test_apply_kelly_sizing_lookback():
    """Test that lookback parameter works correctly."""
    np.random.seed(42)
    n = 200

    returns = pd.Series(np.random.normal(0.001, 0.02, n))
    signals = pd.Series([1] * n)

    lookback = 100
    position_sizes = apply_kelly_sizing(
        signals=signals,
        returns=returns,
        fraction=0.25,
        max_leverage=2.0,
        lookback=lookback,
    )

    # First lookback periods should be zero
    assert (position_sizes[:lookback] == 0).all()

    # After lookback, should have some non-zero positions (if signals are 1)
    # Note: might still be zero if Kelly fraction is negative
    assert len(position_sizes) == n


def test_kelly_formula_theoretical():
    """Test Kelly formula matches theoretical calculation."""
    # Perfect scenario: 60% win rate, 2:1 reward/risk
    # Kelly = (p*b - q) / b = (0.6*2 - 0.4) / 2 = 0.4

    # Create exact returns to match this (need 11 to get 10 position returns after shift)
    wins = [0.02] * 6  # 60% wins
    losses = [-0.01] * 4  # 40% losses
    returns = pd.Series([0.0] + wins + losses)  # Add dummy first value
    signals = pd.Series([1] * 11)

    win_rate, win_loss_ratio, kelly_frac = calculate_kelly_parameters(returns, signals)

    expected_win_rate = 0.6
    expected_ratio = 2.0
    expected_kelly = (expected_win_rate * expected_ratio - (1 - expected_win_rate)) / expected_ratio

    assert abs(win_rate - expected_win_rate) < 0.05, f"Win rate {win_rate} != {expected_win_rate}"
    assert abs(win_loss_ratio - expected_ratio) < 0.15, f"Ratio {win_loss_ratio} != {expected_ratio}"
    assert abs(kelly_frac - expected_kelly) < 0.1, f"Kelly {kelly_frac} != {expected_kelly}"
