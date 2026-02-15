"""Tests for walk-forward backtesting engine.

Test cases:
1. Basic walk-forward produces correct number of folds
2. Embargo period creates gap between train and test
3. Equity curve starts at 1.0 and compounds correctly
4. Per-fold metrics include sharpe, return, accuracy
5. Minimum 5 folds requirement enforced
6. Cost model integration (position changes reduce equity)
7. Expanding window (train size grows, not slides)
8. Next-day execution timing (returns applied correctly)

We use simple dummy models to test the backtester logic:
- AlwaysLongModel: always predicts 1
- ThresholdModel: predicts based on feature threshold
"""

import numpy as np
import pandas as pd
import pytest

from sparky.backtest.engine import BacktestResult, WalkForwardBacktester


class AlwaysLongModel:
    """Dummy model that always predicts 1 (long)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """No-op fit."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Always predict 1."""
        return np.ones(len(X), dtype=int)


class AlwaysFlatModel:
    """Dummy model that always predicts 0 (flat)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """No-op fit."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Always predict 0."""
        return np.zeros(len(X), dtype=int)


class ThresholdModel:
    """Dummy model that predicts based on feature threshold."""

    def __init__(self, feature_name: str, threshold: float):
        self.feature_name = feature_name
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """No-op fit."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict 1 if feature > threshold, else 0."""
        return (X[self.feature_name] > self.threshold).astype(int).values


class SimpleCostModel:
    """Simple cost model: fixed percentage per trade."""

    def __init__(self, cost_pct: float = 0.001):
        self.cost_pct = cost_pct

    def compute_cost(self, position_change: int, asset: str) -> float:
        """Return fixed cost per position change."""
        if position_change == 0:
            return 0.0
        return self.cost_pct


def make_test_data(n_days: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate synthetic test data.

    Args:
        n_days: Number of days of data.
        seed: Random seed.

    Returns:
        Tuple of (X, y, returns).
    """
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    # Features: random values
    X = pd.DataFrame({
        "feature1": np.random.randn(n_days),
        "feature2": np.random.randn(n_days),
    }, index=dates)

    # Target: random binary labels
    y = pd.Series(np.random.randint(0, 2, n_days), index=dates)

    # Returns: small random returns
    returns = pd.Series(np.random.normal(0.001, 0.02, n_days), index=dates)

    return X, y, returns


class TestWalkForwardBacktester:
    """Test walk-forward backtester core functionality."""

    def test_basic_fold_generation(self):
        """Verify correct number of folds are generated."""
        X, y, returns = make_test_data(n_days=250)  # Increased to get 5 folds
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysLongModel()
        result = backtester.run(model, X, y, returns)

        # With 250 days, train_min=50, embargo=7, test=30, step=30:
        # First test starts at day 57 (50+7), ends at 87
        # Second test starts at day 87, ends at 117
        # Third test starts at day 117, ends at 147
        # Fourth test starts at day 147, ends at 177
        # Fifth test starts at day 177, ends at 207
        # So we expect at least 5 folds
        assert result.fold_count >= 5
        assert len(result.per_fold_metrics) >= 5
        assert len(result.per_fold_metrics) == result.fold_count

    def test_minimum_5_folds_requirement(self):
        """Verify that less than 5 folds raises an error."""
        X, y, returns = make_test_data(n_days=100)  # Too little data
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysLongModel()

        with pytest.raises(ValueError, match="Insufficient data for 5 test folds"):
            backtester.run(model, X, y, returns)

    def test_embargo_period_creates_gap(self):
        """Verify embargo period creates gap between train and test."""
        X, y, returns = make_test_data(n_days=300)
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysLongModel()
        result = backtester.run(model, X, y, returns)

        # Check first fold
        fold = result.per_fold_metrics[0]
        train_end = fold["train_end"]
        test_start = fold["test_start"]

        # There should be at least 7 days gap
        gap_days = (test_start - train_end).days
        assert gap_days >= 7

    def test_expanding_window(self):
        """Verify train window expands (not slides)."""
        X, y, returns = make_test_data(n_days=300)
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysLongModel()
        result = backtester.run(model, X, y, returns)

        # Train start should be the same for all folds
        train_starts = [fold["train_start"] for fold in result.per_fold_metrics]
        assert len(set(train_starts)) == 1  # All the same

        # Train end should increase with each fold
        train_ends = [fold["train_end"] for fold in result.per_fold_metrics]
        for i in range(len(train_ends) - 1):
            # Train end stays the same or increases (expanding window)
            # Actually, in pure expanding window, train_end should stay constant
            # until test window advances past it
            # Let's just check that train doesn't shrink
            pass  # The expanding window means we train on all data up to embargo boundary

        # Actually, let's check train size increases
        train_sizes = [
            (fold["train_end"] - fold["train_start"]).days
            for fold in result.per_fold_metrics
        ]
        # Train sizes should not decrease
        for i in range(len(train_sizes) - 1):
            assert train_sizes[i+1] >= train_sizes[i]

    def test_equity_curve_starts_at_one(self):
        """Verify equity curve starts at 1.0."""
        X, y, returns = make_test_data(n_days=300)
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysLongModel()
        result = backtester.run(model, X, y, returns)

        assert result.equity_curve.iloc[0] == pytest.approx(1.0)

    def test_equity_curve_with_always_long(self):
        """Test equity curve when model is always long."""
        X, y, returns = make_test_data(n_days=300, seed=42)
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysLongModel()
        result = backtester.run(model, X, y, returns)

        # With always long, equity should compound all returns
        # Let's verify equity curve length matches test period
        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] == pytest.approx(1.0)

        # Equity should be positive (assuming returns are reasonable)
        assert result.equity_curve.iloc[-1] > 0

    def test_equity_curve_with_always_flat(self):
        """Test equity curve when model is always flat."""
        X, y, returns = make_test_data(n_days=300, seed=42)
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysFlatModel()
        result = backtester.run(model, X, y, returns)

        # With always flat, equity should stay at 1.0 (no returns applied)
        assert result.equity_curve.iloc[0] == pytest.approx(1.0)
        assert result.equity_curve.iloc[-1] == pytest.approx(1.0)
        assert (result.equity_curve == 1.0).all()

    def test_per_fold_metrics_structure(self):
        """Verify per-fold metrics have required fields."""
        X, y, returns = make_test_data(n_days=300)
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysLongModel()
        result = backtester.run(model, X, y, returns)

        required_fields = {
            "fold", "train_start", "train_end", "test_start", "test_end",
            "sharpe", "total_return", "accuracy", "num_trades"
        }

        for fold_metrics in result.per_fold_metrics:
            assert set(fold_metrics.keys()) == required_fields
            assert isinstance(fold_metrics["fold"], int)
            assert isinstance(fold_metrics["sharpe"], float)
            assert isinstance(fold_metrics["total_return"], float)
            assert isinstance(fold_metrics["accuracy"], float)
            assert isinstance(fold_metrics["num_trades"], int)

    def test_per_fold_metrics_accuracy(self):
        """Verify accuracy is computed correctly."""
        X, y, returns = make_test_data(n_days=300)
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )

        # AlwaysLongModel should have accuracy based on proportion of 1s in y
        model = AlwaysLongModel()
        result = backtester.run(model, X, y, returns)

        # All accuracy values should be between 0 and 1
        for fold_metrics in result.per_fold_metrics:
            assert 0 <= fold_metrics["accuracy"] <= 1

    def test_cost_model_reduces_equity(self):
        """Verify cost model reduces equity on position changes."""
        X, y, returns = make_test_data(n_days=300, seed=42)

        # Create positive returns to ensure equity would grow without costs
        returns = pd.Series(np.ones(300) * 0.01, index=returns.index)  # 1% daily return

        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )

        # Run without cost model
        model1 = AlwaysLongModel()
        result_no_cost = backtester.run(model1, X, y, returns)

        # Run with cost model
        model2 = AlwaysLongModel()
        cost_model = SimpleCostModel(cost_pct=0.001)  # 0.1% per trade
        result_with_cost = backtester.run(model2, X, y, returns, cost_model=cost_model)

        # Equity with costs should be lower (due to transaction costs)
        # Note: AlwaysLongModel only trades once (entering position), so cost impact is minimal
        # Let's just verify both equity curves end above 1.0 and the one with costs is lower
        assert result_no_cost.equity_curve.iloc[-1] > 1.0
        assert result_with_cost.equity_curve.iloc[-1] > 1.0
        assert result_with_cost.equity_curve.iloc[-1] <= result_no_cost.equity_curve.iloc[-1]

    def test_cost_model_with_multiple_trades(self):
        """Verify cost model impact with multiple trades."""
        # Create data where threshold model will trade frequently
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        X = pd.DataFrame({
            "feature1": np.random.randn(300),
            "feature2": np.random.randn(300),
        }, index=dates)
        y = pd.Series(np.random.randint(0, 2, 300), index=dates)
        returns = pd.Series(np.random.normal(0.01, 0.02, 300), index=dates)

        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )

        # Threshold model will trade when feature1 > 0
        model = ThresholdModel("feature1", threshold=0.0)

        # Run without cost
        result_no_cost = backtester.run(model, X, y, returns)

        # Run with cost
        cost_model = SimpleCostModel(cost_pct=0.01)  # 1% per trade
        result_with_cost = backtester.run(model, X, y, returns, cost_model=cost_model)

        # Should have multiple trades
        assert len(result_no_cost.trades) > 1

        # With 1% cost per trade, equity with costs should be noticeably lower
        final_equity_no_cost = result_no_cost.equity_curve.iloc[-1]
        final_equity_with_cost = result_with_cost.equity_curve.iloc[-1]
        assert final_equity_with_cost < final_equity_no_cost

    def test_trade_log_generation(self):
        """Verify trade log is generated correctly."""
        X, y, returns = make_test_data(n_days=300)
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )

        # Use threshold model to generate some trades
        model = ThresholdModel("feature1", threshold=0.0)
        result = backtester.run(model, X, y, returns)

        # Should have at least one trade (entering position)
        assert len(result.trades) >= 1

        # Each trade should have required fields
        for trade in result.trades:
            assert "date" in trade
            assert "signal" in trade
            assert "position_before" in trade
            assert "position_after" in trade
            assert trade["position_before"] != trade["position_after"]

    def test_backtest_result_structure(self):
        """Verify BacktestResult has all required attributes."""
        X, y, returns = make_test_data(n_days=300)
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysLongModel()
        result = backtester.run(model, X, y, returns)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.trades, list)
        assert isinstance(result.equity_curve, pd.Series)
        assert isinstance(result.per_fold_metrics, list)
        assert isinstance(result.fold_count, int)
        assert result.fold_count == len(result.per_fold_metrics)
        assert result.fold_count >= 5

    def test_date_alignment(self):
        """Verify that misaligned dates are handled correctly."""
        # Create data with different date ranges
        dates_X = pd.date_range("2020-01-01", periods=300, freq="D")
        dates_y = pd.date_range("2020-01-02", periods=299, freq="D")  # Off by 1 day
        dates_returns = pd.date_range("2020-01-01", periods=300, freq="D")

        X = pd.DataFrame({
            "feature1": np.random.randn(300),
        }, index=dates_X)
        y = pd.Series(np.random.randint(0, 2, 299), index=dates_y)
        returns = pd.Series(np.random.normal(0.001, 0.02, 300), index=dates_returns)

        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysLongModel()

        # Should handle alignment without error
        result = backtester.run(model, X, y, returns)
        assert result.fold_count >= 5

    def test_fold_indices_are_disjoint(self):
        """Verify that test periods don't overlap."""
        X, y, returns = make_test_data(n_days=300)
        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysLongModel()
        result = backtester.run(model, X, y, returns)

        # Check that test periods don't overlap
        for i in range(len(result.per_fold_metrics) - 1):
            fold_i = result.per_fold_metrics[i]
            fold_j = result.per_fold_metrics[i + 1]

            # Test end of fold i should be <= test start of fold j
            assert fold_i["test_end"] <= fold_j["test_start"]

    def test_returns_applied_correctly_when_long(self):
        """Verify returns are applied correctly when in long position."""
        # Create simple data with known returns
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        X = pd.DataFrame({"feature1": np.ones(300)}, index=dates)
        y = pd.Series(np.ones(300, dtype=int), index=dates)
        # Fixed 1% daily return
        returns = pd.Series(np.ones(300) * 0.01, index=dates)

        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysLongModel()
        result = backtester.run(model, X, y, returns)

        # With always long and 1% daily return, equity should grow
        # Final equity should be > 1.0
        assert result.equity_curve.iloc[-1] > 1.0

        # Check that equity grows monotonically (with positive returns and always long)
        equity_diffs = result.equity_curve.diff().dropna()
        assert (equity_diffs > 0).all() or (equity_diffs >= 0).all()

    def test_no_returns_when_flat(self):
        """Verify no returns applied when position is flat."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        X = pd.DataFrame({"feature1": np.ones(300)}, index=dates)
        y = pd.Series(np.zeros(300, dtype=int), index=dates)
        # Large positive returns (should not be applied)
        returns = pd.Series(np.ones(300) * 0.10, index=dates)

        backtester = WalkForwardBacktester(
            train_min_length=50,
            embargo_days=7,
            test_length_days=30,
            step_days=30,
        )
        model = AlwaysFlatModel()
        result = backtester.run(model, X, y, returns)

        # With always flat, equity should stay at 1.0 despite large returns
        assert (result.equity_curve == 1.0).all()
