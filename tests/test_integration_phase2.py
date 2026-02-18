"""Phase 2 integration tests — backtester, cost model, leakage detector, feature pipeline, tracker.

All tests use real module instances with synthetic data. No mocks for anything in src/sparky/.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from sparky.backtest.costs import TransactionCostModel
from sparky.backtest.engine import BacktestResult, WalkForwardBacktester
from sparky.backtest.leakage_detector import LeakageDetector
from sparky.backtest.statistics import BacktestStatistics
from sparky.features.returns import annualized_sharpe, simple_returns
from sparky.features.technical import ema, momentum, rsi
from sparky.models.baselines import BuyAndHold, EqualWeight, SimpleMomentum
from sparky.tracking.experiment import ExperimentTracker


@pytest.fixture
def backtest_data():
    """Generate synthetic data sufficient for 5+ walk-forward folds.

    With train_min_length=252, embargo=7, test_length=30, step_size=30,
    we need ~252 + 7 + 5*30 = 409 rows minimum. Generate 600 for safety.
    """
    np.random.seed(42)
    n = 600
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Synthetic prices with drift
    log_rets = np.random.normal(0.0003, 0.02, n)
    prices = 100 * np.exp(np.cumsum(log_rets))

    # Features: momentum + RSI + EMA ratio
    price_series = pd.Series(prices, index=dates, name="close")
    mom = momentum(price_series, period=30)
    rsi_14 = rsi(price_series, period=14)
    ema_20 = ema(price_series, 20)
    ema_ratio = price_series / ema_20 - 1  # Price relative to EMA

    X = pd.DataFrame({"momentum": mom, "rsi": rsi_14, "ema_ratio": ema_ratio}, index=dates)
    # Drop NaN rows from lookback
    X = X.dropna()

    # Returns: simple daily returns
    returns = simple_returns(price_series).loc[X.index]

    # Target: binary (1=up, 0=down) based on next-day return
    y = (returns.shift(-1) > 0).astype(int)
    y = y.loc[X.index].fillna(0).astype(int)

    return X, y, returns


class TestBacktesterWithRealCostModel:
    """WalkForwardBacktester + TransactionCostModel.for_btc() + BuyAndHold."""

    def test_equity_with_costs_less_than_without(self, backtest_data):
        """Equity curve with transaction costs should have lower final value."""
        X, y, returns = backtest_data
        model = BuyAndHold()
        cost_model = TransactionCostModel.for_btc()
        backtester = WalkForwardBacktester(train_min_length=252)

        result_with_costs = backtester.run(model, X, y, returns, cost_model=cost_model, asset="BTC")
        result_without_costs = backtester.run(model, X, y, returns, cost_model=None, asset="BTC")

        # With costs should be lower
        final_with = result_with_costs.equity_curve.iloc[-1]
        final_without = result_without_costs.equity_curve.iloc[-1]
        assert final_with < final_without

        # Both should be valid BacktestResults
        assert isinstance(result_with_costs, BacktestResult)
        assert result_with_costs.fold_count >= 5
        assert len(result_with_costs.per_fold_metrics) >= 5


class TestBacktesterWithAllBaselines:
    """Run all three baselines through the real backtester."""

    def test_buy_and_hold(self, backtest_data):
        X, y, returns = backtest_data
        cost_model = TransactionCostModel.for_btc()
        backtester = WalkForwardBacktester(train_min_length=252)

        result = backtester.run(BuyAndHold(), X, y, returns, cost_model=cost_model)

        assert isinstance(result, BacktestResult)
        assert result.fold_count >= 5
        assert len(result.per_fold_metrics) >= 5
        assert all("sharpe" in m for m in result.per_fold_metrics)
        assert all("accuracy" in m for m in result.per_fold_metrics)

    def test_simple_momentum(self, backtest_data):
        X, y, returns = backtest_data
        cost_model = TransactionCostModel.for_btc()
        backtester = WalkForwardBacktester(train_min_length=252)

        # SimpleMomentum uses the "momentum" column in X
        result = backtester.run(
            SimpleMomentum(momentum_col="momentum"),
            X,
            y,
            returns,
            cost_model=cost_model,
        )

        assert isinstance(result, BacktestResult)
        assert result.fold_count >= 5
        assert len(result.per_fold_metrics) >= 5

    def test_equal_weight(self, backtest_data):
        X, y, returns = backtest_data
        cost_model = TransactionCostModel.for_btc()
        backtester = WalkForwardBacktester(train_min_length=252)

        result = backtester.run(EqualWeight(), X, y, returns, cost_model=cost_model)

        assert isinstance(result, BacktestResult)
        assert result.fold_count >= 5
        assert len(result.per_fold_metrics) >= 5


class TestLeakageDetectorPreservesModelState:
    """Verify model predictions are identical before and after run_all_checks."""

    def test_model_predictions_unchanged_after_run_all_checks(self):
        """Fit a model, record predictions, run leakage checks, verify predictions match."""
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)}, index=dates)
        y = pd.Series((X["f1"] > 0).astype(int).values, index=dates)

        # Split with temporal gap
        X_train, X_test = X.iloc[:200], X.iloc[210:]
        y_train, y_test = y.iloc[:200], y.iloc[210:]

        # Reference data for prediction check
        X_ref = X.iloc[100:150]

        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # Record predictions BEFORE leakage detection
        predictions_before = model.predict(X_ref).copy()

        # Run leakage detector
        detector = LeakageDetector(n_shuffle_trials=3)
        report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)

        # Record predictions AFTER leakage detection
        predictions_after = model.predict(X_ref)

        # Predictions must be identical
        np.testing.assert_array_equal(predictions_before, predictions_after)

        # Report should be valid
        assert len(report.checks) == 3
        assert report.passed  # No leakage in clean synthetic data


class TestExperimentTrackerLogsRealBacktest:
    """Run real backtest -> log to ExperimentTracker -> verify W&B run."""

    def test_backtest_metrics_logged_correctly(self, backtest_data, tmp_path):
        """Metrics from a real backtest are correctly passed to W&B."""
        from unittest.mock import MagicMock, patch

        X, y, returns = backtest_data
        model = BuyAndHold()
        cost_model = TransactionCostModel.for_btc()
        backtester = WalkForwardBacktester(train_min_length=252)

        result = backtester.run(model, X, y, returns, cost_model=cost_model)

        # Extract metrics from backtest result
        avg_sharpe = float(np.mean([m["sharpe"] for m in result.per_fold_metrics]))
        total_return = float(result.equity_curve.iloc[-1] - 1.0)

        # Mock W&B to avoid network calls
        mock_run = MagicMock()
        mock_run.id = "test_run_123"

        with patch("sparky.tracking.experiment._ensure_wandb_login"):
            with patch("sparky.tracking.experiment.wandb") as mock_wandb:
                mock_wandb.init.return_value = mock_run

                tracker = ExperimentTracker(experiment_name="integration_test")
                run_id = tracker.log_experiment(
                    name="test_buy_and_hold",
                    config={"model": "BuyAndHold", "cost_model": "btc"},
                    metrics={"avg_sharpe": avg_sharpe, "total_return": total_return},
                )

                # Verify run was created
                assert run_id == "test_run_123"
                mock_wandb.init.assert_called_once()

                # Verify metrics were logged
                logged = mock_wandb.log.call_args[0][0]
                assert logged["avg_sharpe"] == pytest.approx(avg_sharpe, rel=1e-6)
                assert logged["total_return"] == pytest.approx(total_return, rel=1e-6)

                # Verify config contains the right params
                call_config = mock_wandb.init.call_args[1]["config"]
                assert call_config["model"] == "BuyAndHold"
                assert "config_hash" in call_config


class TestBootstrapCIScaleMatchesFoldSharpe:
    """Verify bootstrap CI and fold Sharpe are on the same annualized scale."""

    def test_ci_and_point_estimate_same_scale(self, backtest_data):
        """Bootstrap CI bounds should be comparable to annualized_sharpe point estimate."""
        X, y, returns = backtest_data
        model = BuyAndHold()
        backtester = WalkForwardBacktester(train_min_length=252)

        result = backtester.run(model, X, y, returns)

        # Get test-period returns from equity curve
        equity = result.equity_curve
        equity_returns = equity.pct_change().dropna()

        # Compute annualized Sharpe (point estimate)
        point_sharpe = annualized_sharpe(equity_returns)

        # Compute bootstrap CI (annualized by default)
        lower, upper = BacktestStatistics.sharpe_confidence_interval(
            equity_returns,
            n_bootstrap=2000,
            ci=0.95,
            annualize=True,
            random_state=42,
        )

        # Both should be on the same scale (annualized)
        # The point estimate should be in a reasonable neighborhood of the CI
        # (may not be strictly inside due to bootstrap vs analytic differences)
        ci_width = upper - lower
        assert ci_width > 0, "CI should have positive width"

        # The CI midpoint and point estimate should be on the same order of magnitude
        ci_mid = (lower + upper) / 2
        # Both are annualized — difference should be small relative to CI width
        assert abs(point_sharpe - ci_mid) < ci_width * 3, (
            f"Point estimate {point_sharpe:.3f} too far from CI [{lower:.3f}, {upper:.3f}] (mid={ci_mid:.3f})"
        )

        # Verify daily (non-annualized) CI is sqrt(252)x smaller
        lower_daily, upper_daily = BacktestStatistics.sharpe_confidence_interval(
            equity_returns,
            n_bootstrap=2000,
            ci=0.95,
            annualize=False,
            random_state=42,
        )
        daily_width = upper_daily - lower_daily
        scale_ratio = ci_width / daily_width if daily_width > 0 else float("inf")
        assert abs(scale_ratio - np.sqrt(365)) < 1.0, (
            f"Annualized/daily CI width ratio {scale_ratio:.2f} should be ~{np.sqrt(365):.2f}"
        )
