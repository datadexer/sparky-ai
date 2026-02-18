"""Tests for XGBoost model implementation and ModelProtocol compliance."""

import numpy as np
import pandas as pd
import pytest

from sparky.backtest.costs import TransactionCostModel
from sparky.backtest.engine import WalkForwardBacktester
from sparky.models.xgboost_model import XGBoostModel


@pytest.fixture
def synthetic_data():
    """Generate synthetic feature matrix and labels for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 5

    # Random features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
        index=pd.date_range("2020-01-01", periods=n_samples, freq="D", tz="UTC"),
    )

    # Random binary labels (balanced)
    y = pd.Series(
        np.random.randint(0, 2, size=n_samples),
        index=X.index,
        name="target",
    )

    # Synthetic returns (random walk)
    returns = pd.Series(
        np.random.randn(n_samples) * 0.02,  # 2% daily volatility
        index=X.index,
        name="returns",
    )

    return X, y, returns


def test_protocol_compliance():
    """Test that XGBoostModel implements ModelProtocol (fit/predict)."""
    model = XGBoostModel(random_state=42)

    # Check methods exist
    assert hasattr(model, "fit"), "Model must have fit() method"
    assert hasattr(model, "predict"), "Model must have predict() method"

    # Check signatures
    import inspect

    fit_sig = inspect.signature(model.fit)
    assert "X" in fit_sig.parameters, "fit() must accept X parameter"
    assert "y" in fit_sig.parameters, "fit() must accept y parameter"

    predict_sig = inspect.signature(model.predict)
    assert "X" in predict_sig.parameters, "predict() must accept X parameter"


def test_fit_and_predict(synthetic_data):
    """Test basic fit and predict functionality."""
    X, y, _ = synthetic_data

    model = XGBoostModel(random_state=42, n_estimators=10)
    model.fit(X, y)

    predictions = model.predict(X)

    # Check output shape
    assert len(predictions) == len(X), "Predictions must match input length"

    # Check output type
    assert isinstance(predictions, np.ndarray), "Predictions must be numpy array"


def test_binary_output(synthetic_data):
    """Test that predictions are strictly {0, 1}."""
    X, y, _ = synthetic_data

    model = XGBoostModel(random_state=42, n_estimators=10)
    model.fit(X, y)

    predictions = model.predict(X)

    # Check only {0, 1} values
    unique_values = np.unique(predictions)
    assert set(unique_values).issubset({0, 1}), "Predictions must be {0, 1} only"


def test_determinism(synthetic_data):
    """Test that same random_state produces identical predictions."""
    X, y, _ = synthetic_data

    # Train two models with same seed
    model1 = XGBoostModel(random_state=42, n_estimators=10)
    model1.fit(X, y)
    pred1 = model1.predict(X)

    model2 = XGBoostModel(random_state=42, n_estimators=10)
    model2.fit(X, y)
    pred2 = model2.predict(X)

    # Should be identical
    np.testing.assert_array_equal(pred1, pred2, err_msg="Same seed must produce identical predictions")

    # Different seed should give different predictions
    model3 = XGBoostModel(random_state=99, n_estimators=10)
    model3.fit(X, y)
    pred3 = model3.predict(X)

    assert not np.array_equal(pred1, pred3), "Different seeds should produce different predictions"


def test_backtester_integration(synthetic_data):
    """Test integration with WalkForwardBacktester (MANDATORY per CLAUDE.md)."""
    X, y, returns = synthetic_data

    # Initialize real backtester (no mocks)
    model = XGBoostModel(random_state=42, n_estimators=10)
    cost_model = TransactionCostModel.for_btc()  # 0.13% per trade
    backtester = WalkForwardBacktester(
        train_min_length=100,
        embargo_days=7,
        test_length=30,
        step_size=30,
    )

    # Run backtest
    result = backtester.run(model, X, y, returns, cost_model=cost_model, asset="BTC")

    # Verify result structure
    assert result is not None, "Backtester must return a result"
    assert hasattr(result, "equity_curve"), "Result must have equity_curve"
    assert hasattr(result, "fold_count"), "Result must have fold_count"
    assert hasattr(result, "per_fold_metrics"), "Result must have per_fold_metrics"

    # Verify equity curve is reasonable
    assert len(result.equity_curve) > 0, "Equity curve must not be empty"
    assert result.equity_curve.iloc[0] == 1.0, "Equity curve must start at 1.0"
    assert result.fold_count > 0, "Must have at least one fold"


def test_feature_importances(synthetic_data):
    """Test feature importance extraction."""
    X, y, _ = synthetic_data

    model = XGBoostModel(random_state=42, n_estimators=10)

    # Should fail before training
    with pytest.raises(ValueError, match="not trained"):
        model.get_feature_importances()

    # Train and extract
    model.fit(X, y)
    importances = model.get_feature_importances()

    # Check structure
    assert isinstance(importances, pd.DataFrame), "Importances must be DataFrame"
    assert "feature" in importances.columns, "Must have 'feature' column"
    assert "importance" in importances.columns, "Must have 'importance' column"
    assert len(importances) == X.shape[1], "Must have importance for each feature"

    # Check sorted descending
    assert importances["importance"].is_monotonic_decreasing, "Importances must be sorted descending"


def test_nan_handling(synthetic_data):
    """Test that XGBoost handles NaN features correctly."""
    X, y, _ = synthetic_data

    # Introduce NaN in some features
    X_with_nan = X.copy()
    X_with_nan.iloc[10:20, 0] = np.nan
    X_with_nan.iloc[30:40, 2] = np.nan

    model = XGBoostModel(random_state=42, n_estimators=10)
    model.fit(X_with_nan, y)

    predictions = model.predict(X_with_nan)

    # Should still produce valid predictions
    assert len(predictions) == len(X_with_nan)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_probability_predictions(synthetic_data):
    """Test probability prediction interface."""
    X, y, _ = synthetic_data

    model = XGBoostModel(random_state=42, n_estimators=10)
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    # Check shape (n_samples, 2)
    assert probabilities.shape == (len(X), 2), "Probabilities must be (n_samples, 2)"

    # Check probabilities sum to 1
    prob_sums = probabilities.sum(axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5, err_msg="Probabilities must sum to 1")

    # Check probabilities in [0, 1]
    assert np.all(probabilities >= 0), "Probabilities must be >= 0"
    assert np.all(probabilities <= 1), "Probabilities must be <= 1"
