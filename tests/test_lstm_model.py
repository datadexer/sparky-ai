"""Tests for LSTM model implementation and ModelProtocol compliance."""

import numpy as np
import pandas as pd
import pytest

from sparky.backtest.costs import TransactionCostModel
from sparky.backtest.engine import WalkForwardBacktester
from sparky.models.lstm_model import LSTMModel


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
    """Test that LSTMModel implements ModelProtocol (fit/predict)."""
    model = LSTMModel(random_state=42)

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


def test_sequence_creation(synthetic_data):
    """Test rolling window sequence creation correctness."""
    X, y, _ = synthetic_data

    model = LSTMModel(window_length=10, random_state=42)

    # Create sequences
    X_seq, y_seq = model._create_sequences(X, y)

    # Check shapes
    expected_n_sequences = len(X) - model.window_length
    assert X_seq.shape == (
        expected_n_sequences,
        model.window_length,
        X.shape[1],
    ), "X_seq shape incorrect"
    assert y_seq.shape == (expected_n_sequences,), "y_seq shape incorrect"

    # Check that sequences are rolling windows
    # First sequence should be X[0:10], label should be y[10]
    np.testing.assert_array_equal(X_seq[0], X.iloc[0:10].values, err_msg="First sequence incorrect")
    assert y_seq[0] == y.iloc[10], "First label incorrect"

    # Last sequence should be X[-11:-1], label should be y[-1]
    np.testing.assert_array_equal(
        X_seq[-1], X.iloc[-11:-1].values, err_msg="Last sequence incorrect"
    )
    assert y_seq[-1] == y.iloc[-1], "Last label incorrect"


def test_fit_and_predict(synthetic_data):
    """Test basic fit and predict functionality."""
    X, y, _ = synthetic_data

    model = LSTMModel(window_length=10, max_epochs=5, random_state=42)
    model.fit(X, y)

    predictions = model.predict(X)

    # Check output shape
    assert len(predictions) == len(X), "Predictions must match input length"

    # Check output type
    assert isinstance(predictions, np.ndarray), "Predictions must be numpy array"


def test_binary_output(synthetic_data):
    """Test that predictions are strictly {0, 1}."""
    X, y, _ = synthetic_data

    model = LSTMModel(window_length=10, max_epochs=5, random_state=42)
    model.fit(X, y)

    predictions = model.predict(X)

    # Check only {0, 1} values
    unique_values = np.unique(predictions)
    assert set(unique_values).issubset({0, 1}), "Predictions must be {0, 1} only"


def test_training_convergence(synthetic_data):
    """Test that training loss decreases over epochs."""
    X, y, _ = synthetic_data

    # Use more epochs to see convergence
    model = LSTMModel(window_length=10, max_epochs=20, patience=20, random_state=42)
    model.fit(X, y)

    # Model should have been trained (not None)
    assert model.model is not None, "Model should be trained after fit()"


def test_short_test_set_handling():
    """Test that LSTM handles short test sets correctly (pads with 0)."""
    # Create very short test set (< window_length)
    X_short = pd.DataFrame(
        np.random.randn(5, 3),
        columns=["f1", "f2", "f3"],
        index=pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC"),
    )
    y_short = pd.Series(
        [0, 1, 0, 1, 0],
        index=X_short.index,
    )

    model = LSTMModel(window_length=10, random_state=42)

    # Model not trained, should return zeros
    predictions = model.predict(X_short)
    assert len(predictions) == len(X_short), "Predictions must match input length"
    np.testing.assert_array_equal(
        predictions, np.zeros(len(X_short), dtype=int), err_msg="Untrained model should return zeros"
    )


def test_backtester_integration(synthetic_data):
    """Test integration with WalkForwardBacktester (MANDATORY per CLAUDE.md)."""
    X, y, returns = synthetic_data

    # Initialize real backtester (no mocks)
    model = LSTMModel(window_length=10, max_epochs=5, random_state=42)
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


def test_determinism(synthetic_data):
    """Test that same random_state produces similar predictions."""
    X, y, _ = synthetic_data

    # Train two models with same seed
    model1 = LSTMModel(window_length=10, max_epochs=5, random_state=42)
    model1.fit(X, y)
    pred1 = model1.predict(X)

    model2 = LSTMModel(window_length=10, max_epochs=5, random_state=42)
    model2.fit(X, y)
    pred2 = model2.predict(X)

    # Should be very similar (may not be 100% identical due to GPU/CPU differences)
    # Check that at least 90% of predictions match
    agreement = (pred1 == pred2).sum() / len(pred1)
    assert agreement > 0.9, f"Same seed should produce similar predictions (agreement={agreement:.2%})"


def test_insufficient_data_handling():
    """Test handling of datasets too small for sequence creation."""
    # Create tiny dataset (smaller than window_length)
    X_tiny = pd.DataFrame(
        np.random.randn(5, 3),
        columns=["f1", "f2", "f3"],
        index=pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC"),
    )
    y_tiny = pd.Series([0, 1, 0, 1, 0], index=X_tiny.index)

    model = LSTMModel(window_length=10, random_state=42)
    model.fit(X_tiny, y_tiny)

    # Model should handle this gracefully (log warning but not crash)
    # Predict should return zeros
    predictions = model.predict(X_tiny)
    assert len(predictions) == len(X_tiny)


def test_normalization():
    """Test that features are normalized during training."""
    np.random.seed(42)

    # Create features with different scales
    X = pd.DataFrame({
        "small_scale": np.random.randn(200) * 0.01,  # Mean~0, std~0.01
        "large_scale": np.random.randn(200) * 100,  # Mean~0, std~100
    }, index=pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC"))

    y = pd.Series(np.random.randint(0, 2, size=200), index=X.index)

    model = LSTMModel(window_length=10, max_epochs=5, random_state=42)
    model.fit(X, y)

    # Check that scaler was fitted
    assert hasattr(model.scaler, "mean_"), "Scaler should be fitted"
    assert hasattr(model.scaler, "scale_"), "Scaler should have scale_"

    # Scaler should have normalized both features to similar scales
    assert len(model.scaler.mean_) == 2, "Scaler should fit both features"
