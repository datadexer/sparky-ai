"""Tests for data leakage detector module.

Tests cover:
- Shuffled-label test: model predicting shuffled data above/below threshold
- Temporal boundary test: overlapping vs non-overlapping train/test timestamps
- Index overlap audit: overlapping indices detection
- run_all_checks: integration test for all checks
- Model preservation: run_all_checks does not corrupt the model
- LeakageReport: passed/failed check reporting
"""

import numpy as np
import pandas as pd
import pytest

from sparky.backtest.leakage_detector import (
    LeakageCheckResult,
    LeakageDetector,
    LeakageReport,
)


class DummyModel:
    """Simple model for testing that always predicts majority class."""

    def __init__(self, predict_value=1):
        self.predict_value = predict_value

    def fit(self, X, y):
        """Fit does nothing."""
        return self

    def predict(self, X):
        """Always predict the same value."""
        return np.full(len(X), self.predict_value)


class LeakyModel:
    """Model that memorizes data (simulates leakage)."""

    def fit(self, X, y):
        """Store training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def predict(self, X):
        """Return high accuracy even on shuffled data (simulate leakage)."""
        # Always predict 1 with high accuracy (simulates leakage)
        return np.ones(len(X), dtype=int)


@pytest.fixture
def temporal_data_no_overlap():
    """Create temporal data with proper train/test split (no overlap)."""
    # Training data: Jan 1-10, 2025
    train_dates = pd.date_range("2025-01-01", "2025-01-10", freq="D")
    X_train = pd.DataFrame(
        {
            "feature_1": np.random.randn(len(train_dates)),
            "feature_2": np.random.randn(len(train_dates)),
        },
        index=train_dates,
    )
    y_train = pd.Series(np.random.randint(0, 2, len(train_dates)), index=train_dates, name="target")

    # Test data: Jan 15-20, 2025 (gap of 5 days)
    test_dates = pd.date_range("2025-01-15", "2025-01-20", freq="D")
    X_test = pd.DataFrame(
        {
            "feature_1": np.random.randn(len(test_dates)),
            "feature_2": np.random.randn(len(test_dates)),
        },
        index=test_dates,
    )
    y_test = pd.Series(np.random.randint(0, 2, len(test_dates)), index=test_dates, name="target")

    return X_train, y_train, X_test, y_test


@pytest.fixture
def temporal_data_with_overlap():
    """Create temporal data with overlapping train/test split."""
    # Training data: Jan 1-15, 2025
    train_dates = pd.date_range("2025-01-01", "2025-01-15", freq="D")
    X_train = pd.DataFrame(
        {
            "feature_1": np.random.randn(len(train_dates)),
            "feature_2": np.random.randn(len(train_dates)),
        },
        index=train_dates,
    )
    y_train = pd.Series(np.random.randint(0, 2, len(train_dates)), index=train_dates, name="target")

    # Test data: Jan 10-20, 2025 (overlaps with training)
    test_dates = pd.date_range("2025-01-10", "2025-01-20", freq="D")
    X_test = pd.DataFrame(
        {
            "feature_1": np.random.randn(len(test_dates)),
            "feature_2": np.random.randn(len(test_dates)),
        },
        index=test_dates,
    )
    y_test = pd.Series(np.random.randint(0, 2, len(test_dates)), index=test_dates, name="target")

    return X_train, y_train, X_test, y_test


@pytest.fixture
def non_temporal_data():
    """Create data without datetime index."""
    X_train = pd.DataFrame(
        {
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
        }
    )
    y_train = pd.Series(np.random.randint(0, 2, 100), name="target")

    X_test = pd.DataFrame(
        {
            "feature_1": np.random.randn(50),
            "feature_2": np.random.randn(50),
        }
    )
    y_test = pd.Series(np.random.randint(0, 2, 50), name="target")

    return X_train, y_train, X_test, y_test


class TestShuffledLabelTest:
    """Tests for shuffled-label leakage detection."""

    def test_fails_when_predicting_noise_above_threshold(self, temporal_data_no_overlap):
        """Shuffled-label test fails when model predicts shuffled data above threshold."""
        X_train, y_train, X_test, y_test = temporal_data_no_overlap

        # Create a model that always predicts 1 with high accuracy
        # Need test data where always predicting 1 gives high accuracy
        y_test_leaky = pd.Series([1, 1, 1, 1, 1, 1], index=y_test.index[:6])
        X_test_leaky = X_test.iloc[:6]

        # Leaky model that achieves high accuracy even on noise
        model = LeakyModel()
        detector = LeakageDetector(shuffled_accuracy_threshold=0.55, n_shuffle_trials=3)

        result = detector.shuffled_label_test(model, X_train, y_train, X_test_leaky, y_test_leaky)

        # Should fail because leaky model predicts noise too well (100% accuracy on all 1s)
        assert not result.passed
        assert result.check_name == "shuffled_label"
        assert "FAIL" in result.detail or "leakage" in result.detail.lower()

    def test_passes_when_predicting_noise_below_threshold(self, temporal_data_no_overlap):
        """Shuffled-label test passes when model predicts shuffled data below threshold."""
        X_train, y_train, X_test, y_test = temporal_data_no_overlap

        # Dummy model with low accuracy
        # Make y_test balanced so random prediction gives ~0.5 accuracy
        y_test = pd.Series([0] * 3 + [1] * 3, index=y_test.index[:6])
        X_test = X_test.iloc[:6]

        model = DummyModel(predict_value=1)
        detector = LeakageDetector(shuffled_accuracy_threshold=0.55, n_shuffle_trials=3)

        result = detector.shuffled_label_test(model, X_train, y_train, X_test, y_test)

        # Should pass because dummy model has ~50% accuracy on balanced data
        assert result.passed
        assert result.check_name == "shuffled_label"
        assert "PASS" in result.detail

    def test_multiple_shuffle_trials(self, temporal_data_no_overlap):
        """Shuffled-label test runs multiple trials and averages."""
        X_train, y_train, X_test, y_test = temporal_data_no_overlap

        model = DummyModel(predict_value=0)
        n_trials = 5
        detector = LeakageDetector(n_shuffle_trials=n_trials)

        result = detector.shuffled_label_test(model, X_train, y_train, X_test, y_test)

        # Should have metric value (mean accuracy)
        assert result.metric_value >= 0.0
        assert result.metric_value <= 1.0

    def test_handles_model_fit_failure(self, temporal_data_no_overlap):
        """Shuffled-label test handles model fitting failures gracefully."""
        X_train, y_train, X_test, y_test = temporal_data_no_overlap

        class FailingModel:
            def fit(self, X, y):
                raise ValueError("Fit failed")

            def predict(self, X):
                return np.zeros(len(X))

        model = FailingModel()
        detector = LeakageDetector(n_shuffle_trials=3)

        result = detector.shuffled_label_test(model, X_train, y_train, X_test, y_test)

        # Should pass when test cannot run
        assert result.passed
        assert "Could not run" in result.detail


class TestTemporalBoundaryTest:
    """Tests for temporal boundary leakage detection."""

    def test_fails_with_overlapping_timestamps(self, temporal_data_with_overlap):
        """Temporal boundary test fails when train/test timestamps overlap."""
        X_train, y_train, X_test, y_test = temporal_data_with_overlap

        detector = LeakageDetector()
        result = detector.temporal_boundary_test(X_train, X_test)

        assert not result.passed
        assert result.check_name == "temporal_boundary"
        assert "FAIL" in result.detail
        assert "overlap" in result.detail.lower()

    def test_passes_with_gap_between_train_test(self, temporal_data_no_overlap):
        """Temporal boundary test passes when there's a gap between train and test."""
        X_train, y_train, X_test, y_test = temporal_data_no_overlap

        detector = LeakageDetector()
        result = detector.temporal_boundary_test(X_train, X_test)

        assert result.passed
        assert result.check_name == "temporal_boundary"
        assert "PASS" in result.detail
        assert result.metric_value > 0  # Gap in days

    def test_skips_non_datetime_index(self, non_temporal_data):
        """Temporal boundary test skips non-datetime indices."""
        X_train, y_train, X_test, y_test = non_temporal_data

        detector = LeakageDetector()
        result = detector.temporal_boundary_test(X_train, X_test)

        assert result.passed
        assert "Non-datetime index" in result.detail

    def test_reports_gap_in_days(self, temporal_data_no_overlap):
        """Temporal boundary test reports the gap between train and test in days."""
        X_train, y_train, X_test, y_test = temporal_data_no_overlap

        detector = LeakageDetector()
        result = detector.temporal_boundary_test(X_train, X_test)

        # There's a 5-day gap in the fixture
        assert result.metric_value >= 4  # At least 4 days gap
        assert "Gap:" in result.detail


class TestIndexOverlapAudit:
    """Tests for index overlap audit."""

    def test_fails_with_overlapping_indices(self):
        """Feature timestamp audit fails when train/test have overlapping indices."""
        # Create overlapping timestamps
        dates = pd.date_range("2025-01-01", "2025-01-20", freq="D")

        X_train = pd.DataFrame(
            {
                "feature_1": np.random.randn(15),
            },
            index=dates[:15],
        )  # Jan 1-15

        X_test = pd.DataFrame(
            {
                "feature_1": np.random.randn(10),
            },
            index=dates[10:20],
        )  # Jan 11-20 (overlaps with train)

        detector = LeakageDetector()
        result = detector.index_overlap_audit(X_train, X_test)

        assert not result.passed
        assert result.check_name == "index_overlap_audit"
        assert "FAIL" in result.detail
        assert "overlapping" in result.detail.lower()
        assert result.metric_value > 0  # Number of overlapping timestamps

    def test_passes_with_no_overlap(self, temporal_data_no_overlap):
        """Feature timestamp audit passes when there's no index overlap."""
        X_train, y_train, X_test, y_test = temporal_data_no_overlap

        detector = LeakageDetector()
        result = detector.index_overlap_audit(X_train, X_test)

        assert result.passed
        assert result.check_name == "index_overlap_audit"
        assert "PASS" in result.detail
        assert result.metric_value == 0.0

    def test_skips_non_datetime_index(self, non_temporal_data):
        """Feature timestamp audit skips non-datetime indices."""
        X_train, y_train, X_test, y_test = non_temporal_data

        detector = LeakageDetector()
        result = detector.index_overlap_audit(X_train, X_test)

        assert result.passed
        assert "Non-datetime index" in result.detail


class TestRunAllChecks:
    """Integration tests for run_all_checks."""

    def test_reports_all_passed(self, temporal_data_no_overlap):
        """run_all_checks reports all passed when no leakage detected."""
        X_train, y_train, X_test, y_test = temporal_data_no_overlap

        # Use dummy model with low accuracy
        y_test = pd.Series([0] * 3 + [1] * 3, index=y_test.index[:6])
        X_test = X_test.iloc[:6]

        model = DummyModel(predict_value=1)
        detector = LeakageDetector(n_shuffle_trials=2)

        report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)

        assert isinstance(report, LeakageReport)
        assert report.passed
        assert len(report.checks) == 3  # Three checks
        assert all(check.passed for check in report.checks)
        assert len(report.failed_checks) == 0

    def test_reports_failures(self, temporal_data_with_overlap):
        """run_all_checks reports failures when leakage detected."""
        X_train, y_train, X_test, y_test = temporal_data_with_overlap

        model = DummyModel()
        detector = LeakageDetector()

        report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)

        assert isinstance(report, LeakageReport)
        assert not report.passed  # Should fail due to temporal overlap
        assert len(report.failed_checks) > 0

    def test_failed_checks_property(self, temporal_data_with_overlap):
        """LeakageReport.failed_checks returns only failed checks."""
        X_train, y_train, X_test, y_test = temporal_data_with_overlap

        model = DummyModel()
        detector = LeakageDetector()

        report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)

        failed = report.failed_checks
        assert len(failed) > 0
        assert all(not check.passed for check in failed)

    def test_all_three_checks_run(self, temporal_data_no_overlap):
        """run_all_checks executes all three leakage checks."""
        X_train, y_train, X_test, y_test = temporal_data_no_overlap

        model = DummyModel()
        detector = LeakageDetector(n_shuffle_trials=2)

        report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)

        assert len(report.checks) == 3
        check_names = {check.check_name for check in report.checks}
        assert check_names == {"shuffled_label", "temporal_boundary", "index_overlap_audit"}


class TestLeakageCheckResult:
    """Tests for LeakageCheckResult dataclass."""

    def test_check_result_creation(self):
        """LeakageCheckResult can be created with required fields."""
        result = LeakageCheckResult(check_name="test_check", passed=True, detail="Test passed", metric_value=0.5)

        assert result.check_name == "test_check"
        assert result.passed is True
        assert result.detail == "Test passed"
        assert result.metric_value == 0.5

    def test_check_result_default_metric(self):
        """LeakageCheckResult has default metric_value of 0.0."""
        result = LeakageCheckResult(check_name="test_check", passed=False, detail="Test failed")

        assert result.metric_value == 0.0


class TestLeakageReport:
    """Tests for LeakageReport dataclass."""

    def test_report_creation(self):
        """LeakageReport can be created with checks."""
        checks = [
            LeakageCheckResult("check1", True, "Pass"),
            LeakageCheckResult("check2", False, "Fail"),
        ]

        report = LeakageReport(checks=checks, passed=False)

        assert len(report.checks) == 2
        assert not report.passed

    def test_failed_checks_property(self):
        """LeakageReport.failed_checks filters to failed checks only."""
        checks = [
            LeakageCheckResult("check1", True, "Pass"),
            LeakageCheckResult("check2", False, "Fail 1"),
            LeakageCheckResult("check3", False, "Fail 2"),
        ]

        report = LeakageReport(checks=checks, passed=False)
        failed = report.failed_checks

        assert len(failed) == 2
        assert all(not check.passed for check in failed)
        assert {check.check_name for check in failed} == {"check2", "check3"}


class TestDetectorConfiguration:
    """Tests for LeakageDetector configuration."""

    def test_default_threshold(self):
        """LeakageDetector uses default threshold from constant."""
        from sparky.backtest.leakage_detector import SHUFFLED_ACCURACY_THRESHOLD

        detector = LeakageDetector()
        assert detector.shuffled_accuracy_threshold == SHUFFLED_ACCURACY_THRESHOLD
        assert detector.shuffled_accuracy_threshold == 0.55

    def test_custom_threshold(self):
        """LeakageDetector accepts custom threshold."""
        detector = LeakageDetector(shuffled_accuracy_threshold=0.6, n_shuffle_trials=10)

        assert detector.shuffled_accuracy_threshold == 0.6
        assert detector.n_shuffle_trials == 10

    def test_default_shuffle_trials(self):
        """LeakageDetector uses default number of shuffle trials."""
        detector = LeakageDetector()
        assert detector.n_shuffle_trials == 5


class TestModelPreservation:
    """Tests that run_all_checks does not corrupt the model."""

    def test_model_predictions_unchanged_after_run_all_checks(self):
        """Model predictions must be identical before and after run_all_checks."""
        from sklearn.tree import DecisionTreeClassifier

        np.random.seed(42)
        n_train, n_test = 100, 20
        train_dates = pd.date_range("2025-01-01", periods=n_train, freq="D")
        test_dates = pd.date_range("2025-05-01", periods=n_test, freq="D")

        X_train = pd.DataFrame({"f1": np.random.randn(n_train), "f2": np.random.randn(n_train)}, index=train_dates)
        y_train = pd.Series(np.random.randint(0, 2, n_train), index=train_dates)
        X_test = pd.DataFrame({"f1": np.random.randn(n_test), "f2": np.random.randn(n_test)}, index=test_dates)
        y_test = pd.Series(np.random.randint(0, 2, n_test), index=test_dates)

        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # Get predictions BEFORE leakage checks
        predictions_before = model.predict(X_test).copy()

        # Run all leakage checks (shuffled-label test used to corrupt the model)
        detector = LeakageDetector(n_shuffle_trials=5)
        detector.run_all_checks(model, X_train, y_train, X_test, y_test)

        # Get predictions AFTER leakage checks
        predictions_after = model.predict(X_test)

        # Predictions must be identical
        np.testing.assert_array_equal(predictions_before, predictions_after)
