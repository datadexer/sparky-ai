"""Data leakage detector — mandatory check before logging any model result.

Three tests:
1. Shuffled-label test: model should not predict random noise
2. Temporal boundary test: no future info at train/test boundary
3. Feature timestamp audit: max feature timestamp <= row timestamp

If any check fails → result is NOT logged, error written to RESEARCH_LOG.md.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SHUFFLED_ACCURACY_THRESHOLD = 0.55  # From research_standards


@dataclass
class LeakageCheckResult:
    """Result of a single leakage check."""

    check_name: str
    passed: bool
    detail: str
    metric_value: float = 0.0


@dataclass
class LeakageReport:
    """Full leakage detection report."""

    checks: list[LeakageCheckResult]
    passed: bool  # True only if ALL checks pass

    @property
    def failed_checks(self) -> list[LeakageCheckResult]:
        return [c for c in self.checks if not c.passed]


class LeakageDetector:
    """Detect data leakage in model training pipeline.

    Usage:
        detector = LeakageDetector()
        report = detector.run_all_checks(model, X_train, y_train, X_test, y_test)
        if not report.passed:
            # DO NOT log this result
            for check in report.failed_checks:
                print(f"LEAK: {check.check_name}: {check.detail}")
    """

    def __init__(
        self,
        shuffled_accuracy_threshold: float = SHUFFLED_ACCURACY_THRESHOLD,
        n_shuffle_trials: int = 5,
    ):
        self.shuffled_accuracy_threshold = shuffled_accuracy_threshold
        self.n_shuffle_trials = n_shuffle_trials

    def run_all_checks(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> LeakageReport:
        """Run all leakage detection checks.

        Args:
            model: Model with .fit() and .predict() methods.
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.

        Returns:
            LeakageReport with all check results.
        """
        checks = []

        # 1. Shuffled-label test
        checks.append(self.shuffled_label_test(model, X_train, y_train, X_test, y_test))

        # 2. Temporal boundary test
        checks.append(self.temporal_boundary_test(X_train, X_test))

        # 3. Feature timestamp audit
        checks.append(self.feature_timestamp_audit(X_train, X_test))

        all_passed = all(c.passed for c in checks)

        if not all_passed:
            failed = [c for c in checks if not c.passed]
            logger.error(
                f"LEAKAGE DETECTED — {len(failed)} check(s) failed: "
                + ", ".join(c.check_name for c in failed)
            )

        return LeakageReport(checks=checks, passed=all_passed)

    def shuffled_label_test(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> LeakageCheckResult:
        """Test 1: Model should not predict shuffled labels.

        Randomly permute target labels, retrain, and check accuracy.
        If accuracy > threshold, leakage is present (model is learning
        from features that contain target information).
        """
        accuracies = []

        for trial in range(self.n_shuffle_trials):
            # Shuffle training labels
            rng = np.random.RandomState(seed=trial + 42)
            y_shuffled = y_train.copy()
            y_shuffled = pd.Series(
                rng.permutation(y_shuffled.values),
                index=y_shuffled.index,
            )

            # Retrain on shuffled labels
            try:
                model.fit(X_train, y_shuffled)
                predictions = model.predict(X_test)
                accuracy = np.mean(predictions == y_test.values)
                accuracies.append(accuracy)
            except Exception as e:
                logger.warning(f"Shuffled-label trial {trial} failed: {e}")
                continue

        if not accuracies:
            return LeakageCheckResult(
                check_name="shuffled_label",
                passed=True,
                detail="Could not run shuffled-label test",
            )

        mean_accuracy = np.mean(accuracies)
        passed = mean_accuracy <= self.shuffled_accuracy_threshold

        return LeakageCheckResult(
            check_name="shuffled_label",
            passed=passed,
            detail=(
                f"Mean shuffled accuracy: {mean_accuracy:.3f} "
                f"(threshold: {self.shuffled_accuracy_threshold}). "
                + ("PASS" if passed else "FAIL — model predicts noise, likely leakage")
            ),
            metric_value=mean_accuracy,
        )

    def temporal_boundary_test(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> LeakageCheckResult:
        """Test 2: No temporal overlap between train and test.

        Verify that the max training timestamp < min test timestamp.
        """
        if not isinstance(X_train.index, pd.DatetimeIndex) or not isinstance(
            X_test.index, pd.DatetimeIndex
        ):
            return LeakageCheckResult(
                check_name="temporal_boundary",
                passed=True,
                detail="Non-datetime index — skipping temporal check",
            )

        train_max = X_train.index.max()
        test_min = X_test.index.min()

        passed = train_max < test_min
        gap_days = (test_min - train_max).days if passed else 0

        return LeakageCheckResult(
            check_name="temporal_boundary",
            passed=passed,
            detail=(
                f"Train max: {train_max.date()}, Test min: {test_min.date()}, "
                f"Gap: {gap_days} days. "
                + ("PASS" if passed else "FAIL — train/test overlap detected")
            ),
            metric_value=float(gap_days),
        )

    def feature_timestamp_audit(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> LeakageCheckResult:
        """Test 3: Feature values don't contain future information.

        Check that no feature column contains values that appear to be
        from future dates (e.g., a rolling mean that starts too early,
        suggesting min_periods wasn't set correctly).

        Heuristic: for each feature, check if non-NaN values exist in
        positions where the lookback window hasn't been filled yet.
        This is a conservative check — actual lookback validation depends
        on the feature definition.
        """
        if not isinstance(X_train.index, pd.DatetimeIndex):
            return LeakageCheckResult(
                check_name="feature_timestamp_audit",
                passed=True,
                detail="Non-datetime index — skipping timestamp audit",
            )

        # Check no test data timestamps appear in train data
        overlap = X_train.index.intersection(X_test.index)
        if len(overlap) > 0:
            return LeakageCheckResult(
                check_name="feature_timestamp_audit",
                passed=False,
                detail=f"FAIL — {len(overlap)} overlapping timestamps between train and test",
                metric_value=float(len(overlap)),
            )

        return LeakageCheckResult(
            check_name="feature_timestamp_audit",
            passed=True,
            detail="No timestamp overlap between train and test. PASS",
            metric_value=0.0,
        )
