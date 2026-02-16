"""Feature selection to prevent overfitting.

Systematic feature reduction before model training:
1. Correlation filter — drop redundant features (>0.85 pairwise)
2. Importance threshold — drop noise features (importance < 0.01)
3. Stability test — flag features with unstable importance across folds

Max features cap: 20 (from research_standards in trading_rules.yaml).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_FEATURES = 20  # Hard ceiling from research_standards


@dataclass
class SelectionResult:
    """Result of feature selection process."""

    selected_features: list[str]
    dropped_features: list[dict]  # [{name, reason, detail}] — actually removed features
    flagged_features: list[dict] = field(default_factory=list)  # [{name, reason, detail}] — flagged but kept
    correlation_matrix: Optional[pd.DataFrame] = None
    importance_scores: Optional[dict[str, float]] = None
    stability_scores: Optional[dict[str, float]] = None


class FeatureSelector:
    """Systematic feature selection pipeline.

    Usage:
        selector = FeatureSelector()
        result = selector.select(X, y, model_class=XGBClassifier)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.85,
        importance_threshold: float = 0.01,
        stability_variance_threshold: float = 0.3,
        max_features: int = MAX_FEATURES,
        n_stability_folds: int = 10,
    ):
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.stability_variance_threshold = stability_variance_threshold
        self.max_features = max_features
        self.n_stability_folds = n_stability_folds

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model=None,
    ) -> SelectionResult:
        """Run full selection pipeline.

        Args:
            X: Feature matrix.
            y: Target variable.
            model: Model with .fit() and .feature_importances_ (e.g., XGBClassifier).
                   If None, skips importance and stability steps.

        Returns:
            SelectionResult with selected features and diagnostics.
        """
        dropped = []
        flagged = []
        remaining = list(X.columns)

        # Step 1: Correlation filter
        remaining, corr_dropped, corr_matrix = self._correlation_filter(X, y, remaining)
        dropped.extend(corr_dropped)

        # Step 2: Importance threshold (requires model)
        importance_scores = None
        if model is not None and len(remaining) > 0:
            remaining, imp_dropped, importance_scores = self._importance_filter(
                X[remaining], y, model
            )
            dropped.extend(imp_dropped)

        # Step 3: Stability test (requires model)
        stability_scores = None
        if model is not None and len(remaining) > 0:
            stability_scores, stability_flagged = self._stability_test(
                X[remaining], y, model
            )
            for feat_info in stability_flagged:
                logger.warning(
                    f"Unstable feature '{feat_info['name']}': "
                    f"importance variance = {feat_info['detail']}"
                )
            # Don't drop unstable features, just flag them
            flagged.extend(stability_flagged)

        # Step 4: Cap at max_features
        if len(remaining) > self.max_features:
            if importance_scores:
                # Keep top features by importance
                sorted_by_imp = sorted(
                    remaining,
                    key=lambda f: importance_scores.get(f, 0),
                    reverse=True,
                )
                capped = sorted_by_imp[self.max_features:]
                remaining = sorted_by_imp[:self.max_features]
                for f in capped:
                    dropped.append({
                        "name": f,
                        "reason": "max_features_cap",
                        "detail": f"Exceeded {self.max_features} feature limit",
                    })
            else:
                remaining = remaining[:self.max_features]

        logger.info(
            f"Feature selection: {len(remaining)} selected, "
            f"{len(dropped)} dropped, {len(flagged)} flagged"
        )

        return SelectionResult(
            selected_features=remaining,
            dropped_features=dropped,
            flagged_features=flagged,
            correlation_matrix=corr_matrix,
            importance_scores=importance_scores,
            stability_scores=stability_scores,
        )

    def _correlation_filter(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str],
    ) -> tuple[list[str], list[dict], pd.DataFrame]:
        """Drop features with pairwise correlation > threshold.

        When two features are correlated, keep the one with higher
        univariate correlation with the target.
        """
        if len(features) <= 1:
            return features, [], pd.DataFrame()

        X_sub = X[features]
        corr_matrix = X_sub.corr()
        target_corr = X_sub.corrwith(y).abs()

        to_drop = set()
        dropped = []

        for i in range(len(features)):
            if features[i] in to_drop:
                continue
            for j in range(i + 1, len(features)):
                if features[j] in to_drop:
                    continue
                if abs(corr_matrix.iloc[i, j]) > self.correlation_threshold:
                    # Drop the one with lower target correlation
                    fi, fj = features[i], features[j]
                    if target_corr.get(fi, 0) >= target_corr.get(fj, 0):
                        to_drop.add(fj)
                        dropped.append({
                            "name": fj,
                            "reason": "correlation",
                            "detail": f"corr={corr_matrix.iloc[i, j]:.3f} with {fi}",
                        })
                    else:
                        to_drop.add(fi)
                        dropped.append({
                            "name": fi,
                            "reason": "correlation",
                            "detail": f"corr={corr_matrix.iloc[i, j]:.3f} with {fj}",
                        })

        remaining = [f for f in features if f not in to_drop]
        return remaining, dropped, corr_matrix

    def _importance_filter(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
    ) -> tuple[list[str], list[dict], dict[str, float]]:
        """Drop features with importance < threshold."""
        # Fit model to get importances
        model.fit(X, y)
        importances = model.feature_importances_

        scores = dict(zip(X.columns, importances))
        dropped = []
        remaining = []

        for feat, imp in scores.items():
            if imp < self.importance_threshold:
                dropped.append({
                    "name": feat,
                    "reason": "low_importance",
                    "detail": f"importance={imp:.4f} < {self.importance_threshold}",
                })
            else:
                remaining.append(feat)

        return remaining, dropped, scores

    def _stability_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
    ) -> tuple[dict[str, float], list[dict]]:
        """Test feature importance stability across k-fold splits.

        Returns variance of importance per feature. Flags features with
        variance > threshold (unstable — importance depends heavily on data split).
        """
        n = len(X)
        fold_size = n // self.n_stability_folds
        if fold_size < 10:
            logger.warning("Not enough data for stability test")
            return {}, []

        importance_matrix = []

        for fold in range(self.n_stability_folds):
            start = fold * fold_size
            end = start + fold_size
            # Train on everything except this fold
            mask = np.ones(n, dtype=bool)
            mask[start:end] = False
            X_train = X.iloc[mask]
            y_train = y.iloc[mask]

            model.fit(X_train, y_train)
            importance_matrix.append(dict(zip(X.columns, model.feature_importances_)))

        imp_df = pd.DataFrame(importance_matrix)
        variances = imp_df.var().to_dict()

        flagged = []
        for feat, var in variances.items():
            if var > self.stability_variance_threshold:
                flagged.append({
                    "name": feat,
                    "reason": "unstable_importance",
                    "detail": f"variance={var:.4f} > {self.stability_variance_threshold}",
                })

        return variances, flagged
