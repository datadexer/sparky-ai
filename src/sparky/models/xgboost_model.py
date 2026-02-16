"""XGBoost binary classifier for crypto signal prediction.

Implements ModelProtocol for backtester compatibility. Uses conservative
hyperparameters optimized for high-noise crypto data: shallow trees, strong
regularization, and probability calibration for robust predictions.

Default hyperparameters:
- max_depth=5: Shallow trees generalize better on noisy crypto data
- learning_rate=0.1: Conservative to prevent overfitting
- subsample=0.8, colsample_bytree=0.8: Sample diversity
- min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0: Strong regularization
- objective="binary:logistic": Binary classification with calibrated probabilities
"""

import logging

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost binary classifier for crypto signal prediction.

    Implements ModelProtocol (fit/predict) for backtester compatibility.
    Predicts {0=flat, 1=long} signals with probability calibration.

    Args:
        max_depth: Maximum tree depth (default 5 for shallow, regularized trees).
        learning_rate: Boosting learning rate (default 0.1).
        n_estimators: Number of boosting rounds (default 100).
        subsample: Row sampling fraction (default 0.8).
        colsample_bytree: Column sampling fraction (default 0.8).
        min_child_weight: Minimum sum of instance weight in a child (default 5).
        reg_alpha: L1 regularization (default 0.1).
        reg_lambda: L2 regularization (default 1.0).
        random_state: Random seed for reproducibility.
        eval_metric: Evaluation metric (default "logloss").
        use_label_encoder: Disable sklearn label encoder (default False).
    """

    def __init__(
        self,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        eval_metric: str = "logloss",
        use_label_encoder: bool = False,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.use_label_encoder = use_label_encoder

        self.model = XGBClassifier(
            objective="binary:logistic",
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            eval_metric=self.eval_metric,
            use_label_encoder=self.use_label_encoder,
            tree_method="hist",
            device="cuda",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train XGBoost model on features X and binary labels y.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels {0, 1} where 1=long, 0=flat.
        """
        # Handle NaN in features (XGBoost supports NaN natively)
        # But log warning if too many NaN
        nan_fraction = X.isna().sum().sum() / (X.shape[0] * X.shape[1])
        if nan_fraction > 0.3:
            logger.warning(
                f"High NaN fraction in features: {nan_fraction:.1%}. "
                "XGBoost will handle this, but consider imputation."
            )

        # Fit model
        self.model.fit(X, y, verbose=False)
        logger.info(f"XGBoost trained on {len(X)} samples with {X.shape[1]} features")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary signals {0, 1} for features X.

        Uses 0.5 probability threshold for classification.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Binary signals as numpy array {0=flat, 1=long}.
        """
        # XGBoost's predict() returns class labels {0, 1} directly
        predictions = self.model.predict(X)
        return predictions.astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for features X.

        Returns:
            Array of shape (n_samples, 2) with [P(class=0), P(class=1)].
        """
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> pd.DataFrame:
        """Extract feature importances from trained model.

        Returns:
            DataFrame with columns ['feature', 'importance'], sorted descending.
        """
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model not trained yet — call fit() first")

        importances = pd.DataFrame({
            "feature": self.model.feature_names_in_,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return importances
