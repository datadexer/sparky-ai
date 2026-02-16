"""Holdout data guard — prevents unauthorized OOS evaluation.

Any script that loads data for model evaluation MUST call
HoldoutGuard.check_data_boundaries() before proceeding.
Violations are logged and raise HoldoutViolation.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

POLICY_PATH = Path("configs/holdout_policy.yaml")
OOS_LOG_PATH = Path("results/oos_evaluations.jsonl")


class HoldoutViolation(Exception):
    """Raised when code attempts to use holdout data without authorization."""
    pass


class HoldoutGuard:
    """Enforces holdout data boundaries.

    Usage:
        guard = HoldoutGuard()

        # During research (default) — blocks OOS data
        guard.check_data_boundaries(df, asset="btc")

        # When OOS approved — requires explicit authorization
        guard.authorize_oos_evaluation(
            model_name="catboost_v3",
            approach_family="tree_ensemble",
            approved_by="research-business-manager",
            in_sample_sharpe=0.85
        )
        guard.check_data_boundaries(df, asset="btc", oos_authorized=True)
    """

    def __init__(self, policy_path: Optional[Path] = None):
        self.policy_path = policy_path or POLICY_PATH
        with open(self.policy_path) as f:
            self.policy = yaml.safe_load(f)
        self._oos_authorization = None

    def get_oos_boundary(self, asset: str) -> tuple[str, int]:
        """Get OOS start date and embargo days for an asset."""
        asset_policy = self.policy["holdout_periods"].get(
            asset, self.policy["holdout_periods"]["cross_asset"]
        )
        return asset_policy["oos_start"], asset_policy["embargo_days"]

    def get_max_training_date(self, asset: str) -> pd.Timestamp:
        """Get the latest date allowed for training data."""
        oos_start, embargo_days = self.get_oos_boundary(asset)
        oos_ts = pd.Timestamp(oos_start, tz="UTC")
        return oos_ts - pd.Timedelta(days=embargo_days)

    def check_data_boundaries(
        self,
        df: pd.DataFrame,
        asset: str,
        oos_authorized: bool = False,
        purpose: str = "training"
    ) -> None:
        """Check that data respects holdout boundaries.

        Args:
            df: DataFrame with DatetimeIndex
            asset: Asset name (btc, eth, etc.)
            oos_authorized: Whether OOS evaluation has been explicitly authorized
            purpose: "training", "validation", or "oos_evaluation"

        Raises:
            HoldoutViolation: If data crosses into holdout without authorization
        """
        if df.empty:
            return

        oos_start, embargo_days = self.get_oos_boundary(asset)
        oos_ts = pd.Timestamp(oos_start, tz="UTC")
        embargo_ts = oos_ts - pd.Timedelta(days=embargo_days)

        data_max = df.index.max()
        if hasattr(data_max, 'tz') and data_max.tz is None:
            data_max = data_max.tz_localize("UTC")

        if purpose in ("training", "validation"):
            if data_max >= embargo_ts:
                raise HoldoutViolation(
                    f"HOLDOUT VIOLATION: {purpose} data for {asset} extends to "
                    f"{data_max.date()} but must end before {embargo_ts.date()} "
                    f"(embargo boundary). OOS starts {oos_start}."
                )

        elif purpose == "oos_evaluation":
            if not oos_authorized:
                raise HoldoutViolation(
                    f"HOLDOUT VIOLATION: OOS evaluation for {asset} attempted "
                    f"without authorization. Get approval from RBM or AK first."
                )
            if self._oos_authorization is None:
                raise HoldoutViolation(
                    f"HOLDOUT VIOLATION: OOS evaluation requested but no "
                    f"authorization record. Call authorize_oos_evaluation() first."
                )

        logger.info(
            f"[HOLDOUT GUARD] Data boundary check PASSED for {asset} "
            f"({purpose}): data ends {data_max.date()}, "
            f"embargo starts {embargo_ts.date()}"
        )

    def authorize_oos_evaluation(
        self,
        model_name: str,
        approach_family: str,
        approved_by: str,
        in_sample_sharpe: float,
    ) -> None:
        """Record OOS evaluation authorization."""
        valid_approvers = self.policy["policy"]["approvers"]
        if approved_by not in valid_approvers:
            raise HoldoutViolation(
                f"OOS evaluation can only be approved by {valid_approvers}, "
                f"not '{approved_by}'"
            )

        self._oos_authorization = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "approach_family": approach_family,
            "approved_by": approved_by,
            "in_sample_sharpe": in_sample_sharpe,
        }
        logger.info(
            f"[HOLDOUT GUARD] OOS evaluation authorized for {model_name} "
            f"by {approved_by}"
        )

    def log_oos_result(self, oos_sharpe: float, verdict: str) -> None:
        """Log OOS evaluation result to append-only log."""
        if self._oos_authorization is None:
            raise HoldoutViolation("Cannot log OOS result without authorization")

        entry = {
            **self._oos_authorization,
            "oos_sharpe": oos_sharpe,
            "verdict": verdict,
        }

        OOS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OOS_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(
            f"[HOLDOUT GUARD] OOS result logged: {entry['model_name']} "
            f"Sharpe={oos_sharpe}, verdict={verdict}"
        )
        self._oos_authorization = None  # One evaluation per authorization
