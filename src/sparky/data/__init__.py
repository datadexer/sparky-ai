"""Sparky data pipeline.

ALL timestamps in the Sparky data pipeline MUST be timezone-aware (UTC).
Use assert_tz_aware() at pipeline boundaries to enforce this invariant.
"""

import pandas as pd


def assert_tz_aware(df: pd.DataFrame, context: str = "") -> None:
    """Assert that a DataFrame's DatetimeIndex is timezone-aware (UTC).

    Call this at data pipeline boundaries: after loading data, before saving,
    and when passing DataFrames between modules.

    Args:
        df: DataFrame to validate.
        context: Description of where this check is happening (for error messages).

    Raises:
        ValueError: If DatetimeIndex is timezone-naive.
    """
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        raise ValueError(
            f"Timezone-naive DatetimeIndex detected{f' at {context}' if context else ''}. "
            f"ALL timestamps must be timezone-aware (UTC). "
            f"Fix: df.index = df.index.tz_localize('UTC')"
        )
