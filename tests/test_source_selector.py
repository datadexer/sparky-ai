"""Tests for source selector module."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sparky.data.source_selector import (
    BGEOMETRICS_EXCLUSIVE,
    DIVERGENCE_THRESHOLD,
    OVERLAPPING_METRICS,
    SourceScore,
    SourceSelector,
)


@pytest.fixture
def reference_date():
    """Fixed reference date for testing."""
    return pd.Timestamp("2024-01-15", tz="UTC")


@pytest.fixture
def sample_dates():
    """Sample date range for test data."""
    return pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")


@pytest.fixture
def bgeometrics_df(sample_dates):
    """Sample BGeometrics DataFrame with overlapping and exclusive metrics."""
    return pd.DataFrame(
        {
            # Overlapping metrics
            "hash_rate": np.linspace(100, 110, len(sample_dates)),
            "active_addresses": np.linspace(500, 600, len(sample_dates)),
            # Exclusive metrics
            "mvrv_zscore": np.linspace(1.0, 1.5, len(sample_dates)),
            "sopr": np.linspace(1.01, 1.05, len(sample_dates)),
            "nupl": np.linspace(0.5, 0.6, len(sample_dates)),
        },
        index=sample_dates,
    )


@pytest.fixture
def coinmetrics_df(sample_dates):
    """Sample CoinMetrics DataFrame with overlapping metrics."""
    return pd.DataFrame(
        {
            "HashRate": np.linspace(100, 110, len(sample_dates)),
            "AdrActCnt": np.linspace(500, 600, len(sample_dates)),
        },
        index=sample_dates,
    )


@pytest.fixture
def blockchain_com_df(sample_dates):
    """Sample Blockchain.com DataFrame (reference source)."""
    return pd.DataFrame(
        {
            "hash_rate": np.linspace(100, 110, len(sample_dates)),
            "active_addresses": np.linspace(500, 600, len(sample_dates)),
        },
        index=sample_dates,
    )


class TestSourceSelector:
    """Tests for SourceSelector class."""

    def test_select_btc_onchain_with_all_sources(
        self, bgeometrics_df, coinmetrics_df, blockchain_com_df, reference_date
    ):
        """Test that select_btc_onchain returns unified DataFrame with all sources available."""
        selector = SourceSelector()
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=bgeometrics_df,
            coinmetrics_df=coinmetrics_df,
            blockchain_com_df=blockchain_com_df,
            reference_date=reference_date,
        )

        # Check result DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(bgeometrics_df)

        # Should have overlapping + exclusive metrics
        expected_metrics = set(OVERLAPPING_METRICS.keys()) | set(BGEOMETRICS_EXCLUSIVE)
        # Filter to only metrics that exist in our test data
        expected_in_result = {"hash_rate", "active_addresses", "mvrv_zscore", "sopr", "nupl"}
        assert set(result.columns) == expected_in_result

        # Check scores were generated
        assert len(scores) > 0
        assert all(isinstance(s, SourceScore) for s in scores)

        # Check that exactly one source was selected per overlapping metric
        for metric in OVERLAPPING_METRICS.keys():
            metric_scores = [s for s in scores if s.metric == metric]
            selected_count = sum(s.selected for s in metric_scores)
            assert selected_count == 1, f"Expected 1 selected source for {metric}, got {selected_count}"

    def test_select_best_source_for_overlapping_metrics(
        self, bgeometrics_df, reference_date
    ):
        """Test that the best source is selected based on completeness and freshness."""
        # Create coinmetrics data that is less complete than bgeometrics
        dates = bgeometrics_df.index
        incomplete_cm_df = pd.DataFrame(
            {
                "HashRate": [100, np.nan, 102, np.nan, 104, np.nan, 106, np.nan, 108, 110],
                "AdrActCnt": [500, np.nan, 520, np.nan, 540, np.nan, 560, np.nan, 580, 600],
            },
            index=dates,
        )

        selector = SourceSelector()
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=bgeometrics_df,
            coinmetrics_df=incomplete_cm_df,
            blockchain_com_df=None,
            reference_date=reference_date,
        )

        # BGeometrics should be selected due to better completeness
        hash_rate_scores = [s for s in scores if s.metric == "hash_rate"]
        selected = [s for s in hash_rate_scores if s.selected]
        assert len(selected) == 1
        assert selected[0].source == "bgeometrics"
        assert selected[0].completeness == 1.0

        # Check CoinMetrics has lower completeness
        cm_score = [s for s in hash_rate_scores if s.source == "coinmetrics"][0]
        assert cm_score.completeness < 1.0

    def test_bgeometrics_exclusive_metrics_always_included(
        self, bgeometrics_df, reference_date
    ):
        """Test that BGeometrics-exclusive metrics are always included from bgeometrics."""
        selector = SourceSelector()
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=bgeometrics_df,
            coinmetrics_df=None,
            blockchain_com_df=None,
            reference_date=reference_date,
        )

        # Check that exclusive metrics present in test data are in result
        expected_exclusive = ["mvrv_zscore", "sopr", "nupl"]
        for metric in expected_exclusive:
            assert metric in result.columns
            assert not result[metric].isna().all()

        # Check scores for exclusive metrics
        exclusive_scores = [s for s in scores if s.metric in BGEOMETRICS_EXCLUSIVE]
        assert all(s.source == "bgeometrics" for s in exclusive_scores)
        assert all(s.selected for s in exclusive_scores)

    def test_single_source_available(self, bgeometrics_df, reference_date):
        """Test that when only one source is available, it gets selected."""
        selector = SourceSelector()
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=bgeometrics_df,
            coinmetrics_df=None,
            blockchain_com_df=None,
            reference_date=reference_date,
        )

        # All overlapping metrics should come from bgeometrics
        overlapping_scores = [s for s in scores if s.metric in OVERLAPPING_METRICS.keys()]
        assert all(s.source == "bgeometrics" for s in overlapping_scores)
        assert all(s.selected for s in overlapping_scores)

    def test_divergence_warning(self, sample_dates, reference_date, caplog):
        """Test that divergence warning is logged when source diverges >10% from reference."""
        # Create divergent data: coinmetrics differs by >10% from reference
        reference_df = pd.DataFrame(
            {"hash_rate": np.linspace(100, 110, len(sample_dates))},
            index=sample_dates,
        )

        # CoinMetrics data diverges by ~20%
        divergent_cm_df = pd.DataFrame(
            {"HashRate": np.linspace(120, 132, len(sample_dates))},
            index=sample_dates,
        )

        selector = SourceSelector()
        with caplog.at_level(logging.WARNING):
            result, scores = selector.select_btc_onchain(
                bgeometrics_df=None,
                coinmetrics_df=divergent_cm_df,
                blockchain_com_df=reference_df,
                reference_date=reference_date,
            )

        # Check that warning was logged
        assert any("diverges" in record.message for record in caplog.records)

        # Check that MAPE was calculated and exceeds threshold
        hash_rate_scores = [s for s in scores if s.metric == "hash_rate"]
        assert len(hash_rate_scores) > 0
        assert hash_rate_scores[0].reference_mape is not None
        assert hash_rate_scores[0].reference_mape > DIVERGENCE_THRESHOLD

    def test_empty_inputs_produce_empty_output(self, reference_date):
        """Test that empty inputs produce empty DataFrame output."""
        selector = SourceSelector()

        # Test with all None inputs
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=None,
            coinmetrics_df=None,
            blockchain_com_df=None,
            reference_date=reference_date,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert len(scores) == 0

        # Test with empty DataFrames
        empty_df = pd.DataFrame()
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=empty_df,
            coinmetrics_df=empty_df,
            blockchain_com_df=empty_df,
            reference_date=reference_date,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_select_eth_onchain_returns_coinmetrics_directly(self, sample_dates):
        """Test that select_eth_onchain returns CoinMetrics data directly."""
        eth_df = pd.DataFrame(
            {
                "HashRate": np.linspace(200, 220, len(sample_dates)),
                "AdrActCnt": np.linspace(1000, 1100, len(sample_dates)),
                "TxCnt": np.linspace(50000, 60000, len(sample_dates)),
            },
            index=sample_dates,
        )

        selector = SourceSelector()
        result = selector.select_eth_onchain(coinmetrics_df=eth_df)

        # Should return the input DataFrame as-is
        pd.testing.assert_frame_equal(result, eth_df)

    def test_select_eth_onchain_with_empty_input(self):
        """Test that select_eth_onchain handles empty/None input."""
        selector = SourceSelector()

        # Test with None
        result = selector.select_eth_onchain(coinmetrics_df=None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

        # Test with empty DataFrame
        result = selector.select_eth_onchain(coinmetrics_df=pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestSourceScore:
    """Tests for SourceScore computation."""

    def test_score_completeness(self, sample_dates, reference_date):
        """Test that completeness is correctly computed."""
        selector = SourceSelector()

        # Perfect completeness
        complete_series = pd.Series(
            np.linspace(100, 110, len(sample_dates)),
            index=sample_dates,
        )
        score = selector._score_series(
            complete_series, "test_source", "test_metric", None, reference_date
        )
        assert score.completeness == 1.0

        # 50% completeness
        half_complete = pd.Series(
            [100, np.nan, 102, np.nan, 104, np.nan, 106, np.nan, 108, np.nan],
            index=sample_dates,
        )
        score = selector._score_series(
            half_complete, "test_source", "test_metric", None, reference_date
        )
        assert score.completeness == 0.5

        # Empty series
        empty_series = pd.Series([np.nan] * len(sample_dates), index=sample_dates)
        score = selector._score_series(
            empty_series, "test_source", "test_metric", None, reference_date
        )
        assert score.completeness == 0.0

    def test_score_freshness(self, sample_dates, reference_date):
        """Test that freshness is correctly computed in days."""
        selector = SourceSelector()

        # Data with last point on 2024-01-10
        series = pd.Series(
            np.linspace(100, 110, len(sample_dates)),
            index=sample_dates,
        )
        score = selector._score_series(
            series, "test_source", "test_metric", None, reference_date
        )

        # reference_date is 2024-01-15, last data is 2024-01-10
        expected_days = (reference_date - sample_dates[-1]).total_seconds() / 86400
        assert score.freshness_days == pytest.approx(expected_days)

        # Empty series should have high freshness penalty
        empty_series = pd.Series([np.nan] * len(sample_dates), index=sample_dates)
        score = selector._score_series(
            empty_series, "test_source", "test_metric", None, reference_date
        )
        assert score.freshness_days == 999.0

    def test_score_reference_mape(self, sample_dates, reference_date):
        """Test that reference MAPE is correctly computed."""
        selector = SourceSelector()

        # Perfect agreement with reference
        source_series = pd.Series(
            np.linspace(100, 110, len(sample_dates)),
            index=sample_dates,
        )
        reference_series = pd.Series(
            np.linspace(100, 110, len(sample_dates)),
            index=sample_dates,
        )
        score = selector._score_series(
            source_series, "test_source", "test_metric", reference_series, reference_date
        )
        assert score.reference_mape is not None
        assert score.reference_mape == pytest.approx(0.0, abs=1e-10)

        # 10% error
        divergent_series = pd.Series(
            np.linspace(110, 121, len(sample_dates)),  # 10% higher
            index=sample_dates,
        )
        score = selector._score_series(
            divergent_series, "test_source", "test_metric", reference_series, reference_date
        )
        assert score.reference_mape is not None
        assert score.reference_mape == pytest.approx(0.1, rel=0.01)

        # No reference provided
        score = selector._score_series(
            source_series, "test_source", "test_metric", None, reference_date
        )
        assert score.reference_mape is None

    def test_score_attributes(self, sample_dates, reference_date):
        """Test that SourceScore contains all expected attributes."""
        selector = SourceSelector()
        series = pd.Series(
            np.linspace(100, 110, len(sample_dates)),
            index=sample_dates,
        )

        score = selector._score_series(
            series, "bgeometrics", "hash_rate", None, reference_date
        )

        assert score.source == "bgeometrics"
        assert score.metric == "hash_rate"
        assert isinstance(score.completeness, float)
        assert isinstance(score.freshness_days, float)
        assert score.reference_mape is None or isinstance(score.reference_mape, float)
        assert isinstance(score.selected, bool)
        assert not score.selected  # Default is False


class TestScoringLogic:
    """Tests for scoring logic that picks the best source."""

    def test_more_complete_source_wins(self, sample_dates, reference_date):
        """Test that a more complete source is preferred."""
        # BGeometrics: 100% complete
        bg_df = pd.DataFrame(
            {"hash_rate": np.linspace(100, 110, len(sample_dates))},
            index=sample_dates,
        )

        # CoinMetrics: 50% complete
        cm_df = pd.DataFrame(
            {"HashRate": [100, np.nan, 102, np.nan, 104, np.nan, 106, np.nan, 108, 110]},
            index=sample_dates,
        )

        selector = SourceSelector()
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=bg_df,
            coinmetrics_df=cm_df,
            blockchain_com_df=None,
            reference_date=reference_date,
        )

        selected = [s for s in scores if s.selected and s.metric == "hash_rate"][0]
        assert selected.source == "bgeometrics"
        assert selected.completeness > 0.9

    def test_fresher_source_wins(self, reference_date):
        """Test that a fresher source is preferred when completeness is similar."""
        # BGeometrics: ends on 2024-01-10 (5 days old)
        old_dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
        bg_df = pd.DataFrame(
            {"hash_rate": np.linspace(100, 110, len(old_dates))},
            index=old_dates,
        )

        # CoinMetrics: ends on 2024-01-14 (1 day old)
        fresh_dates = pd.date_range("2024-01-05", "2024-01-14", freq="D", tz="UTC")
        cm_df = pd.DataFrame(
            {"HashRate": np.linspace(105, 114, len(fresh_dates))},
            index=fresh_dates,
        )

        selector = SourceSelector()
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=bg_df,
            coinmetrics_df=cm_df,
            blockchain_com_df=None,
            reference_date=reference_date,
        )

        selected = [s for s in scores if s.selected and s.metric == "hash_rate"][0]
        # CoinMetrics should win due to freshness
        assert selected.source == "coinmetrics"
        assert selected.freshness_days < 2.0

    def test_better_agreement_with_reference_wins(self, sample_dates, reference_date):
        """Test that better agreement with reference is preferred."""
        # Reference data
        ref_df = pd.DataFrame(
            {"hash_rate": np.linspace(100, 110, len(sample_dates))},
            index=sample_dates,
        )

        # BGeometrics: perfect match with reference
        bg_df = pd.DataFrame(
            {"hash_rate": np.linspace(100, 110, len(sample_dates))},
            index=sample_dates,
        )

        # CoinMetrics: 20% divergence from reference
        cm_df = pd.DataFrame(
            {"HashRate": np.linspace(120, 132, len(sample_dates))},
            index=sample_dates,
        )

        selector = SourceSelector()
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=bg_df,
            coinmetrics_df=cm_df,
            blockchain_com_df=ref_df,
            reference_date=reference_date,
        )

        selected = [s for s in scores if s.selected and s.metric == "hash_rate"][0]
        # BGeometrics should win due to perfect agreement
        assert selected.source == "bgeometrics"
        assert selected.reference_mape is not None
        assert selected.reference_mape < 0.01


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_columns_in_source(self, sample_dates, reference_date):
        """Test handling when expected columns are missing from source."""
        # DataFrame without the expected columns
        bg_df = pd.DataFrame(
            {"some_other_metric": np.linspace(1, 10, len(sample_dates))},
            index=sample_dates,
        )

        selector = SourceSelector()
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=bg_df,
            coinmetrics_df=None,
            blockchain_com_df=None,
            reference_date=reference_date,
        )

        # Should handle gracefully with empty result
        assert result.empty
        assert len(scores) == 0

    def test_misaligned_indices(self, reference_date):
        """Test handling when source indices don't align."""
        dates1 = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
        dates2 = pd.date_range("2024-01-05", "2024-01-15", freq="D", tz="UTC")

        bg_df = pd.DataFrame(
            {"hash_rate": np.linspace(100, 110, len(dates1))},
            index=dates1,
        )

        cm_df = pd.DataFrame(
            {"HashRate": np.linspace(105, 116, len(dates2))},
            index=dates2,
        )

        selector = SourceSelector()
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=bg_df,
            coinmetrics_df=cm_df,
            blockchain_com_df=None,
            reference_date=reference_date,
        )

        # Should still produce a result
        assert not result.empty
        assert "hash_rate" in result.columns

    def test_all_nan_series(self, sample_dates, reference_date):
        """Test handling of series with all NaN values."""
        bg_df = pd.DataFrame(
            {"hash_rate": [np.nan] * len(sample_dates)},
            index=sample_dates,
        )

        cm_df = pd.DataFrame(
            {"HashRate": [np.nan] * len(sample_dates)},
            index=sample_dates,
        )

        selector = SourceSelector()
        result, scores = selector.select_btc_onchain(
            bgeometrics_df=bg_df,
            coinmetrics_df=cm_df,
            blockchain_com_df=None,
            reference_date=reference_date,
        )

        # Should still select a source even with all NaNs
        assert "hash_rate" in result.columns
        hash_rate_scores = [s for s in scores if s.metric == "hash_rate"]
        assert len(hash_rate_scores) == 2  # Both sources scored
        assert sum(s.selected for s in hash_rate_scores) == 1  # One selected

    def test_timezone_handling(self, sample_dates):
        """Test handling of timezone-naive and timezone-aware indices."""
        # Timezone-naive dates
        naive_dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        bg_df = pd.DataFrame(
            {"hash_rate": np.linspace(100, 110, len(naive_dates))},
            index=naive_dates,
        )

        # Should handle without errors
        selector = SourceSelector()
        reference_date_utc = pd.Timestamp("2024-01-15", tz="UTC")

        result, scores = selector.select_btc_onchain(
            bgeometrics_df=bg_df,
            coinmetrics_df=None,
            blockchain_com_df=None,
            reference_date=reference_date_utc,
        )

        assert not result.empty
        assert len(scores) > 0
        # Freshness should still be computed
        assert all(s.freshness_days >= 0 for s in scores)
