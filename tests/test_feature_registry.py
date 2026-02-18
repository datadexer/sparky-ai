"""Tests for feature registry and matrix builder."""

import logging

import pandas as pd
import pytest

from sparky.features.registry import FeatureDefinition, FeatureRegistry


@pytest.fixture
def registry():
    """Create a fresh FeatureRegistry."""
    return FeatureRegistry()


@pytest.fixture
def sample_data():
    """Create sample DataFrames for testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    price_df = pd.DataFrame(
        {"close": [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]},
        index=dates,
    )
    volume_df = pd.DataFrame(
        {"volume": [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1400, 1350, 1450]},
        index=dates,
    )
    return {"price": price_df, "volume": volume_df}


@pytest.fixture
def simple_feature():
    """Create a simple feature definition."""
    return FeatureDefinition(
        name="close_price",
        category="technical",
        compute_fn=lambda data: data["price"]["close"],
        input_columns=["close"],
        lookback=0,
        description="Raw close price",
    )


class TestFeatureRegistration:
    """Tests for feature registration and retrieval."""

    def test_register_and_retrieve(self, registry, simple_feature):
        """Test basic registration and retrieval."""
        registry.register(simple_feature)
        retrieved = registry.get("close_price")
        assert retrieved.name == "close_price"
        assert retrieved.category == "technical"
        assert retrieved.description == "Raw close price"

    def test_overwrite_existing_feature_with_warning(self, registry, simple_feature, caplog):
        """Test that overwriting an existing feature logs a warning."""
        registry.register(simple_feature)

        # Register another feature with the same name
        new_feature = FeatureDefinition(
            name="close_price",
            category="market_context",
            compute_fn=lambda data: data["price"]["close"] * 2,
            input_columns=["close"],
        )

        with caplog.at_level(logging.WARNING):
            registry.register(new_feature)

        assert "Overwriting existing feature: close_price" in caplog.text

        # Verify the new feature replaced the old one
        retrieved = registry.get("close_price")
        assert retrieved.category == "market_context"

    def test_get_nonexistent_feature_raises_error(self, registry):
        """Test that getting a nonexistent feature raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            registry.get("nonexistent_feature")
        assert "Feature 'nonexistent_feature' not registered" in str(exc_info.value)


class TestListFeatures:
    """Tests for listing features with filters."""

    def test_list_all_features(self, registry):
        """Test listing all features without filters."""
        registry.register(
            FeatureDefinition(
                name="rsi_14",
                category="technical",
                compute_fn=lambda data: data["price"]["close"] / 100,
                input_columns=["close"],
            )
        )
        registry.register(
            FeatureDefinition(
                name="active_addresses",
                category="onchain_btc",
                compute_fn=lambda data: data["volume"]["volume"],
                input_columns=["volume"],
                asset="btc",
            )
        )

        features = registry.list_features()
        assert len(features) == 2
        assert "rsi_14" in features
        assert "active_addresses" in features

    def test_list_features_by_category(self, registry):
        """Test filtering features by category."""
        registry.register(
            FeatureDefinition(
                name="rsi_14",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
            )
        )
        registry.register(
            FeatureDefinition(
                name="macd",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
            )
        )
        registry.register(
            FeatureDefinition(
                name="active_addresses",
                category="onchain_btc",
                compute_fn=lambda data: data["volume"]["volume"],
                input_columns=["volume"],
            )
        )

        technical_features = registry.list_features(category="technical")
        assert len(technical_features) == 2
        assert "rsi_14" in technical_features
        assert "macd" in technical_features
        assert "active_addresses" not in technical_features

    def test_list_features_by_asset(self, registry):
        """Test filtering features by asset."""
        registry.register(
            FeatureDefinition(
                name="btc_specific",
                category="onchain_btc",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                asset="btc",
            )
        )
        registry.register(
            FeatureDefinition(
                name="eth_specific",
                category="onchain_eth",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                asset="eth",
            )
        )
        registry.register(
            FeatureDefinition(
                name="universal",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                asset="all",
            )
        )

        btc_features = registry.list_features(asset="btc")
        assert len(btc_features) == 2
        assert "btc_specific" in btc_features
        assert "universal" in btc_features
        assert "eth_specific" not in btc_features

    def test_list_features_by_category_and_asset(self, registry):
        """Test filtering features by both category and asset."""
        registry.register(
            FeatureDefinition(
                name="btc_onchain",
                category="onchain_btc",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                asset="btc",
            )
        )
        registry.register(
            FeatureDefinition(
                name="btc_technical",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                asset="btc",
            )
        )
        registry.register(
            FeatureDefinition(
                name="eth_onchain",
                category="onchain_eth",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                asset="eth",
            )
        )

        filtered = registry.list_features(category="onchain_btc", asset="btc")
        assert len(filtered) == 1
        assert "btc_onchain" in filtered


class TestBuildFeatureMatrix:
    """Tests for building feature matrices."""

    def test_build_feature_matrix_computes_correctly(self, registry, sample_data):
        """Test that build_feature_matrix computes features correctly."""
        registry.register(
            FeatureDefinition(
                name="close_price",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
            )
        )
        registry.register(
            FeatureDefinition(
                name="close_doubled",
                category="technical",
                compute_fn=lambda data: data["price"]["close"] * 2,
                input_columns=["close"],
            )
        )

        matrix = registry.build_feature_matrix(
            asset="btc",
            feature_names=["close_price", "close_doubled"],
            data=sample_data,
            drop_na_rows=False,
        )

        assert len(matrix) == 10
        assert list(matrix.columns) == ["close_price", "close_doubled"]
        assert matrix["close_price"].iloc[0] == 100
        assert matrix["close_doubled"].iloc[0] == 200
        assert matrix["close_price"].iloc[-1] == 109
        assert matrix["close_doubled"].iloc[-1] == 218

    def test_valid_from_masking(self, registry, sample_data):
        """Test that features return NaN before valid_from date."""
        registry.register(
            FeatureDefinition(
                name="restricted_feature",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                valid_from="2024-01-06",  # 6th day
            )
        )

        matrix = registry.build_feature_matrix(
            asset="btc",
            feature_names=["restricted_feature"],
            data=sample_data,
            drop_na_rows=False,
        )

        # First 5 days should be NaN
        assert matrix["restricted_feature"].iloc[:5].isna().all()
        # Days 6 onward should have values
        assert matrix["restricted_feature"].iloc[5:].notna().all()
        assert matrix["restricted_feature"].iloc[5] == 104  # Jan 6 value

    def test_drop_na_rows_removes_all_nan_rows(self, registry, sample_data):
        """Test that drop_na_rows removes rows where ALL features are NaN."""
        # Create two features with different valid_from dates
        registry.register(
            FeatureDefinition(
                name="early_feature",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                valid_from="2024-01-03",
            )
        )
        registry.register(
            FeatureDefinition(
                name="late_feature",
                category="technical",
                compute_fn=lambda data: data["price"]["close"] * 2,
                input_columns=["close"],
                valid_from="2024-01-07",
            )
        )

        matrix = registry.build_feature_matrix(
            asset="btc",
            feature_names=["early_feature", "late_feature"],
            data=sample_data,
            drop_na_rows=True,
        )

        # First 2 days (Jan 1-2) should be dropped (all NaN)
        # Days from Jan 3 onward should be kept
        assert len(matrix) == 8  # 10 - 2 = 8
        assert matrix.index[0] == pd.Timestamp("2024-01-03", tz="UTC")

    def test_drop_na_rows_keeps_partial_nan_rows(self, registry, sample_data):
        """Test that drop_na_rows keeps rows with partial NaN values."""
        registry.register(
            FeatureDefinition(
                name="early_feature",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                valid_from="2024-01-03",
            )
        )
        registry.register(
            FeatureDefinition(
                name="late_feature",
                category="technical",
                compute_fn=lambda data: data["price"]["close"] * 2,
                input_columns=["close"],
                valid_from="2024-01-07",
            )
        )

        matrix = registry.build_feature_matrix(
            asset="btc",
            feature_names=["early_feature", "late_feature"],
            data=sample_data,
            drop_na_rows=True,
        )

        # Days 3-6 should be kept even though late_feature is NaN
        jan_3_row = matrix.loc[pd.Timestamp("2024-01-03", tz="UTC")]
        assert pd.notna(jan_3_row["early_feature"])
        assert pd.isna(jan_3_row["late_feature"])

        jan_6_row = matrix.loc[pd.Timestamp("2024-01-06", tz="UTC")]
        assert pd.notna(jan_6_row["early_feature"])
        assert pd.isna(jan_6_row["late_feature"])

    def test_skips_features_not_applicable_for_asset(self, registry, sample_data, caplog):
        """Test that features not applicable for the asset are skipped."""
        registry.register(
            FeatureDefinition(
                name="btc_only",
                category="onchain_btc",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                asset="btc",
            )
        )
        registry.register(
            FeatureDefinition(
                name="eth_only",
                category="onchain_eth",
                compute_fn=lambda data: data["price"]["close"] * 2,
                input_columns=["close"],
                asset="eth",
            )
        )

        with caplog.at_level(logging.INFO):
            matrix = registry.build_feature_matrix(
                asset="btc",
                feature_names=["btc_only", "eth_only"],
                data=sample_data,
            )

        # Only btc_only should be computed
        assert "btc_only" in matrix.columns
        assert "eth_only" not in matrix.columns
        assert "Skipped feature 'eth_only': not applicable for btc" in caplog.text

    def test_handles_compute_fn_errors_gracefully(self, registry, sample_data, caplog):
        """Test that compute_fn errors are handled gracefully."""
        registry.register(
            FeatureDefinition(
                name="good_feature",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
            )
        )
        registry.register(
            FeatureDefinition(
                name="bad_feature",
                category="technical",
                compute_fn=lambda data: data["nonexistent"]["column"],
                input_columns=["nonexistent"],
            )
        )

        with caplog.at_level(logging.WARNING):
            matrix = registry.build_feature_matrix(
                asset="btc",
                feature_names=["good_feature", "bad_feature"],
                data=sample_data,
            )

        # good_feature should be computed
        assert "good_feature" in matrix.columns
        # bad_feature should be skipped
        assert "bad_feature" not in matrix.columns
        assert "Failed to compute feature 'bad_feature'" in caplog.text

    def test_handles_non_series_return_gracefully(self, registry, sample_data, caplog):
        """Test that compute_fn returning non-Series is handled gracefully."""
        registry.register(
            FeatureDefinition(
                name="good_feature",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
            )
        )
        registry.register(
            FeatureDefinition(
                name="returns_dataframe",
                category="technical",
                compute_fn=lambda data: data["price"],  # Returns DataFrame, not Series
                input_columns=["close"],
            )
        )

        with caplog.at_level(logging.INFO):
            matrix = registry.build_feature_matrix(
                asset="btc",
                feature_names=["good_feature", "returns_dataframe"],
                data=sample_data,
            )

        assert "good_feature" in matrix.columns
        assert "returns_dataframe" not in matrix.columns
        assert "Skipped feature 'returns_dataframe': compute_fn did not return a Series" in caplog.text

    def test_empty_matrix_when_no_features_computed(self, registry, sample_data, caplog):
        """Test that empty DataFrame is returned when no features are computed."""
        registry.register(
            FeatureDefinition(
                name="bad_feature",
                category="technical",
                compute_fn=lambda data: data["nonexistent"]["column"],
                input_columns=["nonexistent"],
            )
        )

        with caplog.at_level(logging.WARNING):
            matrix = registry.build_feature_matrix(
                asset="btc",
                feature_names=["bad_feature"],
                data=sample_data,
            )

        assert len(matrix) == 0
        assert "No features computed successfully" in caplog.text

    def test_matrix_sorted_by_index(self, registry):
        """Test that feature matrix is sorted by index."""
        # Create unsorted data
        dates = pd.DatetimeIndex(
            [
                "2024-01-05",
                "2024-01-01",
                "2024-01-03",
                "2024-01-02",
                "2024-01-04",
            ],
            tz="UTC",
        )
        unsorted_data = {"price": pd.DataFrame({"close": [105, 101, 103, 102, 104]}, index=dates)}

        registry.register(
            FeatureDefinition(
                name="close_price",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
            )
        )

        matrix = registry.build_feature_matrix(
            asset="btc",
            feature_names=["close_price"],
            data=unsorted_data,
        )

        # Verify sorted
        assert matrix.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
        assert matrix.index[1] == pd.Timestamp("2024-01-02", tz="UTC")
        assert matrix.index[2] == pd.Timestamp("2024-01-03", tz="UTC")
        assert matrix.index[3] == pd.Timestamp("2024-01-04", tz="UTC")
        assert matrix.index[4] == pd.Timestamp("2024-01-05", tz="UTC")


class TestFeatureCount:
    """Tests for feature_count property."""

    def test_feature_count_empty(self, registry):
        """Test feature_count returns 0 for empty registry."""
        assert registry.feature_count == 0

    def test_feature_count_after_registration(self, registry):
        """Test feature_count returns correct count after registration."""
        registry.register(
            FeatureDefinition(
                name="feature1",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
            )
        )
        assert registry.feature_count == 1

        registry.register(
            FeatureDefinition(
                name="feature2",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
            )
        )
        assert registry.feature_count == 2

    def test_feature_count_after_overwrite(self, registry):
        """Test feature_count doesn't increase when overwriting."""
        registry.register(
            FeatureDefinition(
                name="feature1",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
            )
        )
        assert registry.feature_count == 1

        # Overwrite
        registry.register(
            FeatureDefinition(
                name="feature1",
                category="market_context",
                compute_fn=lambda data: data["price"]["close"] * 2,
                input_columns=["close"],
            )
        )
        assert registry.feature_count == 1


class TestValidFromWithTimezones:
    """Tests for valid_from with different timezone scenarios."""

    def test_valid_from_with_naive_datetime_index(self, registry):
        """Test valid_from works with naive datetime index."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")  # No tz
        data = {"price": pd.DataFrame({"close": range(100, 110)}, index=dates)}

        registry.register(
            FeatureDefinition(
                name="restricted",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                valid_from="2024-01-05",
            )
        )

        matrix = registry.build_feature_matrix(
            asset="btc",
            feature_names=["restricted"],
            data=data,
            drop_na_rows=False,
        )

        # First 4 days should be NaN
        assert matrix["restricted"].iloc[:4].isna().all()
        # Day 5 onward should have values
        assert matrix["restricted"].iloc[4:].notna().all()

    def test_valid_from_with_utc_datetime_index(self, registry):
        """Test valid_from works with UTC datetime index."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        data = {"price": pd.DataFrame({"close": range(100, 110)}, index=dates)}

        registry.register(
            FeatureDefinition(
                name="restricted",
                category="technical",
                compute_fn=lambda data: data["price"]["close"],
                input_columns=["close"],
                valid_from="2024-01-05",
            )
        )

        matrix = registry.build_feature_matrix(
            asset="btc",
            feature_names=["restricted"],
            data=data,
            drop_na_rows=False,
        )

        # First 4 days should be NaN
        assert matrix["restricted"].iloc[:4].isna().all()
        # Day 5 onward should have values
        assert matrix["restricted"].iloc[4:].notna().all()
