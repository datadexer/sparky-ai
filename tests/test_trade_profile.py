import pandas as pd
from sparky.analysis.trade_profile import (
    extract_trades,
    trade_statistics,
    trade_clustering,
)


def _make_index(n, freq="D"):
    return pd.date_range("2020-01-01", periods=n, freq=freq)


class TestExtractTrades:
    def test_known_sequence(self):
        idx = _make_index(10)
        positions = pd.Series([0, 1, 1, 1, 0, -1, -1, 0, 1, 0], index=idx)
        prices = pd.Series([100, 100, 105, 110, 108, 108, 100, 95, 95, 100], index=idx)

        trades = extract_trades(positions, prices)
        assert len(trades) == 3
        # First trade: long from day 1 to day 4
        assert trades.iloc[0]["direction"] == 1
        assert trades.iloc[0]["entry_date"] == idx[1]
        assert trades.iloc[0]["exit_date"] == idx[4]
        # Second trade: short from day 5 to day 7
        assert trades.iloc[1]["direction"] == -1
        # Third trade: long from day 8 to day 9
        assert trades.iloc[2]["direction"] == 1

    def test_empty_positions(self):
        idx = _make_index(5)
        positions = pd.Series([0, 0, 0, 0, 0], index=idx)
        prices = pd.Series([100, 101, 102, 103, 104], index=idx)
        trades = extract_trades(positions, prices)
        assert len(trades) == 0

    def test_long_return_correct(self):
        idx = _make_index(4)
        positions = pd.Series([0, 1, 1, 0], index=idx)
        prices = pd.Series([100, 100, 110, 120], index=idx)
        trades = extract_trades(positions, prices)
        assert len(trades) == 1
        assert abs(trades.iloc[0]["return_pct"] - 0.20) < 1e-10


class TestTradeStatistics:
    def test_known_win_rate(self):
        trades = pd.DataFrame(
            {
                "entry_date": pd.date_range("2020-01-01", periods=4, freq="10D"),
                "exit_date": pd.date_range("2020-01-05", periods=4, freq="10D"),
                "direction": [1, 1, 1, 1],
                "duration_days": [5, 5, 5, 5],
                "entry_price": [100, 100, 100, 100],
                "exit_price": [110, 90, 120, 95],
                "return_pct": [0.10, -0.10, 0.20, -0.05],
            }
        )
        stats = trade_statistics(trades)
        assert stats["n_trades"] == 4
        assert stats["win_rate"] == 0.5
        assert stats["max_consec_wins"] == 1
        assert stats["max_consec_losses"] == 1

    def test_empty_trades(self):
        trades = pd.DataFrame(
            columns=["entry_date", "exit_date", "direction", "duration_days", "entry_price", "exit_price", "return_pct"]
        )
        stats = trade_statistics(trades)
        assert stats["n_trades"] == 0


class TestTradeClustering:
    def test_bunched_trades(self):
        trades = pd.DataFrame(
            {
                "entry_date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05", "2020-02-01"]),
                "exit_date": pd.to_datetime(["2020-01-02", "2020-01-04", "2020-01-06", "2020-02-05"]),
                "return_pct": [0.01, -0.01, 0.02, 0.01],
            }
        )
        result = trade_clustering(trades, threshold_days=5)
        assert result["n_clusters"] == 2
        assert result["avg_cluster_size"] == 2.0

    def test_single_trade(self):
        trades = pd.DataFrame(
            {
                "entry_date": pd.to_datetime(["2020-01-01"]),
                "exit_date": pd.to_datetime(["2020-01-05"]),
                "return_pct": [0.05],
            }
        )
        result = trade_clustering(trades)
        assert result["n_clusters"] == 1
