import numpy as np
import pandas as pd
from sparky.analysis.edge_attribution import (
    regime_attribution,
    signal_contribution,
    temporal_stability,
)


def _make_index(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


class TestRegimeAttribution:
    def test_known_regime_split(self):
        np.random.seed(42)
        idx = _make_index(100)
        returns = pd.Series(np.random.randn(100) * 0.01, index=idx)
        positions = pd.Series(np.ones(100), index=idx)
        regimes = pd.Series(["bull"] * 50 + ["bear"] * 50, index=idx)

        result = regime_attribution(returns, positions, regimes)
        assert set(result["regime"]) == {"bear", "bull"}
        assert result["n_obs"].sum() == 100
        assert abs(result["frac_pnl"].sum() - 1.0) < 1e-6

    def test_single_regime(self):
        idx = _make_index(50)
        returns = pd.Series(0.01 * np.ones(50), index=idx)
        positions = pd.Series(np.ones(50), index=idx)
        regimes = pd.Series(["bull"] * 50, index=idx)

        result = regime_attribution(returns, positions, regimes)
        assert len(result) == 1
        assert result.iloc[0]["frac_time"] == 1.0


class TestSignalContribution:
    def test_orthogonal_signals(self):
        np.random.seed(42)
        idx = _make_index(500)
        sig1 = pd.Series(np.random.randn(500), index=idx)
        sig2 = pd.Series(np.random.randn(500), index=idx)
        returns = pd.Series(np.random.randn(500) * 0.01, index=idx)
        positions = pd.Series(np.sign(sig1.values), index=idx)

        result = signal_contribution(returns, {"s1": sig1, "s2": sig2}, positions)
        assert len(result) == 2
        assert all(
            col in result.columns for col in ["signal", "corr_with_returns", "corr_with_position", "marginal_ic"]
        )


class TestTemporalStability:
    def test_detects_structural_break(self):
        np.random.seed(123)
        idx = _make_index(600)
        # First 300: positive strategy returns; last 300: negative
        strat_ret = np.concatenate([np.abs(np.random.randn(300)) * 0.01, -np.abs(np.random.randn(300)) * 0.01])
        returns = pd.Series(strat_ret, index=idx)
        positions = pd.Series(np.ones(600), index=idx)

        result = temporal_stability(returns, positions, window=100)
        assert len(result) > 0
        # Rolling Sharpe should be positive early and negative late
        early = result[result["date"] < idx[350]]
        late = result[result["date"] > idx[400]]
        if len(early) > 0 and len(late) > 0:
            assert early["rolling_sharpe"].mean() > late["rolling_sharpe"].mean()
