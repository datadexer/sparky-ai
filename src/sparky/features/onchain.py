"""Derived features from on-chain data.

BTC features from BGeometrics computed indicators + CoinMetrics raw metrics.
ETH features from CoinMetrics raw metrics + ETH-specific protocol dynamics.

Each function takes a Series or DataFrame and returns a Series.
All features are computed with explicit lookback windows to avoid leakage.
"""

import numpy as np
import pandas as pd

# =============================================================================
# BTC On-Chain Features
# =============================================================================


def hash_ribbon(hash_rate: pd.Series, short: int = 30, long: int = 60) -> pd.Series:
    """Hash ribbon signal: short MA vs long MA of hash rate.

    Bullish when short MA crosses above long MA (miners recovering).
    Returns: 1 (bullish) or 0 (bearish/miner capitulation).
    """
    sma_short = hash_rate.rolling(window=short, min_periods=short).mean()
    sma_long = hash_rate.rolling(window=long, min_periods=long).mean()
    return (sma_short > sma_long).astype(float)


def nvt_zscore(nvt: pd.Series, window: int = 90) -> pd.Series:
    """NVT z-score: how far current NVT is from its rolling mean.

    High z-score (>2) = network overvalued relative to transaction volume.
    Low z-score (<-2) = network undervalued.
    """
    rolling_mean = nvt.rolling(window=window, min_periods=window).mean()
    rolling_std = nvt.rolling(window=window, min_periods=window).std()
    return (nvt - rolling_mean) / rolling_std.replace(0, np.nan)


def mvrv_signal(mvrv_zscore: pd.Series) -> pd.Series:
    """MVRV regime indicator.

    >7 = overheated (sell zone), <0 = undervalued (buy zone), else neutral.
    Returns: -1 (overheated), 1 (undervalued), 0 (neutral).
    """
    result = pd.Series(0.0, index=mvrv_zscore.index)
    result[mvrv_zscore > 7] = -1.0
    result[mvrv_zscore < 0] = 1.0
    return result


def sopr_signal(sopr: pd.Series) -> pd.Series:
    """SOPR capitulation signal.

    SOPR < 1 means coins are being sold at a loss (capitulation = buy signal).
    Returns: 1 (capitulation/buy), 0 (normal).
    """
    return (sopr < 1.0).astype(float)


def address_momentum(active_addresses: pd.Series, period: int = 30) -> pd.Series:
    """Percent change in active addresses over period.

    Rising addresses = growing network usage = bullish.
    """
    return active_addresses.pct_change(periods=period, fill_method=None)


def volume_momentum(tx_volume: pd.Series, period: int = 30) -> pd.Series:
    """Percent change in transaction volume over period."""
    return tx_volume.pct_change(periods=period, fill_method=None)


def nupl_regime(nupl: pd.Series) -> pd.Series:
    """NUPL regime classification.

    Returns categorical float:
    -2 = capitulation (NUPL < 0)
    -1 = hope/fear (0 <= NUPL < 0.25)
     0 = optimism (0.25 <= NUPL < 0.5)
     1 = belief (0.5 <= NUPL < 0.75)
     2 = euphoria (NUPL >= 0.75)
    """
    result = pd.Series(np.nan, index=nupl.index)
    result[nupl < 0] = -2.0
    result[(nupl >= 0) & (nupl < 0.25)] = -1.0
    result[(nupl >= 0.25) & (nupl < 0.5)] = 0.0
    result[(nupl >= 0.5) & (nupl < 0.75)] = 1.0
    result[nupl >= 0.75] = 2.0
    return result


def puell_signal(puell_multiple: pd.Series) -> pd.Series:
    """Puell Multiple buy signal.

    < 0.5 = miner stress = historically strong buy signal.
    Returns: 1 (buy signal), 0 (normal).
    """
    return (puell_multiple < 0.5).astype(float)


def supply_in_profit_extreme(sip: pd.Series) -> pd.Series:
    """Supply in profit cycle extremes.

    >95% = euphoria (cycle top risk), <50% = deep bear (cycle bottom signal).
    Returns: -1 (euphoria/top), 1 (bear/bottom), 0 (normal).
    """
    result = pd.Series(0.0, index=sip.index)
    result[sip > 95] = -1.0
    result[sip < 50] = 1.0
    return result


# =============================================================================
# BTC Adaptive Threshold + STH Features
# =============================================================================


def mvrv_signal_adaptive(mvrv: pd.Series, window: int = 730, upper_pct: int = 90, lower_pct: int = 10) -> pd.Series:
    """Adaptive MVRV signal using rolling percentile thresholds.

    +1 below lower percentile (undervalued), -1 above upper (overvalued), 0 else.
    """
    upper = mvrv.rolling(window=window, min_periods=window).quantile(upper_pct / 100)
    lower = mvrv.rolling(window=window, min_periods=window).quantile(lower_pct / 100)
    result = pd.Series(0.0, index=mvrv.index)
    result[mvrv < lower] = 1.0
    result[mvrv > upper] = -1.0
    return result


def sopr_signal_adaptive(sopr: pd.Series, window: int = 730, upper_pct: int = 95, lower_pct: int = 5) -> pd.Series:
    """Adaptive SOPR signal respecting SOPR=1 as regime boundary.

    +1 below lower percentile (capitulation), -1 above upper (profit taking), 0 else.
    Additionally: SOPR < 1 always gets +1 (capitulation regardless of percentile).
    """
    upper = sopr.rolling(window=window, min_periods=window).quantile(upper_pct / 100)
    lower = sopr.rolling(window=window, min_periods=window).quantile(lower_pct / 100)
    result = pd.Series(0.0, index=sopr.index)
    result[sopr < lower] = 1.0
    result[sopr > upper] = -1.0
    result[(sopr < 1.0) & (result == 0.0)] = 1.0  # SOPR < 1 elevates neutral, not bearish
    return result


def netflow_signal(flow_in: pd.Series, flow_out: pd.Series, window: int = 30) -> pd.Series:
    """Net exchange flow z-score signal.

    Positive z-score (net inflow) = bearish (coins moving to exchanges for selling).
    Negative z-score (net outflow) = bullish (coins leaving exchanges).
    Returns z-score (not clipped to [-1,1]).
    """
    net = flow_in - flow_out
    rolling_mean = net.rolling(window=window, min_periods=window).mean()
    rolling_std = net.rolling(window=window, min_periods=window).std()
    zscore = (net - rolling_mean) / rolling_std.replace(0, np.nan)
    return -zscore.fillna(0)


def sth_mvrv_signal(sth_mvrv: pd.Series, window: int = 365, upper_pct: int = 90, lower_pct: int = 10) -> pd.Series:
    """Short-term holder MVRV signal with adaptive thresholds.

    Same pattern as mvrv_signal_adaptive but shorter default window
    since STH cycles are faster.
    """
    upper = sth_mvrv.rolling(window=window, min_periods=window).quantile(upper_pct / 100)
    lower = sth_mvrv.rolling(window=window, min_periods=window).quantile(lower_pct / 100)
    result = pd.Series(0.0, index=sth_mvrv.index)
    result[sth_mvrv < lower] = 1.0
    result[sth_mvrv > upper] = -1.0
    return result


def sth_sopr_signal(sth_sopr: pd.Series, window: int = 365, upper_pct: int = 95, lower_pct: int = 5) -> pd.Series:
    """Short-term holder SOPR signal for capitulation detection.

    STH capitulation is cleaner signal than aggregate SOPR.
    +1 below lower percentile, -1 above upper, 0 else.
    STH SOPR < 1 always gets +1 (capitulation).
    """
    upper = sth_sopr.rolling(window=window, min_periods=window).quantile(upper_pct / 100)
    lower = sth_sopr.rolling(window=window, min_periods=window).quantile(lower_pct / 100)
    result = pd.Series(0.0, index=sth_sopr.index)
    result[sth_sopr < lower] = 1.0
    result[sth_sopr > upper] = -1.0
    result[(sth_sopr < 1.0) & (result == 0.0)] = 1.0
    return result


# =============================================================================
# ETH-Specific On-Chain Features
# =============================================================================


def gas_fee_zscore(fee_total_usd: pd.Series, window: int = 90) -> pd.Series:
    """Z-score of ETH gas fees.

    High gas = high network demand = bullish activity signal.
    Valid from: Aug 2021 (EIP-1559), but FeeTotUSD exists earlier.
    """
    rolling_mean = fee_total_usd.rolling(window=window, min_periods=window).mean()
    rolling_std = fee_total_usd.rolling(window=window, min_periods=window).std()
    return (fee_total_usd - rolling_mean) / rolling_std.replace(0, np.nan)


def eth_address_momentum(active_addresses: pd.Series, period: int = 30) -> pd.Series:
    """ETH active address momentum (pct change).

    Same calculation as BTC but ETH addresses are heavily DeFi/contract-driven.
    """
    return active_addresses.pct_change(periods=period, fill_method=None)


def eth_transfer_value_zscore(transfer_value_adj: pd.Series, window: int = 60) -> pd.Series:
    """Z-score of ETH adjusted transfer value.

    Large value transfers often precede major DeFi events or whale positioning.
    """
    rolling_mean = transfer_value_adj.rolling(window=window, min_periods=window).mean()
    rolling_std = transfer_value_adj.rolling(window=window, min_periods=window).std()
    return (transfer_value_adj - rolling_mean) / rolling_std.replace(0, np.nan)


def eth_nvt_signal(market_cap: pd.Series, transfer_value_adj: pd.Series, window: int = 90) -> pd.Series:
    """ETH NVT signal (z-score of market cap / transfer value).

    NVT works for ETH but uses adjusted transfer value (excludes contract churn).
    """
    nvt = market_cap / transfer_value_adj.replace(0, np.nan)
    rolling_mean = nvt.rolling(window=window, min_periods=window).mean()
    rolling_std = nvt.rolling(window=window, min_periods=window).std()
    return (nvt - rolling_mean) / rolling_std.replace(0, np.nan)


def eth_btc_ratio_momentum(eth_price: pd.Series, btc_price: pd.Series, period: int = 30) -> pd.Series:
    """ETH/BTC ratio momentum.

    ETH outperformance signals risk-on altcoin rotation.
    """
    ratio = eth_price / btc_price.replace(0, np.nan)
    return ratio.pct_change(periods=period, fill_method=None)


def eth_btc_correlation_regime(eth_returns: pd.Series, btc_returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling correlation between ETH and BTC returns.

    Low correlation = ETH trading on own fundamentals (more alpha opportunity).
    High correlation = macro-driven, both moving together.
    """
    return eth_returns.rolling(window=window, min_periods=window).corr(btc_returns)
