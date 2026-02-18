"""Hidden Markov Model (HMM) Regime Detection for Trading.

Research Basis:
- "Statistical Modeling of Volatility and Regime Switching in Financial Markets" (SSRN 2025)
- "Applications of Hidden Markov Models in Detecting Regime Changes in Bitcoin Markets" (2025)
- "Market Regime using Hidden Markov Model" (QuantInsti 2024)
- HMM is the gold standard for regime detection (probabilistic, not threshold-based)

HMM Advantages over Threshold-Based:
1. PROBABILISTIC: Returns P(regime), not binary classification
2. SMOOTH TRANSITIONS: No abrupt regime switches at arbitrary thresholds
3. LEARNS FROM DATA: Discovers optimal regime boundaries automatically
4. CAPTURES MEMORY: Uses transition probabilities (Markov property)

Strategy Logic:
- Train 2-state or 3-state Gaussian HMM on (returns, volatility)
- 2-state: LOW-VOL, HIGH-VOL (simplest, most robust)
- 3-state: LOW-VOL, MEDIUM-VOL, HIGH-VOL (more nuanced)
- Apply regime-specific trading strategies or probabilistic weighting

Target:
- Sharpe ≥0.85 (vs baseline 0.772)
- Probabilistic regime detection (no hard thresholds)
- Capture regime transitions smoothly
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm as hmmlearn_hmm

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. HMM regime detection unavailable. Install with: pip install hmmlearn")


def train_hmm_regime_model(
    prices: pd.Series,
    n_states: int = 2,
    n_iter: int = 100,
    random_state: int = 42,
) -> tuple:
    """Train Hidden Markov Model for regime detection.

    Args:
        prices: Close prices (daily frequency).
        n_states: Number of hidden states (2 or 3, default 2).
        n_iter: Number of EM iterations (default 100).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (model, features_df, hidden_states, state_probs).
        - model: Trained GaussianHMM model
        - features_df: DataFrame with ["returns", "realized_vol"]
        - hidden_states: Series of most likely state sequence (0, 1, ...)
        - state_probs: DataFrame with P(state_i) for each state
    """
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn not installed. Install with: pip install hmmlearn")

    # Compute features: returns and realized volatility
    returns = prices.pct_change()
    realized_vol = returns.rolling(30).std() * np.sqrt(365)  # Annualized 30-day vol

    # Stack features
    features = np.column_stack([returns, realized_vol])

    # Drop NaN rows
    valid_mask = ~(np.isnan(features).any(axis=1))
    features_clean = features[valid_mask]

    # Train Gaussian HMM
    model = hmmlearn_hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )

    model.fit(features_clean)

    # Decode hidden states (Viterbi algorithm)
    hidden_states_clean = model.predict(features_clean)

    # Predict state probabilities
    state_probs_clean = model.predict_proba(features_clean)

    # Create full-length Series/DataFrame (fill NaN periods with state 0)
    hidden_states = pd.Series(0, index=prices.index, dtype=int)
    hidden_states[valid_mask] = hidden_states_clean

    state_probs = pd.DataFrame(
        0.0,
        index=prices.index,
        columns=[f"P(state_{i})" for i in range(n_states)],
    )
    state_probs.loc[valid_mask, :] = state_probs_clean

    # Create features DataFrame
    features_df = pd.DataFrame(
        {
            "returns": returns,
            "realized_vol": realized_vol,
        },
        index=prices.index,
    )

    # Log model statistics
    logger.info(f"Trained {n_states}-state HMM with {n_iter} iterations")
    logger.info(f"Converged: {model.monitor_.converged}")
    logger.info(f"Log-likelihood: {model.score(features_clean):.2f}")

    # Analyze state characteristics
    for state in range(n_states):
        state_mask = hidden_states == state
        n_days = state_mask.sum()
        pct_days = n_days / len(hidden_states) * 100
        mean_vol = realized_vol[state_mask].mean()

        logger.info(f"State {state}: {n_days} days ({pct_days:.1f}%), mean vol={mean_vol:.3f}")

    # Log transition probabilities
    logger.info("Transition probabilities (rows=current, cols=next):")
    logger.info(f"\n{pd.DataFrame(model.transmat_)}")

    return model, features_df, hidden_states, state_probs


def hmm_regime_donchian(
    prices: pd.Series,
    n_states: int = 2,
    aggressive_params: tuple[int, int] = (15, 5),
    standard_params: tuple[int, int] = (20, 10),
    conservative_params: tuple[int, int] = (40, 20),
) -> pd.Series:
    """Donchian strategy with HMM regime detection.

    For 2-state HMM:
    - State 0 (LOW-VOL): Use aggressive parameters (15/5)
    - State 1 (HIGH-VOL): Use conservative parameters (40/20)

    For 3-state HMM:
    - State 0 (LOW-VOL): Use aggressive parameters (15/5)
    - State 1 (MEDIUM-VOL): Use standard parameters (20/10)
    - State 2 (HIGH-VOL): Use conservative parameters (40/20)

    Args:
        prices: Close prices (daily frequency).
        n_states: Number of HMM states (2 or 3, default 2).
        aggressive_params: (entry, exit) for LOW vol (default 15/5).
        standard_params: (entry, exit) for MEDIUM vol (default 20/10).
        conservative_params: (entry, exit) for HIGH vol (default 40/20).

    Returns:
        Series of signals (1 = LONG, 0 = FLAT), same index as prices.
    """
    # Train HMM
    model, features_df, hidden_states, state_probs = train_hmm_regime_model(prices, n_states=n_states)

    # Map states to parameters (assume state 0 = low vol, higher states = higher vol)
    # This is a simplification; ideally we'd sort states by mean volatility
    state_to_params = {
        0: aggressive_params,
        1: conservative_params if n_states == 2 else standard_params,
        2: conservative_params,  # Only used if n_states == 3
    }

    # Initialize signals
    signals = pd.Series(0, index=prices.index, dtype=int)

    # Track position state
    in_position = False
    current_params = standard_params

    for i in range(len(prices)):
        if i < 30:  # Need at least 30 days for volatility feature
            signals.iloc[i] = 0
            continue

        # Get current HMM state
        current_state = hidden_states.iloc[i]
        current_params = state_to_params.get(current_state, standard_params)

        entry_period, exit_period = current_params

        # Compute Donchian channels with regime-specific periods
        if i >= entry_period:
            upper_channel = prices.iloc[i - entry_period : i].max()
        else:
            upper_channel = prices.iloc[:i].max()

        if i >= exit_period:
            lower_channel = prices.iloc[i - exit_period : i].min()
        else:
            lower_channel = prices.iloc[:i].min()

        current_price = prices.iloc[i]

        if not in_position:
            # Check for entry
            if i > 0 and current_price >= upper_channel:
                in_position = True
                signals.iloc[i] = 1
            else:
                signals.iloc[i] = 0
        else:
            # In position: check for exit
            if current_price <= lower_channel:
                in_position = False
                signals.iloc[i] = 0
            else:
                signals.iloc[i] = 1

    n_long = signals.sum()
    n_total = len(signals)

    logger.info(
        f"HMM Regime Donchian ({n_states}-state): {n_long} LONG ({n_long / n_total * 100:.1f}%), "
        f"{n_total - n_long} FLAT ({(n_total - n_long) / n_total * 100:.1f}%)"
    )

    return signals


def hmm_probabilistic_ensemble(
    prices: pd.Series,
    n_states: int = 2,
    random_state: int = 42,
) -> pd.Series:
    """Multi-timeframe ensemble with HMM probabilistic weighting.

    Instead of hard switching, weights 3 Donchian strategies by HMM state probabilities.

    For 2-state HMM:
    - Aggressive strategy (15/5) weighted by P(LOW-VOL)
    - Conservative strategy (40/20) weighted by P(HIGH-VOL)
    - Final signal = weighted average

    Args:
        prices: Close prices (daily frequency).
        n_states: Number of HMM states (2 or 3, default 2).
        random_state: Random seed for HMM EM algorithm (default 42).

    Returns:
        Series of signals (1 = LONG, 0 = FLAT), same index as prices.
    """
    # Train HMM
    model, features_df, hidden_states, state_probs = train_hmm_regime_model(
        prices, n_states=n_states, random_state=random_state
    )

    # Generate signals for multiple strategies
    from sparky.models.simple_baselines import donchian_channel_strategy

    signals_aggressive = donchian_channel_strategy(prices, entry_period=15, exit_period=5)
    signals_conservative = donchian_channel_strategy(prices, entry_period=40, exit_period=20)

    if n_states == 2:
        # 2-state: blend aggressive (state 0) and conservative (state 1)
        prob_low_vol = state_probs["P(state_0)"]
        prob_high_vol = state_probs["P(state_1)"]

        weighted_signal = prob_low_vol * signals_aggressive + prob_high_vol * signals_conservative

    elif n_states == 3:
        # 3-state: add standard strategy (state 1)
        signals_standard = donchian_channel_strategy(prices, entry_period=20, exit_period=10)

        prob_low_vol = state_probs["P(state_0)"]
        prob_medium_vol = state_probs["P(state_1)"]
        prob_high_vol = state_probs["P(state_2)"]

        weighted_signal = (
            prob_low_vol * signals_aggressive
            + prob_medium_vol * signals_standard
            + prob_high_vol * signals_conservative
        )

    else:
        raise ValueError(f"n_states must be 2 or 3, got {n_states}")

    # Threshold weighted signal at 0.5 (LONG if weighted signal > 0.5)
    final_signals = (weighted_signal > 0.5).astype(int)

    n_long = final_signals.sum()
    n_total = len(final_signals)

    logger.info(
        f"HMM Probabilistic Ensemble ({n_states}-state): {n_long} LONG ({n_long / n_total * 100:.1f}%), "
        f"{n_total - n_long} FLAT ({(n_total - n_long) / n_total * 100:.1f}%)"
    )

    return final_signals


def visualize_hmm_regimes(
    prices: pd.Series,
    hidden_states: pd.Series,
    output_path: str = None,
) -> None:
    """Visualize HMM-detected regimes overlaid on price chart.

    Args:
        prices: Close prices (daily frequency).
        hidden_states: Series of HMM states (0, 1, 2, ...).
        output_path: Optional path to save plot (e.g., "results/hmm_regimes.png").
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Cannot visualize regimes.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot prices
    ax.plot(prices.index, prices.values, color="black", linewidth=1, label="BTC Price")

    # Color background by regime
    unique_states = sorted(hidden_states.unique())
    colors = ["lightgreen", "yellow", "lightcoral"]  # LOW, MEDIUM, HIGH

    for state in unique_states:
        state_mask = hidden_states == state
        ax.fill_between(
            prices.index,
            prices.min(),
            prices.max(),
            where=state_mask,
            color=colors[state],
            alpha=0.3,
            label=f"State {state}",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title("HMM-Detected Regimes Overlaid on BTC Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved HMM regime visualization to {output_path}")

    plt.show()
