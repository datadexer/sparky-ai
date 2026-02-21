import numpy as np
import pandas as pd


def extract_trades(
    positions: pd.Series,
    prices: pd.Series,
) -> pd.DataFrame:
    """Extract individual trades from a position series."""
    idx = positions.index.intersection(prices.index)
    pos, px = positions.loc[idx], prices.loc[idx]

    trades = []
    in_trade = False
    entry_date = entry_price = direction = None

    for i in range(len(pos)):
        p = pos.iloc[i]
        if not in_trade and p != 0:
            in_trade = True
            entry_date = idx[i]
            entry_price = px.iloc[min(i + 1, len(px) - 1)]
            direction = int(np.sign(p))
        elif in_trade and (p == 0 or (p != 0 and int(np.sign(p)) != direction)):
            exit_date = idx[i]
            exit_price = px.iloc[i]
            ret = direction * (exit_price / entry_price - 1)
            duration = (exit_date - entry_date).days
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "direction": direction,
                    "duration_days": duration,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "return_pct": float(ret),
                }
            )
            # If we flipped direction, start new trade
            if p != 0 and int(np.sign(p)) != direction:
                entry_date = idx[i]
                entry_price = px.iloc[min(i + 1, len(px) - 1)]
                direction = int(np.sign(p))
            else:
                in_trade = False

    # Close any open trade at the end
    if in_trade:
        exit_date = idx[-1]
        exit_price = px.iloc[-1]
        ret = direction * (exit_price / entry_price - 1)
        duration = (exit_date - entry_date).days
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "direction": direction,
                "duration_days": duration,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "return_pct": float(ret),
            }
        )

    return pd.DataFrame(trades)


def trade_statistics(trades: pd.DataFrame) -> dict:
    """Compute trade-level summary statistics."""
    if len(trades) == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_duration": 0.0,
            "max_consec_wins": 0,
            "max_consec_losses": 0,
            "expectancy": 0.0,
        }

    wins = trades[trades["return_pct"] > 0]
    losses = trades[trades["return_pct"] <= 0]

    win_rate = len(wins) / len(trades)
    avg_win = float(wins["return_pct"].mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses["return_pct"].mean()) if len(losses) > 0 else 0.0

    total_wins = wins["return_pct"].sum() if len(wins) > 0 else 0.0
    total_losses = abs(losses["return_pct"].sum()) if len(losses) > 0 else 0.0
    profit_factor = float(total_wins / total_losses) if total_losses > 0 else float("inf")

    # Consecutive wins/losses
    is_win = (trades["return_pct"] > 0).values
    max_cw = max_cl = cw = cl = 0
    for w in is_win:
        if w:
            cw += 1
            cl = 0
        else:
            cl += 1
            cw = 0
        max_cw = max(max_cw, cw)
        max_cl = max(max_cl, cl)

    return {
        "n_trades": len(trades),
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_duration": float(trades["duration_days"].mean()),
        "max_consec_wins": max_cw,
        "max_consec_losses": max_cl,
        "expectancy": float(trades["return_pct"].mean()),
    }


def trade_clustering(
    trades: pd.DataFrame,
    threshold_days: int = 5,
) -> dict:
    """Analyze temporal clustering of trades."""
    if len(trades) <= 1:
        return {"n_clusters": len(trades), "avg_cluster_size": float(len(trades)), "max_gap_days": 0}

    gaps = []
    for i in range(1, len(trades)):
        gap = (trades.iloc[i]["entry_date"] - trades.iloc[i - 1]["exit_date"]).days
        gaps.append(gap)

    # Cluster: consecutive trades with gaps <= threshold
    clusters = 1
    cluster_sizes = [1]
    for g in gaps:
        if g <= threshold_days:
            cluster_sizes[-1] += 1
        else:
            clusters += 1
            cluster_sizes.append(1)

    return {
        "n_clusters": clusters,
        "avg_cluster_size": float(np.mean(cluster_sizes)),
        "max_gap_days": int(max(gaps)) if gaps else 0,
    }
