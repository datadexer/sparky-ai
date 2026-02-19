"""Log comprehensive final summary to wandb for Layer 4 Session 1."""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import wandb as wb

TAGS = ["layer4_sizing", "donchian", "20260218", "session_001"]
out_dir = Path("results/layer4_sizing_donchian_20260218")

wb.init(project="sparky-ai", entity="datadex_ai", name="layer4_sizing_s001_comprehensive_final", tags=TAGS, reinit=True)

wb.log(
    {
        # Primary metrics
        "sharpe": 1.6333,
        "dsr": 0.9924,
        "max_dd": -0.2478,
        "sharpe_50bps": 1.0919,
        "dsr_50bps": 0.9940,
        # Session stats
        "n_configs_tested": 267,
        "n_trials_cumulative": 627,
        "success_gate_met": 1,
        "n_successes": 4,
        # Best config details
        "best_family": "inverse_vol",
        "best_vol_window": 60,
        "best_target_vol": 0.15,
        "best_avg_position": 0.311,
        "best_n_trades": 156,
        # Sub-period performance
        "sharpe_2017plus": 1.160,
        "max_dd_2017plus": -0.221,
        "sharpe_2020plus": 1.031,
        "max_dd_2020plus": -0.221,
        # Baselines
        "baseline_meta_unsized_sharpe": 1.981,
        "baseline_meta_unsized_max_dd": -0.490,
        "baseline_raw_donchian_sharpe": 1.691,
        "baseline_bh_sharpe": 1.288,
        # Key negative findings
        "kelly_calibrated_best_sharpe": -2.51,
        "kelly_uncalibrated_best_sharpe": -0.10,
        "highconf_filter_best_sharpe": 1.550,
        "highconf_filter_max_dd": -0.490,
        # Close-but-no-cigar configs
        "vol_threshold_best_sharpe": 1.868,
        "vol_threshold_best_max_dd": -0.296,
        "discrete_invvol_best_sharpe": 1.925,
        "atr_based_best_sharpe": 2.388,
        # Cost robustness assessment
        "cost_robust_50bps": 0,  # Sharpe@50bps = 1.09, barely above 1.0
        "fifty_bps_sharpe_above_1": 1,  # technically passes but marginal
    }
)

wb.finish()
print("Final wandb log complete.")
print("\nSESSION 1 SUMMARY:")
print("  Success gate met: YES")
print("  Best config: inv-vol vw=60 tv=0.15")
print("  Sharpe@30bps: 1.633  MaxDD: -0.248  DSR: 0.992")
print("  Sharpe@50bps: 1.092 (marginal)")
print("  Sub-period 2020+: Sharpe=1.031, MaxDD=-0.221")
print("  n_trials cumulative: 627")
print("  n_configs tested this session: 267")
print("\nKey findings:")
print("  1. Kelly sizing FAILS (calibrated and uncalibrated)")
print("  2. inv-vol IS the mechanism — tv=0.15 is the MaxDD gate boundary")
print("  3. All other sizing families fail the MaxDD < -0.25 constraint")
print("  4. 50bps robustness is the outstanding concern")
