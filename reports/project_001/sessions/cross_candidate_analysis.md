# Cross-Candidate Correlation Analysis

**Common period:** 2017-01-01 to 2023-12-02 (2527 days)

## Per-Candidate Stats

| Candidate | Status | Sharpe | MaxDD | Ann. Return |
|-----------|--------|--------|-------|-------------|
| btc_don4h_160_25_iv | PASS | 1.494 | -12.14% | 16.45% |
| btc_don4h_60_20_iv | PASS | 1.392 | -14.65% | 16.09% |
| btc_don8h_82_20_iv | PASS | 1.596 | -19.32% | 26.82% |
| eth_don4h_138_47_flat | FAIL | 2.026 | -35.41% | 206.06% |
| eth_don4h_164_47_flat | FAIL | 2.030 | -30.14% | 200.71% |
| eth_don8h_83_33_iv | CONDITIONAL | 1.868 | -9.20% | 24.78% |

## Pairwise Correlation Matrix

```
                       btc_don4h_160_25_iv  btc_don4h_60_20_iv  btc_don8h_82_20_iv  eth_don4h_138_47_flat  eth_don4h_164_47_flat  eth_don8h_83_33_iv
btc_don4h_160_25_iv                  1.000               0.852               0.901                  0.318                  0.307               0.408
btc_don4h_60_20_iv                   0.852               1.000               0.809                  0.301                  0.283               0.389
btc_don8h_82_20_iv                   0.901               0.809               1.000                  0.352                  0.343               0.452
eth_don4h_138_47_flat                0.318               0.301               0.352                  1.000                  0.980               0.846
eth_don4h_164_47_flat                0.307               0.283               0.343                  0.980                  1.000               0.860
eth_don8h_83_33_iv                   0.408               0.389               0.452                  0.846                  0.860               1.000
```

## High-Correlation Pairs (threshold = 0.7)

- **btc_don4h_160_25_iv** <-> **btc_don4h_60_20_iv**: corr = 0.852
- **btc_don4h_160_25_iv** <-> **btc_don8h_82_20_iv**: corr = 0.901
- **btc_don4h_60_20_iv** <-> **btc_don8h_82_20_iv**: corr = 0.809
- **eth_don4h_138_47_flat** <-> **eth_don4h_164_47_flat**: corr = 0.980
- **eth_don4h_138_47_flat** <-> **eth_don8h_83_33_iv**: corr = 0.846
- **eth_don4h_164_47_flat** <-> **eth_don8h_83_33_iv**: corr = 0.860

## Cross-Asset (BTC <-> ETH) Correlations

- btc_don4h_160_25_iv <-> eth_don4h_138_47_flat: 0.318
- btc_don4h_160_25_iv <-> eth_don4h_164_47_flat: 0.307
- btc_don4h_160_25_iv <-> eth_don8h_83_33_iv: 0.408
- btc_don4h_60_20_iv <-> eth_don4h_138_47_flat: 0.301
- btc_don4h_60_20_iv <-> eth_don4h_164_47_flat: 0.283
- btc_don4h_60_20_iv <-> eth_don8h_83_33_iv: 0.389
- btc_don8h_82_20_iv <-> eth_don4h_138_47_flat: 0.352
- btc_don8h_82_20_iv <-> eth_don4h_164_47_flat: 0.343
- btc_don8h_82_20_iv <-> eth_don8h_83_33_iv: 0.452

## Cluster Representatives

- Cluster ['btc_don4h_160_25_iv', 'btc_don4h_60_20_iv', 'btc_don8h_82_20_iv']:
  - Representative: **btc_don8h_82_20_iv** (Sharpe=1.596)
  - Dropped: btc_don4h_160_25_iv (Sharpe=1.494)
  - Dropped: btc_don4h_60_20_iv (Sharpe=1.392)
- Cluster ['eth_don4h_138_47_flat', 'eth_don4h_164_47_flat', 'eth_don8h_83_33_iv']:
  - Representative: **eth_don8h_83_33_iv** (Sharpe=1.868)
  - Dropped: eth_don4h_138_47_flat (Sharpe=2.026)
  - Dropped: eth_don4h_164_47_flat (Sharpe=2.030)

> **Note on common-period selection:** Representative selected on 2017-2023 common period to ensure fair comparison across 4h/8h timeframes with different data start dates. Full-period Sharpe rankings differ slightly (btc160 leads at 2.319 vs btc82 at 2.220) but the common-period ranking is more comparable.

## Unique Survivors (Portfolio Construction)

- **btc_don8h_82_20_iv** [PASS] — Sharpe=1.596, MaxDD=-19.32%
- **eth_don8h_83_33_iv** [CONDITIONAL] — Sharpe=1.868, MaxDD=-9.20%

## Excluded (FAIL — correlation reference only)

- eth_don4h_138_47_flat — Sharpe=2.026, MaxDD=-35.41%
- eth_don4h_164_47_flat — Sharpe=2.030, MaxDD=-30.14%
