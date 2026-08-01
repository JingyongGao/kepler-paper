# ICLR 2027 cross-scale robustness report

This is a post-confirmatory robustness extension. It is excluded from the original v3.6 p-value and is not a scaling-law fit. No combined confirmatory p-value was computed.

## Scale summaries

| Scale | R2 advances | Ties | R2 delays | Verified onset R0 | Verified onset R2 |
|---|---:|---:|---:|---:|---:|
| 1m | 4 | 0 | 0 | 0/4 | 4/4 |
| 5m | 4 | 0 | 0 | 3/4 | 4/4 |

## Every paired block

| Block | Scale | Rule | Seed | R0 onset | R2 onset | Ordinal delta | Direction | R0 R@5M | R2 R@5M |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| `scale_1m_rule_0_init_0` | 1m | rule_0 | 0 | None | 4000000 | -5 | advance | 0.790687 | 0.044795 |
| `scale_1m_rule_0_init_1` | 1m | rule_0 | 1 | None | 3000000 | -6 | advance | 0.943934 | 0.033878 |
| `scale_1m_rule_1_init_0` | 1m | rule_1 | 0 | None | 4000000 | -5 | advance | 0.765099 | 0.100649 |
| `scale_1m_rule_1_init_1` | 1m | rule_1 | 1 | None | 4000000 | -5 | advance | 0.910074 | 0.254405 |
| `scale_5m_rule_0_init_0` | 5m | rule_0 | 0 | None | 3000000 | -6 | advance | 0.998554 | 0.023875 |
| `scale_5m_rule_0_init_1` | 5m | rule_0 | 1 | 5000000 | 3000000 | -2 | advance | 0.128645 | 0.001191 |
| `scale_5m_rule_1_init_0` | 5m | rule_1 | 0 | 7500000 | 3000000 | -4 | advance | 1.000404 | 0.008529 |
| `scale_5m_rule_1_init_1` | 5m | rule_1 | 1 | 6000000 | 3000000 | -3 | advance | 0.768203 | 0.002444 |

## Frozen outcome details for every paired block

| Block | R0 est. | R2 est. | R0 R@10M | R2 R@10M | R0 center@10M | R2 center@10M | R0 probe@10M | R2 probe@10M | R2 retained@6M | R2 retained@10M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scale_1m_rule_0_init_0` | None | 5000000 | 0.598213 | 0.002717 | 0.049206 | 1.000000 | 0.133295 | 1.000000 | True | True |
| `scale_1m_rule_0_init_1` | None | 5000000 | 0.706267 | 0.003838 | 0.025723 | 0.999970 | 0.089494 | 0.999970 | True | True |
| `scale_1m_rule_1_init_0` | None | 6000000 | 0.578525 | 0.004474 | 0.076564 | 0.999985 | 0.134612 | 1.000000 | True | True |
| `scale_1m_rule_1_init_1` | None | 7500000 | 0.578065 | 0.022096 | 0.047525 | 0.983209 | 0.174855 | 0.988675 | True | True |
| `scale_5m_rule_0_init_0` | None | 4000000 | 0.499197 | 0.000664 | 0.115429 | 1.000000 | 0.171797 | 1.000000 | True | True |
| `scale_5m_rule_0_init_1` | 6000000 | 3000000 | 0.002783 | 0.000064 | 0.998698 | 1.000000 | 0.999531 | 1.000000 | True | True |
| `scale_5m_rule_1_init_0` | 10000000 | 4000000 | 0.030345 | 0.000189 | 0.979485 | 1.000000 | 0.996805 | 1.000000 | True | True |
| `scale_5m_rule_1_init_1` | 10000000 | 4000000 | 0.041254 | 0.000275 | 0.963209 | 1.000000 | 0.967585 | 1.000000 | True | True |

Frozen v3.6 onset, establishment, exact full-support R, center accuracy, probe, and post-removal persistence definitions were reused without change.
All 176 registered checkpoint files passed SHA-256 sidecar verification, and all 16 zero-step sampling-switch checks passed.

Machine-readable results: `artifacts/iclr2027_scale_robustness_results.json`
