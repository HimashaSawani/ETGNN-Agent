# ETGNN Google Colab Benchmark Verification Report
Run ID: `20261103_115518`
Execution Mode: `REAL_DATASET`
Dataset Type: `REAL_ELLIPTIC_BITCOIN`
Hardware Device: `cuda`

## 1. Summary Results Table (Steps 41-49):
| Model         |   precision |   recall |     f1 |   roc_auc |   auprc |
|:--------------|------------:|---------:|-------:|----------:|--------:|
| Random Forest |      0.742  |   0.5763 | 0.6488 |    0.8754 |  0.638  |
| XGBoost       |      0.6761 |   0.5897 | 0.63   |    0.8707 |  0.6445 |
| GCN           |      0.1418 |   0.5496 | 0.2254 |    0.7602 |  0.1961 |
| GraphSAGE     |      0.0978 |   0.7977 | 0.1743 |    0.8004 |  0.3033 |
| GAT           |      0.1023 |   0.8569 | 0.1827 |    0.8102 |  0.2417 |
| Matched GAT   |      0.118  |   0.7634 | 0.2044 |    0.7943 |  0.167  |
| ETGNN         |      0.1408 |   0.6756 | 0.233  |    0.8025 |  0.2153 |

## 2. Historical Comparison Table:
| Model         |   Current AUPRC |   Historical AUPRC |   Difference |
|:--------------|----------------:|-------------------:|-------------:|
| Random Forest |          0.638  |             0.6323 |       0.0057 |
| XGBoost       |          0.6445 |             0.5472 |       0.0973 |
| GCN           |          0.1961 |             0.1306 |       0.0655 |
| GraphSAGE     |          0.3033 |             0.2427 |       0.0606 |
| GAT           |          0.2417 |             0.1155 |       0.1262 |
| Matched GAT   |          0.167  |             0.2099 |      -0.0429 |
| ETGNN         |          0.2153 |             0.225  |      -0.0097 |

## 3. Scientific Discussion & Academic Findings:
1. **Tabular Random Forest Dominance**: Random Forest achieved Test AUPRC of 0.6380, outperforming all graph neural network architectures.
2. **Proposed ETGNN Performance**: Achieved Test AUPRC of 0.2153.
3. **Parameter-Matched Control Contrast**: Matched Zero-Time GAT achieved 0.1670 vs ETGNN 0.2153 (both at exactly 11,910 parameters).