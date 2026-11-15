import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path

# Load probabilities
data = np.load("research_extension/phase9/phase9_all_test_probabilities.npz")
y = data["y_test"]
N = len(y)

comparisons = [
    ("Matched GAT (165f Full)", "Matched GAT (165f Full)", "ETGNN (165f Full)", "Temporal Ablation Control (Same Params)"),
    ("GAT Baseline (165f Full)", "GAT Baseline (165f Full)", "ETGNN (165f Full)", "Static GAT Baseline Control"),
    ("GCN (165f Full)", "GCN (165f Full)", "ETGNN (165f Full)", "Static GCN Baseline Control"),
    ("GraphSAGE (165f Full)", "GraphSAGE (165f Full)", "ETGNN (165f Full)", "Static GraphSAGE Baseline Control"),
    ("MLP (165f Full)", "MLP (165f Full)", "ETGNN (165f Full)", "Tabular Neural Baseline Control"),
    ("Logistic Regression (165f Full)", "Logistic Regression (165f Full)", "ETGNN (165f Full)", "Linear Classical Baseline Control"),
    ("Random Forest (165f Full)", "Random Forest (165f Full)", "ETGNN (165f Full)", "Tree Ensemble Tabular Control"),
    ("XGBoost (165f Full)", "XGBoost (165f Full)", "ETGNN (165f Full)", "Gradient Boosted Tabular Control"),
    ("MLP (93f Local)", "MLP (93f Local)", "ETGNN (93f Local)", "Local-Only Neural Tabular Control"),
    ("Random Forest (93f Local)", "Random Forest (93f Local)", "ETGNN (93f Local)", "Local-Only Tree Tabular Control"),
]

B = 1000
np.random.seed(42)
records = []

for comp_name, model_a_key, model_b_key, desc in comparisons:
    p_a = data[model_a_key]
    p_b = data[model_b_key]

    # Full sample metrics
    prec_a, rec_a, _ = precision_recall_curve(y, p_a)
    auprc_a = auc(rec_a, prec_a)

    prec_b, rec_b, _ = precision_recall_curve(y, p_b)
    auprc_b = auc(rec_b, prec_b)

    diff_point = auprc_a - auprc_b

    # Bootstrap
    diffs = []
    for _ in range(B):
        idx = np.random.choice(N, size=N, replace=True)
        y_b = y[idx]
        if (y_b == 1).sum() == 0 or (y_b == 0).sum() == 0:
            continue
        p_a_b = p_a[idx]
        p_b_b = p_b[idx]

        pa_b, ra_b, _ = precision_recall_curve(y_b, p_a_b)
        pb_b, rb_b, _ = precision_recall_curve(y_b, p_b_b)

        diffs.append(auc(ra_b, pa_b) - auc(rb_b, pb_b))

    diffs = np.array(diffs)
    se = np.std(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    # Two-sided empirical p-value
    if diff_point >= 0:
        p_val = 2 * np.mean(diffs <= 0)
    else:
        p_val = 2 * np.mean(diffs >= 0)
    p_val = min(max(p_val, 1.0 / B), 1.0)

    records.append({
        "Comparison": comp_name,
        "Model_A": model_a_key,
        "Model_B (Baseline)": model_b_key,
        "Description": desc,
        "AUPRC_Model_A": auprc_a,
        "AUPRC_Model_B": auprc_b,
        "Delta_AUPRC (A - B)": diff_point,
        "Bootstrap_SE": se,
        "CI_95_Low": ci_low,
        "CI_95_High": ci_high,
        "P_Value_Raw": p_val,
        "CI_Crosses_Zero": bool(ci_low <= 0 <= ci_high)
    })

df = pd.DataFrame(records)

# Holm-Bonferroni correction
df = df.sort_values(by="P_Value_Raw")
m_tests = len(df)
df["Holm_Rank"] = np.arange(1, m_tests + 1)
df["Holm_Threshold"] = 0.05 / (m_tests - df["Holm_Rank"] + 1)
df["Significant_After_Holm"] = df["P_Value_Raw"] < df["Holm_Threshold"]

out_csv = Path("research_extension/phase9/BASELINE_STATISTICAL_COMPARISONS.csv")
df.to_csv(out_csv, index=False)
print("Saved Statistical Comparisons to:", out_csv)
print(df[["Comparison", "Delta_AUPRC (A - B)", "CI_95_Low", "CI_95_High", "P_Value_Raw", "Significant_After_Holm"]].to_string(index=False))

