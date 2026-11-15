import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

# Paths
LOCAL_NPZ = Path("data/processed/final_test_probabilities.npz")
COLAB_NPZ = Path("results/colab_extracted_20261103/tables/colab_test_probabilities.npz")

local_data = np.load(LOCAL_NPZ)
colab_data = np.load(COLAB_NPZ)

y_local = local_data["y_test"]
y_colab = colab_data["y_test"]

assert np.array_equal(y_local, y_colab), "Ground truth labels mismatch between Local and Colab!"
N = len(y_local)
pos_count = (y_local == 1).sum()
neg_count = (y_local == 0).sum()
print(f"Verified Ground Truth: N={N}, Illicit (class 1)={pos_count}, Licit (class 0)={neg_count}")

# Mapping of models
model_mapping = [
    ("Random Forest", "model_0", "model_Random_Forest"),
    ("XGBoost", "model_3", "model_XGBoost"),
    ("GCN", "model_4", "model_GCN"),
    ("GraphSAGE", "model_5", "model_GraphSAGE"),
    ("GAT", "model_6", "model_GAT"),
    ("Matched GAT", "model_7", "model_Matched_GAT"),
    ("ETGNN", "model_8", "model_ETGNN"),
]

def compute_metrics(y_true, probs):
    p, r, _ = precision_recall_curve(y_true, probs)
    auprc = auc(r, p)
    roc = roc_auc_score(y_true, probs)
    preds = (probs >= 0.5).astype(int)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    return {
        "auprc": auprc,
        "roc_auc": roc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    }

records = []
bootstrap_results = []
np.random.seed(42)
B = 1000

for name, loc_key, col_key in model_mapping:
    p_loc = local_data[loc_key]
    p_col = colab_data[col_key]

    m_loc = compute_metrics(y_local, p_loc)
    m_col = compute_metrics(y_local, p_col)

    # Prob correlations
    r_pearson, _ = pearsonr(p_loc, p_col)
    r_spearman, _ = spearmanr(p_loc, p_col)
    max_abs_diff = np.max(np.abs(p_loc - p_col))
    mean_abs_diff = np.mean(np.abs(p_loc - p_col))

    # Bootstrap paired difference in AUPRC (Colab - Local)
    boot_diffs_auprc = []
    boot_diffs_f1 = []
    indices = np.arange(N)

    for _ in range(B):
        idx = np.random.choice(indices, size=N, replace=True)
        y_b = y_local[idx]
        if (y_b == 1).sum() == 0 or (y_b == 0).sum() == 0:
            continue
        p_l_b = p_loc[idx]
        p_c_b = p_col[idx]

        # Local AUPRC
        prec_l, rec_l, _ = precision_recall_curve(y_b, p_l_b)
        auprc_l = auc(rec_l, prec_l)

        # Colab AUPRC
        prec_c, rec_c, _ = precision_recall_curve(y_b, p_c_b)
        auprc_c = auc(rec_c, prec_c)

        # F1
        f1_l = f1_score(y_b, (p_l_b >= 0.5).astype(int), zero_division=0)
        f1_c = f1_score(y_b, (p_c_b >= 0.5).astype(int), zero_division=0)

        boot_diffs_auprc.append(auprc_c - auprc_l)
        boot_diffs_f1.append(f1_c - f1_l)

    boot_diffs_auprc = np.array(boot_diffs_auprc)
    boot_diffs_f1 = np.array(boot_diffs_f1)

    ci_auprc_low, ci_auprc_high = np.percentile(boot_diffs_auprc, [2.5, 97.5])
    ci_f1_low, ci_f1_high = np.percentile(boot_diffs_f1, [2.5, 97.5])
    p_val_auprc = np.mean(boot_diffs_auprc <= 0) if np.mean(boot_diffs_auprc) > 0 else np.mean(boot_diffs_auprc >= 0)
    p_val_auprc = 2 * min(p_val_auprc, 1 - p_val_auprc)

    records.append({
        "Model": name,
        "Local_AUPRC": m_loc["auprc"],
        "Colab_AUPRC": m_col["auprc"],
        "Delta_AUPRC": m_col["auprc"] - m_loc["auprc"],
        "AUPRC_95_CI_Low": ci_auprc_low,
        "AUPRC_95_CI_High": ci_auprc_high,
        "AUPRC_Diff_Significant": "No" if (ci_auprc_low <= 0 <= ci_auprc_high) else "Yes",
        "Local_F1": m_loc["f1"],
        "Colab_F1": m_col["f1"],
        "Delta_F1": m_col["f1"] - m_loc["f1"],
        "Local_Precision": m_loc["precision"],
        "Colab_Precision": m_col["precision"],
        "Local_Recall": m_loc["recall"],
        "Colab_Recall": m_col["recall"],
        "Local_ROCAUC": m_loc["roc_auc"],
        "Colab_ROCAUC": m_col["roc_auc"],
        "Pearson_R": r_pearson,
        "Spearman_R": r_spearman,
        "Mean_Abs_Prob_Diff": mean_abs_diff,
        "Max_Abs_Prob_Diff": max_abs_diff,
    })

df = pd.DataFrame(records)
out_csv = Path("research_extension/phase8/LOCAL_VS_COLAB_STATISTICAL_RECONCILIATION.csv")
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)
print(df[["Model", "Local_AUPRC", "Colab_AUPRC", "Delta_AUPRC", "AUPRC_95_CI_Low", "AUPRC_95_CI_High", "AUPRC_Diff_Significant", "Pearson_R"]].to_string(index=False))

