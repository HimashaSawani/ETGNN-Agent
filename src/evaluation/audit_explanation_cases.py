import json
from pathlib import Path
import pandas as pd

# Load mappings
mapping_df = pd.read_csv("data/processed/node_mapping.csv").set_index("node_index")

cases_data = [
    ("TP", "connected_tp.json", 1),
    ("FP", "connected_fp.json", 0),
    ("FN", "connected_fn.json", 1),
    ("TN", "connected_tn.json", 0),
]

rows = []
for case_name, filename, true_label in cases_data:
    path = Path("results/explanations") / filename
    with open(path, "r") as f:
        d = json.load(f)
    
    node_idx = d["node_index"]
    tx_id = mapping_df.loc[node_idx, "txId"] if node_idx in mapping_df.index else "N/A"
    time_step = mapping_df.loc[node_idx, "time_step"] if node_idx in mapping_df.index else "N/A"
    
    pred_class = d["predicted_class"] # 1 for TP/FP, 0 for FN/TN
    target_class = pred_class
    
    # Target class probabilities
    p_target_before = d["original_predicted_class_probability"]
    p_target_after = d["probability_after_removing_top_edges"]
    
    # Convert to explicit illicit probabilities P(Y=1)
    if pred_class == 1:
        p_illicit_before = p_target_before
        p_illicit_after = p_target_after
    else:
        p_illicit_before = 1.0 - p_target_before
        p_illicit_after = 1.0 - p_target_after
        
    rows.append({
        "case": case_name,
        "node_index": node_idx,
        "transaction_id": tx_id,
        "time_step": time_step,
        "true_label": true_label,
        "original_predicted_class": pred_class,
        "decision_threshold": 0.50,
        "illicit_probability_before": p_illicit_before,
        "illicit_probability_after": p_illicit_after,
        "target_class": target_class,
        "target_probability_before": p_target_before,
        "target_probability_after": p_target_after,
        "target_prob_drop_pp": (p_target_before - p_target_after) * 100.0,
        "class_flipped": (p_illicit_before >= 0.50 and p_illicit_after < 0.50) or (p_illicit_before < 0.50 and p_illicit_after >= 0.50)
    })

out_df = pd.DataFrame(rows)
out_csv = Path("results/tables/explanation_cases_detailed_audit.csv")
out_df.to_csv(out_csv, index=False)
print("Saved explanation detailed audit to:", out_csv)
print(out_df.to_string(index=False))
