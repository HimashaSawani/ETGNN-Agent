import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import numpy as np
import pandas as pd
import torch
import joblib
import xgboost as xgb
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from src.models.reliability_etgnn import DeepParameterMatchedMLP
from src.models.etgnn import ETGNNClassifier
from src.models.matched_gat import MatchedGATClassifier
from src.models.gat import GATClassifier
from src.models.gcn import GCNClassifier
from src.models.graphsage import GraphSAGEClassifier


def evaluate_binary(y_true, y_prob, threshold=0.5):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    auprc_trapz = float(auc(r, p))
    ap_stepwise = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "auprc": auprc_trapz,
        "ap_stepwise": ap_stepwise,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
    }


def main():
    processed = Path("data/processed")
    nodes = pd.read_csv(processed / "nodes_with_masks.csv")
    features = np.load(processed / "scaled_features.npy", mmap_mode="r")
    test_graph = torch.load(processed / "test_graph.pt", weights_only=False)

    test_mask = nodes["test_mask"].to_numpy(dtype=bool)
    y_test = nodes.loc[test_mask, "label"].to_numpy()
    x = test_graph["x"]
    edge_index = test_graph["edge_index"]
    time_step = test_graph["time_step"]
    dummy_edges = torch.empty((2, 0), dtype=torch.long)

    probabilities = {}

    # 1. Random Forest Full (165f) - Existing Verified Checkpoint
    rf_full = joblib.load("results/checkpoints/random_forest.joblib")
    probabilities["Random Forest (165f Full)"] = rf_full.predict_proba(features[test_mask])[:, 1]

    # 2. Random Forest Local (93f) - Existing Verified Checkpoint
    rf_local = joblib.load("results/checkpoints/rf_local.joblib")
    probabilities["Random Forest (93f Local)"] = rf_local.predict_proba(features[test_mask, :93])[:, 1]

    # 3. XGBoost Full (165f) - Existing Verified Checkpoint
    xgb_full = xgb.XGBClassifier()
    xgb_full.load_model("results/checkpoints/xgboost.json")
    probabilities["XGBoost (165f Full)"] = xgb_full.predict_proba(features[test_mask])[:, 1]

    # 4. Logistic Regression Full (165f) - Phase 9 Isolated Checkpoint
    lr_full = joblib.load("research_extension/phase9/checkpoints/logistic_regression_full.joblib")
    probabilities["Logistic Regression (165f Full)"] = lr_full.predict_proba(features[test_mask])[:, 1]

    # 5. Logistic Regression Local (93f) - Phase 9 Isolated Checkpoint
    lr_local = joblib.load("research_extension/phase9/checkpoints/logistic_regression_local.joblib")
    probabilities["Logistic Regression (93f Local)"] = lr_local.predict_proba(features[test_mask, :93])[:, 1]

    # 6. MLP Full (165f) - Existing Verified Checkpoint (P3-05)
    mlp_full = DeepParameterMatchedMLP(input_dim=165, time_dim=16, hidden_dim1=128, hidden_dim2=64, dropout=0.3)
    mlp_full.load_state_dict(
        torch.load("results/optimization_experiments/checkpoints/P3-05_165f_matched_mlp_s42.pt", weights_only=True)
    )
    mlp_full.eval()
    with torch.no_grad():
        logits = mlp_full(x, dummy_edges, time_step)
        probabilities["MLP (165f Full)"] = torch.softmax(logits[test_mask], dim=1)[:, 1].numpy()

    # 7. MLP Local (93f) - Existing Verified Checkpoint (P3-05)
    mlp_local = DeepParameterMatchedMLP(input_dim=93, time_dim=16, hidden_dim1=128, hidden_dim2=64, dropout=0.3)
    mlp_local.load_state_dict(
        torch.load("results/optimization_experiments/checkpoints/P3-05_93f_matched_mlp_s42.pt", weights_only=True)
    )
    mlp_local.eval()
    with torch.no_grad():
        logits = mlp_local(x[:, :93], dummy_edges, time_step)
        probabilities["MLP (93f Local)"] = torch.softmax(logits[test_mask], dim=1)[:, 1].numpy()

    # 8. GCN Full (165f) - Existing Verified Checkpoint
    gcn = GCNClassifier()
    gcn.load_state_dict(torch.load("results/checkpoints/gcn.pt", weights_only=True))
    gcn.eval()
    with torch.no_grad():
        logits = gcn(x, edge_index)
        probabilities["GCN (165f Full)"] = torch.softmax(logits[test_mask], dim=1)[:, 1].numpy()

    # 9. GraphSAGE Full (165f) - Existing Verified Checkpoint
    sage = GraphSAGEClassifier()
    sage.load_state_dict(torch.load("results/checkpoints/graphsage.pt", weights_only=True))
    sage.eval()
    with torch.no_grad():
        logits = sage(x, edge_index)
        probabilities["GraphSAGE (165f Full)"] = torch.softmax(logits[test_mask], dim=1)[:, 1].numpy()

    # 10. GAT Baseline Full (165f) - Existing Verified Checkpoint
    gat = GATClassifier()
    gat.load_state_dict(torch.load("results/checkpoints/gat.pt", weights_only=True))
    gat.eval()
    with torch.no_grad():
        logits = gat(x, edge_index)
        probabilities["GAT Baseline (165f Full)"] = torch.softmax(logits[test_mask], dim=1)[:, 1].numpy()

    # 11. Matched GAT Full (165f) - Existing Verified Checkpoint
    mgat = MatchedGATClassifier()
    mgat.load_state_dict(torch.load("results/checkpoints/matched_gat.pt", weights_only=True))
    mgat.eval()
    with torch.no_grad():
        logits = mgat(x, edge_index, time_step)
        probabilities["Matched GAT (165f Full)"] = torch.softmax(logits[test_mask], dim=1)[:, 1].numpy()

    # 12. Proposed ETGNN Full (165f) - Existing Verified Checkpoint
    etgnn = ETGNNClassifier()
    etgnn.load_state_dict(torch.load("results/checkpoints/etgnn.pt", weights_only=True))
    etgnn.eval()
    with torch.no_grad():
        logits = etgnn(x, edge_index, time_step)
        probabilities["ETGNN (165f Full)"] = torch.softmax(logits[test_mask], dim=1)[:, 1].numpy()

    # 13. ETGNN Local (93f) - Existing Verified Checkpoint
    etgnn_loc = ETGNNClassifier(input_dim=93)
    etgnn_loc.load_state_dict(torch.load("results/checkpoints/etgnn_local.pt", weights_only=True))
    etgnn_loc.eval()
    with torch.no_grad():
        logits = etgnn_loc(x[:, :93], edge_index, time_step)
        probabilities["ETGNN (93f Local)"] = torch.softmax(logits[test_mask], dim=1)[:, 1].numpy()

    # 14. ETGNN Local No Edges (93f) - Existing Verified Checkpoint
    etgnn_no_e = ETGNNClassifier(input_dim=93)
    etgnn_no_e.load_state_dict(torch.load("results/checkpoints/etgnn_local_no_edges.pt", weights_only=True))
    etgnn_no_e.eval()
    with torch.no_grad():
        logits = etgnn_no_e(x[:, :93], dummy_edges, time_step)
        probabilities["ETGNN (93f Local No Edges)"] = torch.softmax(logits[test_mask], dim=1)[:, 1].numpy()

    rows = []
    for model_name, probs in probabilities.items():
        m = evaluate_binary(y_test, probs)
        rows.append(
            {
                "Model": model_name,
                "AUPRC": m["auprc"],
                "AP_Stepwise": m["ap_stepwise"],
                "ROC_AUC": m["roc_auc"],
                "F1_0.5": m["f1"],
                "Precision_0.5": m["precision"],
                "Recall_0.5": m["recall"],
                "TP": m["tp"],
                "FP": m["fp"],
                "TN": m["tn"],
                "FN": m["fn"],
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(by="AUPRC", ascending=False)

    out_csv = Path("research_extension/phase9/BASELINE_COMPARISON_METRICS.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(df[["Model", "AUPRC", "AP_Stepwise", "ROC_AUC", "F1_0.5", "Precision_0.5", "Recall_0.5"]].to_string(index=False))

    # Build Feature Ablation Table using ONLY existing verified models + Logistic Regression
    # Missing cells are explicitly flagged per Scope Corrections #4 and #8
    ablation_pairs = [
        ("Random Forest", "Random Forest (93f Local)", "Random Forest (165f Full)"),
        ("Logistic Regression", "Logistic Regression (93f Local)", "Logistic Regression (165f Full)"),
        ("MLP (Matched)", "MLP (93f Local)", "MLP (165f Full)"),
        ("ETGNN (GNN)", "ETGNN (93f Local)", "ETGNN (165f Full)"),
        ("XGBoost", None, "XGBoost (165f Full)"),
        ("Matched GAT (0-Time)", None, "Matched GAT (165f Full)"),
    ]

    ablation_rows = []
    for family, loc_key, full_key in ablation_pairs:
        p_full = probabilities[full_key]
        m_full = evaluate_binary(y_test, p_full)

        if loc_key is not None and loc_key in probabilities:
            p_loc = probabilities[loc_key]
            m_loc = evaluate_binary(y_test, p_loc)
            diff_auprc = m_full["auprc"] - m_loc["auprc"]
            diff_f1 = m_full["f1"] - m_loc["f1"]
            ablation_rows.append(
                {
                    "Model_Family": family,
                    "Local_93f_AUPRC": f"{m_loc['auprc']:.4f}",
                    "Full_165f_AUPRC": f"{m_full['auprc']:.4f}",
                    "AUPRC_Gain_From_Aggregates": f"{diff_auprc:+.4f}",
                    "Local_93f_F1": f"{m_loc['f1']:.4f}",
                    "Full_165f_F1": f"{m_full['f1']:.4f}",
                    "F1_Gain_From_Aggregates": f"{diff_f1:+.4f}",
                    "Local_93f_ROCAUC": f"{m_loc['roc_auc']:.4f}",
                    "Full_165f_ROCAUC": f"{m_full['roc_auc']:.4f}",
                    "Precomputed_Aggregates_Help": "Yes" if diff_auprc > 0 else "No",
                    "Provenance_Status": "Verified (Both Local 93f & Full 165f Checkpoints Exist)",
                }
            )
        else:
            ablation_rows.append(
                {
                    "Model_Family": family,
                    "Local_93f_AUPRC": "MISSING",
                    "Full_165f_AUPRC": f"{m_full['auprc']:.4f}",
                    "AUPRC_Gain_From_Aggregates": "N/A (Missing Local 93f)",
                    "Local_93f_F1": "MISSING",
                    "Full_165f_F1": f"{m_full['f1']:.4f}",
                    "F1_Gain_From_Aggregates": "N/A (Missing Local 93f)",
                    "Local_93f_ROCAUC": "MISSING",
                    "Full_165f_ROCAUC": f"{m_full['roc_auc']:.4f}",
                    "Precomputed_Aggregates_Help": "Unverified (No Local 93f Checkpoint)",
                    "Provenance_Status": "Incomplete in Original Repository (Local 93f Not Trained per Scope Correction #4)",
                }
            )

    df_abl = pd.DataFrame(ablation_rows)
    out_abl = Path("research_extension/phase9/FEATURE_ABLATION_RESULTS.csv")
    df_abl.to_csv(out_abl, index=False)
    print(f"\nSaved Feature Ablation Results: {out_abl}")
    print(df_abl.to_string(index=False))

    # Save all test probability vectors for statistical comparison
    np.savez_compressed(
        "research_extension/phase9/phase9_all_test_probabilities.npz",
        y_test=y_test,
        **probabilities,
    )
    print("Saved all test probabilities to research_extension/phase9/phase9_all_test_probabilities.npz")


if __name__ == "__main__":
    main()
