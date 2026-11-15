import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.etgnn import ETGNNClassifier
from src.models.gat import GATClassifier
from src.models.gcn import GCNClassifier
from src.models.graphsage import GraphSAGEClassifier
from src.models.matched_gat import MatchedGATClassifier
from src.models.reliability_etgnn import DeepParameterMatchedMLP
from src.models.time_encoding import encode_time


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    ap = float(average_precision_score(y_true, y_prob))
    p, r, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_trapz = float(auc(r, p))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    y_pred = (y_prob >= threshold).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "AP_Primary": ap,
        "PR_AUC_Trapz": pr_auc_trapz,
        "AP_minus_Trapz": ap - pr_auc_trapz,
        "ROC_AUC": roc_auc,
        "Precision_0.5": prec,
        "Recall_0.5": rec,
        "F1_0.5": f1,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def main():
    out_dir = Path("research_extension/phase10")
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = Path("data/processed")
    nodes = pd.read_csv(processed / "nodes_with_masks.csv")
    features = np.load(processed / "scaled_features.npy", mmap_mode="r")
    val_graph = torch.load(processed / "validation_graph.pt", map_location="cpu", weights_only=False)
    test_graph = torch.load(processed / "test_graph.pt", map_location="cpu", weights_only=False)

    val_mask = nodes["val_mask"].to_numpy(dtype=bool)
    test_mask = nodes["test_mask"].to_numpy(dtype=bool)
    y_val = nodes.loc[val_mask, "label"].to_numpy(dtype=int)
    y_test = nodes.loc[test_mask, "label"].to_numpy(dtype=int)
    t_test = nodes.loc[test_mask, "time_step"].to_numpy(dtype=int)

    # Verify exact ID and label alignment between nodes_with_masks.csv and test_graph.pt
    test_ids_csv = np.flatnonzero(test_mask)
    test_ids_graph = test_graph["original_node_index"][test_graph["test_mask"]].numpy()
    np.testing.assert_array_equal(test_ids_csv, test_ids_graph)
    np.testing.assert_array_equal(y_test, test_graph["y"][test_graph["test_mask"]].numpy())
    np.testing.assert_array_equal(t_test, test_graph["time_step"][test_graph["test_mask"]].numpy())
    print(f"[Verified] Test split alignment: N={len(y_test)}, Illicit={(y_test==1).sum()}, Licit={(y_test==0).sum()}")

    x_test_g = test_graph["x"]
    edge_test_g = test_graph["edge_index"]
    ts_test_g = test_graph["time_step"]
    g_test_mask = test_graph["test_mask"]
    dummy_edges = torch.empty((2, 0), dtype=torch.long)

    # 1. Evaluate all 15 verified local models on held-out Test set (steps 41-49)
    model_specs = [
        ("Random Forest (165f Full)", "Tree Ensemble", "165f Full", "No", "N/A (100 trees)", "results/checkpoints/random_forest.joblib", "Historical"),
        ("RF + Time (165f + 16t)", "Tree Ensemble", "165f + 16t", "No", "N/A (100 trees)", "results/checkpoints/rf_with_time.joblib", "Historical"),
        ("Random Forest (93f Local)", "Tree Ensemble", "93f Local", "No", "N/A (100 trees)", "results/checkpoints/rf_local.joblib", "Historical"),
        ("XGBoost (165f Full)", "Gradient Boosted", "165f Full", "No", "N/A (300 trees)", "results/checkpoints/xgboost.json", "Historical"),
        ("Logistic Regression (165f Full)", "Linear Classifier", "165f Full", "No", "166", "research_extension/phase9/checkpoints/logistic_regression_full.joblib", "Phase 9 Isolated"),
        ("Logistic Regression (93f Local)", "Linear Classifier", "93f Local", "No", "94", "research_extension/phase9/checkpoints/logistic_regression_local.joblib", "Phase 9 Isolated"),
        ("MLP (165f Full, s42)", "Feedforward Neural", "165f + 16t", "No (E=0)", "31682", "results/optimization_experiments/checkpoints/P3-05_165f_matched_mlp_s42.pt", "Phase 3 Existing"),
        ("MLP (93f Local, s42)", "Feedforward Neural", "93f + 16t", "No (E=0)", "22466", "results/optimization_experiments/checkpoints/P3-05_93f_matched_mlp_s42.pt", "Phase 3 Existing"),
        ("GCN (165f Full)", "Static GNN", "165f Full", "Yes (E=234k)", "10754", "results/checkpoints/gcn.pt", "Historical"),
        ("GraphSAGE (165f Full)", "Static GNN", "165f Full", "Yes (E=234k)", "21442", "results/checkpoints/graphsage.pt", "Historical"),
        ("GAT Baseline (165f Full)", "Static GNN", "165f Full", "Yes (E=234k)", "10886", "results/checkpoints/gat.pt", "Historical"),
        ("Matched GAT (165f Full)", "Zero-Time GNN Control", "165f + 16z", "Yes (E=234k)", "11910", "results/checkpoints/matched_gat.pt", "Historical"),
        ("ETGNN (165f Full)", "Temporal GNN (Proposed)", "165f + 16t", "Yes (E=234k)", "11910", "results/checkpoints/etgnn.pt", "Historical"),
        ("ETGNN (93f Local)", "Temporal GNN", "93f + 16t", "Yes (E=234k)", "7302", "results/checkpoints/etgnn_local.pt", "Historical"),
        ("ETGNN (93f Local No Edges)", "Time-Encoded Neural (E=0)", "93f + 16t", "No (E=0)", "7302", "results/checkpoints/etgnn_local_no_edges.pt", "Historical"),
    ]

    probs = {}

    rf_full = joblib.load("results/checkpoints/random_forest.joblib")
    probs["Random Forest (165f Full)"] = rf_full.predict_proba(features[test_mask])[:, 1]

    rf_time = joblib.load("results/checkpoints/rf_with_time.joblib")
    time_test_enc = encode_time(torch.tensor(t_test)).numpy()
    probs["RF + Time (165f + 16t)"] = rf_time.predict_proba(np.concatenate([features[test_mask], time_test_enc], axis=1))[:, 1]

    rf_local = joblib.load("results/checkpoints/rf_local.joblib")
    probs["Random Forest (93f Local)"] = rf_local.predict_proba(features[test_mask, :93])[:, 1]

    xgb_full = xgb.XGBClassifier()
    xgb_full.load_model("results/checkpoints/xgboost.json")
    probs["XGBoost (165f Full)"] = xgb_full.predict_proba(features[test_mask])[:, 1]

    lr_full = joblib.load("research_extension/phase9/checkpoints/logistic_regression_full.joblib")
    probs["Logistic Regression (165f Full)"] = lr_full.predict_proba(features[test_mask])[:, 1]

    lr_local = joblib.load("research_extension/phase9/checkpoints/logistic_regression_local.joblib")
    probs["Logistic Regression (93f Local)"] = lr_local.predict_proba(features[test_mask, :93])[:, 1]

    for seed in [42, 43, 44]:
        m165 = DeepParameterMatchedMLP(input_dim=165, time_dim=16, hidden_dim1=128, hidden_dim2=64, dropout=0.3)
        m165.load_state_dict(torch.load(f"results/optimization_experiments/checkpoints/P3-05_165f_matched_mlp_s{seed}.pt", weights_only=True))
        m165.eval()
        with torch.no_grad():
            probs[f"MLP (165f Full, s{seed})"] = torch.softmax(m165(x_test_g, dummy_edges, ts_test_g)[g_test_mask], dim=1)[:, 1].numpy()

        m93 = DeepParameterMatchedMLP(input_dim=93, time_dim=16, hidden_dim1=128, hidden_dim2=64, dropout=0.3)
        m93.load_state_dict(torch.load(f"results/optimization_experiments/checkpoints/P3-05_93f_matched_mlp_s{seed}.pt", weights_only=True))
        m93.eval()
        with torch.no_grad():
            probs[f"MLP (93f Local, s{seed})"] = torch.softmax(m93(x_test_g[:, :93], dummy_edges, ts_test_g)[g_test_mask], dim=1)[:, 1].numpy()

    gcn = GCNClassifier()
    gcn.load_state_dict(torch.load("results/checkpoints/gcn.pt", weights_only=True))
    gcn.eval()
    with torch.no_grad():
        probs["GCN (165f Full)"] = torch.softmax(gcn(x_test_g, edge_test_g)[g_test_mask], dim=1)[:, 1].numpy()

    sage = GraphSAGEClassifier()
    sage.load_state_dict(torch.load("results/checkpoints/graphsage.pt", weights_only=True))
    sage.eval()
    with torch.no_grad():
        probs["GraphSAGE (165f Full)"] = torch.softmax(sage(x_test_g, edge_test_g)[g_test_mask], dim=1)[:, 1].numpy()

    gat = GATClassifier()
    gat.load_state_dict(torch.load("results/checkpoints/gat.pt", weights_only=True))
    gat.eval()
    with torch.no_grad():
        probs["GAT Baseline (165f Full)"] = torch.softmax(gat(x_test_g, edge_test_g)[g_test_mask], dim=1)[:, 1].numpy()

    mgat = MatchedGATClassifier()
    mgat.load_state_dict(torch.load("results/checkpoints/matched_gat.pt", weights_only=True))
    mgat.eval()
    with torch.no_grad():
        probs["Matched GAT (165f Full)"] = torch.softmax(mgat(x_test_g, edge_test_g, ts_test_g)[g_test_mask], dim=1)[:, 1].numpy()

    etgnn = ETGNNClassifier()
    etgnn.load_state_dict(torch.load("results/checkpoints/etgnn.pt", weights_only=True))
    etgnn.eval()
    with torch.no_grad():
        probs["ETGNN (165f Full)"] = torch.softmax(etgnn(x_test_g, edge_test_g, ts_test_g)[g_test_mask], dim=1)[:, 1].numpy()

    etgnn_loc = ETGNNClassifier(input_dim=93)
    etgnn_loc.load_state_dict(torch.load("results/checkpoints/etgnn_local.pt", weights_only=True))
    etgnn_loc.eval()
    with torch.no_grad():
        probs["ETGNN (93f Local)"] = torch.softmax(etgnn_loc(x_test_g[:, :93], edge_test_g, ts_test_g)[g_test_mask], dim=1)[:, 1].numpy()

    etgnn_noe = ETGNNClassifier(input_dim=93)
    etgnn_noe.load_state_dict(torch.load("results/checkpoints/etgnn_local_no_edges.pt", weights_only=True))
    etgnn_noe.eval()
    with torch.no_grad():
        probs["ETGNN (93f Local No Edges)"] = torch.softmax(etgnn_noe(x_test_g[:, :93], dummy_edges, ts_test_g)[g_test_mask], dim=1)[:, 1].numpy()

    # Verify bitwise match against data/processed/final_test_probabilities.npz for historical models
    hist_npz = np.load("data/processed/final_test_probabilities.npz")
    hist_map = [
        ("Random Forest (165f Full)", "model_0"),
        ("RF + Time (165f + 16t)", "model_1"),
        ("Random Forest (93f Local)", "model_2"),
        ("XGBoost (165f Full)", "model_3"),
        ("GCN (165f Full)", "model_4"),
        ("GraphSAGE (165f Full)", "model_5"),
        ("GAT Baseline (165f Full)", "model_6"),
        ("Matched GAT (165f Full)", "model_7"),
        ("ETGNN (165f Full)", "model_8"),
        ("ETGNN (93f Local)", "model_9"),
        ("ETGNN (93f Local No Edges)", "model_10"),
    ]
    for m_name, k_npz in hist_map:
        max_diff = float(np.max(np.abs(probs[m_name] - hist_npz[k_npz])))
        assert max_diff < 1e-6, f"Mismatch for {m_name}: {max_diff}"
    print("[Verified] All 11 historical checkpoints reproduce final_test_probabilities.npz within < 1e-6")

    # Build Master Local Benchmark Table
    master_rows = []
    for name, family, feat_space, uses_edges, params, ckpt_path, cohort in model_specs:
        p_path = Path(ckpt_path)
        sha = sha256_file(p_path)
        size_b = p_path.stat().st_size
        m = evaluate_all_metrics(y_test, probs[name])
        master_rows.append({
            "Model": name,
            "Paradigm": family,
            "Feature_Space": feat_space,
            "Uses_Graph_Edges": uses_edges,
            "Parameters": params,
            "Cohort": cohort,
            "Checkpoint_Path": ckpt_path,
            "Checkpoint_Bytes": size_b,
            "SHA256": sha,
            **m,
        })

    df_master = pd.DataFrame(master_rows).sort_values(by="AP_Primary", ascending=False)
    df_master.to_csv(out_dir / "PHASE10_MASTER_LOCAL_METRICS.csv", index=False)
    print("\n=== MASTER LOCAL METRICS (Sorted by Primary Metric: Average Precision) ===")
    print(df_master[["Model", "AP_Primary", "PR_AUC_Trapz", "AP_minus_Trapz", "ROC_AUC", "F1_0.5", "Precision_0.5", "Recall_0.5"]].to_string(index=False))

    # Also check MLP multi-seed (s42, s43, s44) on Test vs Validation
    mlp_seed_rows = []
    for feat_label, dim in [("165f Full", 165), ("93f Local", 93)]:
        for seed in [42, 43, 44]:
            key = f"MLP ({feat_label}, s{seed})"
            m_test = evaluate_all_metrics(y_test, probs[key])
            with open(f"results/optimization_experiments/tables/P3-05_{dim}f_matched_mlp_s{seed}.json") as f:
                val_j = json.load(f)
            mlp_seed_rows.append({
                "MLP_Condition": feat_label,
                "Seed": seed,
                "Val_AP_from_JSON": val_j["auprc"],
                "Val_F1_from_JSON": val_j["f1"],
                "Val_ROCAUC_from_JSON": val_j["roc_auc"],
                "Test_AP_Primary": m_test["AP_Primary"],
                "Test_PR_AUC_Trapz": m_test["PR_AUC_Trapz"],
                "Test_F1_0.5": m_test["F1_0.5"],
                "Test_ROCAUC": m_test["ROC_AUC"],
            })
    df_mlp_seeds = pd.DataFrame(mlp_seed_rows)
    df_mlp_seeds.to_csv(out_dir / "PHASE10_MLP_MULTISEED_VAL_VS_TEST.csv", index=False)
    print("\n=== MLP MULTI-SEED VALIDATION (N=7653) VS TEST (N=9973) ===")
    print(df_mlp_seeds.to_string(index=False))

    # 2. Evaluate Colab GPU Run (7 models) with BOTH AP_Primary and PR_AUC_Trapz
    colab_npz = np.load("results/colab_extracted_20261103/tables/colab_test_probabilities.npz")
    np.testing.assert_array_equal(y_test, colab_npz["y_test"])
    colab_map = [
        ("Random Forest (165f Full)", "model_Random_Forest", "results/colab_extracted_20261103/checkpoints/random_forest.joblib"),
        ("XGBoost (165f Full)", "model_XGBoost", "results/colab_extracted_20261103/checkpoints/xgboost.json"),
        ("GCN (165f Full)", "model_GCN", "results/colab_extracted_20261103/checkpoints/gcn.pt"),
        ("GraphSAGE (165f Full)", "model_GraphSAGE", "results/colab_extracted_20261103/checkpoints/graphsage.pt"),
        ("GAT Baseline (165f Full)", "model_GAT", "results/colab_extracted_20261103/checkpoints/gat.pt"),
        ("Matched GAT (165f Full)", "model_Matched_GAT", "results/colab_extracted_20261103/checkpoints/matched_gat.pt"),
        ("ETGNN (165f Full)", "model_ETGNN", "results/colab_extracted_20261103/checkpoints/etgnn.pt"),
    ]
    colab_rows = []
    for m_name, c_key, c_ckpt in colab_map:
        p_col = colab_npz[c_key]
        p_loc = probs[m_name]
        mc = evaluate_all_metrics(y_test, p_col)
        ml = evaluate_all_metrics(y_test, p_loc)
        r_p, _ = pearsonr(p_loc, p_col)
        r_s, _ = spearmanr(p_loc, p_col)
        colab_rows.append({
            "Model": m_name,
            "Local_AP": ml["AP_Primary"],
            "Colab_AP": mc["AP_Primary"],
            "Delta_AP_Colab_minus_Local": mc["AP_Primary"] - ml["AP_Primary"],
            "Local_Trapz_PRAUC": ml["PR_AUC_Trapz"],
            "Colab_Trapz_PRAUC": mc["PR_AUC_Trapz"],
            "Delta_Trapz_Colab_minus_Local": mc["PR_AUC_Trapz"] - ml["PR_AUC_Trapz"],
            "Local_F1": ml["F1_0.5"],
            "Colab_F1": mc["F1_0.5"],
            "Local_ROCAUC": ml["ROC_AUC"],
            "Colab_ROCAUC": mc["ROC_AUC"],
            "Pearson_r": r_p,
            "Spearman_rho": r_s,
            "Colab_Checkpoint_SHA256": sha256_file(Path(c_ckpt)),
        })
    df_colab = pd.DataFrame(colab_rows)
    df_colab.to_csv(out_dir / "PHASE10_LOCAL_VS_COLAB_RECONCILIATION.csv", index=False)
    print("\n=== LOCAL CPU VS COLAB GPU RECONCILIATION (AP vs Trapz PR-AUC) ===")
    print(df_colab[["Model", "Local_AP", "Colab_AP", "Delta_AP_Colab_minus_Local", "Local_Trapz_PRAUC", "Colab_Trapz_PRAUC", "Pearson_r"]].to_string(index=False))

    # 3. Task 3: Paired Bootstrap (B=2000) on Primary Metric (Average Precision) AND Time-Step Clustered Analysis (T=9 steps)
    contrasts = [
        ("C01: ETGNN (165f) vs Matched GAT (165f)", "ETGNN (165f Full)", "Matched GAT (165f Full)", "Primary RQ2 Temporal Control (11,910 params)"),
        ("C02: ETGNN (165f) vs GAT Baseline (165f)", "ETGNN (165f Full)", "GAT Baseline (165f Full)", "Static GAT Baseline Comparison"),
        ("C03: ETGNN (165f) vs GCN (165f)", "ETGNN (165f Full)", "GCN (165f Full)", "Static GCN Baseline Comparison"),
        ("C04: GraphSAGE (165f) vs ETGNN (165f)", "GraphSAGE (165f Full)", "ETGNN (165f Full)", "Spatial Mean-Aggregator GNN Comparison"),
        ("C05: Random Forest (165f) vs ETGNN (165f)", "Random Forest (165f Full)", "ETGNN (165f Full)", "Primary RQ1 Tree Ensemble Comparison"),
        ("C06: XGBoost (165f) vs ETGNN (165f)", "XGBoost (165f Full)", "ETGNN (165f Full)", "Gradient Boosted Tree Comparison"),
        ("C07: MLP (165f, s42) vs ETGNN (165f)", "MLP (165f Full, s42)", "ETGNN (165f Full)", "Full-Feature Non-Relational Neural Control"),
        ("C08: ETGNN (165f) vs Logistic Regression (165f)", "ETGNN (165f Full)", "Logistic Regression (165f Full)", "Full-Feature Linear Baseline Comparison"),
        ("C09: Random Forest (93f) vs ETGNN (93f)", "Random Forest (93f Local)", "ETGNN (93f Local)", "Local-Feature Tree vs GNN Comparison"),
        ("C10: MLP (93f, s42) vs ETGNN (93f)", "MLP (93f Local, s42)", "ETGNN (93f Local)", "Local-Feature MLP vs GNN Comparison"),
        ("C11: Logistic Regression (93f) vs ETGNN (93f)", "Logistic Regression (93f Local)", "ETGNN (93f Local)", "Local-Feature Linear vs GNN Comparison"),
        ("C12: ETGNN No-Edges (93f) vs ETGNN (93f)", "ETGNN (93f Local No Edges)", "ETGNN (93f Local)", "Single-Variable Graph Edge Ablation (7,302 params)"),
    ]

    B = 2000
    rng = np.random.default_rng(42)
    N = len(y_test)
    unique_steps = np.sort(np.unique(t_test))
    assert len(unique_steps) == 9  # steps 41..49

    # Pre-slice indices per time step for fast block bootstrap and per-step AP
    step_indices = {int(ts): np.flatnonzero(t_test == ts) for ts in unique_steps}

    # Pre-generate node bootstrap indices and cluster bootstrap step draws so all contrasts share common random numbers
    boot_node_idx = rng.integers(0, N, size=(B, N))
    boot_step_draws = rng.choice(unique_steps, size=(B, len(unique_steps)), replace=True)
    cluster_idx_list = [np.concatenate([step_indices[int(ts)] for ts in boot_step_draws[b]]) for b in range(B)]

    # Pre-compute node bootstrap AP and cluster bootstrap AP once per unique model in contrasts (parallelized across models)
    unique_contrast_models = sorted({m for _, ma, mb, _ in contrasts for m in (ma, mb)})

    def _compute_model_boots(m_name):
        pm = probs[m_name]
        nb = np.array([
            average_precision_score(y_test[boot_node_idx[b]], pm[boot_node_idx[b]])
            for b in range(B)
        ], dtype=float)
        cb = np.array([
            average_precision_score(y_test[cluster_idx_list[b]], pm[cluster_idx_list[b]])
            for b in range(B)
        ], dtype=float)
        return m_name, nb, cb

    boot_results = Parallel(n_jobs=-1)(delayed(_compute_model_boots)(m) for m in unique_contrast_models)
    node_boot_ap = {m: nb for m, nb, _ in boot_results}
    cluster_boot_ap = {m: cb for m, _, cb in boot_results}

    stat_rows = []
    for cid, m_a, m_b, desc in contrasts:
        pa = probs[m_a]
        pb = probs[m_b]

        ap_a = float(average_precision_score(y_test, pa))
        ap_b = float(average_precision_score(y_test, pb))
        delta_ap = ap_a - ap_b

        pr_p_a, pr_r_a, _ = precision_recall_curve(y_test, pa)
        pr_p_b, pr_r_b, _ = precision_recall_curve(y_test, pb)
        trapz_a = float(auc(pr_r_a, pr_p_a))
        trapz_b = float(auc(pr_r_b, pr_p_b))
        delta_trapz = trapz_a - trapz_b

        # Node-level paired bootstrap on AP (Primary)
        diffs_ap = node_boot_ap[m_a] - node_boot_ap[m_b]

        se_ap = float(np.std(diffs_ap, ddof=1))
        ci_low, ci_high = np.percentile(diffs_ap, [2.5, 97.5])
        # Exact two-sided Monte Carlo bootstrap p-value (Phipson & Smyth 2010: (k + 1)/(B + 1))
        if delta_ap >= 0:
            k_opp = int(np.sum(diffs_ap <= 0))
        else:
            k_opp = int(np.sum(diffs_ap >= 0))
        p_val_exact = min(1.0, 2.0 * (k_opp + 1.0) / (B + 1.0))

        # Per-time-step analysis (T = 9 steps: 41..49)
        step_ap_a = np.array([average_precision_score(y_test[step_indices[int(ts)]], pa[step_indices[int(ts)]]) for ts in unique_steps])
        step_ap_b = np.array([average_precision_score(y_test[step_indices[int(ts)]], pb[step_indices[int(ts)]]) for ts in unique_steps])
        step_diffs = step_ap_a - step_ap_b
        steps_a_wins = int(np.sum(step_diffs > 0))
        mean_step_diff = float(np.mean(step_diffs))
        sd_step_diff = float(np.std(step_diffs, ddof=1))
        _, p_ttest_step = ttest_rel(step_ap_a, step_ap_b)
        _, p_wilcoxon_step = wilcoxon(step_diffs)

        # Time-Step Cluster (Block) Bootstrap (resampling 9 time-step blocks with replacement)
        cluster_diffs = cluster_boot_ap[m_a] - cluster_boot_ap[m_b]
        cluster_se = float(np.std(cluster_diffs, ddof=1))
        cluster_ci_low, cluster_ci_high = np.percentile(cluster_diffs, [2.5, 97.5])

        stat_rows.append({
            "Contrast_ID": cid,
            "Model_A": m_a,
            "Model_B": m_b,
            "Description": desc,
            "AP_Model_A": ap_a,
            "AP_Model_B": ap_b,
            "Delta_AP": delta_ap,
            "Node_Boot_SE": se_ap,
            "Node_CI95_Low": float(ci_low),
            "Node_CI95_High": float(ci_high),
            "Node_P_Value": float(p_val_exact),
            "Node_CI_Crosses_Zero": bool(ci_low <= 0 <= ci_high),
            "Trapz_Model_A": trapz_a,
            "Trapz_Model_B": trapz_b,
            "Delta_Trapz": delta_trapz,
            "Steps_A_Wins_of_9": f"{steps_a_wins}/9",
            "Mean_Step_Delta_AP": mean_step_diff,
            "SD_Step_Delta_AP": sd_step_diff,
            "Step_Paired_T_Pval": float(p_ttest_step),
            "Step_Wilcoxon_Pval": float(p_wilcoxon_step),
            "Cluster_Boot_SE": cluster_se,
            "Cluster_CI95_Low": float(cluster_ci_low),
            "Cluster_CI95_High": float(cluster_ci_high),
            "Cluster_CI_Crosses_Zero": bool(cluster_ci_low <= 0 <= cluster_ci_high),
        })

    df_stat = pd.DataFrame(stat_rows).sort_values(by="Node_P_Value").reset_index(drop=True)
    m_tests = len(df_stat)
    df_stat["Holm_Rank"] = np.arange(1, m_tests + 1)
    df_stat["Holm_Alpha"] = 0.05 / (m_tests - df_stat["Holm_Rank"] + 1)
    # Step-down Holm-Bonferroni: once a rank fails to reject, all higher ranks are False
    raw_pass = (df_stat["Node_P_Value"] <= df_stat["Holm_Alpha"]).to_numpy()
    holm_sig = []
    still_sig = True
    for passed in raw_pass:
        if not passed:
            still_sig = False
        holm_sig.append(still_sig)
    df_stat["Holm_Significant"] = holm_sig

    df_stat.to_csv(out_dir / "PHASE10_STATISTICAL_COMPARISONS.csv", index=False)
    print("\n=== TASK 3: PAIRED BOOTSTRAP (AP Primary) & TIME-STEP CLUSTER SENSITIVITY ===")
    print(df_stat[[
        "Contrast_ID", "Delta_AP", "Node_CI95_Low", "Node_CI95_High",
        "Node_P_Value", "Holm_Significant", "Steps_A_Wins_of_9",
        "Cluster_CI95_Low", "Cluster_CI95_High", "Step_Wilcoxon_Pval"
    ]].to_string(index=False))

    # Save per-time-step AP table for all models
    per_step_rows = []
    for ts in unique_steps:
        idx = step_indices[int(ts)]
        y_ts = y_test[idx]
        row = {
            "time_step": int(ts),
            "N_labeled": int(len(y_ts)),
            "N_illicit": int((y_ts == 1).sum()),
            "prevalence": float(y_ts.mean()),
        }
        for name, *_ in model_specs:
            row[f"AP_{name}"] = float(average_precision_score(y_ts, probs[name][idx]))
        per_step_rows.append(row)
    pd.DataFrame(per_step_rows).to_csv(out_dir / "PHASE10_PER_TIMESTEP_AP.csv", index=False)


if __name__ == "__main__":
    main()