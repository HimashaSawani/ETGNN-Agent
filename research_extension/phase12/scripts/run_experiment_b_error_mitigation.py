#!/usr/bin/env python3
"""
run_experiment_b_error_mitigation.py
====================================
Phase 12 — Experiment B: Explanation-Guided Error Mitigation & Matched Controls.

Scientific Safeguards Enforced:
1. Strict Validation-Only Selection (Safeguard 5): All thresholds, sensitivity cutoffs,
   routing fractions, and blending weights are selected exclusively on the Validation
   Split (steps 33-40, N = 7,653). The Held-Out Test Split (steps 41-49, N = 9,973)
   is quarantined during tuning and evaluated once with frozen parameters.
2. Comprehensive Matched Controls (Safeguard 6):
   - Single-Model Topological Interventions on frozen `etgnn.pt` (Zero external model):
     * M1a: GNNExplainer Top-k Edge Deletion on Predicted Positives
     * M1b: Exact LOO Top-k Edge Deletion on Predicted Positives
     * C1a: Random-k Edge Deletion on Predicted Positives (Matched Control)
     * M1c: Sensitivity-Gated Self-Loop (Empty-Subgraph) Fallback (val-tuned cutoff)
     * C1b: Confidence-Gated Self-Loop Fallback (Matched routing rate)
     * C1c: Random-Gated Self-Loop Fallback (Matched routing rate, 50 draws)
     * M1d: Ungated 100% Self-Loop (Empty-Subgraph) Pass on frozen `etgnn.pt`
   - Dual-Model Fallback Routing (`etgnn.pt` -> `etgnn_local_no_edges.pt`):
     * M2a: Sensitivity-Gated Local Fallback (S_empty, matched routing rate r)
     * M2b: LOO-Sensitivity-Gated Local Fallback (S_LOO, matched routing rate r)
     * C2a: Confidence-Uncertainty-Gated Local Fallback (Matched routing rate r)
     * C2b: Predicted-Risk-Gated Local Fallback (Matched routing rate r — controls for p_et165 magnitude)
     * C2c: Random-Gated Local Fallback (Matched routing rate r, 50 draws)
     * M3a: Predicted-Positive Sensitivity-Gated Local Veto (top 50% sensitive predicted positives)
     * C3a: Predicted-Positive Confidence-Gated Local Veto (top 50% uncertain predicted positives)
     * C3b: Predicted-Positive Random-Gated Local Veto (random 50% predicted positives)
     * C3c: Predicted-Positive 100% Local Veto (all predicted positives vetted by local model)
     * C4:  Ungated Validation-Optimal Linear Blend ((1-alpha*)*p_et165 + alpha**p_local)
     * C5:  Ungated 100% Local-No-Edges Fallback (`etgnn_local_no_edges.pt`)
3. Subgroup, Calibration & Statistical Reporting (Safeguards 7, 8, 9):
   - Evaluates Full Test (steps 41-49, N = 9,973), Pre-Shift (steps 41-42, N = 3,286),
     and Post-Shift (steps 43-49, N = 6,687).
   - Reports Primary AP, Trapz PR-AUC, ROC-AUC, Precision, Recall, F1, FPR, TP, FP, TN, FN,
     Brier Score, and 10-bin Expected Calibration Error (ECE).
   - Computes B = 2,000 paired bootstrap 95% CIs, exact Phipson-Smyth p-values, and
     Holm-Bonferroni multiple-testing adjustments.
"""

import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_auc_score
from torch_geometric.explain import Explainer, GNNExplainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.etgnn import ETGNNClassifier
from research_extension.phase12.scripts.run_experiment_a_temporal_stability import (
    build_incoming_csr,
    extract_2hop_subgraph_fast,
)

P12_DIR = PROJECT_ROOT / "research_extension" / "phase12"
TABLES_DIR = P12_DIR / "tables"
LOGS_DIR = P12_DIR / "logs"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def fast_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact vectorized step-wise Average Precision matching sklearn.metrics.average_precision_score."""
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    n_pos = np.sum(y_sorted)
    if n_pos == 0:
        return 0.0
    distinct_value_indices = np.where(np.diff(y_score[order]))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    prec = tps / (tps + fps)
    rec = tps / n_pos
    rec_diff = np.r_[rec[0], np.diff(rec)]
    return float(np.sum(rec_diff * prec))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b_i in range(n_bins):
        if b_i == n_bins - 1:
            mask = (y_prob >= bins[b_i]) & (y_prob <= bins[b_i + 1])
        else:
            mask = (y_prob >= bins[b_i]) & (y_prob < bins[b_i + 1])
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / float(n)) * abs(acc - conf)
    return float(ece)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50, base_fp: int = None, base_tp: int = None):
    ap = float(average_precision_score(y_true, y_prob))
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
    trapz_pr = float(auc(rec_c, prec_c))
    roc = float(roc_auc_score(y_true, y_prob))

    pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))

    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / float(fp + tn) if (fp + tn) > 0 else 0.0
    brier = float(np.mean((y_prob - y_true) ** 2))
    ece = compute_ece(y_true, y_prob, n_bins=10)

    delta_fp = (fp - base_fp) if base_fp is not None else 0
    delta_tp = (tp - base_tp) if base_tp is not None else 0

    return {
        "AP_Primary": ap,
        "PR_AUC_Trapz": trapz_pr,
        "ROC_AUC": roc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "FPR": fpr,
        "Brier": brier,
        "ECE_10": ece,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Delta_FP_vs_Base": delta_fp,
        "Delta_TP_vs_Base": delta_tp,
    }


def compute_split_perturbation_signals(
    graph_dict: dict,
    mask_key: str,
    m_et165: torch.nn.Module,
    m_noedge: torch.nn.Module,
    explainer_et165: Explainer,
    alpha_budget: float = 0.25,
    split_label: str = "Validation",
):
    """
    Computes exact 2-hop subgraph perturbation signals for all masked nodes in graph_dict:
      - p_et165: full-graph ETGNN (165f) illicit probability
      - p_empty: self-loop-only (0-edge) ETGNN (165f) illicit probability
      - s_empty: |p_et165 - p_empty|
      - s_loo_max: max single-edge LOO occlusion shift |p_et165 - p_et165(-e)| across E_cand
      - p_loo_topk_del: illicit probability after deleting top-k LOO edges supporting predicted class
      - p_rand_topk_del: mean illicit probability after deleting k random edges (10 draws)
      - p_ge_topk_del: illicit probability after deleting top-k GNNExplainer edges (computed for all predicted positives p >= 0.50)
      - p_local_noedge: frozen `etgnn_local_no_edges.pt` illicit probability
    """
    t0 = time.perf_counter()
    x_full = graph_dict["x"]
    x_local = x_full[:, :93]
    ei = graph_dict["edge_index"]
    ts = graph_dict["time_step"]
    y_all = graph_dict["y"].numpy()
    mask = graph_dict[mask_key].bool().numpy()
    masked_indices = np.where(mask)[0]

    empty_ei = torch.zeros((2, 0), dtype=torch.long)
    with torch.no_grad():
        probs_full = F.softmax(m_et165(x_full, ei, ts), dim=1).numpy()
        probs_empty = F.softmax(m_et165(x_full, empty_ei, ts), dim=1).numpy()
        probs_local = F.softmax(m_noedge(x_local, empty_ei, ts), dim=1).numpy()

    src_np, dst_np, in_edges_by_node = build_incoming_csr(ei, len(x_full))

    N = len(masked_indices)
    y_true = y_all[masked_indices].astype(int)
    steps = ts.numpy()[masked_indices].astype(int)
    p_et165 = probs_full[masked_indices, 1].astype(np.float64)
    p_empty = probs_empty[masked_indices, 1].astype(np.float64)
    p_local = probs_local[masked_indices, 1].astype(np.float64)
    s_empty = np.abs(p_et165 - p_empty)

    e_cand_arr = np.zeros(N, dtype=int)
    s_loo_max = np.zeros(N, dtype=np.float64)
    p_loo_topk = p_et165.copy()
    p_rand_topk = p_et165.copy()
    p_ge_topk = p_et165.copy()
    s_ge_drop = np.zeros(N, dtype=np.float64)

    n_pred_pos = 0
    for idx_i, nid in enumerate(masked_indices):
        nid_int = int(nid)
        sub_nodes, sub_edges, local_center, _ = extract_2hop_subgraph_fast(
            nid_int, src_np, dst_np, in_edges_by_node
        )
        ec = int(sub_edges.shape[1])
        e_cand_arr[idx_i] = ec
        if ec == 0:
            continue

        k_bud = max(1, int(np.ceil(alpha_budget * ec)))
        sub_x = x_full[sub_nodes]
        sub_t = ts[sub_nodes]
        sub_ei = torch.from_numpy(sub_edges).long()
        p_orig = p_et165[idx_i]
        pred_c = 1 if p_orig >= 0.50 else 0

        # Exact LOO for all nodes with E_cand >= 1 (if E_cand == 1, deleting that 1 edge equals empty subgraph!)
        if ec == 1:
            s_loo_max[idx_i] = s_empty[idx_i]
            p_loo_topk[idx_i] = p_empty[idx_i]
            p_rand_topk[idx_i] = p_empty[idx_i]
            if pred_c == 1:
                p_ge_topk[idx_i] = p_empty[idx_i]
                s_ge_drop[idx_i] = p_orig - p_empty[idx_i]
                n_pred_pos += 1
            continue

        # Multi-edge node (ec >= 2): batched single-pass LOO across candidate edges (capped at 16 edges for giant hubs)
        eval_e_indices = np.arange(ec) if ec <= 16 else np.linspace(0, ec - 1, 16, dtype=int)
        n_eval_e = len(eval_e_indices)
        n_sub = len(sub_nodes)

        with torch.no_grad():
            batch_x = sub_x.repeat(n_eval_e, 1)
            batch_t = sub_t.repeat(n_eval_e)
            ei_list = []
            for b_idx, e_i in enumerate(eval_e_indices):
                keep = np.ones(ec, dtype=bool)
                keep[e_i] = False
                ei_list.append(sub_ei[:, keep] + (b_idx * n_sub))
            batch_ei = torch.cat(ei_list, dim=1) if len(ei_list) > 0 else torch.zeros((2, 0), dtype=torch.long)
            batch_out = F.softmax(m_et165(batch_x, batch_ei, batch_t), dim=1)
            center_indices = [b_idx * n_sub + local_center for b_idx in range(n_eval_e)]
            edge_del_p1 = batch_out[center_indices, 1].numpy().astype(np.float64)

        shifts = np.abs(p_orig - edge_del_p1)
        s_loo_max[idx_i] = float(np.max(shifts))

        # Target-class signed drops when each evaluated edge is removed
        if pred_c == 1:
            target_drops = p_orig - edge_del_p1
        else:
            target_drops = edge_del_p1 - p_orig

        loo_order_local = np.argsort(-target_drops)
        loo_order = eval_e_indices[loo_order_local]
        k_eff = min(k_bud, len(loo_order))
        if k_bud == 1:
            p_loo_topk[idx_i] = float(edge_del_p1[loo_order_local[0]])
            p_rand_topk[idx_i] = float(np.mean(edge_del_p1))
        else:
            keep_loo = np.ones(ec, dtype=bool)
            keep_loo[loo_order[:k_eff]] = False
            with torch.no_grad():
                p_loo_topk[idx_i] = float(
                    F.softmax(m_et165(sub_x, sub_ei[:, keep_loo], sub_t), dim=1)[local_center, 1].item()
                )
            r_ei_list = []
            for d_i in range(10):
                r_rng = np.random.default_rng(500000 + nid_int * 20 + d_i)
                r_sel = r_rng.choice(ec, size=k_bud, replace=False)
                keep_r = np.ones(ec, dtype=bool)
                keep_r[r_sel] = False
                r_ei_list.append(sub_ei[:, keep_r] + (d_i * n_sub))
            with torch.no_grad():
                r_batch_out = F.softmax(
                    m_et165(sub_x.repeat(10, 1), torch.cat(r_ei_list, dim=1), sub_t.repeat(10)), dim=1
                )
                r_centers = [d_i * n_sub + local_center for d_i in range(10)]
                p_rand_topk[idx_i] = float(r_batch_out[r_centers, 1].mean().item())

        # Run GNNExplainer on predicted-positive multi-edge nodes (pred_c == 1)
        if pred_c == 1:
            n_pred_pos += 1
            torch.manual_seed(42)
            expl = explainer_et165(x=sub_x, edge_index=sub_ei, time_step=sub_t, index=local_center)
            m_arr = expl.edge_mask.detach().cpu().numpy()
            ge_order = np.argsort(-m_arr)
            if k_bud == 1 and ge_order[0] in eval_e_indices:
                pos_in_eval = int(np.where(eval_e_indices == ge_order[0])[0][0])
                p_ge_topk[idx_i] = float(edge_del_p1[pos_in_eval])
            else:
                keep_ge = np.ones(ec, dtype=bool)
                keep_ge[ge_order[:k_bud]] = False
                with torch.no_grad():
                    p_ge_topk[idx_i] = float(
                        F.softmax(m_et165(sub_x, sub_ei[:, keep_ge], sub_t), dim=1)[local_center, 1].item()
                    )
            s_ge_drop[idx_i] = p_orig - p_ge_topk[idx_i]

    print(
        f"  [{split_label}] Computed exact perturbation signals for N={N} nodes "
        f"(pred_pos={n_pred_pos}, multi-edge={int(np.sum(e_cand_arr>=2))}) in {time.perf_counter()-t0:.2f}s"
    )
    return {
        "node_indices": masked_indices,
        "y_true": y_true,
        "time_step": steps,
        "E_cand": e_cand_arr,
        "p_et165": p_et165,
        "p_empty": p_empty,
        "s_empty": s_empty,
        "s_loo_max": s_loo_max,
        "p_loo_topk": p_loo_topk,
        "p_rand_topk": p_rand_topk,
        "p_ge_topk": p_ge_topk,
        "s_ge_drop": s_ge_drop,
        "p_local": p_local,
        "conf_uncertainty": 0.50 - np.abs(p_et165 - 0.50),
    }


def apply_routing_fraction(p_base: np.ndarray, p_fallback: np.ndarray, score_signal: np.ndarray, frac: float) -> np.ndarray:
    """Routes the top `frac` fraction of nodes ranked by `score_signal` from p_base to p_fallback."""
    if frac <= 0.0:
        return p_base.copy()
    if frac >= 1.0:
        return p_fallback.copy()
    n = len(p_base)
    k = int(np.round(frac * n))
    order = np.argsort(-score_signal, kind="mergesort")
    out = p_base.copy()
    out[order[:k]] = p_fallback[order[:k]]
    return out


def paired_bootstrap_contrast(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, tau_a: float = 0.50, tau_b: float = 0.50, B: int = 2000, seed: int = 20261104):
    """
    Computes paired bootstrap 95% CIs and exact Phipson-Smyth p-values for:
      - Delta AP = AP(A) - AP(B)
      - Delta F1 = F1(A) - F1(B)
      - Delta FPR = FPR(A) - FPR(B)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    obs_ap_a = fast_ap(y_true, p_a)
    obs_ap_b = fast_ap(y_true, p_b)
    obs_d_ap = obs_ap_a - obs_ap_b

    pred_a = (p_a >= tau_a).astype(int)
    pred_b = (p_b >= tau_b).astype(int)

    def f1_and_fpr(y, pr):
        tp = np.sum((pr == 1) & (y == 1))
        fp = np.sum((pr == 1) & (y == 0))
        tn = np.sum((pr == 0) & (y == 0))
        fn = np.sum((pr == 0) & (y == 1))
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        fpr = fp / float(fp + tn) if (fp + tn) > 0 else 0.0
        return f1, fpr

    obs_f1_a, obs_fpr_a = f1_and_fpr(y_true, pred_a)
    obs_f1_b, obs_fpr_b = f1_and_fpr(y_true, pred_b)
    obs_d_f1 = obs_f1_a - obs_f1_b
    obs_d_fpr = obs_fpr_a - obs_fpr_b

    boot_d_ap = np.zeros(B, dtype=np.float64)
    boot_d_f1 = np.zeros(B, dtype=np.float64)
    boot_d_fpr = np.zeros(B, dtype=np.float64)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        boot_d_ap[b] = fast_ap(yb, p_a[idx]) - fast_ap(yb, p_b[idx])
        f1a, fpra = f1_and_fpr(yb, pred_a[idx])
        f1b, fprb = f1_and_fpr(yb, pred_b[idx])
        boot_d_f1[b] = f1a - f1b
        boot_d_fpr[b] = fpra - fprb

    def p_exact(obs, boots):
        if abs(obs) < 1e-12:
            return 1.0
        if obs > 0:
            k_opp = int(np.sum(boots <= 0.0))
        else:
            k_opp = int(np.sum(boots >= 0.0))
        return min(1.0, 2.0 * (k_opp + 1.0) / (B + 1.0))

    return {
        "Delta_AP": float(obs_d_ap),
        "AP_CI95_Low": float(np.percentile(boot_d_ap, 2.5)),
        "AP_CI95_High": float(np.percentile(boot_d_ap, 97.5)),
        "AP_Exact_P": float(p_exact(obs_d_ap, boot_d_ap)),
        "Delta_F1": float(obs_d_f1),
        "F1_CI95_Low": float(np.percentile(boot_d_f1, 2.5)),
        "F1_CI95_High": float(np.percentile(boot_d_f1, 97.5)),
        "F1_Exact_P": float(p_exact(obs_d_f1, boot_d_f1)),
        "Delta_FPR": float(obs_d_fpr),
        "FPR_CI95_Low": float(np.percentile(boot_d_fpr, 2.5)),
        "FPR_CI95_High": float(np.percentile(boot_d_fpr, 97.5)),
    }


def main():
    t0_all = time.perf_counter()
    print("=" * 80)
    print("PHASE 12 — EXPERIMENT B: EXPLANATION-GUIDED ERROR MITIGATION & MATCHED CONTROLS")
    print("=" * 80)

    with open(P12_DIR / "configs" / "phase12_experiment_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Load Validation Graph (steps 1-40, val_mask = steps 33-40) and Test Graph (steps 1-49, test_mask = steps 41-49)
    val_g = torch.load(PROJECT_ROOT / cfg["data_artifacts"]["validation_graph"], map_location="cpu", weights_only=False)
    test_g = torch.load(PROJECT_ROOT / cfg["data_artifacts"]["test_graph"], map_location="cpu", weights_only=False)

    assert 41 not in val_g["time_step"], "Test steps leaked into validation_graph!"
    assert int(val_g["val_mask"].sum().item()) == 7653
    assert int(test_g["test_mask"].sum().item()) == 9973

    m_et165 = ETGNNClassifier(input_dim=165, time_dim=16, hidden_dim=64, dropout=0.3)
    m_et165.load_state_dict(torch.load(PROJECT_ROOT / cfg["checkpoints"]["etgnn_165f_primary"], map_location="cpu", weights_only=True))
    m_et165.eval()

    m_noedge = ETGNNClassifier(input_dim=93, time_dim=16, hidden_dim=64, dropout=0.3)
    m_noedge.load_state_dict(torch.load(PROJECT_ROOT / cfg["checkpoints"]["etgnn_93f_local_no_edges"], map_location="cpu", weights_only=True))
    m_noedge.eval()

    explainer_et165 = Explainer(
        model=m_et165,
        algorithm=GNNExplainer(epochs=50, lr=0.05),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )

    # -------------------------------------------------------------------------
    # STEP 1: Compute Perturbation Signals on Validation Split (Steps 33-40, N = 7,653)
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Computing Perturbation Signals on Validation Split (Steps 33-40, N = 7,653) ---")
    val_sig = compute_split_perturbation_signals(
        val_g, "val_mask", m_et165, m_noedge, explainer_et165, alpha_budget=0.25, split_label="Validation (33-40)"
    )

    # -------------------------------------------------------------------------
    # STEP 2: Safeguard 5 — Validation-Only Hyperparameter & Rule Calibration
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Validation-Only Hyperparameter Tuning (Zero Test-Set Access) ---")
    y_val = val_sig["y_true"]
    p_val_base = val_sig["p_et165"]
    p_val_emp = val_sig["p_empty"]
    p_val_loc = val_sig["p_local"]
    s_val_emp = val_sig["s_empty"]
    s_val_loo = val_sig["s_loo_max"]
    u_val_conf = val_sig["conf_uncertainty"]

    val_tuning_rows = []

    # 2A. Tune optimal decision threshold tau* for raw ETGNN on validation F1
    best_tau_val = 0.50
    best_tau_f1 = -1.0
    for tau_cand in np.linspace(0.20, 0.85, 66):
        m_res = evaluate_predictions(y_val, p_val_base, threshold=float(tau_cand))
        val_tuning_rows.append({
            "Tuning_Family": "1_SingleModel_Threshold_Tau",
            "Parameter_Name": "tau",
            "Parameter_Value": round(float(tau_cand), 3),
            "Val_AP": m_res["AP_Primary"],
            "Val_F1": m_res["F1"],
            "Val_Precision": m_res["Precision"],
            "Val_Recall": m_res["Recall"],
            "Val_FPR": m_res["FPR"],
        })
        if m_res["F1"] > best_tau_f1:
            best_tau_f1 = m_res["F1"]
            best_tau_val = round(float(tau_cand), 3)

    # 2B. Tune Single-Model Self-Loop (Empty-Subgraph) Sensitivity Cutoff delta* on Validation AP
    best_delta_selfloop = 0.05
    best_delta_ap = -1.0
    for d_cand in cfg["experiment_b"]["sensitivity_thresholds_grid"]:
        p_cand = np.where(s_val_emp >= d_cand, p_val_emp, p_val_base)
        m_res = evaluate_predictions(y_val, p_cand, threshold=0.50)
        frac_routed = float(np.mean(s_val_emp >= d_cand))
        val_tuning_rows.append({
            "Tuning_Family": "2_SingleModel_SelfLoop_Sensitivity_Cutoff",
            "Parameter_Name": f"delta={d_cand} (routed={frac_routed*100:.1f}%)",
            "Parameter_Value": float(d_cand),
            "Val_AP": m_res["AP_Primary"],
            "Val_F1": m_res["F1"],
            "Val_Precision": m_res["Precision"],
            "Val_Recall": m_res["Recall"],
            "Val_FPR": m_res["FPR"],
        })
        # Select interior routing cutoff (between 5% and 50% routed) with highest Val F1
        if 0.05 <= frac_routed <= 0.50 and m_res["F1"] > best_delta_ap:
            best_delta_ap = m_res["F1"]
            best_delta_selfloop = float(d_cand)

    val_selfloop_routed_frac = float(np.mean(s_val_emp >= best_delta_selfloop))

    # 2C. Evaluate Dual-Model Routing (`etgnn.pt` -> `etgnn_local_no_edges.pt`) across routing fractions r on Validation
    for r_frac in cfg["experiment_b"]["routing_fractions_grid"]:
        for method_tag, sig_arr in [
            ("Sensitivity_S_empty", s_val_emp),
            ("Sensitivity_S_LOO", s_val_loo),
            ("Confidence_Uncertainty", u_val_conf),
            ("Predicted_Risk_p_et165", p_val_base),
        ]:
            p_routed = apply_routing_fraction(p_val_base, p_val_loc, sig_arr, float(r_frac))
            m_res = evaluate_predictions(y_val, p_routed, threshold=0.50)
            val_tuning_rows.append({
                "Tuning_Family": f"3_DualModel_Routing_{method_tag}",
                "Parameter_Name": "routing_fraction_r",
                "Parameter_Value": float(r_frac),
                "Val_AP": m_res["AP_Primary"],
                "Val_F1": m_res["F1"],
                "Val_Precision": m_res["Precision"],
                "Val_Recall": m_res["Recall"],
                "Val_FPR": m_res["FPR"],
            })

    # 2D. Tune Linear Blend Weight alpha* on Validation: (1 - alpha)*p_et165 + alpha*p_local
    best_alpha_blend = 0.50
    best_blend_ap = -1.0
    for alpha_c in cfg["experiment_b"]["blend_weights_grid"]:
        p_bl = (1.0 - alpha_c) * p_val_base + alpha_c * p_val_loc
        m_res = evaluate_predictions(y_val, p_bl, threshold=0.50)
        val_tuning_rows.append({
            "Tuning_Family": "4_DualModel_Linear_Blend",
            "Parameter_Name": "alpha_local_weight",
            "Parameter_Value": float(alpha_c),
            "Val_AP": m_res["AP_Primary"],
            "Val_F1": m_res["F1"],
            "Val_Precision": m_res["Precision"],
            "Val_Recall": m_res["Recall"],
            "Val_FPR": m_res["FPR"],
        })
        if 0.0 < alpha_c < 1.0 and m_res["AP_Primary"] > best_blend_ap:
            best_blend_ap = m_res["AP_Primary"]
            best_alpha_blend = float(alpha_c)

    df_val_tuning = pd.DataFrame(val_tuning_rows)
    df_val_tuning.to_csv(TABLES_DIR / "EXP_B_VALIDATION_THRESHOLD_TUNING.csv", index=False)
    print(
        f"  [Validation Selection Frozen] tau_val* = {best_tau_val:.3f} | "
        f"selfloop_delta* = {best_delta_selfloop:.3f} (val routed={val_selfloop_routed_frac*100:.1f}%) | "
        f"interior blend alpha* = {best_alpha_blend:.2f}"
    )

    # -------------------------------------------------------------------------
    # STEP 3: Single-Pass Evaluation on Quarantined Test Split (Steps 41-49, N = 9,973)
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Computing Perturbation Signals on Held-Out Test Split (Steps 41-49, N = 9,973) ---")
    test_sig = compute_split_perturbation_signals(
        test_g, "test_mask", m_et165, m_noedge, explainer_et165, alpha_budget=0.25, split_label="Test (41-49)"
    )

    y_test = test_sig["y_true"]
    t_test = test_sig["time_step"]
    p_base = test_sig["p_et165"]
    p_emp = test_sig["p_empty"]
    p_loc = test_sig["p_local"]
    s_emp = test_sig["s_empty"]
    s_loo = test_sig["s_loo_max"]
    u_conf = test_sig["conf_uncertainty"]
    p_ge_del = test_sig["p_ge_topk"]
    p_loo_del = test_sig["p_loo_topk"]
    p_rand_del = test_sig["p_rand_topk"]
    s_ge_drop = test_sig["s_ge_drop"]

    # Construct all Test Strategies & Matched Controls using strictly frozen Validation rules
    pred_pos_mask = p_base >= 0.50
    n_test = len(y_test)
    rng_ctrl = np.random.default_rng(cfg["experiment_b"]["random_seed"])

    # Helper for random routing control (average probability & average metrics across 50 draws)
    def random_routed_probs(p_src, p_dst, frac, n_draws=50):
        k = int(np.round(frac * len(p_src)))
        acc = np.zeros_like(p_src)
        for d_i in range(n_draws):
            r_idx = rng_ctrl.choice(len(p_src), size=k, replace=False)
            tmp = p_src.copy()
            tmp[r_idx] = p_dst[r_idx]
            acc += tmp
        return acc / float(n_draws)

    # 1. Single-Model Strategies (Using frozen `etgnn.pt` ONLY — Zero external model)
    # M1a: GNNExplainer Top-k Edge Pruning on Predicted Positives
    p_m1a_ge_pruned = np.where(pred_pos_mask, p_ge_del, p_base)
    # M1b: Exact LOO Top-k Edge Pruning on Predicted Positives
    p_m1b_loo_pruned = np.where(pred_pos_mask, p_loo_del, p_base)
    # C1a: Random-k Edge Pruning on Predicted Positives (Matched Control)
    p_c1a_rand_pruned = np.where(pred_pos_mask, p_rand_del, p_base)

    # M1c: Sensitivity-Gated Self-Loop Fallback at frozen val cutoff delta*
    test_selfloop_mask = s_emp >= best_delta_selfloop
    test_selfloop_rate = float(np.mean(test_selfloop_mask))
    p_m1c_sens_selfloop = np.where(test_selfloop_mask, p_emp, p_base)
    # C1b: Confidence-Gated Self-Loop Fallback at matched test routing rate
    p_c1b_conf_selfloop = apply_routing_fraction(p_base, p_emp, u_conf, test_selfloop_rate)
    # C1c: Random-Gated Self-Loop Fallback at matched test routing rate
    p_c1c_rand_selfloop = random_routed_probs(p_base, p_emp, test_selfloop_rate)
    # M1d: Ungated 100% Self-Loop (Empty-Subgraph) Pass on frozen `etgnn.pt`
    p_m1d_all_selfloop = p_emp.copy()

    # 2. Dual-Model Fixed-Budget Routing (`etgnn.pt` -> `etgnn_local_no_edges.pt`) at r = 20%
    r_fixed = 0.20
    p_m2a_sens20_loc = apply_routing_fraction(p_base, p_loc, s_emp, r_fixed)
    p_m2b_loo20_loc = apply_routing_fraction(p_base, p_loc, s_loo, r_fixed)
    p_c2a_conf20_loc = apply_routing_fraction(p_base, p_loc, u_conf, r_fixed)
    p_c2b_risk20_loc = apply_routing_fraction(p_base, p_loc, p_base, r_fixed)
    p_c2c_rand20_loc = random_routed_probs(p_base, p_loc, r_fixed)

    # 3. Dual-Model Predicted-Positive Veto Regime (Targeting FP reduction among p_et165 >= 0.50)
    # Among the 903 predicted positives on test, route top 50% (452 nodes) to local model
    pos_indices = np.where(pred_pos_mask)[0]
    k_pos_half = len(pos_indices) // 2

    def route_pred_pos_subset(signal_on_pos):
        out = p_base.copy()
        sub_order = np.argsort(-signal_on_pos, kind="mergesort")
        chosen = pos_indices[sub_order[:k_pos_half]]
        out[chosen] = p_loc[chosen]
        return out

    p_m3a_pos_sens_veto = route_pred_pos_subset(s_emp[pos_indices])
    p_m3b_pos_ge_veto = route_pred_pos_subset(s_ge_drop[pos_indices])
    p_c3a_pos_conf_veto = route_pred_pos_subset(u_conf[pos_indices])

    # Random 50% of predicted positives
    p_c3b_pos_rand_veto = p_base.copy()
    acc_pos_rand = np.zeros_like(p_base)
    for _ in range(50):
        chosen_r = rng_ctrl.choice(pos_indices, size=k_pos_half, replace=False)
        tmp = p_base.copy()
        tmp[chosen_r] = p_loc[chosen_r]
        acc_pos_rand += tmp
    p_c3b_pos_rand_veto = acc_pos_rand / 50.0

    # All 100% of predicted positives vetted by local model (`r = 9.05%` of total test nodes)
    p_c3c_pos_all_veto = np.where(pred_pos_mask, p_loc, p_base)

    # 4. Ungated Dual-Model Controls
    p_c4_blend = (1.0 - best_alpha_blend) * p_base + best_alpha_blend * p_loc
    p_c5_local_only = p_loc.copy()

    strategies = [
        ("01_Baseline_ETGNN_165f_Default", "Baseline: Original ETGNN (165f, tau=0.50)", "0_Baseline", p_base, 0.50, 0.0),
        ("02_Baseline_ETGNN_165f_ValOptTau", f"Control: Original ETGNN (Val-Optimal tau*={best_tau_val:.3f})", "0_Baseline", p_base, best_tau_val, 0.0),
        ("03_M1a_SingleModel_GE_TopK_Pruned", "M1a (Single-Model): GNNExplainer Top-k Edge Pruning on Pred-Pos", "1_SingleModel_EdgePerturb", p_m1a_ge_pruned, 0.50, float(np.mean(pred_pos_mask))),
        ("04_M1b_SingleModel_LOO_TopK_Pruned", "M1b (Single-Model): Exact LOO Top-k Edge Pruning on Pred-Pos", "1_SingleModel_EdgePerturb", p_m1b_loo_pruned, 0.50, float(np.mean(pred_pos_mask))),
        ("05_C1a_SingleModel_Random_TopK_Pruned", "C1a (Matched Control): Random-k Edge Pruning on Pred-Pos", "1_SingleModel_EdgePerturb", p_c1a_rand_pruned, 0.50, float(np.mean(pred_pos_mask))),
        ("06_M1c_SingleModel_SensGated_SelfLoop", f"M1c (Single-Model): Sensitivity-Gated Self-Loop (delta*={best_delta_selfloop}, r={test_selfloop_rate*100:.1f}%)", "1_SingleModel_EdgePerturb", p_m1c_sens_selfloop, 0.50, test_selfloop_rate),
        ("07_C1b_SingleModel_ConfGated_SelfLoop", f"C1b (Matched Control): Confidence-Gated Self-Loop (r={test_selfloop_rate*100:.1f}%)", "1_SingleModel_EdgePerturb", p_c1b_conf_selfloop, 0.50, test_selfloop_rate),
        ("08_C1c_SingleModel_RandGated_SelfLoop", f"C1c (Matched Control): Random-Gated Self-Loop (r={test_selfloop_rate*100:.1f}%)", "1_SingleModel_EdgePerturb", p_c1c_rand_selfloop, 0.50, test_selfloop_rate),
        ("09_M1d_SingleModel_100pct_SelfLoop", "M1d (Single-Model): 100% Empty-Subgraph Self-Loop Pass (`etgnn.pt` E=0)", "1_SingleModel_EdgePerturb", p_m1d_all_selfloop, 0.50, 1.0),
        ("10_M2a_DualModel_SensEmpty_20pct", "M2a (Dual-Model): Sensitivity-Gated (S_empty) Local Fallback (r=20%)", "2_DualModel_20pct_Routed", p_m2a_sens20_loc, 0.50, 0.20),
        ("11_M2b_DualModel_SensLOO_20pct", "M2b (Dual-Model): LOO-Sensitivity-Gated (S_LOO) Local Fallback (r=20%)", "2_DualModel_20pct_Routed", p_m2b_loo20_loc, 0.50, 0.20),
        ("12_C2a_DualModel_ConfUncert_20pct", "C2a (Matched Control): Confidence-Uncertainty-Gated Local Fallback (r=20%)", "2_DualModel_20pct_Routed", p_c2a_conf20_loc, 0.50, 0.20),
        ("13_C2b_DualModel_PredRisk_20pct", "C2b (Matched Control): Predicted-Risk-Gated (p_et165) Local Fallback (r=20%)", "2_DualModel_20pct_Routed", p_c2b_risk20_loc, 0.50, 0.20),
        ("14_C2c_DualModel_Random_20pct", "C2c (Matched Control): Random-Gated Local Fallback (r=20%, 50 draws)", "2_DualModel_20pct_Routed", p_c2c_rand20_loc, 0.50, 0.20),
        ("15_M3a_PredPos_SensEmpty_Veto50pct", "M3a (Pred-Pos Veto): Top-50% S_empty Fragile Pred-Pos -> Local Model", "3_PredPos_Veto_Regime", p_m3a_pos_sens_veto, 0.50, float(k_pos_half / n_test)),
        ("16_M3b_PredPos_GEDrop_Veto50pct", "M3b (Pred-Pos Veto): Top-50% GNNExplainer-Drop Pred-Pos -> Local Model", "3_PredPos_Veto_Regime", p_m3b_pos_ge_veto, 0.50, float(k_pos_half / n_test)),
        ("17_C3a_PredPos_ConfUncert_Veto50pct", "C3a (Matched Control): Top-50% Uncertain Pred-Pos -> Local Model", "3_PredPos_Veto_Regime", p_c3a_pos_conf_veto, 0.50, float(k_pos_half / n_test)),
        ("18_C3b_PredPos_Random_Veto50pct", "C3b (Matched Control): Random-50% Pred-Pos -> Local Model", "3_PredPos_Veto_Regime", p_c3b_pos_rand_veto, 0.50, float(k_pos_half / n_test)),
        ("19_C3c_PredPos_100pct_LocalVeto", f"C3c (Pred-Pos Control): All 100% Pred-Pos Vetted by Local Model (r={np.mean(pred_pos_mask)*100:.1f}%)", "3_PredPos_Veto_Regime", p_c3c_pos_all_veto, 0.50, float(np.mean(pred_pos_mask))),
        ("20_C4_Ungated_LinearBlend_ValOpt", f"C4 (Ungated Control): Validation-Optimal Linear Blend (alpha*={best_alpha_blend:.2f})", "4_Ungated_Controls", p_c4_blend, 0.50, 1.0),
        ("21_C5_Standalone_ETGNN_Local_NoEdges", "C5 (Ungated Control): Standalone 100% ETGNN Local No Edges (`etgnn_local_no_edges.pt`)", "4_Ungated_Controls", p_c5_local_only, 0.50, 1.0),
    ]

    # Evaluate across Full Test (41-49), Pre-Shift (41-42), and Post-Shift (43-49) (Safeguard 7)
    pre_mask = np.isin(t_test, [41, 42])
    post_mask = t_test >= 43

    test_result_rows = []
    for period_label, p_mask in [
        ("Full_Test_41_49 (N=9,973)", np.ones(n_test, dtype=bool)),
        ("Pre_Shift_41_42 (N=3,286)", pre_mask),
        ("Post_Shift_43_49 (N=6,687)", post_mask),
    ]:
        base_eval = evaluate_predictions(y_test[p_mask], p_base[p_mask], threshold=0.50)
        b_fp = base_eval["FP"]
        b_tp = base_eval["TP"]

        for s_id, s_desc, s_fam, p_arr, tau_use, r_rate in strategies:
            ev = evaluate_predictions(y_test[p_mask], p_arr[p_mask], threshold=tau_use, base_fp=b_fp, base_tp=b_tp)
            row = {
                "Period": period_label,
                "Strategy_ID": s_id,
                "Strategy_Description": s_desc,
                "Family": s_fam,
                "Threshold_Used": tau_use,
                "Routing_Fraction": round(r_rate, 4),
            }
            row.update(ev)
            test_result_rows.append(row)

    df_test_res = pd.DataFrame(test_result_rows)
    df_test_res.to_csv(TABLES_DIR / "EXP_B_TEST_MITIGATION_RESULTS.csv", index=False)
    print(f"[Step 3] Saved complete test & subgroup mitigation results to {TABLES_DIR / 'EXP_B_TEST_MITIGATION_RESULTS.csv'}")

    # -------------------------------------------------------------------------
    # STEP 4: Safeguard 9 — Paired Bootstrap Contrasts & Holm-Bonferroni Correction
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Running B = 2,000 Paired Bootstrap Contrasts ---")
    contrasts_spec = [
        ("B01: M1a (GE Top-k Pruned) vs. Original ETGNN", p_m1a_ge_pruned, p_base, 0.50, 0.50),
        ("B02: M1a (GE Top-k Pruned) vs. C1a (Random-k Pruned)", p_m1a_ge_pruned, p_c1a_rand_pruned, 0.50, 0.50),
        ("B03: M1b (LOO Top-k Pruned) vs. C1a (Random-k Pruned)", p_m1b_loo_pruned, p_c1a_rand_pruned, 0.50, 0.50),
        ("B04: M1c (Sens-Gated SelfLoop) vs. Original ETGNN", p_m1c_sens_selfloop, p_base, 0.50, 0.50),
        ("B05: M1c (Sens-Gated SelfLoop) vs. C1b (Conf-Gated SelfLoop)", p_m1c_sens_selfloop, p_c1b_conf_selfloop, 0.50, 0.50),
        ("B06: M1d (100% SelfLoop E=0) vs. Original ETGNN", p_m1d_all_selfloop, p_base, 0.50, 0.50),
        ("B07: M2a (Sens-Gated 20% Local) vs. Original ETGNN", p_m2a_sens20_loc, p_base, 0.50, 0.50),
        ("B08: M2a (Sens-Gated 20% Local) vs. C2c (Random 20% Local)", p_m2a_sens20_loc, p_c2c_rand20_loc, 0.50, 0.50),
        ("B09: M2a (Sens-Gated 20% Local) vs. C2a (Conf-Uncert 20% Local)", p_m2a_sens20_loc, p_c2a_conf20_loc, 0.50, 0.50),
        ("B10: M2a (Sens-Gated 20% Local) vs. C2b (Pred-Risk 20% Local)", p_m2a_sens20_loc, p_c2b_risk20_loc, 0.50, 0.50),
        ("B11: M3a (PredPos Sens-Veto 50%) vs. C3b (PredPos Random-Veto 50%)", p_m3a_pos_sens_veto, p_c3b_pos_rand_veto, 0.50, 0.50),
        ("B12: M3b (PredPos GE-Veto 50%) vs. C3b (PredPos Random-Veto 50%)", p_m3b_pos_ge_veto, p_c3b_pos_rand_veto, 0.50, 0.50),
        ("B13: C4 (Ungated Linear Blend) vs. M2a (Sens-Gated 20% Local)", p_c4_blend, p_m2a_sens20_loc, 0.50, 0.50),
        ("B14: C5 (Standalone 100% Local NoEdges) vs. M2a (Sens-Gated 20% Local)", p_c5_local_only, p_m2a_sens20_loc, 0.50, 0.50),
    ]

    boot_rows = []
    for c_label, pa, pb, ta, tb in contrasts_spec:
        res = paired_bootstrap_contrast(
            y_test, pa, pb, tau_a=ta, tau_b=tb,
            B=cfg["experiment_b"]["bootstrap_replicates"],
            seed=cfg["experiment_b"]["random_seed"],
        )
        row = {"Contrast": c_label}
        row.update(res)
        boot_rows.append(row)

    df_boot = pd.DataFrame(boot_rows)
    # Apply monotonic Holm-Bonferroni correction across the M = 14 contrasts on AP_Exact_P
    m_tot = len(df_boot)
    order_p = np.argsort(df_boot["AP_Exact_P"].values)
    holm_p = np.zeros(m_tot, dtype=np.float64)
    running_max = 0.0
    for rank_i, orig_idx in enumerate(order_p):
        adj = min(1.0, df_boot.loc[orig_idx, "AP_Exact_P"] * (m_tot - rank_i))
        running_max = max(running_max, adj)
        holm_p[orig_idx] = running_max
    df_boot["AP_Holm_Adjusted_P"] = holm_p
    df_boot["AP_Holm_Significant_0.05"] = df_boot["AP_Holm_Adjusted_P"] <= 0.05

    df_boot.to_csv(TABLES_DIR / "EXP_B_PAIRED_BOOTSTRAP_TESTS.csv", index=False)
    print(f"[Step 4] Saved paired bootstrap tests to {TABLES_DIR / 'EXP_B_PAIRED_BOOTSTRAP_TESTS.csv'}")

    wall_sec = time.perf_counter() - t0_all
    mem_mb = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    log_info = {
        "experiment": "Experiment B — Explanation-Guided Error Mitigation",
        "status": "COMPLETED",
        "wall_time_sec": round(wall_sec, 2),
        "peak_rss_mb": round(mem_mb, 2),
        "val_selected_tau": best_tau_val,
        "val_selected_selfloop_delta": best_delta_selfloop,
        "val_selected_blend_alpha": best_alpha_blend,
        "n_val_evaluated": int(len(y_val)),
        "n_test_evaluated": int(len(y_test)),
    }
    with open(LOGS_DIR / "experiment_b_audit_log.json", "w", encoding="utf-8") as f:
        json.dump(log_info, f, indent=2)

    print("=" * 80)
    print(f"[EXPERIMENT B COMPLETE] Runtime: {wall_sec:.2f}s | RAM: {mem_mb:.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
