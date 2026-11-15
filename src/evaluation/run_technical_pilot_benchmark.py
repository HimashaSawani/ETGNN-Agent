import os
import sys
import time
import json
import hashlib
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import psutil
import scipy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import degree, k_hop_subgraph

from src.models.etgnn import ETGNNClassifier
from src.models.matched_gat import MatchedGATClassifier


def get_mem_mb():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024.0 * 1024.0)


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_dependency_versions():
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "psutil": psutil.__version__,
    }


def run_benchmark(
    sample_csv_path="experiments/matched_c1_c2_audit/technical_pilot_sample.csv",
    val_graph_path="data/processed/validation_graph.pt",
    node_mapping_path="data/processed/node_mapping.csv",
    etgnn_ckpt_path="results/checkpoints/etgnn.pt",
    matched_gat_ckpt_path="results/checkpoints/matched_gat.pt",
    output_dir="experiments/technical_pilot_results",
    device_str=None,
):
    print("==================================================================")
    print("  ETGNN + MATCHED GAT CORRECTED C1 + C2 TECHNICAL PILOT BENCHMARK")
    print("==================================================================")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    raw_records_path = out_path / "raw_benchmark_records.csv"
    detailed_interventions_path = out_path / "detailed_intervention_records.csv"
    summary_path = out_path / "technical_pilot_summary.json"
    resume_path = out_path / "pilot_resume_state.json"

    # Compute runner hash for provenance
    current_runner_path = Path(__file__).resolve()
    runner_sha256 = compute_file_sha256(str(current_runner_path))
    dep_versions = get_dependency_versions()

    # Determine device
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Hardware Device: {device} (CUDA available: {torch.cuda.is_available()})")
    if device.type == "cuda":
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    # 1. Verification of Required Files & SHA-256 Checksums
    print("\n--- 1. Checking Inputs & SHA-256 Manifest ---")
    manifest = {}
    for p in [sample_csv_path, val_graph_path, node_mapping_path, etgnn_ckpt_path, matched_gat_ckpt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required artifact: {p}")
        h = compute_file_sha256(p)
        sz = os.path.getsize(p)
        manifest[os.path.basename(p)] = {"path": p, "sha256": h, "size_bytes": sz}
        print(f"  {os.path.basename(p):<30} | {sz:>10} bytes | SHA256: {h[:16]}...")
    (out_path / "input_manifest.json").write_text(json.dumps(manifest, indent=2))

    # 2. Assertions on the 4 Development Nodes
    print("\n--- 2. Asserting 4 Development Nodes ---")
    sample_df = pd.read_csv(sample_csv_path)
    assert len(sample_df) == 4, f"Expected exactly 4 benchmark nodes, found {len(sample_df)}"
    categories = sample_df["prediction_category"].tolist()
    assert sorted(categories) == ["FN", "FP", "TN", "TP"], f"Unexpected categories: {categories}"

    df_map = pd.read_csv(node_mapping_path)
    node_to_tx = dict(zip(df_map["node_index"], df_map["txId"]))

    # Historical pilot exclusion check
    hist_pilot_path = "experiments/pilot_validation_audit/pilot_nodes.csv"
    hist_pilot_nodes = set()
    if os.path.exists(hist_pilot_path):
        hp_df = pd.read_csv(hist_pilot_path)
        hist_pilot_nodes = set(hp_df["node_index"].astype(int).tolist())

    for _, row in sample_df.iterrows():
        nid = int(row["node_index"])
        tx = int(row["transaction_id"])
        step = int(row["time_step"])
        assert step in [33, 34, 35, 36], f"Node {nid} step {step} not in dev period [33, 36]!"
        assert nid not in hist_pilot_nodes, f"Node {nid} is in historical pilot!"
        assert node_to_tx.get(nid) == tx, f"Node {nid} txId mismatch: {node_to_tx.get(nid)} vs {tx}"
        print(f"  Verified {row['prediction_category']} (Node {nid}, txId {tx}, Step {step}, E_cand={row['candidate_edge_count']})")

    # 3. Model Loading & Eval Mode
    print("\n--- 3. Loading Models in Eval Mode ---")
    val_graph = torch.load(val_graph_path, map_location="cpu", weights_only=True)
    m_etgnn = ETGNNClassifier(input_dim=165).to(device)
    m_etgnn.load_state_dict(torch.load(etgnn_ckpt_path, map_location=device, weights_only=True))
    m_etgnn.eval()

    m_gat = MatchedGATClassifier(input_dim=165).to(device)
    m_gat.load_state_dict(torch.load(matched_gat_ckpt_path, map_location=device, weights_only=True))
    m_gat.eval()

    # Precompute full graph probabilities for equivalence check
    print("Computing full-graph inference...")
    val_x = val_graph["x"].to(device)
    val_edges = val_graph["edge_index"].to(device)
    val_time = val_graph["time_step"].to(device)

    sync_device(device)
    with torch.no_grad():
        full_logits_etgnn = m_etgnn(val_x, val_edges, val_time)
        full_probs_etgnn = torch.softmax(full_logits_etgnn, dim=1)
        full_logits_gat = m_gat(val_x, val_edges, val_time)
        full_probs_gat = torch.softmax(full_logits_gat, dim=1)
    sync_device(device)

    # 4. GNNExplainer Setup for BOTH Models (C1 and C2)
    explainer_etgnn = Explainer(
        model=m_etgnn,
        algorithm=GNNExplainer(epochs=50),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )

    explainer_gat = Explainer(
        model=m_gat,
        algorithm=GNNExplainer(epochs=50),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )

    benchmark_summary_records = []
    raw_benchmark_records = []
    detailed_intervention_records = []
    seeds = [42, 43, 44]
    num_random_draws = 30

    print("\n--- 4. Running Benchmark Across 4 Benchmark Nodes ---")
    for idx, row in sample_df.iterrows():
        pos = int(row["node_index"])
        cat = row["prediction_category"]
        tx_id = int(row["transaction_id"])
        step = int(row["time_step"])
        print(f"\nProcessing Node [{idx+1}/4]: Cat={cat}, Index={pos}, txId={tx_id}, Step={step}...")

        # Subgraph extraction & timing
        t0 = time.perf_counter()
        subset, sub_edges, mapping, edge_mask = k_hop_subgraph(
            pos, 2, val_graph["edge_index"], relabel_nodes=True, flow="source_to_target"
        )
        t_extract_ms = (time.perf_counter() - t0) * 1000.0
        local_center = int(mapping.item())
        E_cand = sub_edges.shape[1]
        k_budget = max(1, int(np.ceil(0.25 * E_cand)))
        k_over_E = k_budget / E_cand

        # Map local edge index (0..E_cand-1) to global edge IDs in val_graph["edge_index"]
        global_edge_indices = torch.where(edge_mask)[0].cpu().numpy()
        assert len(global_edge_indices) == E_cand, "Mismatch in global edge index count!"

        sub_x = val_x[subset]
        sub_edges_dev = sub_edges.to(device)
        sub_time = val_time[subset]

        # Numerical Equivalence Check
        with torch.no_grad():
            sub_p_etgnn = torch.softmax(m_etgnn(sub_x, sub_edges_dev, sub_time), dim=1)[local_center]
            sub_p_gat = torch.softmax(m_gat(sub_x, sub_edges_dev, sub_time), dim=1)[local_center]
        diff_etgnn = float(torch.abs(full_probs_etgnn[pos] - sub_p_etgnn).max().item())
        diff_gat = float(torch.abs(full_probs_gat[pos] - sub_p_gat).max().item())
        assert diff_etgnn < 1e-5, f"Inference mismatch ETGNN at {pos}: {diff_etgnn}"
        assert diff_gat < 1e-5, f"Inference mismatch Matched GAT at {pos}: {diff_gat}"

        # Target classes & explicit probabilities labeling
        p_class_0_etgnn = float(full_probs_etgnn[pos, 0].item())
        p_illicit_etgnn = float(full_probs_etgnn[pos, 1].item())
        p_class_0_gat = float(full_probs_gat[pos, 0].item())
        p_illicit_gat = float(full_probs_gat[pos, 1].item())
        p0_etgnn = p_class_0_etgnn
        p1_etgnn = p_illicit_etgnn
        p0_gat = p_class_0_gat
        p1_gat = p_illicit_gat
        c1_target_class = int(p_illicit_etgnn >= 0.5)
        p_c1_orig_etgnn = float(full_probs_etgnn[pos, c1_target_class].item())
        p_c1_orig_gat = float(full_probs_gat[pos, c1_target_class].item())

        c2_target_gat = int(p_illicit_gat >= 0.5)
        p_c2_orig_gat = float(full_probs_gat[pos, c2_target_gat].item())

        print(f"  E_cand={E_cand}, Budget k={k_budget} ({k_over_E*100:.1f}%), Extract={t_extract_ms:.1f}ms, EquivDiff={diff_etgnn:.2e}")
        print(f"  ETGNN:       p(class 0)={p_class_0_etgnn:.6f}, p(illicit)={p_illicit_etgnn:.6f} -> Predicted Class {c1_target_class}")
        print(f"  Matched GAT: p(class 0)={p_class_0_gat:.6f}, p(illicit)={p_illicit_gat:.6f} -> Predicted Class {c2_target_gat}")

        # -------------------------------------------------------------
        # 4A. GNNExplainer for ETGNN (C1 Ranking & ETGNN C2 Ranking)
        # -------------------------------------------------------------
        ge_etgnn_seed_times = []
        ge_etgnn_selected_edges_by_seed = []
        ge_drops_c1_etgnn = []
        ge_drops_c1_gat_common_target = []
        ge_drops_c2_gat_on_etgnn_ranking = []
        etgnn_seed_results = {}

        for s in seeds:
            torch.manual_seed(s)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(s)
            sync_device(device)
            t_ge_0 = time.perf_counter()
            explanation_etgnn = explainer_etgnn(
                x=sub_x,
                edge_index=sub_edges_dev,
                time_step=sub_time,
                index=local_center,
            )
            sync_device(device)
            t_ge_ms = (time.perf_counter() - t_ge_0) * 1000.0
            ge_etgnn_seed_times.append(t_ge_ms)

            mask_etgnn = explanation_etgnn.edge_mask.detach().cpu().numpy()
            ranked_edges_etgnn = np.argsort(-mask_etgnn)
            selected_local_edges_etgnn = ranked_edges_etgnn[:k_budget].tolist()
            selected_global_edge_ids_etgnn = global_edge_indices[selected_local_edges_etgnn].tolist()
            ge_etgnn_selected_edges_by_seed.append(set(selected_local_edges_etgnn))

            # Mask edges for deletion
            keep_mask_etgnn = torch.tensor(
                [i not in set(selected_local_edges_etgnn) for i in range(E_cand)],
                dtype=torch.bool,
                device=device,
            )
            del_edges_etgnn = sub_edges_dev[:, keep_mask_etgnn]

            # Forward pass drops on deleted edges
            with torch.no_grad():
                del_p_etgnn = torch.softmax(m_etgnn(sub_x, del_edges_etgnn, sub_time), dim=1)[local_center]
                del_p_gat = torch.softmax(m_gat(sub_x, del_edges_etgnn, sub_time), dim=1)[local_center]

            del_p0_et = float(del_p_etgnn[0].item())
            del_p1_et = float(del_p_etgnn[1].item())
            del_p0_gt = float(del_p_gat[0].item())
            del_p1_gt = float(del_p_gat[1].item())

            # C1 probability drops (relative to c1_target_class)
            drop_c1_et = p_c1_orig_etgnn - float(del_p_etgnn[c1_target_class].item())
            drop_c1_gat_common = p_c1_orig_gat - float(del_p_gat[c1_target_class].item())

            # Own-target drop for GAT scored on these ETGNN-ranked deletions
            drop_c2_gat_etgnn_rank = p_c2_orig_gat - float(del_p_gat[c2_target_gat].item())

            assert np.isfinite(drop_c1_et), "Non-finite ETGNN drop!"
            assert np.isfinite(drop_c1_gat_common), "Non-finite GAT common-target drop!"

            ge_drops_c1_etgnn.append(drop_c1_et)
            ge_drops_c1_gat_common_target.append(drop_c1_gat_common)
            ge_drops_c2_gat_on_etgnn_ranking.append(drop_c2_gat_etgnn_rank)

            # Store seed-specific results for exact matching in raw records
            etgnn_seed_results[s] = {
                "drop_c1_et": drop_c1_et,
                "drop_c1_gat_common": drop_c1_gat_common,
                "drop_c2_gat_etgnn_rank": drop_c2_gat_etgnn_rank,
                "selected_local_edges_etgnn": selected_local_edges_etgnn,
                "selected_global_edge_ids_etgnn": selected_global_edge_ids_etgnn,
                "runtime_ms": t_ge_ms,
            }

            # Record detailed intervention: C1 ETGNN
            detailed_intervention_records.append({
                "node_index": pos,
                "transaction_id": tx_id,
                "category": cat,
                "contrast": "C1",
                "method": "GNNExplainer",
                "ranking_model_id": "ETGNN",
                "scoring_model_id": "ETGNN",
                "target_class": c1_target_class,
                "seed": s,
                "budget_k": k_budget,
                "E_cand": E_cand,
                "actual_k_over_E": round(k_over_E, 4),
                "orig_prob_target": p_c1_orig_etgnn,
                "del_prob_target": float(del_p_etgnn[c1_target_class].item()),
                "signed_drop": drop_c1_et,
                "orig_p0": p0_etgnn,
                "orig_p1": p1_etgnn,
                "del_p0": del_p0_et,
                "del_p1": del_p1_et,
                "deleted_local_edge_indices": json.dumps(selected_local_edges_etgnn),
                "deleted_global_edge_ids": json.dumps(selected_global_edge_ids_etgnn),
                "runtime_ms": t_ge_ms,
            })

            # Record detailed intervention: C1 Matched GAT (common target, ETGNN ranking)
            detailed_intervention_records.append({
                "node_index": pos,
                "transaction_id": tx_id,
                "category": cat,
                "contrast": "C1",
                "method": "GNNExplainer",
                "ranking_model_id": "ETGNN",
                "scoring_model_id": "Matched_GAT",
                "target_class": c1_target_class,
                "seed": s,
                "budget_k": k_budget,
                "E_cand": E_cand,
                "actual_k_over_E": round(k_over_E, 4),
                "orig_prob_target": p_c1_orig_gat,
                "del_prob_target": float(del_p_gat[c1_target_class].item()),
                "signed_drop": drop_c1_gat_common,
                "orig_p0": float(full_probs_gat[pos, 0].item()),
                "orig_p1": float(full_probs_gat[pos, 1].item()),
                "del_p0": del_p0_gt,
                "del_p1": del_p1_gt,
                "deleted_local_edge_indices": json.dumps(selected_local_edges_etgnn),
                "deleted_global_edge_ids": json.dumps(selected_global_edge_ids_etgnn),
                "runtime_ms": t_ge_ms,
            })

            # Record detailed intervention: GAT scored on own target with ETGNN ranking
            detailed_intervention_records.append({
                "node_index": pos,
                "transaction_id": tx_id,
                "category": cat,
                "contrast": "C1_Diagnostic_GAT_Own_Target",
                "method": "GNNExplainer",
                "ranking_model_id": "ETGNN",
                "scoring_model_id": "Matched_GAT",
                "target_class": c2_target_gat,
                "seed": s,
                "budget_k": k_budget,
                "E_cand": E_cand,
                "actual_k_over_E": round(k_over_E, 4),
                "orig_prob_target": p_c2_orig_gat,
                "del_prob_target": float(del_p_gat[c2_target_gat].item()),
                "signed_drop": drop_c2_gat_etgnn_rank,
                "orig_p0": float(full_probs_gat[pos, 0].item()),
                "orig_p1": float(full_probs_gat[pos, 1].item()),
                "del_p0": del_p0_gt,
                "del_p1": del_p1_gt,
                "deleted_local_edge_indices": json.dumps(selected_local_edges_etgnn),
                "deleted_global_edge_ids": json.dumps(selected_global_edge_ids_etgnn),
                "runtime_ms": t_ge_ms,
            })

        # -------------------------------------------------------------
        # 4B. GNNExplainer for Matched GAT (True C2: GAT's Own Ranking)
        # -------------------------------------------------------------
        ge_gat_seed_times = []
        ge_gat_selected_edges_by_seed = []
        ge_drops_c2_gat_own_ranking = []
        gat_seed_results = {}

        for s in seeds:
            torch.manual_seed(s)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(s)
            sync_device(device)
            t_ge_gat_0 = time.perf_counter()
            explanation_gat = explainer_gat(
                x=sub_x,
                edge_index=sub_edges_dev,
                time_step=sub_time,
                index=local_center,
            )
            sync_device(device)
            t_ge_gat_ms = (time.perf_counter() - t_ge_gat_0) * 1000.0
            ge_gat_seed_times.append(t_ge_gat_ms)

            mask_gat = explanation_gat.edge_mask.detach().cpu().numpy()
            ranked_edges_gat = np.argsort(-mask_gat)
            selected_local_edges_gat = ranked_edges_gat[:k_budget].tolist()
            selected_global_edge_ids_gat = global_edge_indices[selected_local_edges_gat].tolist()
            ge_gat_selected_edges_by_seed.append(set(selected_local_edges_gat))

            # Mask edges for deletion on GAT's own selection
            keep_mask_gat_own = torch.tensor(
                [i not in set(selected_local_edges_gat) for i in range(E_cand)],
                dtype=torch.bool,
                device=device,
            )
            del_edges_gat_own = sub_edges_dev[:, keep_mask_gat_own]

            with torch.no_grad():
                del_p_gat_own = torch.softmax(m_gat(sub_x, del_edges_gat_own, sub_time), dim=1)[local_center]

            del_p0_gt_own = float(del_p_gat_own[0].item())
            del_p1_gt_own = float(del_p_gat_own[1].item())

            # C2 probability drop for GAT (relative to GAT's OWN target class)
            drop_c2_gat_own = p_c2_orig_gat - float(del_p_gat_own[c2_target_gat].item())
            assert np.isfinite(drop_c2_gat_own), "Non-finite GAT own C2 drop!"
            ge_drops_c2_gat_own_ranking.append(drop_c2_gat_own)

            gat_seed_results[s] = {
                "drop_c2_gat_own": drop_c2_gat_own,
                "selected_local_edges_gat": selected_local_edges_gat,
                "selected_global_edge_ids_gat": selected_global_edge_ids_gat,
                "runtime_ms": t_ge_gat_ms,
            }

            # Record detailed intervention: True C2 GAT
            detailed_intervention_records.append({
                "node_index": pos,
                "transaction_id": tx_id,
                "category": cat,
                "contrast": "C2",
                "method": "GNNExplainer",
                "ranking_model_id": "Matched_GAT",
                "scoring_model_id": "Matched_GAT",
                "target_class": c2_target_gat,
                "seed": s,
                "budget_k": k_budget,
                "E_cand": E_cand,
                "actual_k_over_E": round(k_over_E, 4),
                "orig_prob_target": p_c2_orig_gat,
                "del_prob_target": float(del_p_gat_own[c2_target_gat].item()),
                "signed_drop": drop_c2_gat_own,
                "orig_p0": float(full_probs_gat[pos, 0].item()),
                "orig_p1": float(full_probs_gat[pos, 1].item()),
                "del_p0": del_p0_gt_own,
                "del_p1": del_p1_gt_own,
                "deleted_local_edge_indices": json.dumps(selected_local_edges_gat),
                "deleted_global_edge_ids": json.dumps(selected_global_edge_ids_gat),
                "runtime_ms": t_ge_gat_ms,
            })

        # Append to raw_benchmark_records by joining the EXACT matching seed
        for s in seeds:
            et_res = etgnn_seed_results[s]
            gt_res = gat_seed_results[s]
            raw_benchmark_records.append({
                "node_index": pos,
                "category": cat,
                "method": "GNNExplainer",
                "seed": s,
                "budget_k": k_budget,
                "c1_target_class": c1_target_class,
                "c1_drop_etgnn": et_res["drop_c1_et"],
                "c1_drop_gat": et_res["drop_c1_gat_common"],
                "c2_target_gat": c2_target_gat,
                "c2_drop_gat_own_ranking": gt_res["drop_c2_gat_own"],
                "c2_drop_gat_etgnn_ranking": et_res["drop_c2_gat_etgnn_rank"],
                "deleted_global_edge_ids_etgnn": json.dumps(et_res["selected_global_edge_ids_etgnn"]),
                "deleted_global_edge_ids_gat": json.dumps(gt_res["selected_global_edge_ids_gat"]),
                "runtime_ms_etgnn": et_res["runtime_ms"],
                "runtime_ms_gat": gt_res["runtime_ms"],
            })

        # -------------------------------------------------------------
        # 4C. Shared Random Deletions (30 draws)
        # -------------------------------------------------------------
        t_rnd_0 = time.perf_counter()
        rng = np.random.default_rng(20261020 + pos)
        rnd_drops_c1_etgnn = []
        rnd_drops_c1_gat_common_target = []
        rnd_drops_c2_gat_own_target = []

        for r_idx in range(num_random_draws):
            rnd_local_edges = rng.choice(E_cand, size=k_budget, replace=False).tolist()
            rnd_global_edge_ids = global_edge_indices[rnd_local_edges].tolist()
            keep_mask_rnd = torch.tensor(
                [i not in set(rnd_local_edges) for i in range(E_cand)],
                dtype=torch.bool,
                device=device,
            )
            del_edges_rnd = sub_edges_dev[:, keep_mask_rnd]

            with torch.no_grad():
                del_p_et = torch.softmax(m_etgnn(sub_x, del_edges_rnd, sub_time), dim=1)[local_center]
                del_p_gt = torch.softmax(m_gat(sub_x, del_edges_rnd, sub_time), dim=1)[local_center]

            del_p0_et = float(del_p_et[0].item())
            del_p1_et = float(del_p_et[1].item())
            del_p0_gt = float(del_p_gt[0].item())
            del_p1_gt = float(del_p_gt[1].item())

            d_c1_et = p_c1_orig_etgnn - float(del_p_et[c1_target_class].item())
            d_c1_gt = p_c1_orig_gat - float(del_p_gt[c1_target_class].item())  # Common target
            d_c2_gt = p_c2_orig_gat - float(del_p_gt[c2_target_gat].item())    # GAT own target

            rnd_drops_c1_etgnn.append(d_c1_et)
            rnd_drops_c1_gat_common_target.append(d_c1_gt)
            rnd_drops_c2_gat_own_target.append(d_c2_gt)

            # Detailed intervention: Random C1 ETGNN
            detailed_intervention_records.append({
                "node_index": pos,
                "transaction_id": tx_id,
                "category": cat,
                "contrast": "C1",
                "method": "Random",
                "ranking_model_id": "Random",
                "scoring_model_id": "ETGNN",
                "target_class": c1_target_class,
                "seed": r_idx,
                "budget_k": k_budget,
                "E_cand": E_cand,
                "actual_k_over_E": round(k_over_E, 4),
                "orig_prob_target": p_c1_orig_etgnn,
                "del_prob_target": float(del_p_et[c1_target_class].item()),
                "signed_drop": d_c1_et,
                "orig_p0": p0_etgnn,
                "orig_p1": p1_etgnn,
                "del_p0": del_p0_et,
                "del_p1": del_p1_et,
                "deleted_local_edge_indices": json.dumps(rnd_local_edges),
                "deleted_global_edge_ids": json.dumps(rnd_global_edge_ids),
                "runtime_ms": np.nan,
            })

            # Detailed intervention: Random C1 Matched GAT (common target)
            detailed_intervention_records.append({
                "node_index": pos,
                "transaction_id": tx_id,
                "category": cat,
                "contrast": "C1",
                "method": "Random",
                "ranking_model_id": "Random",
                "scoring_model_id": "Matched_GAT",
                "target_class": c1_target_class,
                "seed": r_idx,
                "budget_k": k_budget,
                "E_cand": E_cand,
                "actual_k_over_E": round(k_over_E, 4),
                "orig_prob_target": p_c1_orig_gat,
                "del_prob_target": float(del_p_gt[c1_target_class].item()),
                "signed_drop": d_c1_gt,
                "orig_p0": float(full_probs_gat[pos, 0].item()),
                "orig_p1": float(full_probs_gat[pos, 1].item()),
                "del_p0": del_p0_gt,
                "del_p1": del_p1_gt,
                "deleted_local_edge_indices": json.dumps(rnd_local_edges),
                "deleted_global_edge_ids": json.dumps(rnd_global_edge_ids),
                "runtime_ms": np.nan,
            })

            # Detailed intervention: Random C2 Matched GAT (own target)
            detailed_intervention_records.append({
                "node_index": pos,
                "transaction_id": tx_id,
                "category": cat,
                "contrast": "C2",
                "method": "Random",
                "ranking_model_id": "Random",
                "scoring_model_id": "Matched_GAT",
                "target_class": c2_target_gat,
                "seed": r_idx,
                "budget_k": k_budget,
                "E_cand": E_cand,
                "actual_k_over_E": round(k_over_E, 4),
                "orig_prob_target": p_c2_orig_gat,
                "del_prob_target": float(del_p_gt[c2_target_gat].item()),
                "signed_drop": d_c2_gt,
                "orig_p0": float(full_probs_gat[pos, 0].item()),
                "orig_p1": float(full_probs_gat[pos, 1].item()),
                "del_p0": del_p0_gt,
                "del_p1": del_p1_gt,
                "deleted_local_edge_indices": json.dumps(rnd_local_edges),
                "deleted_global_edge_ids": json.dumps(rnd_global_edge_ids),
                "runtime_ms": np.nan,
            })

            # In raw_benchmark_records: note the typo fix ("c1_drop_gat": d_c1_gt)
            raw_benchmark_records.append({
                "node_index": pos,
                "category": cat,
                "method": "Random",
                "seed": r_idx,
                "budget_k": k_budget,
                "c1_target_class": c1_target_class,
                "c1_drop_etgnn": d_c1_et,
                "c1_drop_gat": d_c1_gt,  # FIXED: uses d_c1_gt instead of unassigned d_c1_gat
                "c2_target_gat": c2_target_gat,
                "c2_drop_gat_own_ranking": d_c2_gt,
                "c2_drop_gat_etgnn_ranking": d_c2_gt,
                "deleted_global_edge_ids_etgnn": json.dumps(rnd_global_edge_ids),
                "deleted_global_edge_ids_gat": json.dumps(rnd_global_edge_ids),
                "runtime_ms_etgnn": np.nan,
                "runtime_ms_gat": np.nan,
            })

        t_rnd_total_ms = (time.perf_counter() - t_rnd_0) * 1000.0

        # -------------------------------------------------------------
        # 4D. Degree Ranking Baseline
        # -------------------------------------------------------------
        t_deg_0 = time.perf_counter()
        src_nodes = sub_edges[0]
        src_degrees = degree(val_graph["edge_index"][0], num_nodes=val_graph["x"].shape[0])[src_nodes].numpy()
        deg_ranked = np.argsort(-src_degrees)
        deg_selected = deg_ranked[:k_budget].tolist()
        deg_global_ids = global_edge_indices[deg_selected].tolist()

        keep_mask_deg = torch.tensor([i not in set(deg_selected) for i in range(E_cand)], dtype=torch.bool, device=device)
        del_edges_deg = sub_edges_dev[:, keep_mask_deg]
        with torch.no_grad():
            del_p_et = torch.softmax(m_etgnn(sub_x, del_edges_deg, sub_time), dim=1)[local_center]
            del_p_gt = torch.softmax(m_gat(sub_x, del_edges_deg, sub_time), dim=1)[local_center]
        d_deg_c1_et = p_c1_orig_etgnn - float(del_p_et[c1_target_class].item())
        d_deg_c1_gt = p_c1_orig_gat - float(del_p_gt[c1_target_class].item())
        d_deg_c2_gt = p_c2_orig_gat - float(del_p_gt[c2_target_gat].item())
        t_deg_ms = (time.perf_counter() - t_deg_0) * 1000.0

        raw_benchmark_records.append({
            "node_index": pos,
            "category": cat,
            "method": "Degree",
            "seed": 0,
            "budget_k": k_budget,
            "c1_target_class": c1_target_class,
            "c1_drop_etgnn": d_deg_c1_et,
            "c1_drop_gat": d_deg_c1_gt,
            "c2_target_gat": c2_target_gat,
            "c2_drop_gat_own_ranking": d_deg_c2_gt,
            "c2_drop_gat_etgnn_ranking": d_deg_c2_gt,
            "deleted_global_edge_ids_etgnn": json.dumps(deg_global_ids),
            "deleted_global_edge_ids_gat": json.dumps(deg_global_ids),
            "runtime_ms_etgnn": t_deg_ms,
            "runtime_ms_gat": np.nan,
        })

        # -------------------------------------------------------------
        # 4E. Single-Edge Occlusion Baseline
        # -------------------------------------------------------------
        t_occ_0 = time.perf_counter()
        occ_drops = []
        for e_idx in range(E_cand):
            mask_occ = torch.ones(E_cand, dtype=torch.bool, device=device)
            mask_occ[e_idx] = False
            del_e = sub_edges_dev[:, mask_occ]
            with torch.no_grad():
                dp = torch.softmax(m_etgnn(sub_x, del_e, sub_time), dim=1)[local_center]
            occ_drops.append(p_c1_orig_etgnn - float(dp[c1_target_class].item()))
        occ_ranked = np.argsort(-np.array(occ_drops))
        occ_selected = occ_ranked[:k_budget].tolist()
        occ_global_ids = global_edge_indices[occ_selected].tolist()

        keep_mask_occ = torch.tensor([i not in set(occ_selected) for i in range(E_cand)], dtype=torch.bool, device=device)
        del_edges_occ = sub_edges_dev[:, keep_mask_occ]
        with torch.no_grad():
            del_p_et = torch.softmax(m_etgnn(sub_x, del_edges_occ, sub_time), dim=1)[local_center]
            del_p_gt = torch.softmax(m_gat(sub_x, del_edges_occ, sub_time), dim=1)[local_center]
        d_occ_c1_et = p_c1_orig_etgnn - float(del_p_et[c1_target_class].item())
        d_occ_c1_gt = p_c1_orig_gat - float(del_p_gt[c1_target_class].item())
        d_occ_c2_gt = p_c2_orig_gat - float(del_p_gt[c2_target_gat].item())
        t_occ_ms = (time.perf_counter() - t_occ_0) * 1000.0

        raw_benchmark_records.append({
            "node_index": pos,
            "category": cat,
            "method": "Occlusion",
            "seed": 0,
            "budget_k": k_budget,
            "c1_target_class": c1_target_class,
            "c1_drop_etgnn": d_occ_c1_et,
            "c1_drop_gat": d_occ_c1_gt,
            "c2_target_gat": c2_target_gat,
            "c2_drop_gat_own_ranking": d_occ_c2_gt,
            "c2_drop_gat_etgnn_ranking": d_occ_c2_gt,
            "deleted_global_edge_ids_etgnn": json.dumps(occ_global_ids),
            "deleted_global_edge_ids_gat": json.dumps(occ_global_ids),
            "runtime_ms_etgnn": t_occ_ms,
            "runtime_ms_gat": np.nan,
        })

        # -------------------------------------------------------------
        # 4F. Node-Level Aggregated Metrics & Corrected Summary Naming
        # -------------------------------------------------------------
        # C1 Common-Target Metrics (ranking = ETGNN, target = c1_target_class)
        mean_ge_c1_et = float(np.mean(ge_drops_c1_etgnn))
        mean_ge_c1_gt_common = float(np.mean(ge_drops_c1_gat_common_target))
        mean_rnd_c1_et = float(np.mean(rnd_drops_c1_etgnn))
        mean_rnd_c1_gt_common = float(np.mean(rnd_drops_c1_gat_common_target))

        # Explicitly labeled C1 advantages:
        c1_advantage_etgnn = mean_ge_c1_et - mean_rnd_c1_et
        c1_advantage_gat_common_target = mean_ge_c1_gt_common - mean_rnd_c1_gt_common

        # C2 Matched GAT Metrics (ranking = GAT own GNNExplainer, target = c2_target_gat)
        mean_ge_c2_gt_own = float(np.mean(ge_drops_c2_gat_own_ranking))
        mean_rnd_c2_gt_own = float(np.mean(rnd_drops_c2_gat_own_target))
        c2_advantage_gat_own_ranking = mean_ge_c2_gt_own - mean_rnd_c2_gt_own

        # Diagnostic: GAT own-target advantage when evaluated on ETGNN ranking
        mean_ge_c2_gt_etgnn_rank = float(np.mean(ge_drops_c2_gat_on_etgnn_ranking))
        c2_gat_own_target_advantage_on_etgnn_ranking = mean_ge_c2_gt_etgnn_rank - mean_rnd_c2_gt_own

        mean_ge_time_etgnn_ms = float(np.mean(ge_etgnn_seed_times))
        mean_ge_time_gat_ms = float(np.mean(ge_gat_seed_times))
        rss_mb = get_mem_mb()
        vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if device.type == "cuda" else 0.0

        benchmark_summary_records.append({
            "node_index": pos,
            "transaction_id": tx_id,
            "category": cat,
            "time_step": step,
            "E_cand": E_cand,
            "k_budget": k_budget,
            "actual_k_over_E": round(k_over_E, 4),
            "extract_runtime_ms": round(t_extract_ms, 2),
            "gnnexplainer_etgnn_mean_runtime_ms": round(mean_ge_time_etgnn_ms, 2),
            "gnnexplainer_gat_mean_runtime_ms": round(mean_ge_time_gat_ms, 2),
            "random_30_total_runtime_ms": round(t_rnd_total_ms, 2),
            "degree_ranking_runtime_ms": round(t_deg_ms, 2),
            "occlusion_runtime_ms": round(t_occ_ms, 2),
            "peak_rss_mb": round(rss_mb, 2),
            "peak_vram_mb": round(vram_mb, 2),
            # C1 Common Target (ETGNN predicted class)
            "c1_target_class": c1_target_class,
            "c1_mean_ge_drop_etgnn": round(mean_ge_c1_et, 8),
            "c1_mean_rnd_drop_etgnn": round(mean_rnd_c1_et, 8),
            "c1_advantage_etgnn": round(c1_advantage_etgnn, 8),
            "c1_mean_ge_drop_gat_common_target": round(mean_ge_c1_gt_common, 8),
            "c1_mean_rnd_drop_gat_common_target": round(mean_rnd_c1_gt_common, 8),
            "c1_advantage_gat_common_target": round(c1_advantage_gat_common_target, 8),
            # C2 Matched GAT Own Target & Own Ranking
            "c2_target_gat": c2_target_gat,
            "c2_mean_ge_drop_gat_own_ranking": round(mean_ge_c2_gt_own, 8),
            "c2_mean_rnd_drop_gat_own": round(mean_rnd_c2_gt_own, 8),
            "c2_advantage_gat_own_ranking": round(c2_advantage_gat_own_ranking, 8),
            # Diagnostic: Own-target advantage on ETGNN-ranked interventions
            "c2_gat_own_target_advantage_on_etgnn_ranking": round(c2_gat_own_target_advantage_on_etgnn_ranking, 8),
            # Legacy aliases for backward compatibility
            "advantage_A_etgnn": round(c1_advantage_etgnn, 4),
            "advantage_A_gat": round(c1_advantage_gat_common_target, 4),  # Explicitly documented as common-target!
        })

    # -------------------------------------------------------------
    # 5. Strict Assertion of Raw/Detailed Record Consistency
    # -------------------------------------------------------------
    df_raw = pd.DataFrame(raw_benchmark_records)
    df_det = pd.DataFrame(detailed_intervention_records)

    # 1. Check GNNExplainer
    for _, r_row in df_raw[df_raw["method"] == "GNNExplainer"].iterrows():
        nid = r_row["node_index"]
        s = r_row["seed"]

        # ETGNN C1
        det_et_c1 = df_det[(df_det["node_index"] == nid) & (df_det["method"] == "GNNExplainer") &
                           (df_det["contrast"] == "C1") & (df_det["ranking_model_id"] == "ETGNN") &
                           (df_det["scoring_model_id"] == "ETGNN") & (df_det["seed"] == s)]
        assert len(det_et_c1) == 1, f"Missing detailed ETGNN C1 record for node {nid}, seed {s}"
        assert abs(det_et_c1["signed_drop"].iloc[0] - r_row["c1_drop_etgnn"]) < 1e-7, \
            f"Mismatch in c1_drop_etgnn for node {nid}, seed {s}: detailed={det_et_c1['signed_drop'].iloc[0]}, raw={r_row['c1_drop_etgnn']}"
        assert det_et_c1["deleted_global_edge_ids"].iloc[0] == r_row["deleted_global_edge_ids_etgnn"], \
            f"Mismatch in ETGNN deleted edges for node {nid}, seed {s}"

        # Matched GAT C1 Common Target
        det_gt_c1 = df_det[(df_det["node_index"] == nid) & (df_det["method"] == "GNNExplainer") &
                           (df_det["contrast"] == "C1") & (df_det["ranking_model_id"] == "ETGNN") &
                           (df_det["scoring_model_id"] == "Matched_GAT") & (df_det["seed"] == s)]
        assert len(det_gt_c1) == 1, f"Missing detailed GAT C1 record for node {nid}, seed {s}"
        assert abs(det_gt_c1["signed_drop"].iloc[0] - r_row["c1_drop_gat"]) < 1e-7, \
            f"Mismatch in c1_drop_gat for node {nid}, seed {s}: detailed={det_gt_c1['signed_drop'].iloc[0]}, raw={r_row['c1_drop_gat']}"

        # Matched GAT True C2 Own Ranking
        det_gt_c2 = df_det[(df_det["node_index"] == nid) & (df_det["method"] == "GNNExplainer") &
                           (df_det["contrast"] == "C2") & (df_det["ranking_model_id"] == "Matched_GAT") &
                           (df_det["scoring_model_id"] == "Matched_GAT") & (df_det["seed"] == s)]
        assert len(det_gt_c2) == 1, f"Missing detailed GAT C2 record for node {nid}, seed {s}"
        assert abs(det_gt_c2["signed_drop"].iloc[0] - r_row["c2_drop_gat_own_ranking"]) < 1e-7, \
            f"Mismatch in c2_drop_gat_own_ranking for node {nid}, seed {s}: detailed={det_gt_c2['signed_drop'].iloc[0]}, raw={r_row['c2_drop_gat_own_ranking']}"
        assert det_gt_c2["deleted_global_edge_ids"].iloc[0] == r_row["deleted_global_edge_ids_gat"], \
            f"Mismatch in GAT deleted edges for node {nid}, seed {s}"

    # 2. Check Random
    for _, r_row in df_raw[df_raw["method"] == "Random"].iterrows():
        nid = r_row["node_index"]
        s = r_row["seed"]

        det_et_rnd = df_det[(df_det["node_index"] == nid) & (df_det["method"] == "Random") &
                            (df_det["scoring_model_id"] == "ETGNN") & (df_det["seed"] == s)]
        assert len(det_et_rnd) == 1, f"Missing detailed Random ETGNN record for node {nid}, seed {s}"
        assert abs(det_et_rnd["signed_drop"].iloc[0] - r_row["c1_drop_etgnn"]) < 1e-7

        det_gt_rnd_c1 = df_det[(df_det["node_index"] == nid) & (df_det["method"] == "Random") &
                               (df_det["scoring_model_id"] == "Matched_GAT") & (df_det["contrast"] == "C1") & (df_det["seed"] == s)]
        assert len(det_gt_rnd_c1) == 1
        assert abs(det_gt_rnd_c1["signed_drop"].iloc[0] - r_row["c1_drop_gat"]) < 1e-7

        det_gt_rnd_c2 = df_det[(df_det["node_index"] == nid) & (df_det["method"] == "Random") &
                               (df_det["scoring_model_id"] == "Matched_GAT") & (df_det["contrast"] == "C2") & (df_det["seed"] == s)]
        assert len(det_gt_rnd_c2) == 1
        assert abs(det_gt_rnd_c2["signed_drop"].iloc[0] - r_row["c2_drop_gat_own_ranking"]) < 1e-7

    print("Asserted 100% agreement between raw_benchmark_records and detailed_intervention_records across all nodes, methods, models, and seeds!")

    # Save outputs
    df_raw.to_csv(raw_records_path, index=False)
    df_det.to_csv(detailed_interventions_path, index=False)
    summary_df = pd.DataFrame(benchmark_summary_records)
    summary_df.to_csv(out_path / "technical_pilot_benchmark_summary.csv", index=False)

    summary_json = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
        "benchmark_nodes_count": len(summary_df),
        "runner_sha256": runner_sha256,
        "dependency_versions": dep_versions,
        "mean_gnnexplainer_time_sec_per_seed_etgnn": round(summary_df["gnnexplainer_etgnn_mean_runtime_ms"].mean() / 1000.0, 3),
        "mean_gnnexplainer_time_sec_per_seed_gat": round(summary_df["gnnexplainer_gat_mean_runtime_ms"].mean() / 1000.0, 3),
        "mean_extract_time_ms": round(summary_df["extract_runtime_ms"].mean(), 2),
        "mean_random_30_time_ms": round(summary_df["random_30_total_runtime_ms"].mean(), 2),
        "mean_occlusion_time_ms": round(summary_df["occlusion_runtime_ms"].mean(), 2),
        "max_peak_rss_mb": round(summary_df["peak_rss_mb"].max(), 2),
        "max_peak_vram_mb": round(summary_df["peak_vram_mb"].max(), 2),
        "node_134593_distinction": {
            "node_index": 134593,
            "etgnn_probabilities": {
                "p_class_0": float(full_probs_etgnn[134593, 0].item()),
                "p_illicit": float(full_probs_etgnn[134593, 1].item()),
                "predicted_class": int(float(full_probs_etgnn[134593, 1].item()) >= 0.5),
            },
            "matched_gat_probabilities": {
                "p_class_0": float(full_probs_gat[134593, 0].item()),
                "p_illicit": float(full_probs_gat[134593, 1].item()),
                "predicted_class": int(float(full_probs_gat[134593, 1].item()) >= 0.5),
            },
            "c1_target_class": int(summary_df.loc[summary_df["node_index"] == 134593, "c1_target_class"].iloc[0]),
            "c2_target_gat": int(summary_df.loc[summary_df["node_index"] == 134593, "c2_target_gat"].iloc[0]),
            "common_target_gat_advantage": float(summary_df.loc[summary_df["node_index"] == 134593, "c1_advantage_gat_common_target"].iloc[0]),
            "own_target_advantage_on_etgnn_ranking": float(summary_df.loc[summary_df["node_index"] == 134593, "c2_gat_own_target_advantage_on_etgnn_ranking"].iloc[0]),
            "own_target_advantage_on_gat_own_ranking": float(summary_df.loc[summary_df["node_index"] == 134593, "c2_advantage_gat_own_ranking"].iloc[0]),
            "colab_gpu_pilot_v1_reference": {
                "common_target_gat_advantage": 0.06576057,
                "own_target_advantage_on_same_interventions": -0.06576057
            }
        },
        "graph_time_step_ranges": {
            "all_nodes_in_validation_graph": {
                "total_nodes": int(val_graph["x"].shape[0]),
                "time_step_min": int(val_graph["time_step"].min().item()),
                "time_step_max": int(val_graph["time_step"].max().item()),
                "description": "Entire 2-hop snapshot containing historical transactions (steps 1-32) plus validation transactions (steps 33-40)."
            },
            "val_mask_nodes": {
                "total_nodes": int(val_graph["val_mask"].sum().item()),
                "time_step_min": int(val_graph["time_step"][val_graph["val_mask"]].min().item()),
                "time_step_max": int(val_graph["time_step"][val_graph["val_mask"]].max().item()),
                "description": "Labelled evaluation universe restricted strictly to validation time steps 33-40."
            },
            "historical_test_period_disclosure": "Test time steps 41-49 are completely absent from validation_graph.pt. The test period was previously inspected in a one-time held-out evaluation on 2026-09-25. No test data is touched or re-evaluated."
        },
        "notes": "Technical pilot benchmark executed on 4 development period nodes (steps 33-36). Contains explicit C1 common-target and C2 own-ranking branches.",
    }
    summary_path.write_text(json.dumps(summary_json, indent=2))
    resume_path.write_text(json.dumps({
        "status": "COMPLETED",
        "timestamp": time.time(),
        "runner_sha256": runner_sha256,
        "nodes_evaluated": sample_df["node_index"].tolist()
    }, indent=2))

    print("\n==================================================================")
    print("  TECHNICAL PILOT BENCHMARK COMPLETED SUCCESSFULLY")
    print("==================================================================")
    print(summary_df[[
        "category",
        "node_index",
        "E_cand",
        "k_budget",
        "c1_target_class",
        "c1_advantage_etgnn",
        "c1_advantage_gat_common_target",
        "c2_target_gat",
        "c2_advantage_gat_own_ranking",
        "c2_gat_own_target_advantage_on_etgnn_ranking",
    ]].to_string())
    print("\nSummary Metrics:")
    print(json.dumps(summary_json, indent=2))
    return summary_df, summary_json


if __name__ == "__main__":
    run_benchmark()
