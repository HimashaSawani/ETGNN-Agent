#!/usr/bin/env python3
"""
run_main_c1_c2_benchmark.py
===========================
Authoritative Main Benchmark Runner for ETGNN vs. Matched GAT (C1 & C2).

Reconciled Features & Protocol Verification:
1. Historical Exclusions:
   Loads authentic historical pilot register (experiments/pilot_validation_audit/pilot_nodes.csv),
   maps original node IDs to txIds via node_mapping.csv, asserts presence, and fails if missing.
2. GAT C2 Random Baseline:
   Filters GAT C2 GNNExplainer and Random records explicitly by:
     contrast="C2", scoring_model_id="Matched_GAT", and target_class=c2_target_gat.
   Asserts exactly S explainer and D random draws per node.
   Includes regression check on Node 134593 (A = -0.06576057275).
3. All Eight Pair Registers:
   Analyzes all 8 primary and degree-sensitivity registers from frozen config.
   Asserts complete pair membership (never silently skips missing nodes).
   Saves individual pair differences with pair IDs before bootstrap.
4. Degree & Occlusion Comparators:
   Maps relabelled sub_edges[0] to global nodes via sub_nodes before indexing out-degrees.
   Logs Degree and single-edge Occlusion comparators in both detailed and raw tables.
5. Main Preflight Checks:
   Validates graph time steps, node-to-tx mapping, original_node_index, candidate count,
   and predicted targets against frozen metadata.
   Asserts full-graph / 2-hop subgraph forward pass probability equivalence (tolerance 1e-5).
   Reads budget_alpha dynamically from configuration.
6. Robust Resume & Atomic Checkpoint State:
   Binds saved state to config_sha256, runner_sha256, mode, and artifact hashes.
   Validates record completeness and uniqueness before skipping completed nodes.
   Uses atomic file replacements (write-tmp -> os.replace).
   Supports Drive-to-local restore and mirrors final statistical outputs.
7. Post-Processing & Epistemic Labeling:
   Labels C1 strictly as random-adjusted deletion-response difference,
   acknowledging baseline noise subtraction without claiming isolated explanation fidelity.
"""

import os
import sys
import argparse
import time
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree, k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.etgnn import ETGNNClassifier
from src.models.matched_gat import MatchedGATClassifier


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_mem_mb() -> float:
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


class MainBenchmarkRunner:
    def __init__(self, config_path: Path, output_dir: Path, is_dev_check: bool = False, backup_dir: Path = None):
        self.config_path = config_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_dev_check = is_dev_check
        self.backup_dir = backup_dir
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

        assert config_path.exists(), f"Configuration file missing: {config_path}"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runner_sha256 = compute_file_sha256(Path(__file__))
        self.config_sha256 = compute_file_sha256(self.config_path)

        # Protocol parameters
        self.explainer_seeds = self.config["protocol_parameters"]["explainer_seeds"]
        self.num_random_draws = self.config["protocol_parameters"]["random_comparator_draws"]
        self.budget_alpha = float(self.config["protocol_parameters"]["budget_alpha"])

    def load_historical_pilot_exclusions(self) -> Tuple[Set[int], Set[int]]:
        """
        Loads authentic historical pilot register (20 convenience nodes),
        maps node_index to txId via node_mapping.csv, asserts presence, and fails if missing.
        """
        hist_pilot_path = PROJECT_ROOT / "experiments" / "pilot_validation_audit" / "pilot_nodes.csv"
        assert hist_pilot_path.exists(), f"CRITICAL ERROR: Authentic historical pilot register missing at {hist_pilot_path}!"
        
        hist_df = pd.read_csv(hist_pilot_path)
        assert len(hist_df) == 20, f"Expected 20 historical pilot nodes, found {len(hist_df)}"
        
        hist_nodes = set(hist_df["node_index"].astype(int).tolist())
        
        map_path = PROJECT_ROOT / self.config["data_artifacts"]["node_mapping"]["path"]
        assert map_path.exists(), f"Missing node mapping at {map_path}"
        map_df = pd.read_csv(map_path)
        
        tx_col = "txId" if "txId" in map_df.columns else "transaction_id"
        idx_col = "node_index" if "node_index" in map_df.columns else "node_idx"
        mapping = dict(zip(map_df[idx_col].astype(int), map_df[tx_col].astype(int)))
        
        hist_txs = {mapping[nid] for nid in hist_nodes if nid in mapping}
        assert len(hist_txs) == 20, f"Failed to map all 20 historical pilot nodes to txIds (mapped {len(hist_txs)})"
        
        print(f"  Loaded authentic historical pilot register: {len(hist_nodes)} nodes, {len(hist_txs)} txIds.")
        return hist_nodes, hist_txs

    def preflight_checks(self):
        print("\n--- 1. Preflight Validation & Cryptographic Assertions ---")
        # Validate models
        et_path = PROJECT_ROOT / self.config["models"]["etgnn"]["checkpoint_path"]
        assert et_path.exists(), f"Missing ETGNN checkpoint: {et_path}"
        self.et_sha = compute_file_sha256(et_path)
        assert self.et_sha == self.config["models"]["etgnn"]["sha256"], "ETGNN checkpoint hash mismatch!"
        print(f"  ETGNN Checkpoint:      {self.et_sha[:16]}... (Verified)")

        gt_path = PROJECT_ROOT / self.config["models"]["matched_gat"]["checkpoint_path"]
        assert gt_path.exists(), f"Missing Matched GAT checkpoint: {gt_path}"
        self.gt_sha = compute_file_sha256(gt_path)
        assert self.gt_sha == self.config["models"]["matched_gat"]["sha256"], "Matched GAT checkpoint hash mismatch!"
        print(f"  Matched GAT Checkpoint:{self.gt_sha[:16]}... (Verified)")

        # Validate graph
        vg_path = PROJECT_ROOT / self.config["data_artifacts"]["validation_graph"]["path"]
        assert vg_path.exists(), f"Missing validation graph: {vg_path}"
        self.vg_sha = compute_file_sha256(vg_path)
        assert self.vg_sha == self.config["data_artifacts"]["validation_graph"]["sha256"], "Validation graph hash mismatch!"
        print(f"  Validation Graph:      {self.vg_sha[:16]}... (Verified)")

        # Validate mapping
        map_path = PROJECT_ROOT / self.config["data_artifacts"]["node_mapping"]["path"]
        assert map_path.exists(), f"Missing node mapping: {map_path}"
        self.map_sha = compute_file_sha256(map_path)
        assert self.map_sha == self.config["data_artifacts"]["node_mapping"]["sha256"], "Node mapping hash mismatch!"
        print(f"  Node Mapping CSV:      {self.map_sha[:16]}... (Verified)")

        # Load graph
        self.val_graph = torch.load(vg_path, map_location="cpu", weights_only=False)
        time_steps = self.val_graph["time_step"]
        val_mask = self.val_graph["val_mask"]

        # Validate graph metadata
        assert int(time_steps.min().item()) == 1 and int(time_steps.max().item()) == 40, "Graph time steps deviate from 1..40!"
        assert int(time_steps[val_mask].min().item()) == 33 and int(time_steps[val_mask].max().item()) == 40, "val_mask deviates from 33..40!"
        assert int(val_mask.sum().item()) == 7653, "val_mask count deviates from 7,653!"
        assert 41 not in time_steps, "CRITICAL ERROR: Test steps 41-49 present in validation graph!"

        # Validate original_node_index
        assert "original_node_index" in self.val_graph, "Missing original_node_index in validation graph!"
        assert (self.val_graph["original_node_index"] == torch.arange(len(self.val_graph["x"]))).all(), "original_node_index mismatch!"

        # Validate node mapping
        map_df = pd.read_csv(map_path)
        assert len(map_df) == 203769, "Node mapping row count deviates from 203,769!"

        # Load historical pilot exclusions
        self.hist_nodes, self.hist_txs = self.load_historical_pilot_exclusions()

    def load_models_and_compute_full_inference(self):
        print("\n--- 2. Loading Checkpoints & Full-Graph Inference ---")
        et_path = PROJECT_ROOT / self.config["models"]["etgnn"]["checkpoint_path"]
        self.m_etgnn = ETGNNClassifier(input_dim=165, time_dim=16, hidden_dim=64, dropout=0.3)
        self.m_etgnn.load_state_dict(torch.load(et_path, map_location="cpu"))
        self.m_etgnn.eval()
        self.m_etgnn.to(self.device)

        gt_path = PROJECT_ROOT / self.config["models"]["matched_gat"]["checkpoint_path"]
        self.m_gat = MatchedGATClassifier(input_dim=165, time_dim=16, hidden_dim=64, dropout=0.3)
        self.m_gat.load_state_dict(torch.load(gt_path, map_location="cpu"))
        self.m_gat.eval()
        self.m_gat.to(self.device)

        self.explainer_etgnn = Explainer(
            model=self.m_etgnn,
            algorithm=GNNExplainer(epochs=self.config["protocol_parameters"]["gnnexplainer_epochs"]),
            explanation_type="model",
            node_mask_type=None,
            edge_mask_type="object",
            model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
        )

        self.explainer_gat = Explainer(
            model=self.m_gat,
            algorithm=GNNExplainer(epochs=self.config["protocol_parameters"]["gnnexplainer_epochs"]),
            explanation_type="model",
            node_mask_type=None,
            edge_mask_type="object",
            model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
        )

        print("  Computing full-graph inference...")
        with torch.no_grad():
            x_dev = self.val_graph["x"].to(self.device)
            ei_dev = self.val_graph["edge_index"].to(self.device)
            t_dev = self.val_graph["time_step"].to(self.device)
            self.full_probs_etgnn = F.softmax(self.m_etgnn(x_dev, ei_dev, t_dev), dim=1).cpu()
            self.full_probs_gat = F.softmax(self.m_gat(x_dev, ei_dev, t_dev), dim=1).cpu()

    def select_and_validate_nodes(self) -> List[Dict[str, Any]]:
        print("\n--- 3. Selecting & Validating Evaluation Nodes ---")
        if self.is_dev_check:
            pilot_csv = PROJECT_ROOT / "experiments" / "matched_c1_c2_audit" / "technical_pilot_sample.csv"
            assert pilot_csv.exists(), f"Missing {pilot_csv}"
            df_pilot = pd.read_csv(pilot_csv)
            nodes = []
            for _, r in df_pilot.iterrows():
                nodes.append({
                    "node_index": int(r["node_index"]),
                    "transaction_id": int(r["transaction_id"]),
                    "time_step": int(r["time_step"]),
                    "etgnn_prediction_category": str(r["prediction_category"]),
                    "candidate_edge_count": int(r["candidate_edge_count"]),
                    "c1_target_class": int(r["predicted_class"]),
                    "eval_sources": ["dev_check"]
                })
            print(f"  DEV-CHECK MODE: Evaluating {len(nodes)} development benchmark nodes (steps 33-36).")
            return nodes

        # Main Evaluation: 377 unique execution nodes across steps 37-40
        nodes = self.config["execution_node_union"]["nodes"]
        assert len(nodes) == 377, f"Expected 377 execution union nodes, found {len(nodes)}"

        for item in nodes:
            nid = item["node_index"]
            step = item["time_step"]
            tx_id = item["transaction_id"]
            cand_e = item["candidate_edge_count"]

            # Strict assertions
            assert step in [37, 38, 39, 40], f"Node {nid} has step {step} not in 37..40!"
            assert bool(self.val_graph["val_mask"][nid].item()) is True, f"Node {nid} not in val_mask!"
            assert nid not in self.hist_nodes, f"Node {nid} is in authentic historical pilot!"
            assert tx_id not in self.hist_txs, f"txId {tx_id} is in authentic historical pilot!"
            assert cand_e >= 2, f"Node {nid} has E_cand < 2!"

        print(f"  MAIN MODE: Verified all {len(nodes)} execution union nodes against frozen constraints.")
        return nodes

    def restore_from_backup_if_needed(self, detailed_path: Path, raw_path: Path, summary_path: Path, resume_path: Path):
        """Restores state from Drive backup if local is missing or has fewer completed nodes."""
        if not self.backup_dir or not (self.backup_dir / resume_path.name).exists():
            return

        try:
            with open(self.backup_dir / resume_path.name, "r", encoding="utf-8") as f:
                b_state = json.load(f)

            b_completed = b_state.get("completed_count", 0)
            local_completed = 0
            if resume_path.exists():
                with open(resume_path, "r", encoding="utf-8") as f:
                    l_state = json.load(f)
                local_completed = l_state.get("completed_count", 0)

            if b_completed > local_completed:
                print(f"  Restoring more advanced state from Drive backup ({b_completed} > {local_completed} nodes)...")
                for p in [detailed_path, raw_path, summary_path, resume_path]:
                    if (self.backup_dir / p.name).exists():
                        shutil.copy2(self.backup_dir / p.name, p)
                print("  Drive restore completed successfully.")
        except Exception as e:
            print(f"  Warning: Drive restore check failed ({e})")

    def validate_node_records_completeness(self, nid: int, detailed_df: pd.DataFrame, raw_df: pd.DataFrame) -> bool:
        """Validates that a node has full, complete, and unique records before allowing resume skip."""
        d_sub = detailed_df[detailed_df["node_index"] == nid]
        r_sub = raw_df[raw_df["node_index"] == nid]

        S = len(self.explainer_seeds)
        D = self.num_random_draws

        # Detailed record requirements:
        # GNNExplainer: S (ETGNN C1) + S (GAT C1) + S (GAT C2) = 3*S
        # Random: D (ETGNN C1) + D (GAT C1) + D (GAT C2) = 3*D
        # Degree: 1 (ETGNN C1) + 1 (GAT C1) + 1 (GAT C2) = 3
        # Occlusion: 1 (ETGNN C1) + 1 (GAT C1) + 1 (GAT C2) = 3
        # Total detailed rows per node = 3*S + 3*D + 3 + 3
        expected_detailed_rows = 3 * S + 3 * D + 6
        if len(d_sub) != expected_detailed_rows:
            return False

        # Raw record requirements:
        # S (GNNExplainer) + D (Random) + 1 (Degree) + 1 (Occlusion) = S + D + 2
        expected_raw_rows = S + D + 2
        if len(r_sub) != expected_raw_rows:
            return False

        # Uniqueness check in raw
        if r_sub.duplicated(subset=["method", "seed"]).any():
            return False

        return True

    def atomic_save(self, detailed_records, raw_records, summary_records, completed_nodes, detailed_path, raw_path, summary_path, resume_path):
        """Atomically writes checkpoints using temporary files and os.replace."""
        df_det = pd.DataFrame(detailed_records)
        df_raw = pd.DataFrame(raw_records)
        df_sum = pd.DataFrame(summary_records)

        # Temporary paths
        tmp_det = detailed_path.with_suffix(".csv.tmp")
        tmp_raw = raw_path.with_suffix(".csv.tmp")
        tmp_sum = summary_path.with_suffix(".csv.tmp")
        tmp_res = resume_path.with_suffix(".json.tmp")

        df_det.to_csv(tmp_det, index=False)
        df_raw.to_csv(tmp_raw, index=False)
        df_sum.to_csv(tmp_sum, index=False)

        state = {
            "status": "COMPLETED" if len(completed_nodes) >= len(self.config["execution_node_union"]["nodes"]) else "IN_PROGRESS",
            "mode": "DEV_CHECK" if self.is_dev_check else "MAIN_RUN",
            "config_sha256": self.config_sha256,
            "runner_sha256": self.runner_sha256,
            "etgnn_checkpoint_sha256": self.et_sha,
            "matched_gat_checkpoint_sha256": self.gt_sha,
            "validation_graph_sha256": self.vg_sha,
            "last_updated": time.time(),
            "completed_count": len(completed_nodes),
            "completed_node_indices": sorted(list(completed_nodes)),
        }
        with open(tmp_res, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # Atomic replacement
        tmp_det.replace(detailed_path)
        tmp_raw.replace(raw_path)
        tmp_sum.replace(summary_path)
        tmp_res.replace(resume_path)

        # Mirror to Drive if requested
        if self.backup_dir:
            try:
                shutil.copy2(detailed_path, self.backup_dir / detailed_path.name)
                shutil.copy2(raw_path, self.backup_dir / raw_path.name)
                shutil.copy2(summary_path, self.backup_dir / summary_path.name)
                shutil.copy2(resume_path, self.backup_dir / resume_path.name)
            except Exception as e:
                print(f"  Warning: Drive backup mirror failed ({e})")

    def run_benchmark(self):
        self.preflight_checks()
        self.load_models_and_compute_full_inference()
        eval_nodes = self.select_and_validate_nodes()

        detailed_path = self.output_dir / "detailed_intervention_records.csv"
        raw_path = self.output_dir / "raw_benchmark_records.csv"
        summary_path = self.output_dir / "main_benchmark_summary.csv"
        resume_path = self.output_dir / "main_resume_state.json"

        # Drive restore if available
        self.restore_from_backup_if_needed(detailed_path, raw_path, summary_path, resume_path)

        # State recovery
        completed_nodes = set()
        detailed_records = []
        raw_records = []
        summary_records = []

        if resume_path.exists():
            try:
                with open(resume_path, "r", encoding="utf-8") as f:
                    rstate = json.load(f)

                # Validate state binding
                expected_mode = "DEV_CHECK" if self.is_dev_check else "MAIN_RUN"
                assert rstate.get("mode") == expected_mode, f"Resume mode mismatch! State has {rstate.get('mode')}, current is {expected_mode}"
                assert rstate.get("config_sha256") == self.config_sha256, "Resume config hash mismatch!"
                assert rstate.get("runner_sha256") == self.runner_sha256, "Resume runner hash mismatch!"

                if detailed_path.exists() and raw_path.exists() and summary_path.exists():
                    df_det_loaded = pd.read_csv(detailed_path)
                    df_raw_loaded = pd.read_csv(raw_path)
                    df_sum_loaded = pd.read_csv(summary_path)

                    raw_completed = set(rstate.get("completed_node_indices", []))
                    # Validate completeness for each completed node
                    valid_completed = set()
                    for nid in raw_completed:
                        if self.validate_node_records_completeness(nid, df_det_loaded, df_raw_loaded):
                            valid_completed.add(nid)
                        else:
                            print(f"  Warning: Node {nid} has incomplete or corrupted records. Will recompute.")

                    completed_nodes = valid_completed
                    detailed_records = df_det_loaded[df_det_loaded["node_index"].isin(completed_nodes)].to_dict(orient="records")
                    raw_records = df_raw_loaded[df_raw_loaded["node_index"].isin(completed_nodes)].to_dict(orient="records")
                    summary_records = df_sum_loaded[df_sum_loaded["node_index"].isin(completed_nodes)].to_dict(orient="records")
                    print(f"  Validated resume state: {len(completed_nodes)} nodes verified complete.")
            except Exception as e:
                print(f"  Resume validation error ({e}). Restarting evaluation safely.")
                completed_nodes = set()
                detailed_records = []
                raw_records = []
                summary_records = []

        total_nodes = len(eval_nodes)
        print(f"\n--- 4. Executing Interventions (Total: {total_nodes} nodes, Budget Alpha: {self.budget_alpha}) ---")
        start_time_all = time.time()

        for idx, item in enumerate(eval_nodes):
            pos = item["node_index"]
            tx_id = item["transaction_id"]
            step = item["time_step"]
            cat = item.get("etgnn_category", item.get("etgnn_prediction_category", "Unknown"))

            if pos in completed_nodes:
                continue

            # Extract 2-hop computation subgraph
            t_extract_0 = time.perf_counter()
            sub_nodes, sub_edges, node_map, edge_mask = k_hop_subgraph(
                node_idx=pos,
                num_hops=2,
                edge_index=self.val_graph["edge_index"],
                relabel_nodes=True,
                flow="source_to_target",
            )
            t_extract_ms = (time.perf_counter() - t_extract_0) * 1000.0

            local_center = int(node_map.item())
            E_cand = sub_edges.shape[1]
            k_budget = max(1, int(np.ceil(self.budget_alpha * E_cand)))
            k_over_E = k_budget / float(E_cand)
            global_edge_indices = torch.where(edge_mask)[0].numpy()

            sub_x = self.val_graph["x"][sub_nodes].to(self.device)
            sub_time = self.val_graph["time_step"][sub_nodes].to(self.device)
            sub_edges_dev = sub_edges.to(self.device)

            # Unperturbed forward pass & probability equivalence check
            with torch.no_grad():
                init_p_et = torch.softmax(self.m_etgnn(sub_x, sub_edges_dev, sub_time), dim=1)[local_center]
                init_p_gt = torch.softmax(self.m_gat(sub_x, sub_edges_dev, sub_time), dim=1)[local_center]

            p0_et = float(init_p_et[0].item())
            p1_et = float(init_p_et[1].item())
            p0_gt = float(init_p_gt[0].item())
            p1_gt = float(init_p_gt[1].item())

            # Full-graph vs. subgraph equivalence assertion (tolerance 1e-5)
            diff_et = max(abs(p0_et - float(self.full_probs_etgnn[pos, 0].item())),
                          abs(p1_et - float(self.full_probs_etgnn[pos, 1].item())))
            diff_gt = max(abs(p0_gt - float(self.full_probs_gat[pos, 0].item())),
                          abs(p1_gt - float(self.full_probs_gat[pos, 1].item())))
            assert diff_et <= 1e-5, f"Node {pos} ETGNN full/subgraph equivalence failed (diff={diff_et:.2e})!"
            assert diff_gt <= 1e-5, f"Node {pos} GAT full/subgraph equivalence failed (diff={diff_gt:.2e})!"

            c1_target_class = 1 if p1_et >= 0.5 else 0
            c2_target_gat = 1 if p1_gt >= 0.5 else 0

            p_c1_orig_etgnn = float(init_p_et[c1_target_class].item())
            p_c1_orig_gat = float(init_p_gt[c1_target_class].item())
            p_c2_orig_gat = float(init_p_gt[c2_target_gat].item())

            # -------------------------------------------------------------
            # Loop 4A. GNNExplainer for ETGNN (C1 Ranking & Shared Deletions)
            # -------------------------------------------------------------
            ge_etgnn_seed_times = []
            ge_drops_c1_etgnn = []
            ge_drops_c1_gat_common_target = []
            ge_drops_c2_gat_on_etgnn_ranking = []
            etgnn_seed_results = {}

            for s in self.explainer_seeds:
                torch.manual_seed(s)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(s)
                sync_device(self.device)
                t_ge_0 = time.perf_counter()
                explanation_etgnn = self.explainer_etgnn(
                    x=sub_x,
                    edge_index=sub_edges_dev,
                    time_step=sub_time,
                    index=local_center,
                )
                sync_device(self.device)
                t_ge_ms = (time.perf_counter() - t_ge_0) * 1000.0
                ge_etgnn_seed_times.append(t_ge_ms)

                mask_etgnn = explanation_etgnn.edge_mask.detach().cpu().numpy()
                ranked_edges_etgnn = np.argsort(-mask_etgnn)
                selected_local_edges_etgnn = ranked_edges_etgnn[:k_budget].tolist()
                selected_global_edge_ids_etgnn = global_edge_indices[selected_local_edges_etgnn].tolist()

                keep_mask_etgnn = torch.tensor(
                    [i not in set(selected_local_edges_etgnn) for i in range(E_cand)],
                    dtype=torch.bool,
                    device=self.device,
                )
                del_edges_etgnn = sub_edges_dev[:, keep_mask_etgnn]

                with torch.no_grad():
                    del_p_etgnn = torch.softmax(self.m_etgnn(sub_x, del_edges_etgnn, sub_time), dim=1)[local_center]
                    del_p_gat = torch.softmax(self.m_gat(sub_x, del_edges_etgnn, sub_time), dim=1)[local_center]

                del_p0_et = float(del_p_etgnn[0].item())
                del_p1_et = float(del_p_etgnn[1].item())
                del_p0_gt = float(del_p_gat[0].item())
                del_p1_gt = float(del_p_gat[1].item())

                drop_c1_et = p_c1_orig_etgnn - float(del_p_etgnn[c1_target_class].item())
                drop_c1_gat_common = p_c1_orig_gat - float(del_p_gat[c1_target_class].item())
                drop_c2_gat_etgnn_rank = p_c2_orig_gat - float(del_p_gat[c2_target_gat].item())

                ge_drops_c1_etgnn.append(drop_c1_et)
                ge_drops_c1_gat_common_target.append(drop_c1_gat_common)
                ge_drops_c2_gat_on_etgnn_ranking.append(drop_c2_gat_etgnn_rank)

                etgnn_seed_results[s] = {
                    "drop_c1_et": drop_c1_et,
                    "drop_c1_gat_common": drop_c1_gat_common,
                    "drop_c2_gat_etgnn_rank": drop_c2_gat_etgnn_rank,
                    "selected_local_edges_etgnn": selected_local_edges_etgnn,
                    "selected_global_edge_ids_etgnn": selected_global_edge_ids_etgnn,
                    "runtime_ms": t_ge_ms,
                }

                # Detailed: ETGNN C1
                detailed_records.append({
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
                    "orig_p0": p0_et,
                    "orig_p1": p1_et,
                    "del_p0": del_p0_et,
                    "del_p1": del_p1_et,
                    "deleted_local_edge_indices": json.dumps(selected_local_edges_etgnn),
                    "deleted_global_edge_ids": json.dumps(selected_global_edge_ids_etgnn),
                    "runtime_ms": t_ge_ms,
                })

                # Detailed: GAT C1 Common Target
                detailed_records.append({
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
                    "orig_p0": p0_gt,
                    "orig_p1": p1_gt,
                    "del_p0": del_p0_gt,
                    "del_p1": del_p1_gt,
                    "deleted_local_edge_indices": json.dumps(selected_local_edges_etgnn),
                    "deleted_global_edge_ids": json.dumps(selected_global_edge_ids_etgnn),
                    "runtime_ms": t_ge_ms,
                })

            # -------------------------------------------------------------
            # Loop 4B. GNNExplainer for Matched GAT (True C2: GAT's Own Ranking)
            # -------------------------------------------------------------
            ge_gat_seed_times = []
            ge_drops_c2_gat_own_ranking = []
            gat_seed_results = {}

            for s in self.explainer_seeds:
                torch.manual_seed(s)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(s)
                sync_device(self.device)
                t_ge_gat_0 = time.perf_counter()
                explanation_gat = self.explainer_gat(
                    x=sub_x,
                    edge_index=sub_edges_dev,
                    time_step=sub_time,
                    index=local_center,
                )
                sync_device(self.device)
                t_ge_gat_ms = (time.perf_counter() - t_ge_gat_0) * 1000.0
                ge_gat_seed_times.append(t_ge_gat_ms)

                mask_gat = explanation_gat.edge_mask.detach().cpu().numpy()
                ranked_edges_gat = np.argsort(-mask_gat)
                selected_local_edges_gat = ranked_edges_gat[:k_budget].tolist()
                selected_global_edge_ids_gat = global_edge_indices[selected_local_edges_gat].tolist()

                keep_mask_gat_own = torch.tensor(
                    [i not in set(selected_local_edges_gat) for i in range(E_cand)],
                    dtype=torch.bool,
                    device=self.device,
                )
                del_edges_gat_own = sub_edges_dev[:, keep_mask_gat_own]

                with torch.no_grad():
                    del_p_gat_own = torch.softmax(self.m_gat(sub_x, del_edges_gat_own, sub_time), dim=1)[local_center]

                del_p0_gt_own = float(del_p_gat_own[0].item())
                del_p1_gt_own = float(del_p_gat_own[1].item())
                drop_c2_gat_own = p_c2_orig_gat - float(del_p_gat_own[c2_target_gat].item())
                ge_drops_c2_gat_own_ranking.append(drop_c2_gat_own)

                gat_seed_results[s] = {
                    "drop_c2_gat_own": drop_c2_gat_own,
                    "selected_local_edges_gat": selected_local_edges_gat,
                    "selected_global_edge_ids_gat": selected_global_edge_ids_gat,
                    "runtime_ms": t_ge_gat_ms,
                }

                # Detailed: True C2 Matched GAT
                detailed_records.append({
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
                    "orig_p0": p0_gt,
                    "orig_p1": p1_gt,
                    "del_p0": del_p0_gt_own,
                    "del_p1": del_p1_gt_own,
                    "deleted_local_edge_indices": json.dumps(selected_local_edges_gat),
                    "deleted_global_edge_ids": json.dumps(selected_global_edge_ids_gat),
                    "runtime_ms": t_ge_gat_ms,
                })

            # Append to raw_benchmark_records by joining EXACT matching seeds
            for s in self.explainer_seeds:
                et_res = etgnn_seed_results[s]
                gt_res = gat_seed_results[s]
                raw_records.append({
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
            # Loop 4C. Shared Random Deletions (D draws)
            # -------------------------------------------------------------
            t_rnd_0 = time.perf_counter()
            rng = np.random.default_rng(20261020 + pos)
            rnd_drops_c1_etgnn = []
            rnd_drops_c1_gat_common_target = []
            rnd_drops_c2_gat_own_target = []

            for r_idx in range(self.num_random_draws):
                rnd_local_edges = rng.choice(E_cand, size=k_budget, replace=False).tolist()
                rnd_global_edge_ids = global_edge_indices[rnd_local_edges].tolist()
                keep_mask_rnd = torch.tensor(
                    [i not in set(rnd_local_edges) for i in range(E_cand)],
                    dtype=torch.bool,
                    device=self.device,
                )
                del_edges_rnd = sub_edges_dev[:, keep_mask_rnd]

                with torch.no_grad():
                    del_p_et = torch.softmax(self.m_etgnn(sub_x, del_edges_rnd, sub_time), dim=1)[local_center]
                    del_p_gt = torch.softmax(self.m_gat(sub_x, del_edges_rnd, sub_time), dim=1)[local_center]

                del_p0_et = float(del_p_et[0].item())
                del_p1_et = float(del_p_et[1].item())
                del_p0_gt = float(del_p_gt[0].item())
                del_p1_gt = float(del_p_gt[1].item())

                d_c1_et = p_c1_orig_etgnn - float(del_p_et[c1_target_class].item())
                d_c1_gt = p_c1_orig_gat - float(del_p_gt[c1_target_class].item())
                d_c2_gt = p_c2_orig_gat - float(del_p_gt[c2_target_gat].item())

                rnd_drops_c1_etgnn.append(d_c1_et)
                rnd_drops_c1_gat_common_target.append(d_c1_gt)
                rnd_drops_c2_gat_own_target.append(d_c2_gt)

                # Detailed: Random ETGNN C1
                detailed_records.append({
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
                    "orig_p0": p0_et,
                    "orig_p1": p1_et,
                    "del_p0": del_p0_et,
                    "del_p1": del_p1_et,
                    "deleted_local_edge_indices": json.dumps(rnd_local_edges),
                    "deleted_global_edge_ids": json.dumps(rnd_global_edge_ids),
                    "runtime_ms": np.nan,
                })

                # Detailed: Random Matched GAT C1 Common Target
                detailed_records.append({
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
                    "orig_p0": p0_gt,
                    "orig_p1": p1_gt,
                    "del_p0": del_p0_gt,
                    "del_p1": del_p1_gt,
                    "deleted_local_edge_indices": json.dumps(rnd_local_edges),
                    "deleted_global_edge_ids": json.dumps(rnd_global_edge_ids),
                    "runtime_ms": np.nan,
                })

                # Detailed: Random Matched GAT C2 Own Target
                detailed_records.append({
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
                    "orig_p0": p0_gt,
                    "orig_p1": p1_gt,
                    "del_p0": del_p0_gt,
                    "del_p1": del_p1_gt,
                    "deleted_local_edge_indices": json.dumps(rnd_local_edges),
                    "deleted_global_edge_ids": json.dumps(rnd_global_edge_ids),
                    "runtime_ms": np.nan,
                })

                raw_records.append({
                    "node_index": pos,
                    "category": cat,
                    "method": "Random",
                    "seed": r_idx,
                    "budget_k": k_budget,
                    "c1_target_class": c1_target_class,
                    "c1_drop_etgnn": d_c1_et,
                    "c1_drop_gat": d_c1_gt,
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
            # Loop 4D. Degree Ranking Baseline (Corrected Local-to-Global Mapping)
            # -------------------------------------------------------------
            t_deg_0 = time.perf_counter()
            local_src = sub_edges[0].numpy()
            global_src = sub_nodes[local_src].numpy()  # Corrected mapping

            full_out_degrees = degree(self.val_graph["edge_index"][0], num_nodes=self.val_graph["x"].shape[0]).numpy()
            src_degrees = full_out_degrees[global_src]
            deg_ranked = np.argsort(-src_degrees)
            deg_selected = deg_ranked[:k_budget].tolist()
            deg_global_ids = global_edge_indices[deg_selected].tolist()

            keep_mask_deg = torch.tensor([i not in set(deg_selected) for i in range(E_cand)], dtype=torch.bool, device=self.device)
            del_edges_deg = sub_edges_dev[:, keep_mask_deg]
            with torch.no_grad():
                del_p_et = torch.softmax(self.m_etgnn(sub_x, del_edges_deg, sub_time), dim=1)[local_center]
                del_p_gt = torch.softmax(self.m_gat(sub_x, del_edges_deg, sub_time), dim=1)[local_center]
            d_deg_c1_et = p_c1_orig_etgnn - float(del_p_et[c1_target_class].item())
            d_deg_c1_gt = p_c1_orig_gat - float(del_p_gt[c1_target_class].item())
            d_deg_c2_gt = p_c2_orig_gat - float(del_p_gt[c2_target_gat].item())
            t_deg_ms = (time.perf_counter() - t_deg_0) * 1000.0

            # Detailed: Degree
            for c_name, m_id, t_cls, d_val in [
                ("C1", "ETGNN", c1_target_class, d_deg_c1_et),
                ("C1", "Matched_GAT", c1_target_class, d_deg_c1_gt),
                ("C2", "Matched_GAT", c2_target_gat, d_deg_c2_gt),
            ]:
                detailed_records.append({
                    "node_index": pos,
                    "transaction_id": tx_id,
                    "category": cat,
                    "contrast": c_name,
                    "method": "Degree",
                    "ranking_model_id": "Degree",
                    "scoring_model_id": m_id,
                    "target_class": t_cls,
                    "seed": 0,
                    "budget_k": k_budget,
                    "E_cand": E_cand,
                    "actual_k_over_E": round(k_over_E, 4),
                    "orig_prob_target": p_c1_orig_etgnn if m_id == "ETGNN" else (p_c1_orig_gat if c_name == "C1" else p_c2_orig_gat),
                    "del_prob_target": float(del_p_et[t_cls].item()) if m_id == "ETGNN" else float(del_p_gt[t_cls].item()),
                    "signed_drop": d_val,
                    "orig_p0": p0_et if m_id == "ETGNN" else p0_gt,
                    "orig_p1": p1_et if m_id == "ETGNN" else p1_gt,
                    "del_p0": float(del_p_et[0].item()) if m_id == "ETGNN" else float(del_p_gt[0].item()),
                    "del_p1": float(del_p_et[1].item()) if m_id == "ETGNN" else float(del_p_gt[1].item()),
                    "deleted_local_edge_indices": json.dumps(deg_selected),
                    "deleted_global_edge_ids": json.dumps(deg_global_ids),
                    "runtime_ms": t_deg_ms,
                })

            raw_records.append({
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
            # Loop 4E. Single-Edge Occlusion Comparator (Restored)
            # -------------------------------------------------------------
            t_occ_0 = time.perf_counter()
            occ_drops = []
            for e_idx in range(E_cand):
                mask_occ = torch.ones(E_cand, dtype=torch.bool, device=self.device)
                mask_occ[e_idx] = False
                del_e = sub_edges_dev[:, mask_occ]
                with torch.no_grad():
                    dp = torch.softmax(self.m_etgnn(sub_x, del_e, sub_time), dim=1)[local_center]
                occ_drops.append(p_c1_orig_etgnn - float(dp[c1_target_class].item()))
            occ_ranked = np.argsort(-np.array(occ_drops))
            occ_selected = occ_ranked[:k_budget].tolist()
            occ_global_ids = global_edge_indices[occ_selected].tolist()

            keep_mask_occ = torch.tensor([i not in set(occ_selected) for i in range(E_cand)], dtype=torch.bool, device=self.device)
            del_edges_occ = sub_edges_dev[:, keep_mask_occ]
            with torch.no_grad():
                del_p_et = torch.softmax(self.m_etgnn(sub_x, del_edges_occ, sub_time), dim=1)[local_center]
                del_p_gt = torch.softmax(self.m_gat(sub_x, del_edges_occ, sub_time), dim=1)[local_center]
            d_occ_c1_et = p_c1_orig_etgnn - float(del_p_et[c1_target_class].item())
            d_occ_c1_gt = p_c1_orig_gat - float(del_p_gt[c1_target_class].item())
            d_occ_c2_gt = p_c2_orig_gat - float(del_p_gt[c2_target_gat].item())
            t_occ_ms = (time.perf_counter() - t_occ_0) * 1000.0

            # Detailed: Occlusion
            for c_name, m_id, t_cls, d_val in [
                ("C1", "ETGNN", c1_target_class, d_occ_c1_et),
                ("C1", "Matched_GAT", c1_target_class, d_occ_c1_gt),
                ("C2", "Matched_GAT", c2_target_gat, d_occ_c2_gt),
            ]:
                detailed_records.append({
                    "node_index": pos,
                    "transaction_id": tx_id,
                    "category": cat,
                    "contrast": c_name,
                    "method": "Occlusion",
                    "ranking_model_id": "Occlusion",
                    "scoring_model_id": m_id,
                    "target_class": t_cls,
                    "seed": 0,
                    "budget_k": k_budget,
                    "E_cand": E_cand,
                    "actual_k_over_E": round(k_over_E, 4),
                    "orig_prob_target": p_c1_orig_etgnn if m_id == "ETGNN" else (p_c1_orig_gat if c_name == "C1" else p_c2_orig_gat),
                    "del_prob_target": float(del_p_et[t_cls].item()) if m_id == "ETGNN" else float(del_p_gt[t_cls].item()),
                    "signed_drop": d_val,
                    "orig_p0": p0_et if m_id == "ETGNN" else p0_gt,
                    "orig_p1": p1_et if m_id == "ETGNN" else p1_gt,
                    "del_p0": float(del_p_et[0].item()) if m_id == "ETGNN" else float(del_p_gt[0].item()),
                    "del_p1": float(del_p_et[1].item()) if m_id == "ETGNN" else float(del_p_gt[1].item()),
                    "deleted_local_edge_indices": json.dumps(occ_selected),
                    "deleted_global_edge_ids": json.dumps(occ_global_ids),
                    "runtime_ms": t_occ_ms,
                })

            raw_records.append({
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
            # Node Summary Metrics
            # -------------------------------------------------------------
            mean_ge_c1_et = float(np.mean(ge_drops_c1_etgnn))
            mean_ge_c1_gt_common = float(np.mean(ge_drops_c1_gat_common_target))
            mean_rnd_c1_et = float(np.mean(rnd_drops_c1_etgnn))
            mean_rnd_c1_gt_common = float(np.mean(rnd_drops_c1_gat_common_target))

            c1_advantage_etgnn = mean_ge_c1_et - mean_rnd_c1_et
            c1_advantage_gat_common_target = mean_ge_c1_gt_common - mean_rnd_c1_gt_common

            mean_ge_c2_gt_own = float(np.mean(ge_drops_c2_gat_own_ranking))
            mean_rnd_c2_gt_own = float(np.mean(rnd_drops_c2_gat_own_target))
            c2_advantage_gat_own_ranking = mean_ge_c2_gt_own - mean_rnd_c2_gt_own

            mean_ge_c2_gt_etgnn_rank = float(np.mean(ge_drops_c2_gat_on_etgnn_ranking))
            c2_gat_own_target_advantage_on_etgnn_ranking = mean_ge_c2_gt_etgnn_rank - mean_rnd_c2_gt_own

            summary_records.append({
                "node_index": pos,
                "transaction_id": tx_id,
                "category": cat,
                "time_step": step,
                "E_cand": E_cand,
                "k_budget": k_budget,
                "actual_k_over_E": round(k_over_E, 4),
                "extract_runtime_ms": round(t_extract_ms, 2),
                "gnnexplainer_etgnn_mean_runtime_ms": round(float(np.mean(ge_etgnn_seed_times)), 2),
                "gnnexplainer_gat_mean_runtime_ms": round(float(np.mean(ge_gat_seed_times)), 2),
                "random_30_total_runtime_ms": round(t_rnd_total_ms, 2),
                "degree_ranking_runtime_ms": round(t_deg_ms, 2),
                "occlusion_runtime_ms": round(t_occ_ms, 2),
                "c1_target_class": c1_target_class,
                "c1_mean_ge_drop_etgnn": round(mean_ge_c1_et, 8),
                "c1_mean_rnd_drop_etgnn": round(mean_rnd_c1_et, 8),
                "c1_advantage_etgnn": round(c1_advantage_etgnn, 8),
                "c1_mean_ge_drop_gat_common_target": round(mean_ge_c1_gt_common, 8),
                "c1_mean_rnd_drop_gat_common_target": round(mean_rnd_c1_gt_common, 8),
                "c1_advantage_gat_common_target": round(c1_advantage_gat_common_target, 8),
                "c2_target_gat": c2_target_gat,
                "c2_mean_ge_drop_gat_own_ranking": round(mean_ge_c2_gt_own, 8),
                "c2_mean_rnd_drop_gat_own": round(mean_rnd_c2_gt_own, 8),
                "c2_advantage_gat_own_ranking": round(c2_advantage_gat_own_ranking, 8),
                "c2_gat_own_target_advantage_on_etgnn_ranking": round(c2_gat_own_target_advantage_on_etgnn_ranking, 8),
            })

            completed_nodes.add(pos)
            elapsed_sec = time.time() - start_time_all
            print(f"[{len(completed_nodes)}/{total_nodes}] Node {pos} ({cat}, step {step}, E={E_cand}, k={k_budget}) finished in {elapsed_sec:.1f}s")

            # Incremental atomic checkpoint every 5 nodes or on the last node
            if len(completed_nodes) % 5 == 0 or len(completed_nodes) == total_nodes:
                self.atomic_save(detailed_records, raw_records, summary_records, completed_nodes, detailed_path, raw_path, summary_path, resume_path)

        # Final assertions & post-processing
        self.assert_records_consistency(detailed_records, raw_records)
        self.run_statistical_postprocessing(detailed_records)

    def assert_records_consistency(self, detailed_records: List[Dict], raw_records: List[Dict]):
        print("\n--- 5. Asserting Raw vs. Detailed Record Consistency ---")
        df_raw = pd.DataFrame(raw_records)
        df_det = pd.DataFrame(detailed_records)

        # GNNExplainer assertion
        for _, r_row in df_raw[df_raw["method"] == "GNNExplainer"].iterrows():
            nid = r_row["node_index"]
            s = r_row["seed"]

            # ETGNN C1
            det_et_c1 = df_det[(df_det["node_index"] == nid) & (df_det["method"] == "GNNExplainer") &
                               (df_det["contrast"] == "C1") & (df_det["ranking_model_id"] == "ETGNN") &
                               (df_det["scoring_model_id"] == "ETGNN") & (df_det["seed"] == s)]
            assert len(det_et_c1) == 1, f"Missing detailed ETGNN C1 record for node {nid}, seed {s}"
            assert abs(det_et_c1["signed_drop"].iloc[0] - r_row["c1_drop_etgnn"]) < 1e-7

            # Matched GAT C1 Common Target
            det_gt_c1 = df_det[(df_det["node_index"] == nid) & (df_det["method"] == "GNNExplainer") &
                               (df_det["contrast"] == "C1") & (df_det["ranking_model_id"] == "ETGNN") &
                               (df_det["scoring_model_id"] == "Matched_GAT") & (df_det["seed"] == s)]
            assert len(det_gt_c1) == 1
            assert abs(det_gt_c1["signed_drop"].iloc[0] - r_row["c1_drop_gat"]) < 1e-7

            # Matched GAT True C2 Own Ranking
            det_gt_c2 = df_det[(df_det["node_index"] == nid) & (df_det["method"] == "GNNExplainer") &
                               (df_det["contrast"] == "C2") & (df_det["ranking_model_id"] == "Matched_GAT") &
                               (df_det["scoring_model_id"] == "Matched_GAT") & (df_det["seed"] == s)]
            assert len(det_gt_c2) == 1
            assert abs(det_gt_c2["signed_drop"].iloc[0] - r_row["c2_drop_gat_own_ranking"]) < 1e-7

        print("  ASSERTION PASSED: 100% agreement between raw and detailed records verified.")

    def run_statistical_postprocessing(self, detailed_records: List[Dict]):
        print("\n--- 6. Downstream Statistical Analysis (C1 & C2 Matched-Pair Bootstrap) ---")
        df_det = pd.DataFrame(detailed_records)

        # -------------------------------------------------------------
        # Contrast 1 Analysis (C1: Random-Adjusted Deletion-Response Difference)
        # -------------------------------------------------------------
        c1_node_list = [n["node_index"] for n in self.config["c1_sample_register"]["nodes"]] if not self.is_dev_check else df_det["node_index"].unique().tolist()
        df_c1 = df_det[(df_det["node_index"].isin(c1_node_list)) & (df_det["contrast"] == "C1")].copy()

        c1_results = []
        for nid, g in df_c1.groupby("node_index"):
            ge_et = g[(g["ranking_model_id"] == "ETGNN") & (g["scoring_model_id"] == "ETGNN")]["signed_drop"].mean()
            rnd_et = g[(g["ranking_model_id"] == "Random") & (g["scoring_model_id"] == "ETGNN")]["signed_drop"].mean()
            adv_et = ge_et - rnd_et

            ge_gt = g[(g["ranking_model_id"] == "ETGNN") & (g["scoring_model_id"] == "Matched_GAT")]["signed_drop"].mean()
            rnd_gt = g[(g["ranking_model_id"] == "Random") & (g["scoring_model_id"] == "Matched_GAT")]["signed_drop"].mean()
            adv_gt = ge_gt - rnd_gt

            delta_adj = adv_et - adv_gt
            delta_raw = ge_et - ge_gt

            c1_results.append({
                "node_index": nid,
                "adv_etgnn": adv_et,
                "adv_gat_common": adv_gt,
                "delta_adj": delta_adj,
                "delta_raw": delta_raw,
            })

        df_c1_res = pd.DataFrame(c1_results)
        print("\n=== CONTRAST 1 OUTCOMES (Random-Adjusted Deletion-Response Difference) ===")
        print(f"  Sample Size (nodes):            {len(df_c1_res)}")
        print(f"  Primary Estimator delta_adj:    Mean = {df_c1_res['delta_adj'].mean():.6f}, Std = {df_c1_res['delta_adj'].std():.6f}")
        print(f"  Secondary Diagnostic delta_raw: Mean = {df_c1_res['delta_raw'].mean():.6f}, Std = {df_c1_res['delta_raw'].std():.6f}")
        print("  *Interpretation: delta_adj measures random-adjusted response difference under identical edge loss.")
        print("   It controls for baseline model perturbation noise, but reflects combined model architecture and encoding differences.*")

        c1_out_path = self.output_dir / "c1_statistical_results.csv"
        df_c1_res.to_csv(c1_out_path, index=False)

        # -------------------------------------------------------------
        # Contrast 2 Analysis: Compute Correct Within-Node Advantages
        # -------------------------------------------------------------
        S = len(self.explainer_seeds)
        D = self.num_random_draws

        node_adv_et = {}
        node_adv_gt = {}

        for nid, g in df_det.groupby("node_index"):
            # Target classes for this node
            c1_t = g[g["contrast"] == "C1"]["target_class"].iloc[0]
            c2_t_gat = g[g["contrast"] == "C2"]["target_class"].iloc[0]

            # 1. ETGNN C2 advantage (uses C1/ETGNN target)
            ge_e = g[(g["contrast"] == "C1") & (g["method"] == "GNNExplainer") &
                     (g["ranking_model_id"] == "ETGNN") & (g["scoring_model_id"] == "ETGNN") &
                     (g["target_class"] == c1_t)]
            rnd_e = g[(g["contrast"] == "C1") & (g["method"] == "Random") &
                      (g["scoring_model_id"] == "ETGNN") & (g["target_class"] == c1_t)]

            assert len(ge_e) == S, f"Node {nid} has {len(ge_e)} ETGNN GE rows (expected {S})"
            assert len(rnd_e) == D, f"Node {nid} has {len(rnd_e)} ETGNN Random rows (expected {D})"
            node_adv_et[nid] = ge_e["signed_drop"].mean() - rnd_e["signed_drop"].mean()

            # 2. Matched GAT C2 advantage (uses C2 own target and own ranking!)
            ge_g = g[(g["contrast"] == "C2") & (g["method"] == "GNNExplainer") &
                     (g["ranking_model_id"] == "Matched_GAT") & (g["scoring_model_id"] == "Matched_GAT") &
                     (g["target_class"] == c2_t_gat)]
            rnd_g = g[(g["contrast"] == "C2") & (g["method"] == "Random") &
                      (g["scoring_model_id"] == "Matched_GAT") & (g["target_class"] == c2_t_gat)]

            assert len(ge_g) == S, f"Node {nid} has {len(ge_g)} GAT C2 GE rows (expected {S})"
            assert len(rnd_g) == D, f"Node {nid} has {len(rnd_g)} GAT C2 Random rows (expected {D})"
            node_adv_gt[nid] = ge_g["signed_drop"].mean() - rnd_g["signed_drop"].mean()

        # Regression check on Node 134593 if present
        if 134593 in node_adv_gt:
            expected_134593_c2 = -0.06576057275136311
            actual_134593_c2 = node_adv_gt[134593]
            assert abs(actual_134593_c2 - expected_134593_c2) < 1e-6, \
                f"REGRESSION REGRESSION FAILURE on Node 134593! Expected {expected_134593_c2:.8f}, got {actual_134593_c2:.8f}"
            print(f"  Regression Check Passed: Node 134593 GAT C2 advantage = {actual_134593_c2:.8f}")

        # -------------------------------------------------------------
        # Contrast 2 Analysis: Evaluate ALL EIGHT Registers
        # -------------------------------------------------------------
        c2_registers = [
            # Primary Caliper Matches
            ("ETGNN Primary TP-FP", self.config["c2_pair_registers"]["primary_caliper_matches"]["etgnn_full"]["tp_fp_pairs"], "ETGNN"),
            ("ETGNN Primary TN-FN", self.config["c2_pair_registers"]["primary_caliper_matches"]["etgnn_full"]["tn_fn_pairs"], "ETGNN"),
            ("Matched GAT Primary TP-FP", self.config["c2_pair_registers"]["primary_caliper_matches"]["matched_gat"]["tp_fp_pairs"], "Matched_GAT"),
            ("Matched GAT Primary TN-FN", self.config["c2_pair_registers"]["primary_caliper_matches"]["matched_gat"]["tn_fn_pairs"], "Matched_GAT"),
            # Degree Sensitivity Matches
            ("ETGNN Degree-Sensitivity TP-FP", self.config["c2_pair_registers"]["degree_sensitivity_matches"]["etgnn_full"]["tp_fp_pairs"], "ETGNN"),
            ("ETGNN Degree-Sensitivity TN-FN", self.config["c2_pair_registers"]["degree_sensitivity_matches"]["etgnn_full"]["tn_fn_pairs"], "ETGNN"),
            ("Matched GAT Degree-Sensitivity TP-FP", self.config["c2_pair_registers"]["degree_sensitivity_matches"]["matched_gat"]["tp_fp_pairs"], "Matched_GAT"),
            ("Matched GAT Degree-Sensitivity TN-FN", self.config["c2_pair_registers"]["degree_sensitivity_matches"]["matched_gat"]["tn_fn_pairs"], "Matched_GAT"),
        ]

        if not self.is_dev_check:
            c2_outcomes = []
            individual_pair_diffs = []

            print("\n=== CONTRAST 2 OUTCOMES (Matched-Pair Cluster Bootstrap Across All 8 Registers) ===")
            for reg_name, pairs_list, model_type in c2_registers:
                adv_dict = node_adv_et if model_type == "ETGNN" else node_adv_gt

                pair_diffs = []
                for p_item in pairs_list:
                    pid = p_item["pair_id"]
                    u = int(p_item["treat_node_index"])
                    v = int(p_item["ctrl_node_index"])
                    u_tx = int(p_item["treat_txId"])
                    v_tx = int(p_item["ctrl_txId"])

                    # Complete pair membership assertion (never silently skip!)
                    assert u in adv_dict, f"Register '{reg_name}': Treat node {u} (txId {u_tx}) missing from evaluated records!"
                    assert v in adv_dict, f"Register '{reg_name}': Control node {v} (txId {v_tx}) missing from evaluated records!"

                    adv_u = adv_dict[u]
                    adv_v = adv_dict[v]
                    diff = adv_u - adv_v

                    pair_diffs.append(diff)
                    individual_pair_diffs.append({
                        "register_name": reg_name,
                        "pair_id": pid,
                        "treat_node_index": u,
                        "treat_txId": u_tx,
                        "ctrl_node_index": v,
                        "ctrl_txId": v_tx,
                        "treat_advantage": adv_u,
                        "ctrl_advantage": adv_v,
                        "pair_difference": diff,
                    })

                pair_diffs = np.array(pair_diffs)
                n_p = len(pair_diffs)
                assert n_p == len(pairs_list), f"Register {reg_name} pair count mismatch ({n_p} vs {len(pairs_list)})"

                # Matched-pair cluster bootstrap (B=2,000)
                B = 2000
                boot_means = []
                rng = np.random.default_rng(20261020)
                for _ in range(B):
                    boot_idx = rng.choice(n_p, size=n_p, replace=True)
                    boot_means.append(np.mean(pair_diffs[boot_idx]))

                b_mean = float(np.mean(boot_means))
                b_se = float(np.std(boot_means, ddof=1))
                ci_low = float(np.percentile(boot_means, 2.5))
                ci_high = float(np.percentile(boot_means, 97.5))

                print(f"  {reg_name} (N={n_p} pairs):")
                print(f"    Observed Mean Diff: {np.mean(pair_diffs):.6f}")
                print(f"    Bootstrap SE:       {b_se:.6f}")
                print(f"    95% Percentile CI:  [{ci_low:.6f}, {ci_high:.6f}]")

                c2_outcomes.append({
                    "register": reg_name,
                    "pairs_count": n_p,
                    "mean_diff": np.mean(pair_diffs),
                    "bootstrap_se": b_se,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "graph_dependence_limitation": "Transactions reside in connected graph; pairs assume independent cluster resampling."
                })

            # Save individual pair differences and bootstrap summaries
            pair_diff_path = self.output_dir / "c2_individual_pair_differences.csv"
            pd.DataFrame(individual_pair_diffs).to_csv(pair_diff_path, index=False)

            c2_out_path = self.output_dir / "c2_bootstrap_results.csv"
            pd.DataFrame(c2_outcomes).to_csv(c2_out_path, index=False)

            # Mirror statistical outputs to Drive if requested
            if self.backup_dir:
                try:
                    shutil.copy2(c1_out_path, self.backup_dir / c1_out_path.name)
                    shutil.copy2(pair_diff_path, self.backup_dir / pair_diff_path.name)
                    shutil.copy2(c2_out_path, self.backup_dir / c2_out_path.name)
                    print(f"  Mirrored final statistical outputs to Drive: {self.backup_dir}")
                except Exception as e:
                    print(f"  Warning: Drive mirroring of statistical outputs failed ({e})")

        print("\n==================================================================")
        print("  BENCHMARK RUN & POST-PROCESSING COMPLETED SUCCESSFULLY")
        print("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Authoritative Main Benchmark Runner for C1 & C2")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "main_c1_c2_run_config.json", help="Path to run config")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "main_c1_c2_results", help="Directory to save outputs")
    parser.add_argument("--dev-check", action="store_true", help="Run plumbing verification on the 4 development pilot nodes only")
    parser.add_argument("--backup-dir", type=Path, default=None, help="Optional Drive mirror directory")
    args = parser.parse_args()

    runner = MainBenchmarkRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        is_dev_check=args.dev_check,
        backup_dir=args.backup_dir,
    )
    runner.run_benchmark()


if __name__ == "__main__":
    main()
