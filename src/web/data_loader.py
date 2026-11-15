from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
import torch
try:
    from torch_geometric.explain import Explainer, GNNExplainer
except Exception as e:
    # Fallback when torch_geometric or its dependencies (e.g., sympy) are unavailable.
    Explainer = None
    GNNExplainer = None
    import logging
    logging.getLogger("etgnn_platform").warning(f"Explainer import failed: {e}")
from torch_geometric.utils import k_hop_subgraph, subgraph
from xgboost import XGBClassifier

from src.models.etgnn import ETGNNClassifier
from src.models.matched_gat import MatchedGATClassifier
from src.models.reliability_etgnn import ResidualETGNNClassifier

logger = logging.getLogger("etgnn_platform")
ROOT = Path(__file__).resolve().parents[2]


class ResearchDataLoader:
    """
    Singleton data and model loader for the ETGNN Research Demonstration Platform.
    Loads real artifacts, checkpoints, and datasets strictly in inference mode.
    """

    def __init__(self):
        if threading.current_thread() is threading.main_thread():
            try:
                torch.set_num_threads(2)
            except RuntimeError:
                pass
        self.root = ROOT
        self._lock = threading.RLock()
        self._sha256_cache: Dict[str, Optional[str]] = {}
        self.processed_dir = self.root / "data" / "processed"
        self.checkpoints_dir = self.root / "results" / "checkpoints"
        self.tables_dir = self.root / "results" / "tables"
        self.explanations_dir = self.root / "results" / "explanations"
        self.main_c1_c2_dir = self.root / "results" / "main_c1_c2_results"

        # 1. Load node metadata
        nodes_path = self.processed_dir / "nodes_with_masks.csv"
        if not nodes_path.exists():
            raise FileNotFoundError(f"Missing node metadata artifact: {nodes_path}")
        self.nodes_df = pd.read_csv(nodes_path)
        
        # Build index lookups
        self.tx_to_index: Dict[int, int] = {
            int(tx): int(idx)
            for tx, idx in zip(self.nodes_df["txId"], self.nodes_df["node_index"])
        }
        self.index_to_tx: Dict[int, int] = {
            int(idx): int(tx)
            for tx, idx in zip(self.nodes_df["txId"], self.nodes_df["node_index"])
        }

        # 2. Features mmap
        features_path = self.processed_dir / "scaled_features.npy"
        if not features_path.exists():
            raise FileNotFoundError(f"Missing scaled features: {features_path}")
        self.features = np.load(features_path, mmap_mode="r")

        # 3. Test graph
        graph_path = self.processed_dir / "test_graph.pt"
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing test graph: {graph_path}")
        self.graph = torch.load(graph_path, map_location="cpu", weights_only=True)

        # 4. Checkpoints & Models
        self._load_models()

        # 5. Pre-score full graph for ETGNN and Matched GAT for sub-millisecond lookups
        self._precompute_graph_probabilities()

        # 6. Preload exemplar connected explanations
        self._load_exemplar_explanations()

        # 7. Index intervention records if available
        self._load_intervention_index()

    def _load_models(self):
        # Random Forest
        rf_path = self.checkpoints_dir / "random_forest.joblib"
        if not rf_path.exists():
            raise FileNotFoundError(f"Missing RF checkpoint: {rf_path}")
        self.rf = joblib.load(rf_path)

        # XGBoost
        xgb_path = self.checkpoints_dir / "xgboost.json"
        if not xgb_path.exists():
            raise FileNotFoundError(f"Missing XGB checkpoint: {xgb_path}")
        self.xgb = XGBClassifier(n_jobs=1)
        self.xgb.load_model(str(xgb_path))

        # ETGNN
        etgnn_path = self.checkpoints_dir / "etgnn.pt"
        if not etgnn_path.exists():
            raise FileNotFoundError(f"Missing ETGNN checkpoint: {etgnn_path}")
        self.etgnn = ETGNNClassifier()
        self.etgnn.load_state_dict(
            torch.load(etgnn_path, map_location="cpu", weights_only=True)
        )
        self.etgnn.eval()

        # Matched GAT
        gat_path = self.checkpoints_dir / "matched_gat.pt"
        if not gat_path.exists():
            raise FileNotFoundError(f"Missing Matched GAT checkpoint: {gat_path}")
        self.matched_gat = MatchedGATClassifier()
        self.matched_gat.load_state_dict(
            torch.load(gat_path, map_location="cpu", weights_only=True)
        )
        self.matched_gat.eval()

        # Residual ETGNN (93f Local, Seed 42 - Phase 4 Condition B)
        res_path = ROOT / "results" / "phase4_experiments" / "checkpoints" / "P4-Cond_B_93f_s42.pt"
        if res_path.exists():
            self.residual_etgnn = ResidualETGNNClassifier(input_dim=93, time_dim=16, hidden_dim=64, dropout=0.3)
            self.residual_etgnn.load_state_dict(
                torch.load(res_path, map_location="cpu", weights_only=True)
            )
            self.residual_etgnn.eval()
        else:
            self.residual_etgnn = None

    def _expected_cache_provenance(self) -> Dict[str, Any]:
        """Build cryptographic and structural provenance signature for probability cache validity."""
        return {
            "model_architectures": {
                "etgnn": "ETGNNClassifier:v1.0:in=165:time=16:hidden=64:heads=4:layers=2",
                "matched_gat": "MatchedGATClassifier:v1.0:in=165:time=16:hidden=64:heads=4:layers=2",
            },
            "checkpoint_sha256": {
                "etgnn": self._checkpoint_sha256("results/checkpoints/etgnn.pt"),
                "matched_gat": self._checkpoint_sha256("results/checkpoints/matched_gat.pt"),
            },
            "dataset_identity": "Elliptic_Bitcoin_Dataset_v1_203769_txs_49_steps",
            "graph_identity": {
                "artifact": "data/processed/test_graph.pt",
                "num_nodes": int(self.graph["x"].shape[0]),
                "num_edges": int(self.graph["edge_index"].shape[1]),
            },
            "feature_dimensions": {
                "num_nodes": int(self.graph["x"].shape[0]),
                "input_dim": int(self.graph["x"].shape[1]),
            },
            "preprocessing_version": {
                "scaled_features_rows": int(self.features.shape[0]),
                "scaled_features_cols": int(self.features.shape[1]),
                "train_steps": "1..32",
            },
            "node_ordering": {
                "nodes_count": int(len(self.nodes_df)),
                "first_tx_id": int(self.nodes_df["txId"].iloc[0]),
                "last_tx_id": int(self.nodes_df["txId"].iloc[-1]),
                "nodes_csv_sha256": self._checkpoint_sha256("data/processed/nodes_with_masks.csv"),
            },
            "inference_config": {
                "eval_mode": True,
                "grad_enabled": False,
                "output_transform": "softmax_dim1_class1",
            },
        }

    def verify_probability_cache_provenance(self) -> Dict[str, Any]:
        """Verify whether the on-disk probability cache matches all 8 provenance invariants."""
        cache_path = self.processed_dir / "precomputed_etgnn_gat_probs.npz"
        meta_path = self.processed_dir / "precomputed_etgnn_gat_probs.meta.json"
        expected = self._expected_cache_provenance()
        if not cache_path.exists() or not meta_path.exists():
            return {"valid": False, "reason": "Cache or metadata file missing", "expected": expected}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                saved_meta = json.load(f)
            if saved_meta != expected:
                return {"valid": False, "reason": "Provenance metadata mismatch", "expected": expected, "saved": saved_meta}
            with np.load(cache_path) as cached:
                et_p = cached["etgnn_probs"]
                gat_p = cached["matched_gat_probs"]
                n_expected = expected["feature_dimensions"]["num_nodes"]
                if et_p.shape != (n_expected,) or gat_p.shape != (n_expected,):
                    return {"valid": False, "reason": "Cached probability array shape mismatch"}
                if not (np.isfinite(et_p).all() and np.isfinite(gat_p).all()):
                    return {"valid": False, "reason": "Cached probability array contains non-finite values"}
            return {"valid": True, "provenance": saved_meta}
        except Exception as exc:
            return {"valid": False, "reason": f"Cache verification error: {exc}"}

    def _precompute_graph_probabilities(self):
        """Precompute full graph probabilities for fast lookup across 203k nodes, with provenance-verified disk cache."""
        cache_path = self.processed_dir / "precomputed_etgnn_gat_probs.npz"
        meta_path = self.processed_dir / "precomputed_etgnn_gat_probs.meta.json"
        expected_meta = self._expected_cache_provenance()

        check = self.verify_probability_cache_provenance()
        if check["valid"]:
            with np.load(cache_path) as cached:
                self.etgnn_probs = cached["etgnn_probs"].copy()
                self.matched_gat_probs = cached["matched_gat_probs"].copy()
            return

        logger.info("Recomputing full-graph ETGNN and Matched GAT probabilities (%s)", check.get("reason"))
        with self._lock, torch.no_grad():
            et_logits = self.etgnn(
                self.graph["x"], self.graph["edge_index"], self.graph["time_step"]
            )
            self.etgnn_probs = torch.softmax(et_logits, dim=1)[:, 1].numpy()

            gat_logits = self.matched_gat(
                self.graph["x"], self.graph["edge_index"], self.graph["time_step"]
            )
            self.matched_gat_probs = torch.softmax(gat_logits, dim=1)[:, 1].numpy()

        try:
            np.savez(
                cache_path,
                etgnn_probs=self.etgnn_probs,
                matched_gat_probs=self.matched_gat_probs,
            )
            meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_exemplar_explanations(self):
        self.exemplar_explanations: Dict[int, Dict[str, Any]] = {}
        for case in ("tp", "fp", "tn", "fn"):
            path = self.explanations_dir / f"connected_{case}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    item = json.load(f)
                    node_idx = int(item["node_index"])
                    tx_id = self.index_to_tx.get(node_idx)
                    item["transaction_id"] = tx_id
                    item["is_frozen_research_artifact"] = True
                    item["provenance"] = f"Frozen research artifact: results/explanations/connected_{case}.json"
                    
                    orig_p = float(item.get("original_predicted_class_probability", 0.0))
                    del_p = float(item.get("probability_after_removing_top_edges", orig_p))
                    item["probability_after_intervention"] = del_p
                    item["probability_after_removing_top_edges"] = del_p
                    item["signed_drop"] = float(orig_p - del_p)
                    orig_class = int(orig_p >= 0.5)
                    new_class = int(del_p >= 0.5)
                    item["predicted_class_before"] = orig_class
                    item["predicted_class_after"] = new_class
                    item["class_changed"] = (orig_class != new_class)
                    cand_count = int(item.get("incoming_edges", item.get("subgraph_edges", len(item.get("top_edges", [])))))
                    k = int(item.get("budget_k", len(item.get("top_edges", []))))
                    item["candidate_edges_count"] = cand_count
                    item["budget_k"] = k
                    item["discrete_deletion_pct"] = round((k / max(1, cand_count)) * 100, 2)
                    item["intervention_protocol"] = "k = ceil(0.25 * E_cand) on 1-hop incoming candidate edges"
                    item["trained_model_frozen"] = True
                    
                    self.exemplar_explanations[node_idx] = item

    def _load_intervention_index(self):
        # Load a potentially huge CSV of intervention records in a memory‑efficient way.
        # We only need up to 10 records per node_index for quick lookup.
        self.intervention_records_by_node: Dict[int, List[Dict[str, Any]]] = {}
        path = self.main_c1_c2_dir / "detailed_intervention_records.csv"
        if not path.exists():
            return
        # Iterate over the CSV in chunks to avoid loading the whole file at once.
        # We keep the first 10 rows encountered for each node_index (equivalent to head(10) after grouping).
        chunk_iter = pd.read_csv(path, chunksize=100_000)
        for chunk in chunk_iter:
            # Ensure node_index is integer for consistent dict keys.
            if "node_index" not in chunk.columns:
                continue
            chunk["node_index"] = chunk["node_index"].astype(int)
            for node_idx, group in chunk.groupby("node_index"):
                existing = self.intervention_records_by_node.get(node_idx, [])
                # Append rows from this chunk until we have at most 10 per node.
                needed = 10 - len(existing)
                if needed <= 0:
                    continue
                # Take up to `needed` rows from group (preserving order).
                rows_to_add = group.head(needed).to_dict(orient="records")
                self.intervention_records_by_node[node_idx] = existing + rows_to_add


    def get_nodes_page(
        self,
        limit: int = 25,
        offset: int = 0,
        split: Optional[str] = None,
        label: Optional[int] = None,
        time_step: Optional[int] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Paginated transaction query with filtering."""
        df = self.nodes_df

        if search:
            search_str = search.strip()
            if search_str.isdigit():
                df = df[df["txId"].astype(str).str.contains(search_str)]

        if split == "train":
            df = df[df["train_mask"] == True]
        elif split == "val" or split == "validation":
            df = df[df["val_mask"] == True]
        elif split == "test":
            df = df[df["test_mask"] == True]
        elif split == "unlabelled":
            df = df[
                (df["train_mask"] == False)
                & (df["val_mask"] == False)
                & (df["test_mask"] == False)
            ]

        if label is not None:
            df = df[df["label"] == label]

        if time_step is not None:
            df = df[df["time_step"] == time_step]

        total = len(df)
        page_df = df.iloc[offset : offset + limit]

        items = []
        for _, row in page_df.iterrows():
            idx = int(row["node_index"])
            tx_id = int(row["txId"])
            lbl = int(row["label"])
            sp = (
                "train"
                if bool(row["train_mask"])
                else (
                    "validation"
                    if bool(row["val_mask"])
                    else ("test" if bool(row["test_mask"]) else "unlabelled")
                )
            )
            items.append(
                {
                    "node_index": idx,
                    "transaction_id": tx_id,
                    "time_step": int(row["time_step"]),
                    "actual_label": None if lbl == -1 else lbl,
                    "actual_label_name": (
                        "illicit" if lbl == 1 else ("licit" if lbl == 0 else "unknown")
                    ),
                    "split": sp,
                    "etgnn_prob": float(self.etgnn_probs[idx]),
                    "matched_gat_prob": float(self.matched_gat_probs[idx]),
                    "has_frozen_explanation": idx in self.exemplar_explanations,
                    "has_intervention_record": idx in self.intervention_records_by_node,
                }
            )

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": items,
        }

    def get_transaction(self, tx_id: int) -> Dict[str, Any]:
        """Fetch complete transaction details, features, and model scores."""
        node_idx = self.tx_to_index.get(tx_id)
        if node_idx is None:
            raise KeyError(f"Transaction ID {tx_id} not found in Elliptic dataset.")

        row = self.nodes_df.iloc[node_idx]
        lbl = int(row["label"])
        ts = int(row["time_step"])
        sp = (
            "train"
            if bool(row["train_mask"])
            else (
                "validation"
                if bool(row["val_mask"])
                else ("test" if bool(row["test_mask"]) else "unlabelled")
            )
        )

        feat_vector = np.asarray(self.features[node_idx : node_idx + 1])
        rf_prob = float(self.rf.predict_proba(feat_vector)[0, 1])
        xgb_prob = float(self.xgb.predict_proba(feat_vector)[0, 1])
        et_prob = float(self.etgnn_probs[node_idx])
        gat_prob = float(self.matched_gat_probs[node_idx])

        # Feature summary (local 0..92, aggregate 93..164)
        local_feats = feat_vector[0, :93]
        agg_feats = feat_vector[0, 93:165]

        # Check degree
        edges = self.graph["edge_index"]
        in_deg = int((edges[1] == node_idx).sum())
        out_deg = int((edges[0] == node_idx).sum())

        res_et_prob = 0.0
        if self.residual_etgnn is not None:
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                node_idx,
                2,
                self.graph["edge_index"],
                relabel_nodes=True,
                flow="source_to_target",
                num_nodes=self.graph["x"].size(0),
            )
            sub_x93 = self.graph["x"][subset, :93]
            center_mapped = int(mapping.item())
            sub_time = self.graph["time_step"][subset]
            with torch.no_grad():
                res_logits = self.residual_etgnn(sub_x93, sub_edge_index, sub_time)
                res_et_prob = float(torch.softmax(res_logits[center_mapped : center_mapped + 1], dim=1)[0, 1])

        return {
            "transaction_id": tx_id,
            "node_index": node_idx,
            "time_step": ts,
            "split": sp,
            "actual_label": None if lbl == -1 else lbl,
            "actual_label_name": (
                "illicit" if lbl == 1 else ("licit" if lbl == 0 else "unknown")
            ),
            "graph_degree": {
                "in_degree": in_deg,
                "out_degree": out_deg,
                "total_degree": in_deg + out_deg,
            },
            "features_summary": {
                "total_features": 165,
                "local_features_count": 93,
                "aggregate_features_count": 72,
                "local_mean": float(local_feats.mean()),
                "local_std": float(local_feats.std()),
                "aggregate_mean": float(agg_feats.mean()),
                "aggregate_std": float(agg_feats.std()),
                "sample_features": {
                    f"feat_{i}": float(feat_vector[0, i]) for i in range(10)
                },
            },
            "predictions": {
                "etgnn": {
                    "model_id": "ETGNN",
                    "display_name": "ETGNN (Static Time-Encoded GAT)",
                    "illicit_probability": et_prob,
                    "predicted_class": int(et_prob >= 0.5),
                    "confidence": float(abs(et_prob - 0.5) * 2),
                },
                "matched_gat": {
                    "model_id": "Matched_GAT",
                    "display_name": "Matched GAT (Structural Baseline)",
                    "illicit_probability": gat_prob,
                    "predicted_class": int(gat_prob >= 0.5),
                    "confidence": float(abs(gat_prob - 0.5) * 2),
                },
                "residual_etgnn": {
                    "model_id": "Residual_ETGNN_93f",
                    "display_name": "Residual ETGNN (Ego-Skip GAT, 93f Local)",
                    "illicit_probability": res_et_prob,
                    "predicted_class": int(res_et_prob >= 0.5),
                    "confidence": float(abs(res_et_prob - 0.5) * 2),
                },
                "random_forest": {
                    "model_id": "RandomForest",
                    "display_name": "Random Forest (Tabular)",
                    "illicit_probability": rf_prob,
                    "predicted_class": int(rf_prob >= 0.5),
                    "confidence": float(abs(rf_prob - 0.5) * 2),
                },
                "xgboost": {
                    "model_id": "XGBoost",
                    "display_name": "XGBoost (Tabular)",
                    "illicit_probability": xgb_prob,
                    "predicted_class": int(xgb_prob >= 0.5),
                    "confidence": float(abs(xgb_prob - 0.5) * 2),
                },
            },
            "explanation_status": {
                "has_frozen_explanation": node_idx in self.exemplar_explanations,
                "has_intervention_records": node_idx in self.intervention_records_by_node,
            },
        }

    def get_subgraph(
        self, tx_id: int, num_hops: int = 2, max_nodes: int = 150
    ) -> Dict[str, Any]:
        """
        Extract k-hop neighborhood for interactive graph visualization.
        """
        node_idx = self.tx_to_index.get(tx_id)
        if node_idx is None:
            raise KeyError(f"Transaction ID {tx_id} not found.")

        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx,
            num_hops,
            self.graph["edge_index"],
            relabel_nodes=True,
            flow="source_to_target",
            num_nodes=self.graph["x"].size(0),
        )

        # Cap nodes if neighborhood is excessively dense
        subset_list = subset.tolist()
        total_extracted_nodes = len(subset_list)
        is_truncated = total_extracted_nodes > max_nodes

        if is_truncated:
            # Keep center node at index 0 and first max_nodes - 1 neighbors
            center_mapped = int(mapping.item())
            subset_list = [subset_list[center_mapped]] + [
                n for i, n in enumerate(subset_list) if i != center_mapped
            ][: max_nodes - 1]
            subset = torch.tensor(subset_list, dtype=torch.long)
            # Induce subgraph strictly on capped subset
            sub_edge_index, _ = subgraph(
                subset,
                edge_index=self.graph["edge_index"],
                relabel_nodes=True,
                num_nodes=self.graph["x"].size(0),
            )
            center_idx_in_sub = 0
        else:
            center_idx_in_sub = int(mapping.item())

        # Build node metadata list
        nodes_out = []
        for i, global_idx in enumerate(subset.tolist()):
            row = self.nodes_df.iloc[global_idx]
            lbl = int(row["label"])
            is_center = i == center_idx_in_sub
            et_p = float(self.etgnn_probs[global_idx])
            nodes_out.append(
                {
                    "id": str(global_idx),
                    "subgraph_index": i,
                    "transaction_id": int(row["txId"]),
                    "time_step": int(row["time_step"]),
                    "actual_label": None if lbl == -1 else lbl,
                    "actual_label_name": (
                        "illicit" if lbl == 1 else ("licit" if lbl == 0 else "unknown")
                    ),
                    "etgnn_probability": et_p,
                    "is_center": is_center,
                }
            )

        # Build edge list
        edges_out = []
        srcs = sub_edge_index[0].tolist()
        dsts = sub_edge_index[1].tolist()
        for i, (s, d) in enumerate(zip(srcs, dsts)):
            edges_out.append(
                {
                    "id": f"e_{s}_{d}_{i}",
                    "source": str(subset[s].item()),
                    "target": str(subset[d].item()),
                    "source_subgraph_idx": s,
                    "target_subgraph_idx": d,
                    "is_incident_to_center": (
                        s == center_idx_in_sub or d == center_idx_in_sub
                    ),
                }
            )

        truncation_notice = (
            f"Visualization capped at {len(nodes_out)} nodes (complete {num_hops}-hop neighborhood has {total_extracted_nodes} nodes). "
            "Display is truncated for client-side rendering performance and does not represent the full neighborhood."
            if is_truncated
            else None
        )

        return {
            "center_transaction_id": tx_id,
            "center_node_index": node_idx,
            "num_hops": num_hops,
            "node_count": len(nodes_out),
            "edge_count": len(edges_out),
            "is_truncated": is_truncated,
            "total_untruncated_nodes": total_extracted_nodes,
            "truncation_notice": truncation_notice,
            "nodes": nodes_out,
            "edges": edges_out,
        }

    def predict(
        self, tx_id: int, features_override: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run forward pass for ETGNN, Matched GAT, RF, and XGBoost.
        Supports optional feature override for scenario testing.
        """
        node_idx = self.tx_to_index.get(tx_id)
        if node_idx is None:
            raise KeyError(f"Transaction ID {tx_id} not found in Elliptic dataset.")

        base_features = np.asarray(self.features[node_idx : node_idx + 1]).copy()
        
        has_override = False
        if features_override:
            for k, val in features_override.items():
                if k.startswith("feat_"):
                    try:
                        feat_col = int(k.replace("feat_", ""))
                        if 0 <= feat_col < 165:
                            base_features[0, feat_col] = float(val)
                            has_override = True
                    except ValueError:
                        pass

        with self._lock:
            # Tabular predictions
            rf_p = float(self.rf.predict_proba(base_features)[0, 1])
            xgb_p = float(self.xgb.predict_proba(base_features)[0, 1])

            # GNN predictions
            if not has_override:
                et_p = float(self.etgnn_probs[node_idx])
                gat_p = float(self.matched_gat_probs[node_idx])
            else:
                # Run exact 2-hop subgraph forward pass with overridden features
                subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                    node_idx,
                    2,
                    self.graph["edge_index"],
                    relabel_nodes=True,
                    flow="source_to_target",
                    num_nodes=self.graph["x"].size(0),
                )
                sub_x = self.graph["x"][subset].clone()
                center_mapped = int(mapping.item())
                sub_x[center_mapped] = torch.tensor(base_features[0], dtype=torch.float32)
                sub_time = self.graph["time_step"][subset]

                with torch.no_grad():
                    et_logits = self.etgnn(sub_x, sub_edge_index, sub_time)
                    et_p = float(torch.softmax(et_logits[center_mapped : center_mapped + 1], dim=1)[0, 1])

                    gat_logits = self.matched_gat(sub_x, sub_edge_index, sub_time)
                    gat_p = float(torch.softmax(gat_logits[center_mapped : center_mapped + 1], dim=1)[0, 1])

            # Residual ETGNN (93f Local) prediction
            res_et_p = 0.0
            if self.residual_etgnn is not None:
                subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                    node_idx,
                    2,
                    self.graph["edge_index"],
                    relabel_nodes=True,
                    flow="source_to_target",
                    num_nodes=self.graph["x"].size(0),
                )
                sub_x93 = self.graph["x"][subset, :93].clone()
                center_mapped = int(mapping.item())
                if has_override:
                    sub_x93[center_mapped] = torch.tensor(base_features[0, :93], dtype=torch.float32)
                sub_time = self.graph["time_step"][subset]
                with torch.no_grad():
                    res_logits = self.residual_etgnn(sub_x93, sub_edge_index, sub_time)
                    res_et_p = float(torch.softmax(res_logits[center_mapped : center_mapped + 1], dim=1)[0, 1])

        return {
            "transaction_id": tx_id,
            "node_index": node_idx,
            "has_feature_override": has_override,
            "predictions": {
                "etgnn": {
                    "model_id": "ETGNN",
                    "display_name": "ETGNN (Static Time-Encoded GAT)",
                    "illicit_probability": et_p,
                    "predicted_class": int(et_p >= 0.5),
                    "confidence": float(abs(et_p - 0.5) * 2),
                },
                "matched_gat": {
                    "model_id": "Matched_GAT",
                    "display_name": "Matched GAT (Structural Baseline)",
                    "illicit_probability": gat_p,
                    "predicted_class": int(gat_p >= 0.5),
                    "confidence": float(abs(gat_p - 0.5) * 2),
                },
                "residual_etgnn": {
                    "model_id": "Residual_ETGNN_93f",
                    "display_name": "Residual ETGNN (Ego-Skip GAT, 93f Local)",
                    "illicit_probability": res_et_p,
                    "predicted_class": int(res_et_p >= 0.5),
                    "confidence": float(abs(res_et_p - 0.5) * 2),
                },
                "random_forest": {
                    "model_id": "RandomForest",
                    "display_name": "Random Forest (Tabular)",
                    "illicit_probability": rf_p,
                    "predicted_class": int(rf_p >= 0.5),
                    "confidence": float(abs(rf_p - 0.5) * 2),
                },
                "xgboost": {
                    "model_id": "XGBoost",
                    "display_name": "XGBoost (Tabular)",
                    "illicit_probability": xgb_p,
                    "predicted_class": int(xgb_p >= 0.5),
                    "confidence": float(abs(xgb_p - 0.5) * 2),
                },
            },
        }

    def get_explanation(self, tx_id: int) -> Dict[str, Any]:
        """
        Fetch or compute transaction explanation.
        Priority 1: Frozen connected exemplar artifact (connected_{tp,fp,tn,fn}.json).
        Priority 2: Frozen intervention record from main C1/C2 benchmark.
        Priority 3: Newly generated live demonstration via GNNExplainer (explicitly labeled).
        """
        node_idx = self.tx_to_index.get(tx_id)
        if node_idx is None:
            raise KeyError(f"Transaction ID {tx_id} not found in Elliptic dataset.")

        # Check exemplar artifacts
        if node_idx in self.exemplar_explanations:
            exp = dict(self.exemplar_explanations[node_idx])
            exp["is_frozen_research_artifact"] = True
            exp["scientific_safety_notice"] = (
                "Edge importance reflects model prediction sensitivity under intervention, "
                "not causal proof of illicit transaction routing."
            )
            return exp

        # Check intervention records
        if node_idx in self.intervention_records_by_node:
            records = self.intervention_records_by_node[node_idx]
            first = records[0]
            return {
                "transaction_id": tx_id,
                "node_index": node_idx,
                "is_frozen_research_artifact": True,
                "provenance": "Frozen C1/C2 detailed intervention records",
                "ranking_model_id": first.get("ranking_model_id"),
                "seed": first.get("seed"),
                "budget_k": first.get("budget_k"),
                "signed_drop": first.get("signed_drop"),
                "orig_p1": first.get("orig_p1"),
                "del_p1": first.get("del_p1"),
                "deleted_global_edge_ids": first.get("deleted_global_edge_ids"),
                "runtime_ms": first.get("runtime_ms"),
                "records_count": len(records),
                "records": records,
                "scientific_safety_notice": (
                    "Intervention records demonstrate prediction drop under edge deletion. "
                    "This is an XAI sensitivity metric, not empirical guilt."
                ),
            }

        # Otherwise, generate on-the-fly demonstration explanation
        return self._generate_live_demonstration_explanation(node_idx, tx_id)

    def _generate_live_demonstration_explanation(
        self, node_idx: int, tx_id: int
    ) -> Dict[str, Any]:
        """Compute live GNNExplainer on 2-hop neighborhood."""
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx,
            2,
            self.graph["edge_index"],
            relabel_nodes=True,
            flow="source_to_target",
            num_nodes=self.graph["x"].size(0),
        )

        center_sub_idx = int(mapping.item())
        sub_x = self.graph["x"][subset]
        sub_time = self.graph["time_step"][subset]

        cand_indices = (sub_edge_index[1] == center_sub_idx).nonzero().flatten()
        E_cand = len(cand_indices)
        orig_prob = float(self.etgnn_probs[node_idx])

        if sub_edge_index.numel() == 0 or E_cand == 0:
            return {
                "transaction_id": tx_id,
                "node_index": node_idx,
                "is_frozen_research_artifact": False,
                "is_live_demonstration": True,
                "provenance": "Isolated / Zero-incoming degree node in local time-step",
                "message": "Node has 0 incoming candidate edges in this time-step; no message-passing edges contribute to this node's representation.",
                "disclaimer": "This node has 0 incoming candidate edges; edge attribution is not applicable.",
                "subgraph_nodes": int(len(subset)),
                "subgraph_edges": int(sub_edge_index.shape[1]),
                "original_predicted_class_probability": orig_prob,
                "probability_after_intervention": orig_prob,
                "probability_after_removing_top_edges": orig_prob,
                "signed_drop": 0.0,
                "predicted_class_before": int(orig_prob >= 0.5),
                "predicted_class_after": int(orig_prob >= 0.5),
                "class_changed": False,
                "candidate_edges_count": 0,
                "budget_k": 0,
                "discrete_deletion_pct": 0.0,
                "top_edges": [],
            }

        top_edges = []
        top_edge_indices = []
        try:
            with self._lock:
                torch.manual_seed(42)
                explainer = Explainer(
                    model=self.etgnn,
                    algorithm=GNNExplainer(epochs=35),
                    explanation_type="model",
                    node_mask_type="attributes",
                    edge_mask_type="object",
                    model_config={
                        "mode": "multiclass_classification",
                        "task_level": "node",
                        "return_type": "raw",
                    },
                )

                explanation = explainer(
                    sub_x,
                    sub_edge_index,
                    index=center_sub_idx,
                    time_step=sub_time,
                )

            edge_mask = explanation.edge_mask
            top_k = min(5, edge_mask.numel())
            top_edge_scores, top_edge_indices = torch.topk(edge_mask, k=top_k)

            for score, e_idx in zip(top_edge_scores.tolist(), top_edge_indices.tolist()):
                s_idx = int(sub_edge_index[0, e_idx].item())
                d_idx = int(sub_edge_index[1, e_idx].item())
                top_edges.append(
                    {
                        "source_node_index": int(subset[s_idx].item()),
                        "target_node_index": int(subset[d_idx].item()),
                        "mask_value": float(score),
                    }
                )
        except Exception:
            # Fallback if edge gradients cannot be computed
            top_edges = []
            top_edge_indices = []

        orig_prob = float(self.etgnn_probs[node_idx])

        # Controlled intervention on frozen model
        cand_indices = (sub_edge_index[1] == center_sub_idx).nonzero().flatten()
        E_cand = len(cand_indices)
        import math
        budget_k = max(1, int(math.ceil(0.25 * max(1, E_cand))))
        discrete_pct = round((budget_k / E_cand) * 100, 2) if E_cand > 0 else 0.0

        del_prob = orig_prob
        if len(top_edge_indices) > 0:
            keep_mask = torch.ones(sub_edge_index.size(1), dtype=torch.bool)
            pruned_k = min(budget_k, len(top_edge_indices))
            keep_mask[top_edge_indices[:pruned_k]] = False
            pruned_edges = sub_edge_index[:, keep_mask]
            with torch.no_grad():
                del_logits = self.etgnn(sub_x, pruned_edges, sub_time)
                del_prob = float(
                    torch.softmax(del_logits[center_sub_idx : center_sub_idx + 1], dim=1)[0, 1]
                )

        signed_drop = float(orig_prob - del_prob)
        orig_class = int(orig_prob >= 0.5)
        new_class = int(del_prob >= 0.5)

        return {
            "transaction_id": tx_id,
            "node_index": node_idx,
            "is_frozen_research_artifact": False,
            "is_live_demonstration": True,
            "provenance": "PyG GNNExplainer (epochs=35, seed=42) on local 2-hop neighborhood",
            "disclaimer": "This explanation was newly generated for live demonstration purposes and is NOT frozen C1/C2 experimental evidence.",
            "scientific_safety_notice": "Edge importance values represent gradient-optimized model sensitivity, not causal proof of criminal activity.",
            "subgraph_nodes": int(len(subset)),
            "subgraph_edges": int(sub_edge_index.shape[1]),
            "original_predicted_class_probability": orig_prob,
            "probability_after_intervention": del_prob,
            "probability_after_removing_top_edges": del_prob,
            "signed_drop": signed_drop,
            "predicted_class_before": orig_class,
            "predicted_class_after": new_class,
            "class_changed": (orig_class != new_class),
            "candidate_edges_count": E_cand,
            "budget_k": budget_k,
            "discrete_deletion_pct": discrete_pct,
            "intervention_protocol": "k = ceil(0.25 * E_cand) on 1-hop incoming candidate edges",
            "trained_model_frozen": True,
            "top_edges": top_edges,
        }

    def execute_intervention(self, tx_id: int) -> Dict[str, Any]:
        """
        Execute controlled edge intervention protocol on a transaction with the trained model frozen.
        Protocol: k = ceil(0.25 * E_cand) on 1-hop incoming candidate edges.
        """
        node_idx = self.tx_to_index.get(tx_id)
        if node_idx is None:
            raise KeyError(f"Transaction ID {tx_id} not found in Elliptic dataset.")

        # Check if this node has a frozen exemplar explanation
        if node_idx in self.exemplar_explanations:
            exp = self.exemplar_explanations[node_idx]
            orig_p = float(exp["original_predicted_class_probability"])
            del_p = float(exp["probability_after_intervention"])
            orig_cls = int(orig_p >= 0.5)
            new_cls = int(del_p >= 0.5)
            return {
                "transaction_id": tx_id,
                "node_index": node_idx,
                "case": exp.get("case"),
                "model_id": "ETGNN",
                "checkpoint": "results/checkpoints/etgnn.pt",
                "trained_model_frozen": True,
                "original_probability": orig_p,
                "original_predicted_class": orig_cls,
                "intervened_probability": del_p,
                "intervened_predicted_class": new_cls,
                "score_difference": float(orig_p - del_p),
                "class_changed": (orig_cls != new_cls),
                "candidate_edges_count": exp["candidate_edges_count"],
                "budget_k": exp["budget_k"],
                "discrete_deletion_pct": exp["discrete_deletion_pct"],
                "intervention_protocol": "k = ceil(0.25 * E_cand) on 1-hop incoming candidate edges",
                "is_frozen_research_artifact": True,
                "provenance": exp["provenance"],
                "deleted_edges": exp.get("top_edges", [])[: exp["budget_k"]],
                "epistemic_distinction": (
                    "Individual illustrative intervention. Strictly distinguish from aggregate C1/C2 "
                    "study results (420 matched pairs / 275 balanced validation pairs). Model sensitivity "
                    "under edge deletion is an XAI diagnostic metric and does NOT constitute causal proof of fraud."
                ),
            }

        # Live calculation on 2-hop subgraph
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx,
            2,
            self.graph["edge_index"],
            relabel_nodes=True,
            flow="source_to_target",
            num_nodes=self.graph["x"].size(0),
        )
        center_sub_idx = int(mapping.item())
        sub_x = self.graph["x"][subset]
        sub_time = self.graph["time_step"][subset]

        with torch.no_grad():
            orig_logits = self.etgnn(sub_x, sub_edge_index, sub_time)
            orig_p = float(torch.softmax(orig_logits[center_sub_idx : center_sub_idx + 1], dim=1)[0, 1])

        cand_indices = (sub_edge_index[1] == center_sub_idx).nonzero().flatten()
        E_cand = len(cand_indices)
        import math

        if E_cand == 0:
            return {
                "transaction_id": tx_id,
                "node_index": node_idx,
                "model_id": "ETGNN",
                "checkpoint": "results/checkpoints/etgnn.pt",
                "trained_model_frozen": True,
                "original_probability": orig_p,
                "original_predicted_class": int(orig_p >= 0.5),
                "intervened_probability": orig_p,
                "intervened_predicted_class": int(orig_p >= 0.5),
                "score_difference": 0.0,
                "class_changed": False,
                "candidate_edges_count": 0,
                "budget_k": 0,
                "discrete_deletion_pct": 0.0,
                "message": "Node is an isolated transaction with degree 0 (0 incoming edges in this time-step); no edges eligible for intervention.",
                "epistemic_distinction": "Isolated transaction in out-of-time test graph. Zero intervention possible.",
            }

        budget_k = max(1, int(math.ceil(0.25 * E_cand)))
        discrete_pct = round((budget_k / E_cand) * 100, 2)

        keep_mask = torch.ones(sub_edge_index.size(1), dtype=torch.bool)
        pruned_sub_indices = cand_indices[:budget_k]
        keep_mask[pruned_sub_indices] = False
        pruned_edges = sub_edge_index[:, keep_mask]

        with torch.no_grad():
            del_logits = self.etgnn(sub_x, pruned_edges, sub_time)
            del_p = float(torch.softmax(del_logits[center_sub_idx : center_sub_idx + 1], dim=1)[0, 1])

        orig_cls = int(orig_p >= 0.5)
        new_cls = int(del_p >= 0.5)

        deleted_edges = []
        for e_idx in pruned_sub_indices.tolist():
            s_idx = int(sub_edge_index[0, e_idx].item())
            d_idx = int(sub_edge_index[1, e_idx].item())
            deleted_edges.append(
                {
                    "source_node_index": int(subset[s_idx].item()),
                    "target_node_index": int(subset[d_idx].item()),
                    "direction": "incoming_to_target",
                }
            )

        return {
            "transaction_id": tx_id,
            "node_index": node_idx,
            "model_id": "ETGNN",
            "checkpoint": "results/checkpoints/etgnn.pt",
            "trained_model_frozen": True,
            "original_probability": orig_p,
            "original_predicted_class": orig_cls,
            "intervened_probability": del_p,
            "intervened_predicted_class": new_cls,
            "score_difference": float(orig_p - del_p),
            "class_changed": (orig_cls != new_cls),
            "candidate_edges_count": E_cand,
            "budget_k": budget_k,
            "discrete_deletion_pct": discrete_pct,
            "discrete_pct_explanation": (
                f"With {E_cand} candidate edges, k={budget_k} deletion yields {discrete_pct}% deletion due to integer ceil rounding."
                if E_cand < 4
                else None
            ),
            "deleted_edges": deleted_edges,
            "intervention_protocol": "k = ceil(0.25 * E_cand) on 1-hop incoming candidate edges",
            "is_frozen_research_artifact": False,
            "provenance": "Live controlled edge intervention on frozen ETGNN checkpoint",
            "epistemic_distinction": (
                "Individual illustrative intervention. Strictly distinguish from aggregate C1/C2 "
                "study results (420 matched pairs / 275 balanced validation pairs). Model sensitivity "
                "under edge deletion is an XAI diagnostic metric and does NOT constitute causal proof of fraud."
            ),
        }

    def _checkpoint_sha256(self, rel_path: str) -> Optional[str]:
        """Compute and cache SHA-256 hash of a checkpoint file relative to repository root."""
        if rel_path in self._sha256_cache:
            return self._sha256_cache[rel_path]
        p = self.root / rel_path
        if not p.exists():
            self._sha256_cache[rel_path] = None
            return None
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        digest = h.hexdigest()
        self._sha256_cache[rel_path] = digest
        return digest

    def get_models_metadata(self) -> List[Dict[str, Any]]:
        """Return specifications and cryptographic provenance for all evaluated models."""
        return [
            {
                "id": "etgnn",
                "name": "ETGNN",
                "architecture": "Static Time-Encoded Graph Attention Network (GAT)",
                "category": "Temporal Graph Neural Network (Static)",
                "dataset": "Elliptic Bitcoin Transactions (203,769 nodes)",
                "feature_representation": "Full 165-feature (93 local + 72 1-hop aggregates)",
                "training_protocol": "Chronological split (Train: 1..32, Val: 33..40, Test: 41..49), Adam lr=1e-3",
                "input_dim": 165,
                "time_dim": 16,
                "hidden_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.3,
                "parameters_count": 11910,
                "checkpoint": "results/checkpoints/etgnn.pt",
                "checkpoint_sha256": self._checkpoint_sha256("results/checkpoints/etgnn.pt"),
                "held_out_auprc": 0.2250,
                "validation_auprc": 0.5488,
                "epistemic_status": "Static snapshot GAT with sinusoidal time features (4 heads x 16 dim). Test AUPRC 0.2250.",
            },
            {
                "id": "matched_gat",
                "name": "Matched GAT",
                "architecture": "Ablated Graph Attention Network (Zeroed Time Vector)",
                "category": "Ablated GNN Baseline",
                "dataset": "Elliptic Bitcoin Transactions (203,769 nodes)",
                "feature_representation": "Full 165-feature (93 local + 72 1-hop aggregates)",
                "training_protocol": "Chronological split (Train: 1..32, Val: 33..40, Test: 41..49), Adam lr=1e-3",
                "input_dim": 165,
                "time_dim": 16,
                "hidden_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.3,
                "parameters_count": 11910,
                "checkpoint": "results/checkpoints/matched_gat.pt",
                "checkpoint_sha256": self._checkpoint_sha256("results/checkpoints/matched_gat.pt"),
                "held_out_auprc": 0.2099,
                "validation_auprc": 0.5492,
                "epistemic_status": "Structurally identical to ETGNN (4 heads x 16 dim) but receives zero time encodings. Test AUPRC 0.2099.",
            },
            {
                "id": "random_forest",
                "name": "Random Forest",
                "architecture": "Ensemble of 100 Decision Trees (Scikit-Learn)",
                "category": "Tabular Baseline (Non-graph)",
                "dataset": "Elliptic Bitcoin Transactions (203,769 nodes)",
                "feature_representation": "Full 165-feature (93 local + 72 1-hop aggregates)",
                "training_protocol": "Chronological split (Train: 1..32, Val: 33..40, Test: 41..49), class_weight='balanced'",
                "input_dim": 165,
                "time_dim": 0,
                "parameters_count": None,
                "checkpoint": "results/checkpoints/random_forest.joblib",
                "checkpoint_sha256": self._checkpoint_sha256("results/checkpoints/random_forest.joblib"),
                "held_out_auprc": 0.6323,
                "epistemic_status": "Top performer across held-out test steps (AP 0.6323 vs ETGNN 0.2250). Non-residual GNN message passing introduced topological noise.",
            },
            {
                "id": "xgboost",
                "name": "XGBoost",
                "architecture": "Gradient Boosted Decision Trees",
                "category": "Tabular Baseline (Non-graph)",
                "dataset": "Elliptic Bitcoin Transactions (203,769 nodes)",
                "feature_representation": "Full 165-feature (93 local + 72 1-hop aggregates)",
                "training_protocol": "Chronological split (Train: 1..32, Val: 33..40, Test: 41..49)",
                "input_dim": 165,
                "time_dim": 0,
                "parameters_count": None,
                "checkpoint": "results/checkpoints/xgboost.json",
                "checkpoint_sha256": self._checkpoint_sha256("results/checkpoints/xgboost.json"),
                "held_out_auprc": 0.5472,
                "epistemic_status": "Second-highest baseline on held-out test steps (0.5472), confirming tabular feature superiority over non-residual message passing.",
            },
            {
                "id": "gcn",
                "name": "GCN",
                "architecture": "2-layer Graph Convolutional Network",
                "category": "Standard GNN Baseline",
                "dataset": "Elliptic Bitcoin Transactions (203,769 nodes)",
                "feature_representation": "Full 165-feature (93 local + 72 1-hop aggregates)",
                "training_protocol": "Chronological split (Train: 1..32, Val: 33..40, Test: 41..49)",
                "input_dim": 165,
                "time_dim": 0,
                "hidden_dim": 64,
                "num_layers": 2,
                "parameters_count": 11458,
                "checkpoint": "results/checkpoints/gcn.pt",
                "checkpoint_sha256": self._checkpoint_sha256("results/checkpoints/gcn.pt"),
                "held_out_auprc": 0.1306,
                "epistemic_status": "Isotropic spectral smoothing across heterophilic transaction edges degrades test AUPRC to 0.1306.",
            },
            {
                "id": "graphsage",
                "name": "GraphSAGE",
                "architecture": "2-layer GraphSAGE (Mean Aggregator)",
                "category": "Standard GNN Baseline",
                "dataset": "Elliptic Bitcoin Transactions (203,769 nodes)",
                "feature_representation": "Full 165-feature (93 local + 72 1-hop aggregates)",
                "training_protocol": "Chronological split (Train: 1..32, Val: 33..40, Test: 41..49)",
                "input_dim": 165,
                "time_dim": 0,
                "hidden_dim": 64,
                "num_layers": 2,
                "parameters_count": 11266,
                "checkpoint": "results/checkpoints/graphsage.pt",
                "checkpoint_sha256": self._checkpoint_sha256("results/checkpoints/graphsage.pt"),
                "held_out_auprc": 0.2427,
                "epistemic_status": "Root-node concatenation provides partial ego preservation over GCN/GAT (Test AUPRC 0.2427).",
            },
            {
                "id": "gat",
                "name": "GAT",
                "architecture": "2-layer Graph Attention Network",
                "category": "Standard GNN Baseline",
                "dataset": "Elliptic Bitcoin Transactions (203,769 nodes)",
                "feature_representation": "Full 165-feature (93 local + 72 1-hop aggregates)",
                "training_protocol": "Chronological split (Train: 1..32, Val: 33..40, Test: 41..49)",
                "input_dim": 165,
                "time_dim": 0,
                "hidden_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "parameters_count": 11910,
                "checkpoint": "results/checkpoints/gat.pt",
                "checkpoint_sha256": self._checkpoint_sha256("results/checkpoints/gat.pt"),
                "held_out_auprc": 0.1155,
                "epistemic_status": "Non-residual attention without time encodings (4 heads x 16 dim). Test AUPRC 0.1155.",
            },
            {
                "id": "residual_etgnn_93f",
                "name": "Residual ETGNN (93f Local, True Graph)",
                "architecture": "Ego-Residual Time-Encoded GAT (Phase 4 Cond B)",
                "category": "Phase 4 Controlled Architecture (Local Features)",
                "dataset": "Elliptic Bitcoin Transactions (Validation Split Steps 33..40)",
                "feature_representation": "Local 93-feature (ego-only, 1-hop aggregates stripped)",
                "training_protocol": "Phase 4 5-seed factorial (seeds 42..46), Adam lr=1e-3, Weighted CE",
                "input_dim": 93,
                "time_dim": 16,
                "hidden_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.3,
                "parameters_count": 14472,
                "checkpoint": "results/phase4_experiments/checkpoints/P4-Cond_B_93f_s42.pt",
                "checkpoint_sha256": self._checkpoint_sha256("results/phase4_experiments/checkpoints/P4-Cond_B_93f_s42.pt"),
                "held_out_auprc": 0.4223,
                "validation_auprc": 0.7758,
                "epistemic_status": "Ego-residual skip projections prevent heterophilic feature dilution. Test AP 0.4223 (5-seed mean: 0.3896 +- 0.0562); Pre-shift AP 0.7410; Ego protection bound Delta_empty = 0.0773.",
            },
            {
                "id": "deep_mlp_93f",
                "name": "Deep Parameter-Matched MLP (93f Local)",
                "architecture": "3-Layer Feature-Only MLP Control (Phase 4 Cond A)",
                "category": "Phase 4 Controlled Baseline (Local Features)",
                "dataset": "Elliptic Bitcoin Transactions (Validation Split Steps 33..40)",
                "feature_representation": "Local 93-feature (ego-only, zero graph edges)",
                "training_protocol": "Phase 4 5-seed factorial (seeds 42..46), capacity-matched control",
                "input_dim": 93,
                "time_dim": 16,
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.3,
                "parameters_count": 22466,
                "checkpoint": "results/phase4_experiments/checkpoints/P4-Cond_A_93f_s42.pt",
                "checkpoint_sha256": self._checkpoint_sha256("results/phase4_experiments/checkpoints/P4-Cond_A_93f_s42.pt"),
                "held_out_auprc": None,
                "validation_auprc": 0.7616,
                "epistemic_status": "Strong feature-only 93f baseline (0.7616 +- 0.0563). Statistically indistinguishable from True-Graph Residual ETGNN in mean AUPRC (p=0.6503), though higher seed variance.",
            },
        ]

    def get_benchmark_metrics(self) -> Dict[str, Any]:
        """Return official held-out test results from CSV."""
        path = self.tables_dir / "final_test_results.csv"
        df = pd.read_csv(path)
        records = df.to_dict(orient="records")

        return {
            "dataset": "Elliptic Bitcoin Dataset",
            "split": "Chronological (Train: 1..32, Val: 33..40, Test: 41..49)",
            "test_nodes_count": 9973,
            "test_illicit_count": 524,
            "test_licit_count": 9449,
            "test_illicit_prevalence": 0.0525,
            "validation_illicit_prevalence": 0.1047,
            "metrics": records,
            "primary_ranking_metric": "auprc (Average Precision / Area Under Precision-Recall Curve)",
            "key_finding": "Random Forest (0.6323) outperformed original non-residual ETGNN (0.2250) by +0.4073 AUPRC on held-out test steps 41-49.",
        }

    def get_c1_summary(self) -> Dict[str, Any]:
        """Return frozen C1 experimental summary data."""
        path = self.tables_dir / "c1_response_summary.csv"
        df = pd.read_csv(path)
        return {
            "experiment": "C1: Temporal Explanation-Sensitivity Comparison",
            "contrast": "ETGNN vs Matched GAT under explanation-guided edge deletion",
            "hypothesis": "ETGNN exhibits larger prediction drop than Matched GAT when explanatory edges are deleted.",
            "findings_summary": "INCONCLUSIVE. 95% Confidence Intervals for advantage difference encompass zero across cohorts.",
            "data": df.to_dict(orient="records"),
            "scientific_takeaway": "Sinusoidal time encoding does not produce statistically distinguishable explanation sensitivity.",
        }

    def get_c2_summary(self) -> Dict[str, Any]:
        """Return frozen C2 experimental summary data across 8 registers."""
        path = self.tables_dir / "c2_registers_bootstrap_summary.csv"
        df = pd.read_csv(path)
        return {
            "experiment": "C2: Matched Correct vs Incorrect Explanation Analysis",
            "registers_count": 8,
            "total_matched_pairs": 420,
            "data": df.to_dict(orient="records"),
            "registers": [
                {
                    "name": row["register"],
                    "mean_diff": float(row["mean_diff"]),
                    "ci_95_low": float(row["ci_95_low"]),
                    "ci_95_high": float(row["ci_95_high"]),
                    "bootstrap_se": float(row["bootstrap_se"]),
                    "ci_crosses_zero": bool(row["ci_crosses_zero"]),
                    "interpretation": row["interpretation"],
                }
                for _, row in df.iterrows()
            ],
            "scientific_takeaway": "C2 measures prediction sensitivity under edge deletion, NOT causal ground-truth correctness.",
        }

    def get_ablations_data(self) -> Dict[str, Any]:
        """Return multi-seed ablation and edge removal ablation data."""
        # 1. Multi-seed ablation
        ablation_dir = self.tables_dir / "ablation"
        etgnn_seeds = []
        matched_gat_seeds = []
        for seed in (7, 42, 123):
            et_p = ablation_dir / f"etgnn_seed_{seed}.json"
            gat_p = ablation_dir / f"matched_gat_seed_{seed}.json"
            if et_p.exists():
                with open(et_p) as f:
                    etgnn_seeds.append(json.load(f))
            if gat_p.exists():
                with open(gat_p) as f:
                    matched_gat_seeds.append(json.load(f))

        # 2. Time-step breakdown
        step_path = self.tables_dir / "test_by_time_step.csv"
        step_records = (
            pd.read_csv(step_path).to_dict(orient="records")
            if step_path.exists()
            else []
        )

        # 3. Validation ablation comparison
        validation_ablation = [
            {
                "model": "ETGNN (All Features + Graph Edges)",
                "val_auprc": 0.5488,
                "notes": "Includes 93 local + 72 aggregate features and graph edges",
            },
            {
                "model": "ETGNN Local Only (No Aggregates + Graph Edges)",
                "val_auprc": 0.5401,
                "notes": "Only 93 local features, message passing propagates signals",
            },
            {
                "model": "ETGNN Local No Edges (Zero Graph Edges)",
                "val_auprc": 0.7869,
                "notes": "Removing graph edges causes +0.2468 jump, revealing topological noise in message passing",
            },
        ]

        return {
            "multi_seed_comparison": {
                "etgnn": etgnn_seeds,
                "matched_gat": matched_gat_seeds,
                "finding": "ETGNN (mean val AUPRC ~0.551) and Matched GAT (mean val AUPRC ~0.550) are statistically indistinguishable across seeds.",
            },
            "edge_ablation_anomaly": {
                "phenomenon": "+0.2468 AUPRC jump when removing all edges from local ETGNN",
                "explanation": "Confirms that raw Elliptic graph edges introduce significant inter-class noise that degrades GNN representations below tabular baselines.",
                "data": validation_ablation,
            },
            "temporal_prevalence_volatility": step_records,
        }

    def get_phase4_summary(self) -> Dict[str, Any]:
        """Return verified 40-trial Phase 4 factorial results and Holm-adjusted paired contrasts."""
        csv_path = self.root / "research_extension" / "PHASE4_EXPERIMENT_MATRIX.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing Phase 4 matrix: {csv_path}")

        df = pd.read_csv(csv_path)
        df["condition_id"] = [
            f"Cond_{c}_{int(fd)}f" for c, fd in zip(df["condition"], df["feat_dim"])
        ]
        condition_order = [
            "Cond_B_93f",
            "Cond_A_93f",
            "Cond_D_93f",
            "Cond_C_93f",
            "Cond_A_165f",
            "Cond_C_165f",
            "Cond_D_165f",
            "Cond_B_165f",
        ]
        labels = {
            "Cond_B_93f": "Cond B: Residual ETGNN — True Graph (93f Local)",
            "Cond_A_93f": "Cond A: Deep MLP — Feature Only (93f Local)",
            "Cond_D_93f": "Cond D: Residual ETGNN — Rewired Graph (93f Local)",
            "Cond_C_93f": "Cond C: Residual ETGNN — Empty Edges (93f Local)",
            "Cond_A_165f": "Cond A: Deep MLP — Feature Only (165f Full)",
            "Cond_C_165f": "Cond C: Residual ETGNN — Empty Edges (165f Full)",
            "Cond_D_165f": "Cond D: Residual ETGNN — Rewired Graph (165f Full)",
            "Cond_B_165f": "Cond B: Residual ETGNN — True Graph (165f Full)",
        }

        conditions_summary = []
        for cid in condition_order:
            sub = df[df["condition_id"] == cid].sort_values("seed")
            fd = int(sub["feat_dim"].iloc[0])
            conditions_summary.append(
                {
                    "condition_id": cid,
                    "display_name": labels[cid],
                    "feature_mode": "local_93f" if fd == 93 else "full_165f",
                    "input_dim": fd,
                    "param_count": int(sub["param_count"].iloc[0]),
                    "n_seeds": int(len(sub)),
                    "auprc_mean": float(sub["auprc"].mean()),
                    "auprc_sd": float(sub["auprc"].std(ddof=1)),
                    "f1_mean": float(sub["f1"].mean()),
                    "f1_sd": float(sub["f1"].std(ddof=1)),
                    "precision_mean": float(sub["precision"].mean()),
                    "recall_mean": float(sub["recall"].mean()),
                    "roc_auc_mean": float(sub["roc_auc"].mean()),
                }
            )

        contrast_specs = [
            ("Cond_B_93f", "Cond_D_93f", "93f True Graph vs. Rewired Graph", "Exploratory Stabilization"),
            ("Cond_B_93f", "Cond_A_93f", "93f True Graph vs. Deep MLP", "Null / Indistinguishable"),
            ("Cond_B_93f", "Cond_C_93f", "93f True Graph vs. Empty Edges", "Null / High Variance"),
            ("Cond_B_165f", "Cond_A_165f", "165f True Graph vs. Deep MLP", "Confirmatory Negative (MLP Superior)"),
            ("Cond_B_165f", "Cond_D_165f", "165f True Graph vs. Rewired Graph", "Null / Indistinguishable"),
            ("Cond_B_165f", "Cond_C_165f", "165f True Graph vs. Empty Edges", "Null / Indistinguishable"),
        ]

        raw_contrasts = []
        for c1, c2, cname, verdict in contrast_specs:
            v1 = df[df["condition_id"] == c1].sort_values("seed")["auprc"].values
            v2 = df[df["condition_id"] == c2].sort_values("seed")["auprc"].values
            diffs = v1 - v2
            mean_d = float(np.mean(diffs))
            sd_d = float(np.std(diffs, ddof=1))
            se_d = sd_d / np.sqrt(len(diffs))
            t_crit = float(stats.t.ppf(0.975, df=len(diffs) - 1))
            t_stat, p_unadj = stats.ttest_rel(v1, v2)
            raw_contrasts.append(
                {
                    "contrast": cname,
                    "cond_1": c1,
                    "cond_2": c2,
                    "mean_diff": mean_d,
                    "sd_diff": sd_d,
                    "ci_95_low": mean_d - t_crit * se_d,
                    "ci_95_high": mean_d + t_crit * se_d,
                    "t_stat": float(t_stat),
                    "p_unadjusted": float(p_unadj),
                    "positive_seeds": int(np.sum(diffs > 0)),
                    "total_seeds": int(len(diffs)),
                    "epistemic_verdict": verdict,
                }
            )

        # Apply Holm-Bonferroni across all 6 contrasts
        order = np.argsort([c["p_unadjusted"] for c in raw_contrasts])
        running_max = 0.0
        for rank_idx, orig_idx in enumerate(order):
            adj = min(1.0, raw_contrasts[orig_idx]["p_unadjusted"] * (6 - rank_idx))
            running_max = max(running_max, adj)
            raw_contrasts[orig_idx]["p_holm_6"] = float(running_max)

        return {
            "experiment": "Phase 4: 40-Trial 2x4x5 Factorial Benchmark (Topology x Feature Representation)",
            "evaluation_split": "Elliptic Chronological Validation Split (Steps 33..40, N=7,824 labelled nodes)",
            "total_trials": int(len(df)),
            "matrix_sha256": self._checkpoint_sha256("research_extension/PHASE4_EXPERIMENT_MATRIX.csv"),
            "conditions": conditions_summary,
            "paired_contrasts": raw_contrasts,
            "scientific_takeaway": (
                "On local 93f features, Residual ETGNN (True Graph) achieves the highest mean validation AUPRC "
                "(0.7758 +- 0.0111) and improves over degree-preserving Rewired Graph by +0.0506 (unadjusted p=0.0245, "
                "Holm-adjusted p=0.1223 -> exploratory stabilization). On full 165f features, Deep MLP (0.7189 +- 0.0016) "
                "significantly outperforms True-Graph Residual ETGNN (0.5812 +- 0.0639, Holm-adjusted p=0.0498)."
            ),
        }

    def get_xai_validation_summary(self) -> Dict[str, Any]:
        """Return Phase H Explainable AI validation metrics on frozen 93f ResidualETGNNClassifier."""
        json_path = self.root / "results" / "phase5_xai" / "xai_validation_summary.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing Phase H XAI summary: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return {
            "experiment": "Phase H: GNNExplainer vs. Exact Leave-One-Edge-Out (LOO) Occlusion Oracle on Frozen 93f ResidualETGNNClassifier",
            "evaluation_split": "Elliptic Validation Split (Steps 33..40), Stratified Multi-Edge Cohort (E_cand >= 4, N=30)",
            "summary": summary,
            "scientific_takeaway": (
                "Because the ego-residual pathway bounds edge perturbations (mean empty-subgraph shift 0.0773), "
                "continuous mask relaxation (GNNExplainer) produces near-uniform masks (normalized entropy 0.9378) "
                "whose Top-1 deletion shift (0.0150) is indistinguishable from Random-1 deletion (0.0170, p=0.6827). "
                "By contrast, Exact Leave-One-Edge-Out (LOO) Occlusion captures 3.65x higher sensitivity (0.0547) "
                "and executes 19.0x faster (13.69 ms vs. 260.47 ms per node)."
            ),
        }

    def get_external_validation_summary(self) -> Dict[str, Any]:
        """Return Phase 5 external validation results and statistics on Elliptic++ Actors (56f)."""
        ext_dir = self.root / "results" / "phase5_external"
        val_path = ext_dir / "phase5_validation_statistical_summary.json"
        test_path = ext_dir / "phase5_test_statistical_summary.json"
        xai_path = ext_dir / "external_xai_summary.json"
        manifest_path = self.root / "data" / "processed" / "elliptic_plusplus" / "dataset_quality_manifest.json"

        if not val_path.exists() or not test_path.exists():
            raise FileNotFoundError("External validation statistical summaries not found in results/phase5_external/")

        val_stats = json.loads(val_path.read_text(encoding="utf-8"))
        test_stats = json.loads(test_path.read_text(encoding="utf-8"))
        xai_stats = json.loads(xai_path.read_text(encoding="utf-8")) if xai_path.exists() else {}
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

        return {
            "experiment": "Phase 5: External Validation on Elliptic++ Actors (Wallet Interaction Graph, 56 Features)",
            "provenance_classification": "Partially Overlapping Cross-Representation Validation (Shared Bitcoin 49-step epoch; distinct AddrAddr topology and 56-d actor features)",
            "dataset_manifest": dataset_manifest,
            "validation_summary": val_stats,
            "test_summary": test_stats,
            "external_xai_summary": xai_stats,
            "scientific_verdict": (
                "True AddrAddr graph topology (Condition B) significantly outperforms degree-preserving rewired graph "
                "(Condition D) on both Validation (Mean diff +0.0694 AUPRC, 5/5 seeds positive, Holm p=0.0039) and "
                "Held-Out Test (Mean diff +0.0165 AUPRC, 5/5 seeds positive, Holm p=0.0327). On Validation, Condition B "
                "achieves the highest mean AUPRC (0.4803 vs. 0.4451 for Deep MLP and 0.4378 for Empty Graph), winning on "
                "4/5 seeds (peaking at 0.5570), while seed 46 sensitivity leaves overall MLP superiority statistically indistinguishable."
            ),
        }




    def get_temporal_stability_summary(self) -> Dict[str, Any]:
        """Fetch Phase 12 Experiment A candidate-edge shift census and multi-seed stability audit."""
        census_path = self.root / "research_extension" / "phase12" / "tables" / "EXP_A_CANDIDATE_EDGE_SHIFT_CENSUS.csv"
        stab_path = self.root / "research_extension" / "phase12" / "tables" / "EXP_A_TEMPORAL_STABILITY_SUMMARY.csv"
        
        if not census_path.exists() or not stab_path.exists():
            raise FileNotFoundError("Phase 12 Experiment A tables missing.")
            
        df_census = pd.read_csv(census_path).fillna(0)
        df_stab = pd.read_csv(stab_path).fillna(0)
        
        return {
            "experiment": "Phase 12 Experiment A: Temporal Stability & Candidate-Edge Censorship Census",
            "evaluation_split": "Held-Out Test Steps 41..49 (N=9,973 nodes: 524 illicit, 9,449 licit)",
            "census_summary": {
                "total_nodes": 9973,
                "preshift_nodes": 4342,
                "postshift_nodes": 5631,
                "preshift_illicit_count": 355,
                "postshift_illicit_count": 169,
                "preshift_illicit_zero_edge_pct": 13.52,
                "postshift_illicit_zero_edge_pct": 76.92,
                "preshift_licit_zero_edge_pct": 34.36,
                "postshift_licit_zero_edge_pct": 83.72,
                "census_records": df_census.to_dict(orient="records"),
            },
            "stability_summary": {
                "strata_records": df_stab.to_dict(orient="records"),
                "stratum_2_matched_pairs": 59,
                "tost_equivalence_result": "Failed to reject non-equivalence at delta=+-0.05 and +-0.10 bounds",
            },
            "scientific_takeaway": (
                "Across the step 43 darknet takedown, illicit transactions with zero candidate edges jump from 13.52% to 76.92%. "
                "In Stratum 2 (N=59 matched pairs), failure to reject the null hypothesis (p >= 0.0821) does not prove temporal invariance or equivalence."
            ),
        }

    def get_error_mitigation_summary(self) -> Dict[str, Any]:
        """Fetch Phase 12 Experiment B 21-strategy prospective error mitigation results."""
        mit_path = self.root / "research_extension" / "phase12" / "tables" / "EXP_B_TEST_MITIGATION_RESULTS.csv"
        boot_path = self.root / "research_extension" / "phase12" / "tables" / "EXP_B_PAIRED_BOOTSTRAP_TESTS.csv"
        
        if not mit_path.exists() or not boot_path.exists():
            raise FileNotFoundError("Phase 12 Experiment B tables missing.")
            
        df_mit = pd.read_csv(mit_path).fillna(0)
        df_boot = pd.read_csv(boot_path).fillna(0)
        
        return {
            "experiment": "Phase 12 Experiment B: Prospective Error Mitigation Auditing (21 Strategies)",
            "evaluation_split": "Held-Out Test Steps 41..49 (N=9,973 nodes)",
            "strategies_count": 21,
            "mitigation_results": df_mit.to_dict(orient="records"),
            "paired_bootstrap_contrasts": df_boot.to_dict(orient="records"),
            "scientific_takeaway": (
                "Explanation-guided error mitigations (M1a-M3b) fail to outperform matched random controls (e.g. M1a 0.2721 < C1a 0.2915 AP, p_Holm=0.0140). "
                "Non-explanation predicted-risk cascade (13_C2b, 0.4808 AP) and 100% local veto (19_C3c, 0.4442 AP) succeed as architectural controls, "
                "with explicit disclosure of post-shift true-positive losses (-3 of 5 TP vs default; -9 of 11 TP vs local standalone)."
            ),
        }

@lru_cache(maxsize=1)
def get_data_loader() -> ResearchDataLoader:
    """Singleton getter for data loader."""
    return ResearchDataLoader()
