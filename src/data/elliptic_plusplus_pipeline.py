"""
Phase E: Modular, Memory-Safe Data Pipeline for Elliptic++ Actors (Wallet Interaction Graph).
Implements chunked CSV ingestion, float32/int64 memory optimization, node ID remapping,
directed intra-timestep edge construction, train-only StandardScaler normalization,
temporal integrity assertions, repeated-entity overlap tracking, and manifest generation.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


REQUIRED_RAW_FILES = (
    "wallets_features.csv",
    "wallets_classes.csv",
    "AddrAddr_edgelist.csv",
)


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class EllipticPlusPlusPipeline:
    """
    Leakage-safe, stage-windowed preprocessing pipeline for the Elliptic++ Actors dataset.
    Designed to run safely within an 8 GB RAM CPU-only Windows machine via chunked reading
    and split-specific PyG stage graph construction.
    """

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        train_steps: Tuple[int, int] = (1, 32),
        val_steps: Tuple[int, int] = (33, 40),
        test_steps: Tuple[int, int] = (41, 49),
        expected_feat_dim: int = 56,
        chunk_size: int = 100_000,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.test_steps = test_steps
        self.expected_feat_dim = expected_feat_dim
        self.chunk_size = chunk_size

    def check_raw_files_exist(self) -> Tuple[bool, List[str]]:
        missing = []
        for fname in REQUIRED_RAW_FILES:
            if not (self.raw_dir / fname).exists():
                missing.append(str(self.raw_dir / fname))
        return len(missing) == 0, missing

    def _load_classes_map(self, classes_path: Path) -> Dict[str, int]:
        """
        Load wallet address class labels and map:
          1 (illicit) -> 1
          2 (licit)   -> 0
          3 / unknown -> -1
        """
        label_map: Dict[str, int] = {}
        for chunk in pd.read_csv(classes_path, chunksize=self.chunk_size, dtype=str):
            cols = [c.strip() for c in chunk.columns]
            chunk.columns = cols
            addr_col = cols[0]
            cls_col = cols[1]
            addrs = chunk[addr_col].astype(str).str.strip()
            raw_classes = chunk[cls_col].astype(str).str.strip().str.lower()
            mapped = np.full(len(chunk), -1, dtype=np.int64)
            mapped[raw_classes.isin(["1", "illicit"])] = 1
            mapped[raw_classes.isin(["2", "licit"])] = 0
            label_map.update(zip(addrs.tolist(), mapped.tolist()))
        return label_map

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute full end-to-end preprocessing, validation, train-only scaling,
        stage-graph construction, and quality report generation.
        Uses vectorized integer actor-timestep indexing to stay well under 500 MB RAM
        on 1.268M rows and 2.869M edges.
        """
        import gc

        t0 = time.time()
        ok, missing = self.check_raw_files_exist()
        if not ok:
            raise FileNotFoundError(
                f"Missing required Elliptic++ Actors files: {missing}. "
                "Obtain explicit authorization before downloading external datasets."
            )

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        feat_path = self.raw_dir / "wallets_features.csv"
        cls_path = self.raw_dir / "wallets_classes.csv"
        edge_path = self.raw_dir / "AddrAddr_edgelist.csv"

        # 1. Load label mapping
        label_map = self._load_classes_map(cls_path)

        # 2. Chunked ingestion of wallets_features.csv
        # Map each unique actor address to a compact int32 ID [0 .. U-1]
        addr_to_actor_id: Dict[str, int] = {}
        actor_labels_list: List[int] = []
        obs_actor_ids_list: List[np.ndarray] = []
        time_steps_chunks: List[np.ndarray] = []
        feature_chunks: List[np.ndarray] = []

        # Lookup table: actor_ts_to_node_idx[actor_id, ts] -> global observation node index (or -1)
        capacity = max(len(label_map) + 1024, 4096)
        actor_ts_to_node_idx = np.full((capacity, 50), -1, dtype=np.int32)

        duplicate_obs_dropped = 0
        raw_nan_count = 0
        next_node_idx = 0
        feature_column_names: List[str] = []
        time_step_included_in_x = False

        for chunk in pd.read_csv(feat_path, chunksize=self.chunk_size):
            cols = [c.strip() for c in chunk.columns]
            chunk.columns = cols
            addr_col = cols[0]
            ts_col = cols[1]
            if len(cols) - 2 == self.expected_feat_dim:
                feat_cols = cols[2:]
                time_step_included_in_x = False
            elif len(cols) - 1 == self.expected_feat_dim:
                # Official Elliptic++ wallets_features.csv has 57 columns:
                # 'address' + 56 numeric features ('Time step' + 55 wallet features)
                feat_cols = cols[1:]
                time_step_included_in_x = True
            else:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.expected_feat_dim}, "
                    f"got {len(cols) - 2} (excluding Time step) / {len(cols) - 1} (including Time step)"
                )
            if not feature_column_names:
                feature_column_names = list(feat_cols)

            addrs_series = chunk[addr_col].astype(str).str.strip().tolist()
            ts_arr = chunk[ts_col].to_numpy(dtype=np.int64)
            if len(ts_arr) > 0 and (int(ts_arr.min()) < 1 or int(ts_arr.max()) > 49):
                raise ValueError(
                    f"Invalid time step bounds [{int(ts_arr.min())}, {int(ts_arr.max())}]; expected within [1, 49]."
                )

            keep_indices = []
            keep_actor_ids = []
            keep_ts = []

            for idx, (a, ts_val) in enumerate(zip(addrs_series, ts_arr.tolist())):
                aid = addr_to_actor_id.get(a, -1)
                if aid == -1:
                    aid = len(addr_to_actor_id)
                    addr_to_actor_id[a] = aid
                    actor_labels_list.append(label_map.get(a, -1))
                    if aid >= actor_ts_to_node_idx.shape[0]:
                        new_cap = actor_ts_to_node_idx.shape[0] * 2
                        grown = np.full((new_cap, 50), -1, dtype=np.int32)
                        grown[: actor_ts_to_node_idx.shape[0]] = actor_ts_to_node_idx
                        actor_ts_to_node_idx = grown

                if actor_ts_to_node_idx[aid, ts_val] != -1:
                    duplicate_obs_dropped += 1
                else:
                    actor_ts_to_node_idx[aid, ts_val] = next_node_idx
                    next_node_idx += 1
                    keep_indices.append(idx)
                    keep_actor_ids.append(aid)
                    keep_ts.append(ts_val)

            if keep_indices:
                sub_feats = chunk.iloc[keep_indices][feat_cols].to_numpy(dtype=np.float32)
                raw_nan_count += int(np.isnan(sub_feats).sum() + np.isinf(sub_feats).sum())
                feature_chunks.append(sub_feats)
                obs_actor_ids_list.append(np.asarray(keep_actor_ids, dtype=np.int32))
                time_steps_chunks.append(np.asarray(keep_ts, dtype=np.int64))

        num_unique_actors = len(addr_to_actor_id)
        actor_ts_to_node_idx = actor_ts_to_node_idx[:num_unique_actors]
        del label_map
        gc.collect()

        x_raw = (
            np.vstack(feature_chunks)
            if feature_chunks
            else np.empty((0, self.expected_feat_dim), dtype=np.float32)
        )
        del feature_chunks
        obs_actor_ids = (
            np.concatenate(obs_actor_ids_list)
            if obs_actor_ids_list
            else np.empty(0, dtype=np.int32)
        )
        del obs_actor_ids_list
        time_steps = (
            np.concatenate(time_steps_chunks)
            if time_steps_chunks
            else np.empty(0, dtype=np.int64)
        )
        del time_steps_chunks
        gc.collect()

        num_nodes = int(x_raw.shape[0])
        if num_nodes == 0:
            raise ValueError("No valid node observations found in wallets_features.csv")

        # Validate time step bounds
        min_ts, max_ts = int(time_steps.min()), int(time_steps.max())
        if min_ts < 1 or max_ts > 49:
            raise ValueError(f"Invalid time step bounds [{min_ts}, {max_ts}]; expected within [1, 49].")

        # Map labels via actor_id
        actor_labels_arr = np.asarray(actor_labels_list, dtype=np.int64)
        labels = actor_labels_arr[obs_actor_ids]

        # 3. Define chronological masks and repeated-entity tracking
        in_train_window = (time_steps >= self.train_steps[0]) & (time_steps <= self.train_steps[1])
        in_val_window = (time_steps >= self.val_steps[0]) & (time_steps <= self.val_steps[1])
        in_test_window = (time_steps >= self.test_steps[0]) & (time_steps <= self.test_steps[1])

        is_labelled = labels != -1
        train_mask = in_train_window & is_labelled
        val_mask = in_val_window & is_labelled
        test_mask = in_test_window & is_labelled

        # Track entity overlap across splits using boolean actor masks
        actor_in_train = np.zeros(num_unique_actors, dtype=bool)
        actor_in_val = np.zeros(num_unique_actors, dtype=bool)
        actor_in_test = np.zeros(num_unique_actors, dtype=bool)
        actor_in_train[obs_actor_ids[in_train_window]] = True
        actor_in_val[obs_actor_ids[in_val_window]] = True
        actor_in_test[obs_actor_ids[in_test_window]] = True

        is_unseen_from_train = ~actor_in_train[obs_actor_ids]
        val_unseen_entity_mask = val_mask & is_unseen_from_train
        test_unseen_entity_mask = test_mask & is_unseen_from_train

        # 4. Fit StandardScaler strictly on training window nodes (in_train_window)
        # In-place imputation and scaling to avoid duplicating 220 MB float32 arrays
        train_x_slice = x_raw[in_train_window]
        col_medians = np.nanmedian(np.where(np.isfinite(train_x_slice), train_x_slice, np.nan), axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0).astype(np.float32)
        del train_x_slice

        for j in range(self.expected_feat_dim):
            bad_all = ~np.isfinite(x_raw[:, j])
            if bad_all.any():
                x_raw[bad_all, j] = col_medians[j]

        scaler = StandardScaler()
        scaler.fit(x_raw[in_train_window])
        assert int(scaler.n_samples_seen_) == int(in_train_window.sum()), "Scaler fit sample count mismatch!"

        # Transform in chunks in-place into x_raw (renamed x_scaled) to minimize peak memory
        scale_chunk = 250_000
        for s_start in range(0, num_nodes, scale_chunk):
            s_end = min(num_nodes, s_start + scale_chunk)
            x_raw[s_start:s_end] = scaler.transform(x_raw[s_start:s_end]).astype(np.float32)
        x_scaled = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
        assert np.isfinite(x_scaled).all(), "Scaled features contain non-finite values!"

        # 5. Chunked vectorized ingestion of AddrAddr_edgelist.csv -> intra-timestep directed edges
        self_loops_removed = 0
        unmapped_edges_skipped = 0
        edge_key_chunks: List[np.ndarray] = []

        for chunk in pd.read_csv(edge_path, chunksize=self.chunk_size, dtype=str):
            cols = [c.strip() for c in chunk.columns]
            chunk.columns = cols
            src_col, dst_col = cols[0], cols[1]
            has_ts_col = len(cols) >= 3

            u_ids = np.fromiter(
                (addr_to_actor_id.get(s.strip(), -1) for s in chunk[src_col].astype(str)),
                dtype=np.int32,
                count=len(chunk),
            )
            v_ids = np.fromiter(
                (addr_to_actor_id.get(s.strip(), -1) for s in chunk[dst_col].astype(str)),
                dtype=np.int32,
                count=len(chunk),
            )

            both_known = (u_ids >= 0) & (v_ids >= 0)
            unmapped_edges_skipped += int((~both_known).sum())
            if not both_known.any():
                continue

            u_valid = u_ids[both_known]
            v_valid = v_ids[both_known]

            if has_ts_col:
                ts_raw = pd.to_numeric(chunk[cols[2]].to_numpy()[both_known], errors="coerce")
                ts_valid_mask = np.isfinite(ts_raw) & (ts_raw >= 1) & (ts_raw <= 49)
                if ts_valid_mask.all():
                    ts_int = ts_raw.astype(np.int32)
                    u_node = actor_ts_to_node_idx[u_valid, ts_int]
                    v_node = actor_ts_to_node_idx[v_valid, ts_int]
                    matched = (u_node >= 0) & (v_node >= 0)
                    unmapped_edges_skipped += int((~matched).sum())
                    u_pairs = u_node[matched].astype(np.int64)
                    v_pairs = v_node[matched].astype(np.int64)
                else:
                    # Fallback row-wise when mixed valid/NaN time steps exist
                    u_list, v_list = [], []
                    for ui, vi, tv in zip(u_valid.tolist(), v_valid.tolist(), ts_raw.tolist()):
                        if np.isfinite(tv) and 1 <= int(tv) <= 49:
                            un = int(actor_ts_to_node_idx[ui, int(tv)])
                            vn = int(actor_ts_to_node_idx[vi, int(tv)])
                            if un >= 0 and vn >= 0:
                                u_list.append(un)
                                v_list.append(vn)
                            else:
                                unmapped_edges_skipped += 1
                        else:
                            u_row = actor_ts_to_node_idx[ui, 1:50]
                            v_row = actor_ts_to_node_idx[vi, 1:50]
                            m = (u_row >= 0) & (v_row >= 0)
                            if not m.any():
                                unmapped_edges_skipped += 1
                            else:
                                u_list.extend(u_row[m].tolist())
                                v_list.extend(v_row[m].tolist())
                    u_pairs = np.asarray(u_list, dtype=np.int64)
                    v_pairs = np.asarray(v_list, dtype=np.int64)
            else:
                # Vectorized intersection across all 49 time steps for the chunk
                u_mat = actor_ts_to_node_idx[u_valid, 1:50]
                v_mat = actor_ts_to_node_idx[v_valid, 1:50]
                common_mask = (u_mat >= 0) & (v_mat >= 0)
                row_has_match = common_mask.any(axis=1)
                unmapped_edges_skipped += int((~row_has_match).sum())
                u_pairs = u_mat[common_mask].astype(np.int64)
                v_pairs = v_mat[common_mask].astype(np.int64)

            if len(u_pairs) > 0:
                self_loop_mask = u_pairs == v_pairs
                self_loops_removed += int(self_loop_mask.sum())
                u_clean = u_pairs[~self_loop_mask]
                v_clean = v_pairs[~self_loop_mask]
                if len(u_clean) > 0:
                    edge_key_chunks.append(u_clean * num_nodes + v_clean)

        del addr_to_actor_id, actor_ts_to_node_idx
        gc.collect()

        if edge_key_chunks:
            all_keys = np.concatenate(edge_key_chunks)
            del edge_key_chunks
            total_non_self_loop = len(all_keys)
            unique_keys = np.unique(all_keys)
            del all_keys
            duplicate_edges_removed = int(total_non_self_loop - len(unique_keys))
            u_sorted = unique_keys // num_nodes
            v_sorted = unique_keys % num_nodes
            del unique_keys
            edge_index_all = np.vstack([u_sorted, v_sorted]).astype(np.int64)
            del u_sorted, v_sorted
        else:
            duplicate_edges_removed = 0
            edge_index_all = np.empty((2, 0), dtype=np.int64)
        gc.collect()

        # Verify 100% intra-timestep edges (Delta-t == 0) and zero self-loops
        if edge_index_all.shape[1] > 0:
            assert (edge_index_all[0] != edge_index_all[1]).all(), "Self-loops present!"
            assert (time_steps[edge_index_all[0]] == time_steps[edge_index_all[1]]).all(), "Cross-timestep edges present!"

        # 7. Save stage-windowed PyG graph dictionaries (isolated per split to minimize RAM)
        stages = {
            "train": (self.train_steps, train_mask, "train_mask"),
            "validation": (self.val_steps, val_mask, "val_mask"),
            "test": (self.test_steps, test_mask, "test_mask"),
        }

        stage_stats = {}
        for stage_name, (t_bounds, eval_mask, mask_key) in stages.items():
            t_min, t_max = t_bounds
            node_sel = (time_steps >= t_min) & (time_steps <= t_max)
            global_to_local = np.full(num_nodes, -1, dtype=np.int64)
            selected_indices = np.where(node_sel)[0]
            global_to_local[selected_indices] = np.arange(len(selected_indices), dtype=np.int64)

            if edge_index_all.shape[1] > 0:
                edge_sel = node_sel[edge_index_all[0]] & node_sel[edge_index_all[1]]
                sub_edges_global = edge_index_all[:, edge_sel]
                sub_edges_local = global_to_local[sub_edges_global]
            else:
                sub_edges_local = np.empty((2, 0), dtype=np.int64)

            stage_dict = {
                "x": torch.from_numpy(x_scaled[selected_indices]),
                "edge_index": torch.from_numpy(sub_edges_local).long(),
                "y": torch.from_numpy(labels[selected_indices]).long(),
                "time_step": torch.from_numpy(time_steps[selected_indices]).long(),
                "original_node_index": torch.from_numpy(selected_indices).long(),
                mask_key: torch.from_numpy(eval_mask[selected_indices]).bool(),
                "unseen_entity_mask": torch.from_numpy(
                    (eval_mask & is_unseen_from_train)[selected_indices]
                ).bool(),
            }
            out_pt = self.processed_dir / f"{stage_name}_graph.pt"
            torch.save(stage_dict, out_pt)

            y_lbl = labels[selected_indices][eval_mask[selected_indices]]
            n_licit = int((y_lbl == 0).sum())
            n_illicit = int((y_lbl == 1).sum())
            stage_stats[stage_name] = {
                "time_steps": [t_min, t_max],
                "total_nodes": int(len(selected_indices)),
                "total_edges": int(sub_edges_local.shape[1]),
                "labelled_nodes": int(eval_mask[selected_indices].sum()),
                "unseen_labelled_nodes": int((eval_mask & is_unseen_from_train)[selected_indices].sum()),
                "licit_count": n_licit,
                "illicit_count": n_illicit,
                "illicit_prevalence": round(n_illicit / max(1, n_licit + n_illicit), 6),
                "artifact_path": str(out_pt),
                "sha256": compute_sha256(out_pt),
            }

        # Save scaler and manifest
        scaler_path = self.processed_dir / "train_scaler.joblib"
        joblib.dump(scaler, scaler_path)

        manifest = {
            "dataset": "Elliptic++ Actors",
            "processed_at_epoch": time.time(),
            "processing_wall_time_sec": round(time.time() - t0, 2),
            "feature_dim": self.expected_feat_dim,
            "time_step_included_in_x": time_step_included_in_x,
            "feature_columns": feature_column_names,
            "total_observations": num_nodes,
            "unique_addresses": num_unique_actors,
            "total_intra_step_edges": int(edge_index_all.shape[1]),
            "data_hygiene": {
                "duplicate_observations_dropped": duplicate_obs_dropped,
                "raw_nonfinite_values_imputed": raw_nan_count,
                "self_loops_removed": self_loops_removed,
                "duplicate_edges_removed": duplicate_edges_removed,
                "unmapped_edges_skipped": unmapped_edges_skipped,
            },
            "entity_overlap": {
                "train_unique_actors": int(actor_in_train.sum()),
                "val_unique_actors": int(actor_in_val.sum()),
                "test_unique_actors": int(actor_in_test.sum()),
                "val_actors_seen_in_train": int((actor_in_val & actor_in_train).sum()),
                "test_actors_seen_in_train": int((actor_in_test & actor_in_train).sum()),
            },
            "scaler_verification": {
                "fitted_solely_on_train_steps": list(self.train_steps),
                "n_samples_seen": int(scaler.n_samples_seen_),
                "scaler_sha256": compute_sha256(scaler_path),
            },
            "raw_checksums": {
                fname: compute_sha256(self.raw_dir / fname) for fname in REQUIRED_RAW_FILES
            },
            "stages": stage_stats,
        }

        manifest_path = self.processed_dir / "dataset_quality_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_directory = project_root / "data" / "external" / "elliptic_plusplus"
    processed_directory = project_root / "data" / "processed" / "elliptic_plusplus"
    print(f"[EllipticPlusPlusPipeline] Running on {raw_directory} -> {processed_directory}", flush=True)
    pipe = EllipticPlusPlusPipeline(
        raw_dir=raw_directory,
        processed_dir=processed_directory,
        train_steps=(1, 32),
        val_steps=(33, 40),
        test_steps=(41, 49),
        expected_feat_dim=56,
        chunk_size=150_000,
    )
    out_manifest = pipe.run_pipeline()
    print(json.dumps(out_manifest, indent=2), flush=True)


