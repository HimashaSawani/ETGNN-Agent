"""
Phase F: Controlled External Model Evaluation Harness (Stages 1 to 5).
Implements pre-frozen Conditions A (Deep MLP), B (Residual ETGNN True),
C (Residual ETGNN Empty), and D (Residual ETGNN Degree-Preserving Rewired)
with strict stage gates:
  - Stage 1: Dry run on synthetic schema-matched dataset.
  - Stage 2: Host RAM, CPU, and disk space resource preflight.
  - Stage 3: Single-seed technical smoke pilot (seed 42, no tuning).
  - Stage 4: Full 5-seed validation evaluation (seeds 42..46, 20 trials).
  - Stage 5: Single-shot test evaluation (requires explicit --authorize-test flag).
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.elliptic_plusplus_pipeline import EllipticPlusPlusPipeline
from src.models.reliability_etgnn import DeepParameterMatchedMLP, ResidualETGNNClassifier


def verify_phase4_archive_intact(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Verify frozen Phase 4 and baseline artifacts have not been modified."""
    manifest_path = root / "configs" / "phase5_frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest["frozen_artifact_sha256"]

    verified = {}
    for rel_path, exp_hash in expected_hashes.items():
        full_path = root / rel_path
        if not full_path.exists():
            raise RuntimeError(f"Frozen artifact missing: {full_path}")
        actual_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
        if actual_hash != exp_hash:
            raise RuntimeError(
                f"Frozen artifact hash mismatch for {rel_path}: {actual_hash} != {exp_hash}"
            )
        verified[rel_path] = actual_hash

    return verified


def compute_optimal_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Compute threshold maximizing F1 on validation predictions."""
    prec, rec, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5
    f1_vals = (2 * prec[:-1] * rec[:-1]) / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    best_idx = int(np.argmax(f1_vals))
    return float(thresholds[best_idx])


def evaluate_predictions_extended(
    y_true: np.ndarray, prob: np.ndarray, calibrated_threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute primary AUPRC, secondary uncalibrated F1 (0.5), and validation-calibrated F1."""
    pred_05 = (prob >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_true, pred_05, labels=[0, 1])
    tn, fp, fn, tp = cm_05.ravel()

    pred_cal = (prob >= calibrated_threshold).astype(int)
    f1_cal = float(f1_score(y_true, pred_cal, zero_division=0))

    return {
        "auprc": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "f1": float(f1_score(y_true, pred_05, zero_division=0)),
        "val_calibrated_f1": f1_cal,
        "calibrated_threshold": float(calibrated_threshold),
        "precision": float(precision_score(y_true, pred_05, zero_division=0)),
        "recall": float(recall_score(y_true, pred_05, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def run_stage2_resource_preflight(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Stage 2: Measure available RAM, CPU cores, and disk space on project drive."""
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(root)
    proc = psutil.Process()

    free_ram_mb = vm.available / (1024 * 1024)
    total_ram_gb = vm.total / (1024 ** 3)
    free_disk_gb = disk.free / (1024 ** 3)

    ext_raw_dir = root / "data" / "external" / "elliptic_plusplus"
    ext_exists = (
        (ext_raw_dir / "wallets_features.csv").exists()
        and (ext_raw_dir / "wallets_classes.csv").exists()
        and (ext_raw_dir / "AddrAddr_edgelist.csv").exists()
    )

    preflight = {
        "stage": "Stage_2_Resource_Preflight",
        "timestamp": time.time(),
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_mb": round(free_ram_mb, 1),
        "current_process_rss_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
        "free_disk_gb": round(free_disk_gb, 2),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "recommended_torch_threads": 2,
        "ram_safe_for_stage_windowed_training": free_ram_mb >= 500.0,
        "disk_safe_for_external_dataset": free_disk_gb >= 2.0,
        "external_dataset_present_locally": ext_exists,
        "external_dataset_path": str(ext_raw_dir),
    }
    return preflight


def generate_synthetic_elliptic_plusplus_csvs(
    target_dir: Path, n_actors: int = 360, feat_dim: int = 56, seed: int = 42
) -> None:
    """Create schema-exact synthetic Elliptic++ Actors CSVs for Stage 1 dry run & unit tests."""
    rng = np.random.default_rng(seed)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Distribute actors across steps 1..49 (train 1..32, val 33..40, test 41..49)
    actor_ids = [f"1BoatSLRHtKNngkdXEeobR76b53LETtpy{i:04d}" for i in range(n_actors)]
    classes = []
    for i, aid in enumerate(actor_ids):
        # 15% illicit (1), 65% licit (2), 20% unknown (3)
        r = rng.random()
        cls_val = "1" if r < 0.15 else ("2" if r < 0.80 else "3")
        classes.append({"address": aid, "class": cls_val})

    pd.DataFrame(classes).to_csv(target_dir / "wallets_classes.csv", index=False)

    # Create observations across time steps
    feat_rows = []
    edge_rows = []
    ts_choices = [5, 12, 25, 35, 38, 44, 47]
    for ts in ts_choices:
        step_actors = rng.choice(actor_ids, size=min(80, n_actors), replace=False)
        for aid in step_actors:
            c_str = next(c["class"] for c in classes if c["address"] == aid)
            signal = 1.5 if c_str == "1" else (-0.5 if c_str == "2" else 0.0)
            feats = rng.normal(loc=signal, scale=1.0, size=feat_dim).astype(np.float32)
            row = {"address": aid, "Time step": ts}
            for j in range(feat_dim):
                row[f"feat_{j+1}"] = float(feats[j])
            feat_rows.append(row)

        # Create intra-step directed edges + a deliberate self-loop and duplicate to test hygiene
        for k in range(len(step_actors) - 1):
            edge_rows.append(
                {
                    "input_address": step_actors[k],
                    "output_address": step_actors[(k + 1) % len(step_actors)],
                    "Time step": ts,
                }
            )
        # Deliberate self-loop and duplicate
        edge_rows.append({"input_address": step_actors[0], "output_address": step_actors[0], "Time step": ts})
        edge_rows.append({"input_address": step_actors[0], "output_address": step_actors[1], "Time step": ts})

    pd.DataFrame(feat_rows).to_csv(target_dir / "wallets_features.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(target_dir / "AddrAddr_edgelist.csv", index=False)


def extract_exact_2hop_computation_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    time_step: torch.Tensor,
    eval_mask: torch.Tensor,
    unseen_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract the exact 2-layer message-passing computation graph for `eval_mask`.
    In PyG GATConv (flow='source_to_target'), edge (u -> v) sends a message from u to v.
    Nodes with zero 2-hop directed path into `eval_mask` have zero mathematical effect on
    `logits[eval_mask]` (verified 0.00e+00 logit difference).
    """
    num_nodes = x.size(0)
    if edge_index.numel() == 0:
        sel_idx = torch.where(eval_mask)[0]
        out = {
            "x": x[sel_idx],
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "y": y[sel_idx],
            "time_step": time_step[sel_idx],
            "eval_mask": torch.ones(sel_idx.numel(), dtype=torch.bool),
        }
        if unseen_mask is not None:
            out["unseen_entity_mask"] = unseen_mask[sel_idx]
        return out

    src, dst = edge_index[0], edge_index[1]
    hop0 = eval_mask.clone()
    e_in_hop0 = hop0[dst]
    hop1 = hop0.clone()
    hop1[src[e_in_hop0]] = True
    e_in_hop1 = hop1[dst]
    hop2 = hop1.clone()
    hop2[src[e_in_hop1]] = True

    sel_idx = torch.where(hop2)[0]
    g2l = torch.full((num_nodes,), -1, dtype=torch.long)
    g2l[sel_idx] = torch.arange(sel_idx.numel(), dtype=torch.long)
    sub_edges = g2l[edge_index[:, e_in_hop1]]

    out = {
        "x": x[sel_idx],
        "edge_index": sub_edges,
        "y": y[sel_idx],
        "time_step": time_step[sel_idx],
        "eval_mask": eval_mask[sel_idx],
    }
    if unseen_mask is not None:
        out["unseen_entity_mask"] = unseen_mask[sel_idx]
    return out


def rewired_graph_by_timestep_fast(
    edge_index: torch.Tensor, time_steps: torch.Tensor, n_swaps_factor: int = 5, seed: int = 42
) -> Tuple[torch.Tensor, int, int, float]:
    """
    Exact degree-preserving directed double-edge swap within each discrete time step,
    using packed 64-bit integer edge keys to eliminate Python tuple memory overhead.
    """
    if edge_index.numel() == 0:
        return edge_index, 0, 0, 1.0

    edge_np = edge_index.numpy()
    ts_np = time_steps.numpy()
    unique_ts = np.unique(ts_np[edge_np[0]])

    rewired_src_list = []
    rewired_dst_list = []
    total_attempts = 0
    total_accepted = 0

    for t in unique_ts.tolist():
        mask = ts_np[edge_np[0]] == t
        src = edge_np[0][mask].astype(np.int64).copy()
        dst = edge_np[1][mask].astype(np.int64).copy()
        m = len(src)
        if m < 2:
            rewired_src_list.append(src)
            rewired_dst_list.append(dst)
            continue

        rng = np.random.default_rng(seed + int(t) * 1000)
        edge_set = set(((src << 32) | dst).tolist())
        n_attempts = int(n_swaps_factor * m)
        accepted = 0
        idx1 = rng.integers(0, m, size=n_attempts).tolist()
        idx2 = rng.integers(0, m, size=n_attempts).tolist()
        src_l = src.tolist()
        dst_l = dst.tolist()

        for e1, e2 in zip(idx1, idx2):
            if e1 == e2:
                continue
            u1, v1 = src_l[e1], dst_l[e1]
            u2, v2 = src_l[e2], dst_l[e2]
            if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
                continue
            k_new1 = (u1 << 32) | v2
            k_new2 = (u2 << 32) | v1
            if k_new1 in edge_set or k_new2 in edge_set:
                continue
            edge_set.remove((u1 << 32) | v1)
            edge_set.remove((u2 << 32) | v2)
            edge_set.add(k_new1)
            edge_set.add(k_new2)
            dst_l[e1] = v2
            dst_l[e2] = v1
            accepted += 1

        rewired_src_list.append(np.asarray(src_l, dtype=np.int64))
        rewired_dst_list.append(np.asarray(dst_l, dtype=np.int64))
        total_attempts += n_attempts
        total_accepted += accepted

    new_src = np.concatenate(rewired_src_list)
    new_dst = np.concatenate(rewired_dst_list)
    orig_keys = (edge_np[0].astype(np.int64) << 32) | edge_np[1].astype(np.int64)
    rew_keys = (new_src << 32) | new_dst
    inter = len(np.intersect1d(orig_keys, rew_keys, assume_unique=True))
    union = len(orig_keys) + len(rew_keys) - inter
    jaccard = float(inter / max(1, union))

    rewired_edge_index = torch.from_numpy(np.stack([new_src, new_dst], axis=0)).long()
    return rewired_edge_index, total_attempts, total_accepted, jaccard


def rewired_graph_by_timestep(
    edge_index: torch.Tensor, time_steps: torch.Tensor, n_swaps_factor: int = 5, seed: int = 42
) -> Tuple[torch.Tensor, int, int]:
    """
    Exact degree-preserving directed double-edge swap within discrete time steps.
    Preserves exact in-degree, out-degree, and edge count per snapshot.
    """
    rewired, total_attempts, total_accepted, _ = rewired_graph_by_timestep_fast(
        edge_index=edge_index, time_steps=time_steps, n_swaps_factor=n_swaps_factor, seed=seed
    )
    return rewired, total_attempts, total_accepted


def run_single_external_trial(
    condition: str,
    seed: int,
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    checkpoint_dir: Path,
    max_epochs: int = 50,
    patience: int = 8,
) -> Dict[str, Any]:
    """Train and evaluate a single (condition, seed) trial on validation data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    proc = psutil.Process()

    feat_dim = int(train_data["x"].size(1))
    run_id = f"P5-Cond_{condition}_{feat_dim}f_s{seed}"

    swap_stats = {"attempts": 0, "accepted": 0, "jaccard": 0.0}
    if condition == "A":
        model = DeepParameterMatchedMLP(
            input_dim=feat_dim, time_dim=16, hidden_dim1=128, hidden_dim2=64, dropout=0.3
        )
        edge_tr = torch.empty((2, 0), dtype=torch.long)
        edge_va = torch.empty((2, 0), dtype=torch.long)
    elif condition == "B":
        model = ResidualETGNNClassifier(
            input_dim=feat_dim, time_dim=16, hidden_dim=64, dropout=0.3
        )
        edge_tr = train_data["edge_index"]
        edge_va = val_data["edge_index"]
    elif condition == "C":
        model = ResidualETGNNClassifier(
            input_dim=feat_dim, time_dim=16, hidden_dim=64, dropout=0.3
        )
        edge_tr = torch.empty((2, 0), dtype=torch.long)
        edge_va = torch.empty((2, 0), dtype=torch.long)
    elif condition == "D":
        model = ResidualETGNNClassifier(
            input_dim=feat_dim, time_dim=16, hidden_dim=64, dropout=0.3
        )
        edge_tr, att_tr, acc_tr, jac_tr = rewired_graph_by_timestep_fast(
            train_data["edge_index"], train_data["time_step"], n_swaps_factor=5, seed=seed
        )
        edge_va, att_va, acc_va, _ = rewired_graph_by_timestep_fast(
            val_data["edge_index"], val_data["time_step"], n_swaps_factor=5, seed=seed + 5000
        )
        swap_stats = {"attempts": att_tr + att_va, "accepted": acc_tr + acc_va, "jaccard": jac_tr}
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Extract exact 2-layer computation subgraphs for train_mask and val_mask
    comp_tr = extract_exact_2hop_computation_graph(
        train_data["x"], edge_tr, train_data["y"], train_data["time_step"], train_data["train_mask"]
    )
    comp_va = extract_exact_2hop_computation_graph(
        val_data["x"],
        edge_va,
        val_data["y"],
        val_data["time_step"],
        val_data["val_mask"],
        val_data.get("unseen_entity_mask"),
    )

    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_labels = comp_tr["y"][comp_tr["eval_mask"]]
    n_licit = float((train_labels == 0).sum())
    n_illicit = max(1.0, float((train_labels == 1).sum()))
    weights = torch.tensor([1.0, n_licit / n_illicit], dtype=torch.float32)

    best_auprc = -1.0
    best_epoch = 1
    bad_epochs = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{run_id}.pt"
    y_val = comp_va["y"][comp_va["eval_mask"]].numpy()

    t_start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(comp_tr["x"], comp_tr["edge_index"], comp_tr["time_step"])
        loss = F.cross_entropy(logits[comp_tr["eval_mask"]], train_labels, weight=weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(comp_va["x"], comp_va["edge_index"], comp_va["time_step"])
            prob = torch.softmax(val_logits[comp_va["eval_mask"]], dim=1)[:, 1].numpy()
            auprc = float(average_precision_score(y_val, prob))

        if auprc > best_auprc + 0.0001:
            best_auprc = auprc
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    wall_time = time.time() - t_start
    peak_ram_mb = proc.memory_info().rss / (1024 * 1024)

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    t_inf0 = time.time()
    with torch.no_grad():
        final_logits = model(comp_va["x"], comp_va["edge_index"], comp_va["time_step"])
        final_prob = torch.softmax(final_logits[comp_va["eval_mask"]], dim=1)[:, 1].numpy()
    inf_ms = (time.time() - t_inf0) * 1000.0

    opt_thresh = compute_optimal_f1_threshold(y_val, final_prob)
    metrics = evaluate_predictions_extended(y_val, final_prob, calibrated_threshold=opt_thresh)

    # Compute unseen-entity validation metrics if unseen_entity_mask is available
    unseen_auprc = metrics["auprc"]
    unseen_roc_auc = metrics["roc_auc"]
    unseen_f1 = metrics["f1"]
    if "unseen_entity_mask" in comp_va:
        unseen_in_eval = comp_va["unseen_entity_mask"][comp_va["eval_mask"]].numpy()
        if unseen_in_eval.sum() > 0 and len(np.unique(y_val[unseen_in_eval])) > 1:
            y_un = y_val[unseen_in_eval]
            p_un = final_prob[unseen_in_eval]
            unseen_auprc = float(average_precision_score(y_un, p_un))
            unseen_roc_auc = float(roc_auc_score(y_un, p_un))
            unseen_f1 = float(f1_score(y_un, (p_un >= 0.5).astype(int), zero_division=0))

    metrics.update(
        {
            "run_id": run_id,
            "condition": condition,
            "feat_dim": feat_dim,
            "seed": seed,
            "param_count": param_count,
            "best_epoch": best_epoch,
            "total_epochs": epoch,
            "wall_time_sec": round(wall_time, 3),
            "inference_latency_ms": round(inf_ms, 3),
            "peak_ram_mb": round(peak_ram_mb, 1),
            "checkpoint_path": str(ckpt_path),
            "rewire_attempts": swap_stats["attempts"],
            "rewire_accepted": swap_stats["accepted"],
            "rewire_jaccard": round(swap_stats["jaccard"], 6),
            "unseen_auprc": unseen_auprc,
            "unseen_roc_auc": unseen_roc_auc,
            "unseen_f1": unseen_f1,
        }
    )
    return metrics


def run_stage1_synthetic_dry_run(work_dir: Path) -> Dict[str, Any]:
    """Execute Stage 1 Dry Run on synthetic schema-exact data across all 4 conditions."""
    raw_syn = work_dir / "raw_synthetic"
    proc_syn = work_dir / "processed_synthetic"
    ckpt_syn = work_dir / "checkpoints_synthetic"

    generate_synthetic_elliptic_plusplus_csvs(raw_syn, n_actors=300, feat_dim=56, seed=42)
    pipeline = EllipticPlusPlusPipeline(raw_dir=raw_syn, processed_dir=proc_syn, expected_feat_dim=56)
    manifest = pipeline.run_pipeline()

    train_data = torch.load(proc_syn / "train_graph.pt", map_location="cpu", weights_only=True)
    val_data = torch.load(proc_syn / "validation_graph.pt", map_location="cpu", weights_only=True)

    trial_results = {}
    for cond in ["A", "B", "C", "D"]:
        res = run_single_external_trial(
            condition=cond,
            seed=42,
            train_data=train_data,
            val_data=val_data,
            checkpoint_dir=ckpt_syn,
            max_epochs=6,
            patience=3,
        )
        trial_results[cond] = res

    # Verify parameter equality across B, C, D
    assert (
        trial_results["B"]["param_count"]
        == trial_results["C"]["param_count"]
        == trial_results["D"]["param_count"]
        == 9736
    ), "Parameter count mismatch across B, C, D on 56f!"

    return {
        "stage": "Stage_1_Synthetic_Dry_Run",
        "status": "PASS",
        "pipeline_manifest": manifest,
        "trials": trial_results,
    }


def evaluate_checkpoint_on_test_split(
    condition: str,
    seed: int,
    test_data: Dict[str, torch.Tensor],
    checkpoint_path: Path,
    val_calibrated_threshold: float,
) -> Dict[str, Any]:
    """Single-shot held-out test evaluation (steps 41..49) using frozen trained checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    proc = psutil.Process()

    feat_dim = int(test_data["x"].size(1))
    run_id = f"P5-Cond_{condition}_{feat_dim}f_s{seed}_TEST"

    swap_stats = {"attempts": 0, "accepted": 0, "jaccard": 0.0}
    if condition == "A":
        model = DeepParameterMatchedMLP(
            input_dim=feat_dim, time_dim=16, hidden_dim1=128, hidden_dim2=64, dropout=0.3
        )
        edge_te = torch.empty((2, 0), dtype=torch.long)
    elif condition == "B":
        model = ResidualETGNNClassifier(
            input_dim=feat_dim, time_dim=16, hidden_dim=64, dropout=0.3
        )
        edge_te = test_data["edge_index"]
    elif condition == "C":
        model = ResidualETGNNClassifier(
            input_dim=feat_dim, time_dim=16, hidden_dim=64, dropout=0.3
        )
        edge_te = torch.empty((2, 0), dtype=torch.long)
    elif condition == "D":
        model = ResidualETGNNClassifier(
            input_dim=feat_dim, time_dim=16, hidden_dim=64, dropout=0.3
        )
        edge_te, att_te, acc_te, jac_te = rewired_graph_by_timestep_fast(
            test_data["edge_index"], test_data["time_step"], n_swaps_factor=5, seed=seed + 9000
        )
        swap_stats = {"attempts": att_te, "accepted": acc_te, "jaccard": jac_te}
    else:
        raise ValueError(f"Unknown condition: {condition}")

    comp_te = extract_exact_2hop_computation_graph(
        test_data["x"],
        edge_te,
        test_data["y"],
        test_data["time_step"],
        test_data["test_mask"],
        test_data.get("unseen_entity_mask"),
    )

    param_count = sum(p.numel() for p in model.parameters())
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    y_test = comp_te["y"][comp_te["eval_mask"]].numpy()
    t_inf0 = time.time()
    with torch.no_grad():
        test_logits = model(comp_te["x"], comp_te["edge_index"], comp_te["time_step"])
        test_prob = torch.softmax(test_logits[comp_te["eval_mask"]], dim=1)[:, 1].numpy()
    inf_ms = (time.time() - t_inf0) * 1000.0
    peak_ram_mb = proc.memory_info().rss / (1024 * 1024)

    metrics = evaluate_predictions_extended(
        y_test, test_prob, calibrated_threshold=val_calibrated_threshold
    )

    unseen_auprc = metrics["auprc"]
    unseen_roc_auc = metrics["roc_auc"]
    unseen_f1 = metrics["f1"]
    unseen_cal_f1 = metrics["val_calibrated_f1"]
    if "unseen_entity_mask" in comp_te:
        unseen_in_eval = comp_te["unseen_entity_mask"][comp_te["eval_mask"]].numpy()
        if unseen_in_eval.sum() > 0 and len(np.unique(y_test[unseen_in_eval])) > 1:
            y_un = y_test[unseen_in_eval]
            p_un = test_prob[unseen_in_eval]
            unseen_auprc = float(average_precision_score(y_un, p_un))
            unseen_roc_auc = float(roc_auc_score(y_un, p_un))
            unseen_f1 = float(f1_score(y_un, (p_un >= 0.5).astype(int), zero_division=0))
            unseen_cal_f1 = float(
                f1_score(y_un, (p_un >= val_calibrated_threshold).astype(int), zero_division=0)
            )

    metrics.update(
        {
            "run_id": run_id,
            "condition": condition,
            "feat_dim": feat_dim,
            "seed": seed,
            "param_count": param_count,
            "inference_latency_ms": round(inf_ms, 3),
            "peak_ram_mb": round(peak_ram_mb, 1),
            "checkpoint_path": str(checkpoint_path),
            "rewire_attempts": swap_stats["attempts"],
            "rewire_accepted": swap_stats["accepted"],
            "rewire_jaccard": round(swap_stats["jaccard"], 6),
            "unseen_auprc": unseen_auprc,
            "unseen_roc_auc": unseen_roc_auc,
            "unseen_f1": unseen_f1,
            "unseen_val_calibrated_f1": unseen_cal_f1,
        }
    )
    return metrics


def compute_phase5_statistical_analysis(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    """
    Compute condition summaries (Mean, SD, 95% CI, Median, IQR), paired t-tests,
    Cohen's d_z, and Holm-Bonferroni multiplicity-adjusted p-values (m=3).
    """
    conditions = ["A", "B", "C", "D"]
    cond_names = {
        "A": "DeepParameterMatchedMLP (Feature-Only)",
        "B": "ResidualETGNNClassifier (True AddrAddr Graph)",
        "C": "ResidualETGNNClassifier (Empty Graph Control)",
        "D": "ResidualETGNNClassifier (Degree-Preserving Rewired Graph)",
    }
    metric_cols = [
        "auprc",
        "unseen_auprc",
        "roc_auc",
        "f1",
        "val_calibrated_f1",
        "precision",
        "recall",
        "inference_latency_ms",
        "peak_ram_mb",
    ]
    if "wall_time_sec" in df.columns:
        metric_cols.append("wall_time_sec")

    t_crit = float(stats.t.ppf(0.975, df=4))
    cond_summaries = {}

    for c in conditions:
        sub = df[df["condition"] == c].sort_values("seed")
        c_dict: Dict[str, Any] = {
            "condition": c,
            "description": cond_names[c],
            "n_seeds": len(sub),
            "param_count": int(sub["param_count"].iloc[0]),
            "metrics": {},
        }
        for m in metric_cols:
            if m not in sub.columns:
                continue
            vals = sub[m].to_numpy(dtype=np.float64)
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se = sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ci_low = mean - t_crit * se
            ci_high = mean + t_crit * se
            median = float(np.median(vals))
            q75, q25 = np.percentile(vals, [75, 25])
            c_dict["metrics"][m] = {
                "mean": round(mean, 6),
                "sd": round(sd, 6),
                "ci95_low": round(float(ci_low), 6),
                "ci95_high": round(float(ci_high), 6),
                "median": round(median, 6),
                "iqr": round(float(q75 - q25), 6),
                "min": round(float(np.min(vals)), 6),
                "max": round(float(np.max(vals)), 6),
                "seed_values": [round(float(x), 6) for x in vals.tolist()],
            }
        cond_summaries[c] = c_dict

    # Primary (m=3) and secondary paired contrasts on primary metric (auprc) and unseen_auprc
    contrast_specs = [
        ("B_vs_D_rewired", "B", "D", True, "True AddrAddr Topology vs. Degree-Preserving Rewired Null"),
        ("B_vs_C_empty", "B", "C", True, "True AddrAddr Topology vs. Empty-Edge Ablation"),
        ("B_vs_A_mlp", "B", "A", True, "Residual ETGNN (True Graph) vs. Feature-Only Deep MLP"),
        ("C_vs_A_empty_vs_mlp", "C", "A", False, "Residual ETGNN (Empty Graph) vs. Feature-Only Deep MLP"),
    ]

    def evaluate_contrast_family(metric_key: str) -> List[Dict[str, Any]]:
        results_list = []
        for cid, c1, c2, is_primary, label in contrast_specs:
            v1 = df[df["condition"] == c1].sort_values("seed")[metric_key].to_numpy(dtype=np.float64)
            v2 = df[df["condition"] == c2].sort_values("seed")[metric_key].to_numpy(dtype=np.float64)
            diffs = v1 - v2
            mean_d = float(np.mean(diffs))
            sd_d = float(np.std(diffs, ddof=1))
            se_d = sd_d / np.sqrt(len(diffs))
            ci_l = mean_d - t_crit * se_d
            ci_h = mean_d + t_crit * se_d
            if sd_d > 1e-12:
                t_stat, p_raw = stats.ttest_rel(v1, v2)
                t_stat = float(t_stat)
                p_raw = float(p_raw)
                cohen_dz = float(mean_d / sd_d)
            else:
                t_stat, p_raw, cohen_dz = 0.0, 1.0, 0.0
            pos_count = int(np.sum(diffs > 0))
            neg_count = int(np.sum(diffs < 0))
            results_list.append(
                {
                    "contrast_id": cid,
                    "metric": metric_key,
                    "comparison": f"{c1} - {c2}",
                    "description": label,
                    "is_primary_family": is_primary,
                    "mean_diff": round(mean_d, 6),
                    "sd_diff": round(sd_d, 6),
                    "ci95_low": round(float(ci_l), 6),
                    "ci95_high": round(float(ci_h), 6),
                    "t_stat": round(t_stat, 4),
                    "p_raw": round(p_raw, 6),
                    "cohen_dz": round(cohen_dz, 4),
                    "seed_diffs": [round(float(x), 6) for x in diffs.tolist()],
                    "sign_concordance": f"{pos_count}+ / {neg_count}- (of {len(diffs)})",
                }
            )

        # Apply Holm-Bonferroni across the m=3 primary contrasts
        primary_indices = [i for i, r in enumerate(results_list) if r["is_primary_family"]]
        primary_sorted = sorted(primary_indices, key=lambda idx: results_list[idx]["p_raw"])
        m_fam = len(primary_sorted)
        running_max = 0.0
        for rank, idx in enumerate(primary_sorted):
            adj = min(1.0, results_list[idx]["p_raw"] * (m_fam - rank))
            running_max = max(running_max, adj)
            results_list[idx]["p_holm"] = round(running_max, 6)
            results_list[idx]["statistically_significant_holm_005"] = bool(running_max < 0.05)
            results_list[idx]["practically_significant_002"] = bool(
                abs(results_list[idx]["mean_diff"]) >= 0.020
            )

        for r in results_list:
            if not r["is_primary_family"]:
                r["p_holm"] = None
                r["statistically_significant_holm_005"] = bool(r["p_raw"] < 0.05)
                r["practically_significant_002"] = bool(abs(r["mean_diff"]) >= 0.020)

        return results_list

    return {
        "split": split_name,
        "dataset": "Elliptic++ Actors (56 features, chronological split)",
        "seeds": [42, 43, 44, 45, 46],
        "conditions": cond_summaries,
        "contrasts_auprc": evaluate_contrast_family("auprc"),
        "contrasts_unseen_auprc": evaluate_contrast_family("unseen_auprc"),
        "contrasts_val_calibrated_f1": evaluate_contrast_family("val_calibrated_f1"),
        "contrasts_roc_auc": evaluate_contrast_family("roc_auc"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Controlled External Evaluation Runner")
    parser.add_argument(
        "--stage",
        choices=["stage1_dryrun", "stage2_preflight", "stage3_pilot", "stage4_full", "stage5_test"],
        default="stage2_preflight",
    )
    parser.add_argument("--authorize-test", action="store_true", help="Explicit gate for Stage 5 test set")
    args = parser.parse_args()

    verified_hashes = verify_phase4_archive_intact(PROJECT_ROOT)
    print(f"[Archive Guard] Verified {len(verified_hashes)} frozen Phase 4 & Baseline SHA-256 hashes.", flush=True)

    if args.stage == "stage2_preflight":
        pf = run_stage2_resource_preflight(PROJECT_ROOT)
        out_p = PROJECT_ROOT / "results" / "phase5_Preflight_and_DryRun.json"
        out_p.write_text(json.dumps(pf, indent=2), encoding="utf-8")
        print(json.dumps(pf, indent=2))
    elif args.stage == "stage1_dryrun":
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dr = run_stage1_synthetic_dry_run(Path(tmp))
            pf = run_stage2_resource_preflight(PROJECT_ROOT)
            combined = {"preflight": pf, "dry_run": dr, "frozen_hashes": verified_hashes}
            out_p = PROJECT_ROOT / "results" / "phase5_Preflight_and_DryRun.json"
            out_p.write_text(json.dumps(combined, indent=2), encoding="utf-8")
            print(f"Stage 1 Dry Run & Stage 2 Preflight PASSED. Saved to {out_p}")
    else:
        pf = run_stage2_resource_preflight(PROJECT_ROOT)
        if not pf["external_dataset_present_locally"]:
            print(
                "BLOCKED: External dataset files not found in data/external/elliptic_plusplus/.\n"
                "Per strict execution rules, explicit user authorization is required before downloading "
                "the ~350-500 MB Elliptic++ Actors CSVs or running Stage 3-5 training."
            )
            sys.exit(2)

        proc_dir = PROJECT_ROOT / "data" / "processed" / "elliptic_plusplus"
        out_dir = PROJECT_ROOT / "results" / "phase5_external"
        ckpt_dir = out_dir / "checkpoints"
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if args.stage == "stage3_pilot":
            print("[Stage 3 Pilot] Loading processed train and validation graphs...", flush=True)
            tr_data = torch.load(proc_dir / "train_graph.pt", map_location="cpu", weights_only=True)
            va_data = torch.load(proc_dir / "validation_graph.pt", map_location="cpu", weights_only=True)
            pilot_rows = []
            for cond in ["A", "B", "C", "D"]:
                t_c0 = time.time()
                res = run_single_external_trial(
                    condition=cond,
                    seed=42,
                    train_data=tr_data,
                    val_data=va_data,
                    checkpoint_dir=ckpt_dir,
                    max_epochs=50,
                    patience=8,
                )
                pilot_rows.append(res)
                print(
                    f"  [Pilot s42 Cond {cond}] AUPRC={res['auprc']:.4f} (Unseen={res['unseen_auprc']:.4f}), "
                    f"ROC-AUC={res['roc_auc']:.4f}, F1={res['f1']:.4f}, CalF1={res['val_calibrated_f1']:.4f}, "
                    f"epochs={res['total_epochs']} (best={res['best_epoch']}), time={res['wall_time_sec']:.1f}s, "
                    f"RAM={res['peak_ram_mb']:.1f}MB",
                    flush=True,
                )
            pilot_path = out_dir / "stage3_pilot_summary.json"
            pilot_path.write_text(json.dumps(pilot_rows, indent=2), encoding="utf-8")
            print(f"[Stage 3 Pilot] Saved to {pilot_path}", flush=True)

        elif args.stage == "stage4_full":
            import gc

            print("[Stage 4 Full Validation] Loading processed train and validation graphs...", flush=True)
            tr_data = torch.load(proc_dir / "train_graph.pt", map_location="cpu", weights_only=True)
            va_data = torch.load(proc_dir / "validation_graph.pt", map_location="cpu", weights_only=True)

            # Reuse seed 42 pilot runs or incremental progress if already saved
            cached_trials: Dict[Tuple[str, int], Dict[str, Any]] = {}
            for cache_file in [out_dir / "stage3_pilot_summary.json", out_dir / "stage4_trials_progress.json"]:
                if cache_file.exists():
                    for item in json.loads(cache_file.read_text(encoding="utf-8")):
                        if Path(item["checkpoint_path"]).exists():
                            cached_trials[(item["condition"], int(item["seed"]))] = item

            progress_path = out_dir / "stage4_trials_progress.json"
            all_rows = []
            for seed in [42, 43, 44, 45, 46]:
                for cond in ["A", "B", "C", "D"]:
                    if (cond, seed) in cached_trials:
                        res = cached_trials[(cond, seed)]
                        print(
                            f"  [Cached s{seed} Cond {cond}] AUPRC={res['auprc']:.4f} (Unseen={res['unseen_auprc']:.4f}), "
                            f"CalF1={res['val_calibrated_f1']:.4f}",
                            flush=True,
                        )
                    else:
                        res = run_single_external_trial(
                            condition=cond,
                            seed=seed,
                            train_data=tr_data,
                            val_data=va_data,
                            checkpoint_dir=ckpt_dir,
                            max_epochs=50,
                            patience=8,
                        )
                        gc.collect()
                        print(
                            f"  [Run s{seed} Cond {cond}] AUPRC={res['auprc']:.4f} (Unseen={res['unseen_auprc']:.4f}), "
                            f"ROC-AUC={res['roc_auc']:.4f}, F1={res['f1']:.4f}, CalF1={res['val_calibrated_f1']:.4f}, "
                            f"epochs={res['total_epochs']} (best={res['best_epoch']}), time={res['wall_time_sec']:.1f}s, "
                            f"RAM={res['peak_ram_mb']:.1f}MB",
                            flush=True,
                        )
                    all_rows.append(res)
                    progress_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

            df_val = pd.DataFrame(all_rows)
            csv_path = out_dir / "phase5_validation_matrix.csv"
            df_val.to_csv(csv_path, index=False)

            stat_summary = compute_phase5_statistical_analysis(df_val, split_name="validation_steps_33_to_40")
            stat_path = out_dir / "phase5_validation_statistical_summary.json"
            stat_path.write_text(json.dumps(stat_summary, indent=2), encoding="utf-8")
            print(f"[Stage 4 Full Validation] Saved 20-run matrix to {csv_path} and stats to {stat_path}", flush=True)

        elif args.stage == "stage5_test":
            if not args.authorize_test:
                print("BLOCKED: Stage 5 requires explicit --authorize-test flag.", flush=True)
                sys.exit(2)

            val_csv = out_dir / "phase5_validation_matrix.csv"
            if not val_csv.exists():
                raise FileNotFoundError(f"Run stage4_full before stage5_test (missing {val_csv})")

            df_val = pd.read_csv(val_csv)
            print("[Stage 5 Held-Out Test] Loading test_graph.pt (steps 41..49)...", flush=True)
            te_data = torch.load(proc_dir / "test_graph.pt", map_location="cpu", weights_only=True)

            test_rows = []
            for _, vrow in df_val.iterrows():
                cond = str(vrow["condition"])
                seed = int(vrow["seed"])
                ckpt_p = Path(vrow["checkpoint_path"])
                cal_th = float(vrow["calibrated_threshold"])
                tres = evaluate_checkpoint_on_test_split(
                    condition=cond,
                    seed=seed,
                    test_data=te_data,
                    checkpoint_path=ckpt_p,
                    val_calibrated_threshold=cal_th,
                )
                test_rows.append(tres)
                print(
                    f"  [Test s{seed} Cond {cond}] AUPRC={tres['auprc']:.4f} (Unseen={tres['unseen_auprc']:.4f}), "
                    f"ROC-AUC={tres['roc_auc']:.4f}, F1@0.5={tres['f1']:.4f}, CalF1={tres['val_calibrated_f1']:.4f}, "
                    f"inf={tres['inference_latency_ms']:.1f}ms",
                    flush=True,
                )

            df_test = pd.DataFrame(test_rows)
            test_csv = out_dir / "phase5_test_matrix.csv"
            df_test.to_csv(test_csv, index=False)

            test_stats = compute_phase5_statistical_analysis(df_test, split_name="test_steps_41_to_49")
            test_stat_path = out_dir / "phase5_test_statistical_summary.json"
            test_stat_path.write_text(json.dumps(test_stats, indent=2), encoding="utf-8")
            print(f"[Stage 5 Held-Out Test] Saved 20-run test matrix to {test_csv} and stats to {test_stat_path}", flush=True)


