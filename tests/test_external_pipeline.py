"""
Automated unit tests for Phase E External Dataset Pipeline (Elliptic++ Actors).
Validates chunked CSV ingestion, train-only feature standardization, duplicate & self-loop removal,
chronological split isolation, and PyTorch Geometric stage-graph assembly on synthetic CSV fixtures.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.elliptic_plusplus_pipeline import EllipticPlusPlusPipeline


def _create_synthetic_elliptic_pp_csvs(raw_dir: Path, n_nodes: int = 80) -> None:
    """Create synthetic Elliptic++ Actors CSVs with intentional duplicates, self-loops, and NaN/Inf."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    addresses = [f"1BitcoinAddr{i:05d}" for i in range(n_nodes)]
    # Time steps 1..49 across the 80 nodes
    time_steps = np.linspace(1, 49, n_nodes, dtype=int)
    # Ensure consecutive pairs share a time step so intra-step edges form
    for i in range(0, n_nodes - 1, 2):
        time_steps[i + 1] = time_steps[i]

    feats = rng.normal(0.0, 2.0, size=(n_nodes, 56))
    # Inject a NaN and Inf to verify sanitization
    feats[0, 3] = np.nan
    feats[1, 7] = np.inf

    feat_df = pd.DataFrame({"address": addresses, "Time step": time_steps})
    for j in range(56):
        feat_df[f"feat_{j+1}"] = feats[:, j]
    # Add a duplicate (address, time_step) row to test deduplication
    feat_df = pd.concat([feat_df, feat_df.iloc[[0]]], ignore_index=True)
    feat_df.to_csv(raw_dir / "wallets_features.csv", index=False)

    # Classes: 1 (illicit), 2 (licit), 3 (unknown)
    raw_classes = []
    for i in range(n_nodes):
        if i % 5 == 0:
            raw_classes.append(1)
        elif i % 5 in (1, 2, 3):
            raw_classes.append(2)
        else:
            raw_classes.append(3)
    cls_df = pd.DataFrame({"address": addresses, "class": raw_classes})
    cls_df.to_csv(raw_dir / "wallets_classes.csv", index=False)

    # Directed intra-step edges plus one self-loop, one duplicate edge, and one missing address edge
    src_list = []
    dst_list = []
    for i in range(0, n_nodes - 1, 2):
        src_list.append(addresses[i])
        dst_list.append(addresses[i + 1])
    # Intentional self-loop
    src_list.append(addresses[2])
    dst_list.append(addresses[2])
    # Intentional duplicate edge
    src_list.append(addresses[0])
    dst_list.append(addresses[1])
    # Intentional unknown address edge
    src_list.append("1UnknownExternalAddr")
    dst_list.append(addresses[0])

    edge_df = pd.DataFrame({"input_address": src_list, "output_address": dst_list})
    edge_df.to_csv(raw_dir / "AddrAddr_edgelist.csv", index=False)


def test_elliptic_plusplus_pipeline_end_to_end(tmp_path: Path):
    raw_dir = tmp_path / "raw_elliptic_pp"
    out_dir = tmp_path / "processed_elliptic_pp"
    _create_synthetic_elliptic_pp_csvs(raw_dir, n_nodes=100)

    pipeline = EllipticPlusPlusPipeline(
        raw_dir=raw_dir,
        processed_dir=out_dir,
        train_steps=(1, 32),
        val_steps=(33, 40),
        test_steps=(41, 49),
        chunk_size=25,
    )

    ok, missing = pipeline.check_raw_files_exist()
    assert ok is True
    assert len(missing) == 0

    manifest = pipeline.run_pipeline()
    assert manifest["total_observations"] == 100
    assert manifest["feature_dim"] == 56
    assert manifest["data_hygiene"]["duplicate_observations_dropped"] == 1
    assert manifest["data_hygiene"]["self_loops_removed"] >= 1
    assert manifest["data_hygiene"]["duplicate_edges_removed"] >= 1

    # Verify saved stage graphs
    train_g = torch.load(out_dir / "train_graph.pt", weights_only=False)
    val_g = torch.load(out_dir / "validation_graph.pt", weights_only=False)
    test_g = torch.load(out_dir / "test_graph.pt", weights_only=False)

    # 1. No NaN or Inf in features
    assert torch.isfinite(train_g["x"]).all()
    assert torch.isfinite(val_g["x"]).all()
    assert torch.isfinite(test_g["x"]).all()

    # 2. Train standardization has ~0 mean on train nodes
    train_node_means = train_g["x"].mean(dim=0)
    assert torch.allclose(train_node_means, torch.zeros_like(train_node_means), atol=1e-4)

    # 3. Chronological split isolation (stage-windowed bounds)
    assert int(train_g["time_step"].max().item()) <= 32
    assert 33 <= int(val_g["time_step"].min().item()) and int(val_g["time_step"].max().item()) <= 40
    assert 41 <= int(test_g["time_step"].min().item()) and int(test_g["time_step"].max().item()) <= 49

    # 4. No self-loops in any stage graph
    for g in (train_g, val_g, test_g):
        if g["edge_index"].numel() > 0:
            assert not bool((g["edge_index"][0] == g["edge_index"][1]).any().item())

    # 5. Label encoding strictly in {-1, 0, 1}
    unique_labels = set(torch.unique(test_g["y"]).tolist())
    assert unique_labels.issubset({-1, 0, 1})
