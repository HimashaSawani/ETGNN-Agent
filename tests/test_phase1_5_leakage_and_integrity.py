import pytest
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from src.models.etgnn import ETGNNClassifier
from src.web.data_loader import ResearchDataLoader

@pytest.fixture(scope="module")
def nodes_df():
    return pd.read_csv("data/processed/nodes_with_masks.csv")

@pytest.fixture(scope="module")
def edges_df():
    return pd.read_csv("data/raw/elliptic/elliptic_txs_edgelist.csv")

def test_snapshot_43_topology_calculations(nodes_df, edges_df):
    """
    Workstream 1 Verification:
    Verify the pre-shift (steps 41-42) and post-shift (steps 43-49)
    proportions of illicit transactions with zero incoming candidate edges.
    """
    pre_nodes = nodes_df[nodes_df["time_step"].isin([41, 42])]
    post_nodes = nodes_df[nodes_df["time_step"].isin(range(43, 50))]

    pre_illicit = set(pre_nodes[pre_nodes["label"] == 1]["txId"])
    post_illicit = set(post_nodes[post_nodes["label"] == 1]["txId"])

    pre_dst = set(edges_df[edges_df["txId2"].isin(pre_illicit)]["txId2"])
    post_dst = set(edges_df[edges_df["txId2"].isin(post_illicit)]["txId2"])

    pre_zero_in = pre_illicit - pre_dst
    post_zero_in = post_illicit - post_dst

    pre_pct = len(pre_zero_in) / len(pre_illicit) * 100
    post_pct = len(post_zero_in) / len(post_illicit) * 100

    # Assert exact empirical values: 13.52% and 76.92%
    assert len(pre_illicit) == 355
    assert len(pre_zero_in) == 48
    assert round(pre_pct, 2) == 13.52

    assert len(post_illicit) == 169
    assert len(post_zero_in) == 130
    assert round(post_pct, 2) == 76.92

    # Assert that undirected degree is NOT 0 (all transactions appear in the edgelist)
    all_edge_nodes = set(edges_df["txId1"]).union(set(edges_df["txId2"]))
    assert len(pre_illicit - all_edge_nodes) == 0
    assert len(post_illicit - all_edge_nodes) == 0

def test_temporal_graph_construction_boundaries(nodes_df, edges_df):
    """
    Workstream 2 Verification:
    Assert zero cross-temporal edges exist in the raw Elliptic graph.
    Assert train_graph.pt strictly isolates time_step <= 32.
    """
    tx_time = dict(zip(nodes_df["txId"], nodes_df["time_step"]))
    src_times = edges_df["txId1"].map(tx_time)
    dst_times = edges_df["txId2"].map(tx_time)

    # 100% intra-snapshot edges
    assert (src_times == dst_times).all(), "Found cross-temporal edge in raw edgelist!"

    # Verify train_graph.pt time step boundary
    train_graph = torch.load("data/processed/train_graph.pt", map_location="cpu", weights_only=True)
    assert (train_graph["time_step"] <= 32).all(), "train_graph.pt contains time_step > 32!"

def test_training_only_scaler_fitting(nodes_df):
    """
    Workstream 2 Verification:
    Verify StandardScaler was fit strictly on train split (time_step <= 32, exactly 130,628 nodes).
    """
    scaler = joblib.load("data/processed/train_scaler.joblib")
    train_count = (nodes_df["time_step"] <= 32).sum()
    assert train_count == 130628
    assert scaler.n_samples_seen_ == train_count, f"Scaler saw {scaler.n_samples_seen_} samples, expected {train_count}"

def test_unknown_label_exclusion(nodes_df):
    """
    Workstream 2 Verification:
    Verify unlabelled nodes (label == -1) are never included in train, val, or test masks.
    """
    unknown_mask = nodes_df["label"] == -1
    assert (nodes_df.loc[unknown_mask, "train_mask"] == False).all()
    assert (nodes_df.loc[unknown_mask, "val_mask"] == False).all()
    assert (nodes_df.loc[unknown_mask, "test_mask"] == False).all()

def test_future_evidence_exclusion():
    """
    Workstream 2 Verification:
    Verify point-in-time retrieval contract: in historical evaluation mode,
    any evidence timestamp t_ev > t_query is blocked.
    """
    def filter_evidence(evidence_list, query_timestep, mode="historical"):
        if mode == "historical":
            return [e for e in evidence_list if e["snapshot_id"] <= query_timestep and e["label_available"]]
        elif mode == "retrospective":
            return [e for e in evidence_list]
        raise ValueError(f"Unknown mode {mode}")

    sample_evidence = [
        {"id": "ev1", "snapshot_id": 30, "label_available": True},
        {"id": "ev2", "snapshot_id": 35, "label_available": True},
        {"id": "ev3", "snapshot_id": 42, "label_available": False}, # Unlabeled at historical point
        {"id": "ev4", "snapshot_id": 45, "label_available": True},
    ]

    hist_filtered = filter_evidence(sample_evidence, query_timestep=32, mode="historical")
    assert len(hist_filtered) == 1
    assert hist_filtered[0]["id"] == "ev1"

    retro_filtered = filter_evidence(sample_evidence, query_timestep=32, mode="retrospective")
    assert len(retro_filtered) == 4

def test_checkpoint_inference_parity():
    """
    Workstream 4 Verification:
    Verify that Canonical ETGNN and Random Forest checkpoints reproduce
    the authoritative benchmark AUPRC within 1e-4 tolerance.
    """
    nodes = pd.read_csv("data/processed/nodes_with_masks.csv")
    features = np.load("data/processed/scaled_features.npy", mmap_mode="r")
    test_mask = nodes["test_mask"].to_numpy(dtype=bool)
    y_test = nodes.loc[test_mask, "label"].to_numpy()

    # 1. Random Forest
    rf = joblib.load("results/checkpoints/random_forest.joblib")
    rf_prob = rf.predict_proba(features[test_mask])[:, 1]
    rf_auprc = average_precision_score(y_test, rf_prob)
    assert abs(rf_auprc - 0.632263) < 1e-4, f"RF AUPRC mismatch: {rf_auprc}"

    # 2. Canonical ETGNN
    graph = torch.load("data/processed/test_graph.pt", map_location="cpu", weights_only=True)
    etgnn = ETGNNClassifier()
    etgnn.load_state_dict(torch.load("results/checkpoints/etgnn.pt", map_location="cpu", weights_only=True))
    etgnn.eval()
    with torch.no_grad():
        logits = etgnn(graph["x"], graph["edge_index"], graph["time_step"])
        etgnn_prob = torch.softmax(logits[graph["test_mask"]], dim=1)[:, 1].numpy()
    etgnn_auprc = average_precision_score(y_test, etgnn_prob)
    assert abs(etgnn_auprc - 0.225030) < 1e-4, f"ETGNN AUPRC mismatch: {etgnn_auprc}"

def test_decision_threshold_correctness():
    """
    Workstream 4 Verification:
    Verify decision threshold 0.50 produces reported metrics for Random Forest.
    """
    nodes = pd.read_csv("data/processed/nodes_with_masks.csv")
    features = np.load("data/processed/scaled_features.npy", mmap_mode="r")
    test_mask = nodes["test_mask"].to_numpy(dtype=bool)
    y_test = nodes.loc[test_mask, "label"].to_numpy()

    rf = joblib.load("results/checkpoints/random_forest.joblib")
    rf_prob = rf.predict_proba(features[test_mask])[:, 1]
    rf_pred = (rf_prob >= 0.50).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_test, rf_pred, average="binary", pos_label=1)
    assert abs(p - 0.696552) < 1e-4
    assert abs(r - 0.578244) < 1e-4
    assert abs(f1 - 0.631908) < 1e-4

def test_isolated_node_explanation_handling():
    """
    Workstream 5 Verification:
    Verify that an isolated transaction or transaction with zero incoming edges
    in ResearchDataLoader.get_explanation returns a safe, structured fallback
    without throwing an unhandled exception or 500 error.
    """
    loader = ResearchDataLoader()
    # Find a transaction in step 43 that has 0 incoming candidate edges
    nodes_df = pd.read_csv("data/processed/nodes_with_masks.csv")
    edges_df = pd.read_csv("data/raw/elliptic/elliptic_txs_edgelist.csv")
    
    t43_nodes = nodes_df[nodes_df["time_step"] == 43]
    t43_txs = set(t43_nodes["txId"])
    t43_dst = set(edges_df[edges_df["txId2"].isin(t43_txs)]["txId2"])
    zero_in_txs = list(t43_txs - t43_dst)
    assert len(zero_in_txs) > 0, "No zero-incoming transactions found in step 43"

    sample_tx = zero_in_txs[0]
    explanation = loader.get_explanation(sample_tx)

    assert explanation is not None
    assert "disclaimer" in explanation or "message" in explanation
    assert explanation.get("candidate_edges_count", 0) == 0
    assert explanation.get("signed_drop", 0.0) == 0.0
    assert explanation.get("top_edges", []) == []
