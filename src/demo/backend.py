from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException

from src.models.etgnn import ETGNNClassifier

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="ETGNN Research Demo", version="1.0.0")


@lru_cache(maxsize=1)
def load_state():
    torch.set_num_threads(2)
    processed = ROOT / "data" / "processed"
    checkpoints = ROOT / "results" / "checkpoints"

    nodes = pd.read_csv(processed / "nodes_with_masks.csv")
    features = np.load(processed / "scaled_features.npy", mmap_mode="r")
    by_id = {
        int(tx_id): int(index)
        for tx_id, index in zip(nodes["txId"], nodes["node_index"])
    }

    rf = joblib.load(checkpoints / "random_forest.joblib")

    graph = torch.load(
        processed / "test_graph.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = ETGNNClassifier()
    model.load_state_dict(torch.load(
        checkpoints / "etgnn.pt",
        map_location="cpu",
        weights_only=True,
    ))
    model.eval()
    with torch.no_grad():
        etgnn_prob = torch.softmax(
            model(graph["x"], graph["edge_index"],
                  graph["time_step"]),
            dim=1,
        )[:, 1].numpy()

    if not np.array_equal(
        graph["original_node_index"].numpy(),
        nodes["node_index"].to_numpy(),
    ):
        raise RuntimeError("Graph and transaction mapping do not align")

    explanations = {}
    for case in ("tp", "fp", "fn", "tn"):
        path = ROOT / "results" / "explanations" / f"connected_{case}.json"
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            explanations[int(item["node_index"])] = item

    return nodes, features, by_id, rf, etgnn_prob, explanations


@app.get("/health")
def health():
    return {"status": "ok", "scope": "existing Elliptic transactions"}


@app.get("/examples")
def examples():
    nodes, _, _, _, _, explanations = load_state()
    rows = []
    for index, item in explanations.items():
        node = nodes.iloc[index]
        rows.append({
            "case": item["case"],
            "transaction_id": int(node["txId"]),
            "time_step": int(node["time_step"]),
        })
    return sorted(rows, key=lambda row: row["case"])


@app.get("/transactions/{transaction_id}")
def transaction(transaction_id: int):
    nodes, features, by_id, rf, etgnn_prob, explanations = load_state()
    index = by_id.get(transaction_id)
    if index is None:
        raise HTTPException(status_code=404,
                            detail="Transaction ID not in Elliptic dataset")

    node = nodes.iloc[index]
    rf_prob = float(rf.predict_proba(
        np.asarray(features[index:index + 1])
    )[0, 1])
    et_prob = float(etgnn_prob[index])
    actual = int(node["label"])

    if bool(node["train_mask"]):
        split = "train"
    elif bool(node["val_mask"]):
        split = "validation"
    elif bool(node["test_mask"]):
        split = "test"
    else:
        split = "unlabelled"

    return {
        "transaction_id": transaction_id,
        "time_step": int(node["time_step"]),
        "split": split,
        "actual_class": None if actual == -1 else actual,
        "random_forest": {
            "illicit_probability": rf_prob,
            "predicted_class": int(rf_prob >= 0.5),
        },
        "etgnn": {
            "illicit_probability": et_prob,
            "predicted_class": int(et_prob >= 0.5),
        },
        "explanation": explanations.get(index),
        "note": "Research demo using existing Elliptic transactions and completed time-step graph data.",
    }
