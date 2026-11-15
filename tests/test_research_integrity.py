"""
Automated Research Integrity, Reproducibility, Backend Handler, and Phase 4/5 Audit Tests.
Verifies:
1. Frozen artifact SHA-256 checksums against configs/phase5_frozen_manifest.json
2. Exact 40/40 parity between PHASE4_EXPERIMENT_MATRIX.csv and PHASE4_EXPERIMENT_RESULTS.md
3. Degree-preserving rewiring invariants (exact in-degree/out-degree sequence preservation, zero self-loops)
4. Parameter-count parity between DeepParameterMatchedMLP and ResidualETGNNClassifier
5. Direct Python execution of all FastAPI route handlers, Pydantic validation, live model inference,
   subgraph truncation disclosures, frozen/live explanations, Phase 4 Holm-adjusted statistics,
   and Phase H XAI validation summaries (without requiring loopback socketpair)
6. Cryptographic & structural probability cache provenance and tamper recovery
7. Optional ASGI TestClient verification (marked ENVIRONMENT-BLOCKED when sandbox blocks 127.0.0.1 loopback)
"""

import hashlib
import json
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
import torch

from src.training.run_phase5_external_eval import rewired_graph_by_timestep
from src.models.reliability_etgnn import (
    DeepParameterMatchedMLP,
    ResidualETGNNClassifier,
)
from src.web.app import (
    PredictRequest,
    app,
    get_explanation,
    get_metrics,
    get_models,
    get_research_ablations,
    get_research_c1,
    get_research_c2,
    get_research_phase4,
    get_research_temporal_stability,
    get_research_error_mitigation,
    get_research_xai_validation,
    get_transaction,
    get_transaction_subgraph,
    get_transactions,
    health,
    predict_transaction,
    serve_spa,
)
from src.web.data_loader import get_data_loader
from tests.conftest import ENVIRONMENT_BLOCKED_REASON, LOOPBACK_SOCKETPAIR_AVAILABLE

ROOT = Path(__file__).resolve().parent.parent
EXEMPLAR_TP_TX = 288717023
INVALID_TX = 999999999999


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_frozen_manifest_hashes():
    manifest_path = ROOT / "configs" / "phase5_frozen_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path, expected_hash in manifest["frozen_artifact_sha256"].items():
        full_path = ROOT / rel_path
        assert full_path.exists(), f"Missing frozen artifact: {rel_path}"
        assert _sha256(full_path) == expected_hash, f"SHA-256 mismatch for {rel_path}"


def test_phase4_matrix_integrity_and_markdown_parity():
    csv_path = ROOT / "research_extension" / "PHASE4_EXPERIMENT_MATRIX.csv"
    md_path = ROOT / "research_extension" / "PHASE4_EXPERIMENT_RESULTS.md"
    assert csv_path.exists()
    assert md_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 40
    df["condition_id"] = [
        f"Cond_{c}_{int(fd)}f" for c, fd in zip(df["condition"], df["feat_dim"])
    ]
    assert df["condition_id"].nunique() == 8
    assert df["seed"].nunique() == 5

    md_text = md_path.read_text(encoding="utf-8")
    for _, row in df.iterrows():
        assert str(row["run_id"]) in md_text
        cm_fragment = f"| {int(row['tp'])} | {int(row['fp'])} | {int(row['tn'])} | {int(row['fn'])} |"
        assert cm_fragment in md_text, f"Missing exact confusion matrix in markdown for {row['run_id']}"


def test_degree_preserving_rewiring_invariants():
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2, 3, 4, 5, 1, 2, 3],
            [1, 2, 3, 4, 5, 0, 1, 4, 5, 0],
        ],
        dtype=torch.long,
    )
    time_steps = torch.ones(6, dtype=torch.long)
    rewired, total_attempts, total_accepted = rewired_graph_by_timestep(
        edge_index, time_steps=time_steps, n_swaps_factor=5, seed=1042
    )

    assert rewired.shape == edge_index.shape
    assert total_attempts > 0
    assert not bool((rewired[0] == rewired[1]).any().item())
    orig_out = torch.bincount(edge_index[0], minlength=6)
    orig_in = torch.bincount(edge_index[1], minlength=6)
    new_out = torch.bincount(rewired[0], minlength=6)
    new_in = torch.bincount(rewired[1], minlength=6)
    assert torch.equal(orig_out, new_out)
    assert torch.equal(orig_in, new_in)


def test_parameter_counts_across_93f_165f_and_56f():
    expected_gnn = {93: 14472, 165: 23688, 56: 9736}
    for in_dim, exp_params in expected_gnn.items():
        gnn = ResidualETGNNClassifier(input_dim=in_dim, time_dim=16, hidden_dim=64, dropout=0.3)
        params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
        assert params == exp_params

    mlp_93 = DeepParameterMatchedMLP(input_dim=93, time_dim=16, hidden_dim1=128, hidden_dim2=64, dropout=0.3)
    mlp_165 = DeepParameterMatchedMLP(input_dim=165, time_dim=16, hidden_dim1=128, hidden_dim2=64, dropout=0.3)
    assert sum(p.numel() for p in mlp_93.parameters() if p.requires_grad) == 22466
    assert sum(p.numel() for p in mlp_165.parameters() if p.requires_grad) == 31682


def test_backend_handlers_and_contracts_direct():
    """Verify all FastAPI endpoint functions, inference paths, and schemas directly in Python."""
    # 1. Health & Models & Metrics
    h_data = health()
    assert h_data["status"] == "healthy"
    assert "ETGNN" in h_data["models_loaded"]

    models = {m["id"]: m for m in get_models()}
    assert models["etgnn"]["num_heads"] == 4
    assert models["matched_gat"]["num_heads"] == 4
    assert models["etgnn"]["checkpoint_sha256"] is not None
    assert "residual_etgnn_93f" in models
    assert models["residual_etgnn_93f"]["parameters_count"] == 14472

    m_data = get_metrics()
    assert m_data["test_nodes_count"] == 9973
    assert m_data["test_illicit_prevalence"] == 0.0525
    assert "1..32" in m_data["split"]

    # 2. Transactions pagination, detail, subgraph, and truncation
    page = get_transactions(limit=5, offset=0, split="test", label=1, time_step=None, search=None)
    assert page["total"] > 0
    assert len(page["items"]) == 5
    for item in page["items"]:
        assert item["split"] == "test"
        assert item["actual_label"] == 1

    tx_detail = get_transaction(EXEMPLAR_TP_TX)
    assert tx_detail["transaction_id"] == EXEMPLAR_TP_TX
    assert tx_detail["actual_label"] == 1

    with pytest.raises(HTTPException) as exc_info:
        get_transaction(INVALID_TX)
    assert exc_info.value.status_code == 404

    sub = get_transaction_subgraph(EXEMPLAR_TP_TX, hops=2, max_nodes=10)
    assert sub["is_truncated"] is True
    assert sub["node_count"] <= 10

    # 3. Predictions (cached & live 2-hop feature override) + input validation
    pred_base = predict_transaction(PredictRequest(transaction_id=EXEMPLAR_TP_TX))
    assert pred_base["has_feature_override"] is False
    for m_key in ("etgnn", "matched_gat", "random_forest", "xgboost"):
        assert 0.0 <= pred_base["predictions"][m_key]["illicit_probability"] <= 1.0

    pred_over = predict_transaction(
        PredictRequest(transaction_id=EXEMPLAR_TP_TX, features_override={"feat_0": 3.5, "feat_1": -1.5})
    )
    assert pred_over["has_feature_override"] is True

    with pytest.raises(ValidationError):
        PredictRequest(transaction_id=-1)
    with pytest.raises(ValidationError):
        PredictRequest(transaction_id=EXEMPLAR_TP_TX, features_override={f"feat_{i}": 1.0 for i in range(170)})

    # 4. Explanations & Research summaries (C1, C2, Ablations, Phase 4, Phase H XAI)
    exp_tp = get_explanation(EXEMPLAR_TP_TX)
    assert exp_tp["is_frozen_research_artifact"] is True
    assert exp_tp["case"] == "TP"

    c1_data = get_research_c1()
    assert "INCONCLUSIVE" in c1_data["findings_summary"]
    c2_data = get_research_c2()
    assert c2_data["registers_count"] == 8
    abl_data = get_research_ablations()
    assert "+0.2468" in abl_data["edge_ablation_anomaly"]["phenomenon"]

    p4_data = get_research_phase4()
    assert p4_data["total_trials"] == 40
    assert len(p4_data["conditions"]) == 8
    assert len(p4_data["paired_contrasts"]) == 6
    c_93_rewired = next(
        c for c in p4_data["paired_contrasts"] if c["contrast"] == "93f True Graph vs. Rewired Graph"
    )
    assert round(c_93_rewired["mean_diff"], 4) == 0.0506
    assert round(c_93_rewired["p_unadjusted"], 4) == 0.0245
    assert round(c_93_rewired["p_holm_6"], 4) == 0.1223

    xai_data = get_research_xai_validation()
    assert xai_data["summary"]["cohort_size"] == 30
    assert round(xai_data["summary"]["fidelity_k1_abs_shift"]["exact_oracle_mean"], 4) == 0.0547
    assert round(xai_data["summary"]["stability"]["inter_model_spearman_mean"], 4) == 0.9969

    # Phase 12 Experiment A & B route handlers
    stab_data = get_research_temporal_stability()
    assert "Phase 12 Experiment A" in stab_data["experiment"]
    assert stab_data["census_summary"]["total_nodes"] == 9973
    assert stab_data["census_summary"]["preshift_illicit_zero_edge_pct"] == 13.52
    assert stab_data["census_summary"]["postshift_illicit_zero_edge_pct"] == 76.92
    assert stab_data["stability_summary"]["stratum_2_matched_pairs"] == 59

    mit_data = get_research_error_mitigation()
    assert "Phase 12 Experiment B" in mit_data["experiment"]
    assert mit_data["strategies_count"] == 21
    assert len(mit_data["mitigation_results"]) == 63
    full_strats = [s for s in mit_data["mitigation_results"] if "Full_Test" in s["Period"]]
    assert len(full_strats) == 21
    strat_ids = [s["Strategy_ID"] for s in full_strats]
    assert any("13_C2b" in s for s in strat_ids)
    assert any("19_C3c" in s for s in strat_ids)

    # 5. Path traversal protection on serve_spa
    with pytest.raises(HTTPException) as trav_exc:
        serve_spa("..%2f..%2fsecret.txt")
    assert trav_exc.value.status_code == 403


def test_probability_cache_provenance_and_tamper_recovery():
    loader = get_data_loader()
    prov = loader.verify_probability_cache_provenance()
    assert prov["valid"] is True
    meta = prov["provenance"]
    for required_key in [
        "model_architectures",
        "checkpoint_sha256",
        "dataset_identity",
        "graph_identity",
        "feature_dimensions",
        "preprocessing_version",
        "node_ordering",
        "inference_config",
    ]:
        assert required_key in meta

    # Verify tamper detection: if metadata hash is altered, verify_probability_cache_provenance() returns valid=False
    meta_path = loader.processed_dir / "precomputed_etgnn_gat_probs.meta.json"
    orig_text = meta_path.read_text(encoding="utf-8")
    try:
        tampered = json.loads(orig_text)
        tampered["checkpoint_sha256"]["etgnn"] = "0" * 64
        meta_path.write_text(json.dumps(tampered), encoding="utf-8")
        check_tampered = loader.verify_probability_cache_provenance()
        assert check_tampered["valid"] is False
        assert "mismatch" in check_tampered["reason"].lower()
    finally:
        meta_path.write_text(orig_text, encoding="utf-8")
    assert loader.verify_probability_cache_provenance()["valid"] is True


@pytest.mark.skipif(not LOOPBACK_SOCKETPAIR_AVAILABLE, reason=ENVIRONMENT_BLOCKED_REASON)
def test_fastapi_phase4_and_xai_validation_asgi_transport():
    client = TestClient(app)
    res_p4 = client.get("/api/research/phase4")
    assert res_p4.status_code == 200
    res_xai = client.get("/api/research/xai_validation")
    assert res_xai.status_code == 200
