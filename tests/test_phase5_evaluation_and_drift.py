"""
Phase 5 Comprehensive Verification Test Suite.
Tests all 20 critical areas specified in Phase 5 prompt Section 14:
 1. Agent evaluation dataset validation
 2. Citation metric correctness
 3. Unsupported claim calculations
 4. Retrieval metric calculations
 5. Temporal leakage rejection
 6. Unknown-label handling
 7. Snapshot 43 caveat
 8. Feature drift metrics
 9. Prediction drift metrics
 10. Graph drift metrics
 11. Label drift calculations
 12. Insufficient sample handling
 13. Candidate artifact isolation
 14. Model promotion rules
 15. Feedback provenance
 16. Provider failure handling
 17. Evaluation reproducibility
 18. Report serialization
 19. Dashboard API contracts
 20. Baseline manifest integrity
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
# (TestClient removed to prevent Windows Python 3.13 sandbox loopback socketpair blocking)

from src.agent.evaluation.adversarial_cases import ADVERSARIAL_CASES, get_adversarial_suite
from src.agent.evaluation.datasets import CohortType, PHASE5_EVALUATION_DATASET, get_evaluation_dataset
from src.agent.evaluation.grounding_metrics import GroundingMetricsEvaluator
from src.agent.evaluation.latency_metrics import LatencyMetricsEvaluator
from src.agent.evaluation.retrieval_metrics import RetrievalMetricsEvaluator
from src.agent.evaluation.semantic_citation import (
    SEMANTIC_CITATION_BENCHMARK,
    SemanticCitationEvaluator,
    SupportClassification,
)
from src.agent.evaluation.temporal_metrics import TemporalMetricsEvaluator
from src.agent.evaluation.tool_metrics import ToolMetricsEvaluator
from src.agent.feedback.evaluator import AnalystFeedbackEvaluator
from src.ml.continual.candidate_trainer import SafeCandidateTrainer
from src.ml.continual.promotion_gate import (
    ModelMetrics,
    ModelPromotionGate,
    PromotionDecision,
)
from src.ml.drift.feature_drift import (
    FeatureDriftDetector,
    calculate_js_divergence,
    calculate_ks,
    calculate_psi,
    calculate_wasserstein,
)
from src.ml.drift.graph_drift import GraphTopologyDriftDetector
from src.ml.drift.label_drift import LabelDriftDetector
from src.ml.drift.prediction_drift import PredictionDriftDetector


# 1. Agent evaluation dataset validation
def test_01_evaluation_dataset_structure():
    dataset = get_evaluation_dataset()
    assert len(dataset) >= 10
    cohorts = {item.cohort for item in dataset}
    assert CohortType.ILLICIT_CONNECTED in cohorts
    assert CohortType.SNAPSHOT_43_SHIFT in cohorts
    assert CohortType.TEMPORAL_LEAKAGE_PROBE in cohorts
    assert CohortType.UNKNOWN_LABEL_ABSTENTION in cohorts

    # Check that each fixture has non-empty fields
    for item in dataset:
        assert item.task_id.startswith("EVAL_")
        assert len(item.question) > 10


# 2. Citation metric correctness
def test_02_citation_metric_precision():
    evaluator = GroundingMetricsEvaluator()
    registered = {"EV-TX-100", "EV-GRAPH-100", "EV-RAG-001"}
    text = "Based on [EV-TX-100] and [EV-GRAPH-100], risk is elevated. Also [EV-FAKE-999]."
    res = evaluator.evaluate_report_grounding(
        report_text=text,
        target_tx_id=100,
        registered_evidence_ids=registered,
        expected_numerical_facts={},
    )
    assert res["has_citations"] is True
    assert res["total_citations"] == 3
    assert res["valid_citations"] == 2
    assert abs(res["citation_precision"] - (2 / 3)) < 1e-4


# 3. Unsupported claim calculations
def test_03_unsupported_claim_detection():
    evaluator = GroundingMetricsEvaluator()
    registered = {"EV-TX-200"}
    # Report contains a hallucinated model architecture not in codebase
    text = "Analysis via LSTM neural network indicates illicit pattern citing [EV-TX-200]."
    res = evaluator.evaluate_report_grounding(
        report_text=text,
        target_tx_id=200,
        registered_evidence_ids=registered,
        expected_numerical_facts={},
    )
    assert res["model_identity_correct"] is False
    assert res["unsupported_claim_rate"] > 0.0


# 4. Retrieval metric calculations
def test_04_retrieval_metrics_mrr_and_recall():
    evaluator = RetrievalMetricsEvaluator(k=5)
    retrieved = ["doc_c", "doc_a", "doc_b", "doc_d", "doc_e"]
    relevant = {"doc_a", "doc_f"}
    res = evaluator.evaluate_retrieval_query(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        retrieved_sources={"graph", "literature"},
        expected_sources={"graph", "literature"},
        latency_ms=12.5,
    )
    # doc_a is at rank 2, so MRR = 0.5
    assert res["mrr"] == 0.5
    # Hits in top 5 is 1 (doc_a), relevant count is 2 -> recall = 0.5
    assert res["recall_at_k"] == 0.5
    assert res["precision_at_k"] == 0.2
    assert res["source_coverage"] == 1.0


# 5. Temporal leakage rejection
def test_05_temporal_leakage_rejection():
    evaluator = TemporalMetricsEvaluator()
    # Task has snapshot 43, but query cutoff is 32 (leakage probe)
    res = evaluator.evaluate_task_temporal_integrity(
        task_snapshot=43,
        as_of_snapshot=32,
        retrieved_snapshots=[43],
        labels_exposed=[1],
        leakage_probe_blocked=True,
        is_leakage_probe=True,
    )
    assert res["boundary_preserved"] is False  # attempted to retrieve snapshot 43
    assert res["rejection_preserved"] is True  # agent correctly blocked the probe


# 6. Unknown-label handling
def test_06_unknown_labels_preserved():
    df = pd.DataFrame({
        "txId": [1, 2, 3, 4],
        "time_step": [1, 1, 1, 1],
        "label": [-1, 0, 1, -1],
    })
    detector = LabelDriftDetector()
    rep = detector.analyze_label_and_performance(df, baseline_snapshots=(1, 1), target_snapshots=(1, 1))
    assert rep.overall_unlabelled_count == 2
    assert rep.overall_unlabelled_pct == 0.5
    assert "strictly excluded from supervised evaluation" in rep.unknown_label_safeguard


# 7. Snapshot 43 caveat preservation
def test_07_snapshot_43_caveat():
    detector = GraphTopologyDriftDetector()
    assert "Darknet market takedown hypotheses remain non-causal observational context" in detector.SNAPSHOT_43_CAVEAT

    evaluator = SemanticCitationEvaluator()
    claim_bad = "Evidence EV-RAG-SNAP43 proves that AlphaBay seizure caused the ETGNN AUPRC drop."
    ev = "Snapshot 43 documents graph structural divergence. Darknet market closures represent observational correlation rather than proven causal mechanisms."
    pred = evaluator.deterministic_judge(claim_bad, ev)
    assert pred == SupportClassification.CONTRADICTED


# 8. Feature drift metrics (PSI, KS, Wasserstein, JS)
def test_08_feature_drift_metrics():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 500)
    tgt_same = np.random.normal(0, 1, 500)
    tgt_shifted = np.random.normal(2, 1, 500)

    psi_same = calculate_psi(ref, tgt_same)
    psi_shift = calculate_psi(ref, tgt_shifted)
    assert psi_same < 0.10  # low drift
    assert psi_shift > 0.25  # high drift

    ks_stat, ks_pval = calculate_ks(ref, tgt_shifted)
    assert ks_pval < 0.001

    wass = calculate_wasserstein(ref, tgt_shifted)
    assert wass > 1.0

    js = calculate_js_divergence(ref, tgt_shifted)
    assert 0.0 <= js <= 1.0


# 9. Prediction drift metrics
def test_09_prediction_drift():
    detector = PredictionDriftDetector(tuned_threshold=0.42)
    snaps = np.array([1, 1, 41, 41])
    probs_etgnn = np.array([0.20, 0.30, 0.60, 0.70])
    probs_rf = np.array([0.25, 0.35, 0.40, 0.80])  # disagrees on 3rd item

    rep = detector.analyze_snapshot_predictions(
        snapshot_ids=snaps,
        primary_probs=probs_etgnn,
        secondary_probs=probs_rf,
        baseline_snapshots=(1, 1),
        target_snapshots=(41, 41),
    )
    assert rep.probability_shift_delta > 0.30
    assert rep.threshold_crossing_shift > 0.50
    assert "must not be conflated with true concept drift" in rep.scientific_distinction


# 10. Graph drift metrics
def test_10_graph_drift_calculation():
    detector = GraphTopologyDriftDetector()
    nodes = pd.DataFrame({
        "txId": [101, 102, 103],
        "time_step": [1, 1, 43],
    })
    edges = pd.DataFrame({
        "txId1": [101],
        "txId2": [102],
    })
    rep = detector.analyze_graph_drift(nodes, edges, baseline_snapshots=(1, 1), target_snapshots=(43, 43))
    assert rep.total_snapshots == 2
    # Node 103 in snapshot 43 has zero incoming edges
    s43 = [s for s in rep.snapshot_topologies if s.snapshot_id == 43][0]
    assert s43.zero_in_degree_proportion == 1.0
    assert bool(s43.is_snapshot_43) is True


# 11. Label drift calculations
def test_11_label_drift_prevalence():
    detector = LabelDriftDetector()
    nodes = pd.DataFrame({
        "txId": [1, 2, 3, 4, 5, 6],
        "time_step": [1, 1, 1, 41, 41, 41],
        "label": [0, 0, 1, 0, 1, 1],  # base: 1/3 illicit, tgt: 2/3 illicit
    })
    rep = detector.analyze_label_and_performance(nodes, baseline_snapshots=(1, 1), target_snapshots=(41, 41))
    assert rep.baseline_illicit_prevalence == round(1 / 3, 4)
    assert rep.target_illicit_prevalence == round(2 / 3, 4)
    assert rep.prevalence_shift_delta > 0.30


# 12. Insufficient sample handling
def test_12_insufficient_sample_auprc():
    detector = LabelDriftDetector()
    # Snapshot has only 1 illicit sample, insufficient for stable AUPRC
    nodes = pd.DataFrame({
        "txId": [1, 2],
        "time_step": [5, 5],
        "label": [0, 1],
    })
    rep = detector.analyze_label_and_performance(nodes, baseline_snapshots=(5, 5), target_snapshots=(5, 5))
    s5 = rep.snapshots_analyzed[0]
    assert s5.insufficient_samples is True
    assert s5.auprc is None


# 13. Candidate artifact isolation
def test_13_candidate_isolation_and_safety():
    trainer = SafeCandidateTrainer()
    # Attempting to train on test snapshot 45 must trigger BLOCKED status
    manifest = trainer.plan_candidate_experiment(
        candidate_id="test_cand_blocked",
        base_model_name="ETGNN",
        training_snapshots=[1, 2, 45],
        evaluation_snapshots=[46],
    )
    assert manifest.status == "BLOCKED"
    assert any("Snapshots 41-49 cannot be used as training data" in r for r in manifest.blocking_reasons)
    # Check that candidate manifest is written inside artifacts/phase5_candidates/
    assert Path(manifest.artifact_directory).exists()
    assert "results/checkpoints" not in manifest.artifact_directory


# 14. Model promotion rules
def test_14_model_promotion_rules():
    gate = ModelPromotionGate()
    base_m = ModelMetrics(
        auprc=0.2250, precision=0.26, recall=0.45, f1=0.33, roc_auc=0.80,
        false_positive_rate=0.04, brier_score=0.08, inference_latency_ms=18.0,
        model_size_mb=4.0, seed_auprc_std=0.005,
    )
    # Candidate with good AUPRC (+0.030) and low FPR increase
    cand_good = ModelMetrics(
        auprc=0.2550, precision=0.29, recall=0.48, f1=0.36, roc_auc=0.82,
        false_positive_rate=0.045, brier_score=0.078, inference_latency_ms=19.0,
        model_size_mb=4.0, seed_auprc_std=0.006,
    )
    res_eligible = gate.evaluate_candidate("cand_good", "base", base_m, cand_good, sample_size=500, seeds_evaluated=3)
    assert res_eligible.decision == PromotionDecision.ELIGIBLE_FOR_REVIEW
    assert res_eligible.requires_human_approval is True

    # Candidate with leakage risk
    res_leakage = gate.evaluate_candidate("cand_leak", "base", base_m, cand_good, provenance_has_leakage_risk=True)
    assert res_leakage.decision == PromotionDecision.BLOCKED_BY_LEAKAGE_RISK

    # Candidate with insufficient improvement
    cand_weak = ModelMetrics(
        auprc=0.2260, precision=0.26, recall=0.45, f1=0.33, roc_auc=0.80,
        false_positive_rate=0.04, brier_score=0.08, inference_latency_ms=18.0,
        model_size_mb=4.0, seed_auprc_std=0.005,
    )
    res_rejected = gate.evaluate_candidate("cand_weak", "base", base_m, cand_weak, sample_size=500, seeds_evaluated=3)
    assert res_rejected.decision == PromotionDecision.REJECTED


# 15. Feedback provenance and separation
def test_15_feedback_evaluator_governance():
    evaluator = AnalystFeedbackEvaluator(db_path="data/analyst_feedback.db")
    report = evaluator.analyze_feedback()
    assert report.total_feedback_records >= 6
    assert report.provenance_classification == "ANALYST_OPINION_ONLY"
    assert "MUST NEVER be treated as verified ground-truth fraud labels" in report.scientific_safeguard


# 16. Provider failure handling
def test_16_provider_failure_handling_metric():
    evaluator = ToolMetricsEvaluator()
    task_res = evaluator.evaluate_task_tool_execution(
        executed_tools=["get_transaction_evidence"],
        tool_call_successes=[True],
        expected_tools=["get_transaction_evidence"],
        tool_call_count=1,
        workflow_completed=True,
        abstained=False,
        expected_abstention=False,
        provider_failure_simulated=True,
        provider_failure_handled_properly=True,
    )
    agg = evaluator.aggregate_tool_metrics([task_res])
    assert agg.provider_failure_handling_rate == 1.0


# 17. Evaluation reproducibility (Semantic citation benchmark)
def test_17_semantic_citation_benchmark_reproducibility():
    evaluator = SemanticCitationEvaluator()
    rep1 = evaluator.run_benchmark()
    rep2 = evaluator.run_benchmark()
    assert rep1["accuracy_against_ground_truth"] == rep2["accuracy_against_ground_truth"]
    assert rep1["inter_rater_agreement"] == rep2["inter_rater_agreement"]
    assert rep1["total_items"] == len(SEMANTIC_CITATION_BENCHMARK)


# 18. Report serialization
def test_18_report_serialization():
    drift_file = Path("results/drift/temporal_drift_report.json")
    assert drift_file.exists()
    with open(drift_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "feature_drift" in data
    assert "prediction_drift" in data
    assert "graph_drift" in data
    assert "label_drift" in data

    ablation_file = Path("results/benchmarks/phase5_agent_ablation_results.json")
    assert ablation_file.exists()
    with open(ablation_file, "r", encoding="utf-8") as f:
        ab_data = json.load(f)
    assert len(ab_data["systems"]) == 6


# 19. Dashboard API contracts
def test_19_dashboard_api_contracts():
    from src.web.app import (
        api_get_monitoring_candidates,
        api_get_monitoring_drift,
        api_get_monitoring_evaluation,
        api_get_monitoring_feedback_stats,
    )
    r1 = api_get_monitoring_drift()
    assert "graph_drift" in r1

    r2 = api_get_monitoring_evaluation()
    assert "systems" in r2

    r3 = api_get_monitoring_candidates()
    assert "promotion_gate_rules" in r3

    r4 = api_get_monitoring_feedback_stats()
    assert "scientific_safeguard" in r4


# 20. Baseline manifest integrity
def test_20_baseline_manifest_integrity():
    manifest_path = Path("artifacts/baseline_manifest.json")
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "scientific_inventory" in manifest
    checkpoints = manifest["scientific_inventory"]["checkpoints"]
    assert "etgnn.pt" in checkpoints
    assert checkpoints["etgnn.pt"]["sha256"] == "b2f5535f3b6b1675109dae205934e761d2ad1b80b5409564bbe5889edb79dadc"
    assert "test_set_exposure" in manifest["scientific_disclosures"]
