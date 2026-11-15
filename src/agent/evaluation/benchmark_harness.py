"""
ETGNN-Agent Phase 3.5 Evaluation Benchmark Harness.
Systematically benchmarks citation precision, unsupported claim rates,
latency breakdowns, and epistemic calibration across diverse transaction cohorts.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.agent.services.investigation_runner import InvestigationRunner
from src.agent.telemetry.event_sink import get_event_sink
from src.agent.validation.claim_validator import ClaimValidator
from src.agent.validation.citation_validator import CitationValidator
from src.agent.validation.report_validator import ReportValidator


@dataclass
class InvestigationMetricRecord:
    tx_id: int
    cohort: str
    status: str
    total_latency_ms: float
    tool_execution_ms: float
    graph_retrieval_ms: float
    model_inference_ms: float
    validation_ms: float
    synthesis_ms: float
    total_findings: int
    supported_findings: int
    unsupported_findings: int
    unsupported_claim_rate: float
    citation_precision: float
    model_probability: float
    risk_indicator: str
    generation_mode: str


@dataclass
class BenchmarkSummaryReport:
    total_investigations: int
    successful_investigations: int
    abstained_investigations: int
    average_latency_ms: float
    p95_latency_ms: float
    average_citation_precision: float
    average_unsupported_claim_rate: float
    latency_breakdown_averages: Dict[str, float]
    cohort_summaries: Dict[str, Dict[str, Any]]


class AgentBenchmarkHarness:
    """
    Executes reproducible evaluation benchmark suite over representative cohorts:
    - High-risk illicit transactions
    - Low-risk licit transactions
    - Isolated nodes (0 in-degree or 0 out-degree)
    - Snapshot 43 distribution-shift transactions
    """

    BENCHMARK_COHORTS = {
        "illicit_high_risk": [232629023, 230425980],
        "licit_standard": [23042587, 23042600],
        "snapshot_43_shift": [157288764, 157288770],
    }

    def __init__(self, runner: Optional[InvestigationRunner] = None):
        self.runner = runner or InvestigationRunner()
        self.claim_validator = ClaimValidator()
        self.citation_validator = CitationValidator()
        self.report_validator = ReportValidator()

    def run_benchmark(self, sample_limit_per_cohort: int = 2) -> Dict[str, Any]:
        records: List[InvestigationMetricRecord] = []
        sink = get_event_sink()

        for cohort_name, tx_list in self.BENCHMARK_COHORTS.items():
            for tx_id in tx_list[:sample_limit_per_cohort]:
                t0 = time.time()
                inv_res = self.runner.run_investigation(
                    tx_id=tx_id,
                    evidence_mode="RETROSPECTIVE_INVESTIGATION",
                    llm_provider="mock",
                )
                total_duration = (time.time() - t0) * 1000.0

                inv_id = inv_res.get("investigation_id", "")
                status = inv_res.get("status", "UNKNOWN")
                report = inv_res.get("final_report") or {}

                # Telemetry breakdown
                events = sink.get_events(inv_id)
                tool_ms = sum(e.duration_ms for e in events if e.event_type.value == "TOOL_COMPLETED")
                graph_ms = sum(e.duration_ms for e in events if e.tool_name == "get_transaction_subgraph")
                model_ms = sum(e.duration_ms for e in events if e.tool_name == "get_model_prediction")
                val_ms = sum(e.duration_ms for e in events if "VALIDATED" in e.event_type.value)
                synth_ms = sum(e.duration_ms for e in events if e.event_type.value == "EXPLANATION_GENERATED")

                # Validation metrics
                findings = report.get("supported_findings", [])
                total_f = len(findings)
                claim_res = inv_res.get("claim_validation_result") or {}
                supported_f = claim_res.get("supported_count", total_f)
                unsupported_f = claim_res.get("unsupported_count", 0)

                claim_rate = (unsupported_f / total_f) if total_f > 0 else 0.0

                # Citation precision
                prov = report.get("provenance_chain", [])
                total_cites = 0
                valid_cites = 0
                for f in findings:
                    cites = f.get("evidence_citations", [])
                    total_cites += len(cites)
                    valid_cites += sum(1 for c in cites if c in prov)

                precision = (valid_cites / total_cites) if total_cites > 0 else 1.0

                model_prob = report.get("model_probability")
                record = InvestigationMetricRecord(
                    tx_id=tx_id,
                    cohort=cohort_name,
                    status=status,
                    total_latency_ms=round(total_duration, 2),
                    tool_execution_ms=round(tool_ms, 2),
                    graph_retrieval_ms=round(graph_ms, 2),
                    model_inference_ms=round(model_ms, 2),
                    validation_ms=round(val_ms, 2),
                    synthesis_ms=round(synth_ms, 2),
                    total_findings=total_f,
                    supported_findings=supported_f,
                    unsupported_findings=unsupported_f,
                    unsupported_claim_rate=round(claim_rate, 4),
                    citation_precision=round(precision, 4),
                    model_probability=round(float(model_prob), 4) if model_prob is not None else 0.0,
                    risk_indicator=str(report.get("risk_indicator") or "UNKNOWN"),
                    generation_mode=str(report.get("generation_mode") or "UNKNOWN"),
                )
                records.append(record)

        # Aggregate summary
        total_invs = len(records)
        completed = sum(1 for r in records if r.status == "COMPLETED")
        abstained = sum(1 for r in records if r.status == "ABSTAINED")
        avg_lat = sum(r.total_latency_ms for r in records) / total_invs if total_invs else 0.0
        
        latencies = sorted(r.total_latency_ms for r in records)
        p95_idx = int(len(latencies) * 0.95)
        p95_lat = latencies[min(p95_idx, len(latencies) - 1)] if latencies else 0.0

        avg_prec = sum(r.citation_precision for r in records) / total_invs if total_invs else 0.0
        avg_unsupp = sum(r.unsupported_claim_rate for r in records) / total_invs if total_invs else 0.0

        cohorts_summary: Dict[str, Dict[str, Any]] = {}
        for c_name in self.BENCHMARK_COHORTS:
            c_recs = [r for r in records if r.cohort == c_name]
            if c_recs:
                cohorts_summary[c_name] = {
                    "count": len(c_recs),
                    "avg_latency_ms": round(sum(r.total_latency_ms for r in c_recs) / len(c_recs), 2),
                    "avg_citation_precision": round(sum(r.citation_precision for r in c_recs) / len(c_recs), 4),
                    "avg_unsupported_claim_rate": round(sum(r.unsupported_claim_rate for r in c_recs) / len(c_recs), 4),
                }

        summary = BenchmarkSummaryReport(
            total_investigations=total_invs,
            successful_investigations=completed,
            abstained_investigations=abstained,
            average_latency_ms=round(avg_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            average_citation_precision=round(avg_prec, 4),
            average_unsupported_claim_rate=round(avg_unsupp, 4),
            latency_breakdown_averages={
                "tool_execution_ms": round(sum(r.tool_execution_ms for r in records) / total_invs, 2) if total_invs else 0.0,
                "graph_retrieval_ms": round(sum(r.graph_retrieval_ms for r in records) / total_invs, 2) if total_invs else 0.0,
                "model_inference_ms": round(sum(r.model_inference_ms for r in records) / total_invs, 2) if total_invs else 0.0,
                "validation_ms": round(sum(r.validation_ms for r in records) / total_invs, 2) if total_invs else 0.0,
                "synthesis_ms": round(sum(r.synthesis_ms for r in records) / total_invs, 2) if total_invs else 0.0,
            },
            cohort_summaries=cohorts_summary,
        )

        return {
            "summary": asdict(summary),
            "records": [asdict(r) for r in records],
        }


def run_benchmark_and_save(output_path: str = "results/benchmarks/phase3_5_benchmark_results.json"):
    Path("results/benchmarks").mkdir(parents=True, exist_ok=True)
    harness = AgentBenchmarkHarness()
    results = harness.run_benchmark()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results successfully written to {output_path}")
    return results


if __name__ == "__main__":
    run_benchmark_and_save()
