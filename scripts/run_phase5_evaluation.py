"""
CLI Runner for ETGNN-Agent Phase 5 Scientific Evaluation Suite.
Executes Baselines A, B, C, and Proposed System D and writes reproducible metrics to
results/benchmarks/phase5_evaluation_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.evaluation.phase5_evaluation import run_phase5_evaluation_and_save


def main() -> None:
    out_file = "results/benchmarks/phase5_evaluation_results.json"
    results = run_phase5_evaluation_and_save(out_file)
    print(f"Phase 5 evaluation completed. Saved to {out_file}")
    for sys_key, metrics in results["systems_comparison"].items():
        print(
            f"  [{sys_key}] Factual={metrics['factual_correctness']:.4f} | "
            f"CitValid={metrics['citation_validity']:.4f} | "
            f"UnsuppRate={metrics['unsupported_claim_rate']:.4f} | "
            f"Latency={metrics['average_latency_ms']:.1f}ms"
        )


if __name__ == "__main__":
    main()

