import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from pathlib import Path
from src.agent.evaluation.adversarial_evaluation_suite import run_phase6_adversarial_audit

def main():
    results = run_phase6_adversarial_audit()
    out_path = Path("results/benchmarks/phase6_adversarial_evaluation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Adversarial Suite Execution Complete:")
    print(f"Total Scenarios: {results['total_adversarial_scenarios']}")
    print(f"Passed: {results['defended_successfully']}")
    print(f"Failed: {results['failed_scenarios_count']}")
    print(f"Pass Rate: {results['defense_success_rate'] * 100:.2f}%")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
