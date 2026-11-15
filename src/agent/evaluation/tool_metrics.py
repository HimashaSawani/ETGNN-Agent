"""
Agent Tool and Workflow Execution Metrics.
Evaluates tool selection accuracy, tool-call success, budget compliance,
workflow completion, abstention correctness, failure handling, and retry limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class ToolMetricResult:
    tool_selection_accuracy: float
    tool_call_success_rate: float
    tool_budget_compliance_rate: float
    workflow_completion_rate: float
    correct_abstention_rate: float
    provider_failure_handling_rate: float
    invalid_request_rejection_rate: float
    retry_limit_compliance_rate: float
    sample_sizes: Dict[str, int]
    denominators: Dict[str, str]


class ToolMetricsEvaluator:
    """Evaluates agent tool execution and orchestration workflow integrity."""

    MAX_TOOL_BUDGET = 8
    MAX_RETRIES = 2

    def evaluate_task_tool_execution(
        self,
        executed_tools: List[str],
        tool_call_successes: List[bool],
        expected_tools: List[str],
        tool_call_count: int,
        workflow_completed: bool,
        abstained: bool,
        expected_abstention: bool,
        is_invalid_request: bool = False,
        provider_failure_simulated: bool = False,
        provider_failure_handled_properly: bool = True,
        retries_attempted: int = 0,
    ) -> Dict[str, Any]:
        """Evaluates tool execution dynamics for a single task."""
        # Tool selection accuracy: precision and recall against expected tools
        if expected_tools:
            exec_set = set(executed_tools)
            exp_set = set(expected_tools)
            intersection = exec_set.intersection(exp_set)
            selection_acc = len(intersection) / max(1, len(exp_set))
        else:
            # If no tools were expected (e.g. immediate rejection), 1.0 if executed_tools is empty
            selection_acc = 1.0 if len(executed_tools) == 0 else 0.0

        # Success rate
        total_calls = len(tool_call_successes)
        success_rate = (sum(1 for s in tool_call_successes if s) / max(1, total_calls)) if total_calls > 0 else 1.0

        # Budget compliance
        budget_compliant = tool_call_count <= self.MAX_TOOL_BUDGET

        # Abstention correctness
        if expected_abstention:
            abstention_correct = abstained
        else:
            abstention_correct = not abstained

        # Invalid request rejection
        if is_invalid_request:
            rejection_correct = abstained or ("invalid" in "".join(executed_tools).lower()) or tool_call_count == 0
        else:
            rejection_correct = True

        # Retry limit compliance
        retry_compliant = retries_attempted <= self.MAX_RETRIES

        return {
            "selection_accuracy": selection_acc,
            "success_rate": success_rate,
            "budget_compliant": budget_compliant,
            "workflow_completed": workflow_completed,
            "abstention_correct": abstention_correct,
            "expected_abstention": expected_abstention,
            "rejection_correct": rejection_correct,
            "is_invalid_request": is_invalid_request,
            "provider_failure_simulated": provider_failure_simulated,
            "provider_failure_handled": provider_failure_handled_properly,
            "retry_compliant": retry_compliant,
            "tool_call_count": tool_call_count,
        }

    def aggregate_tool_metrics(self, task_results: List[Dict[str, Any]]) -> ToolMetricResult:
        """Aggregates tool metrics across all benchmark evaluation tasks."""
        n = len(task_results)
        if n == 0:
            return ToolMetricResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {})

        mean_sel = sum(r["selection_accuracy"] for r in task_results) / n
        mean_succ = sum(r["success_rate"] for r in task_results) / n
        mean_bud = sum(1.0 if r["budget_compliant"] else 0.0 for r in task_results) / n
        mean_comp = sum(1.0 if r["workflow_completed"] else 0.0 for r in task_results) / n

        # Abstention subset
        abstention_tasks = [r for r in task_results if r["expected_abstention"]]
        if abstention_tasks:
            mean_abst = sum(1.0 if r["abstention_correct"] else 0.0 for r in abstention_tasks) / len(abstention_tasks)
        else:
            mean_abst = 1.0

        # Invalid request subset
        invalid_tasks = [r for r in task_results if r["is_invalid_request"]]
        if invalid_tasks:
            mean_rej = sum(1.0 if r["rejection_correct"] else 0.0 for r in invalid_tasks) / len(invalid_tasks)
        else:
            mean_rej = 1.0

        # Provider failure subset
        failure_tasks = [r for r in task_results if r["provider_failure_simulated"]]
        if failure_tasks:
            mean_fail_hand = sum(1.0 if r["provider_failure_handled"] else 0.0 for r in failure_tasks) / len(failure_tasks)
        else:
            mean_fail_hand = 1.0

        mean_retry = sum(1.0 if r["retry_compliant"] else 0.0 for r in task_results) / n

        sample_sizes = {
            "total_tasks_evaluated": n,
            "abstention_tasks_evaluated": len(abstention_tasks),
            "invalid_request_tasks_evaluated": len(invalid_tasks),
            "failure_scenarios_evaluated": len(failure_tasks),
            "total_tool_calls_observed": sum(r["tool_call_count"] for r in task_results),
        }

        denominators = {
            "tool_selection_accuracy": "Total evaluation tasks (n)",
            "tool_call_success_rate": "Total tool invocations across all tasks",
            "tool_budget_compliance_rate": "Total evaluation tasks (n)",
            "workflow_completion_rate": "Total evaluation tasks (n)",
            "correct_abstention_rate": "Total tasks requiring intentional abstention",
            "provider_failure_handling_rate": "Total provider failure scenarios simulated",
            "invalid_request_rejection_rate": "Total invalid / out-of-range queries presented",
            "retry_limit_compliance_rate": "Total workflow runs with retries",
        }

        return ToolMetricResult(
            tool_selection_accuracy=round(mean_sel, 4),
            tool_call_success_rate=round(mean_succ, 4),
            tool_budget_compliance_rate=round(mean_bud, 4),
            workflow_completion_rate=round(mean_comp, 4),
            correct_abstention_rate=round(mean_abst, 4),
            provider_failure_handling_rate=round(mean_fail_hand, 4),
            invalid_request_rejection_rate=round(mean_rej, 4),
            retry_limit_compliance_rate=round(mean_retry, 4),
            sample_sizes=sample_sizes,
            denominators=denominators,
        )
