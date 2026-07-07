"""Evaluation framework for LLM Chain Forge."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from chainforge.providers.mock_provider import MockProvider


@dataclass
class TestCase:
    """A single evaluation test case."""
    input: dict[str, Any]
    expected_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class CaseResult:
    """Result for a single test case."""
    test_case: TestCase
    actual_output: str
    exact_match: bool
    latency_ms: float
    token_usage: dict[str, int]
    cost_usd: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Aggregate evaluation report."""
    chain_name: str
    total_cases: int
    results: list[CaseResult]
    metrics: dict[str, float]
    total_cost_usd: float
    total_latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        lines = [
            f"=== Evaluation Report: {self.chain_name} ===",
            f"  Cases: {self.total_cases}",
            f"  Exact Match: {self.metrics.get('exact_match', 0):.1%}",
            f"  Total Cost: ${self.total_cost_usd:.4f}",
            f"  Avg Latency: {self.total_latency_ms / max(self.total_cases, 1):.0f}ms",
        ]
        return "\n".join(lines)


class Evaluator:
    """Evaluate a chain against a set of test cases."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or ["exact_match", "token_efficiency"]

    def evaluate(self, chain, test_cases: list[TestCase]) -> EvaluationReport:
        """Run evaluation on all test cases."""
        results: list[CaseResult] = []
        total_cost = 0.0
        total_latency = 0.0

        for tc in test_cases:
            start = time.time()
            try:
                chain_result = chain.run(tc.input)
                actual = chain_result.output if hasattr(chain_result, "output") else str(chain_result)
                token_usage = getattr(chain_result, "token_usage", {"prompt": 0, "completion": 0})
                cost = getattr(chain_result, "cost", 0.0)
            except Exception as e:
                actual = f"ERROR: {e}"
                token_usage = {"prompt": 0, "completion": 0}
                cost = 0.0

            latency = (time.time() - start) * 1000
            exact = actual.strip().lower() == tc.expected_output.strip().lower()

            result = CaseResult(
                test_case=tc,
                actual_output=actual,
                exact_match=exact,
                latency_ms=latency,
                token_usage=token_usage,
                cost_usd=cost,
            )
            results.append(result)
            total_cost += cost
            total_latency += latency

        exact_rate = sum(r.exact_match for r in results) / max(len(results), 1)
        avg_tokens = sum(r.token_usage.get("completion", 0) for r in results) / max(len(results), 1)

        metrics = {
            "exact_match": exact_rate,
            "avg_completion_tokens": avg_tokens,
            "token_efficiency": exact_rate / max(avg_tokens, 1) * 100,
        }

        return EvaluationReport(
            chain_name=getattr(chain, "name", "unnamed"),
            total_cases=len(test_cases),
            results=results,
            metrics=metrics,
            total_cost_usd=total_cost,
            total_latency_ms=total_latency,
        )

    @staticmethod
    def from_jsonl(path: str) -> list[TestCase]:
        """Load test cases from a JSONL file."""
        import json
        cases = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    cases.append(TestCase(
                        input=d.get("input", {}),
                        expected_output=d.get("expected_output", ""),
                        metadata=d.get("metadata", {}),
                        id=d.get("id", ""),
                    ))
        return cases
