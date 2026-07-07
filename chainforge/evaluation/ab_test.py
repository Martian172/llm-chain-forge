"""A/B Testing for LLM chains."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from scipy import stats as scipy_stats

from chainforge.evaluation.evaluator import Evaluator, TestCase, EvaluationReport


@dataclass
class ABTestReport:
    """Report from an A/B test between two chains."""
    chain_a_name: str
    chain_b_name: str
    report_a: EvaluationReport
    report_b: EvaluationReport
    winner: str
    p_value: float
    effect_size: float
    cost_savings_usd: float
    recommendation: str
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"=== A/B Test: {self.chain_a_name} vs {self.chain_b_name} ===\n"
            f"  Winner: {self.winner}\n"
            f"  P-value: {self.p_value:.4f} {'(significant)' if self.p_value < 0.05 else '(not significant)'}\n"
            f"  Effect size: {self.effect_size:.4f}\n"
            f"  Cost savings: ${self.cost_savings_usd:.4f}\n"
            f"  Recommendation: {self.recommendation}"
        )


class ABTest:
    """Compare two LLM chains on the same test cases."""

    def __init__(self, evaluator: Evaluator | None = None) -> None:
        self.evaluator = evaluator or Evaluator()

    def compare(
        self,
        chain_a,
        chain_b,
        test_cases: list[TestCase],
    ) -> ABTestReport:
        """Run both chains on test cases and compare results."""
        report_a = self.evaluator.evaluate(chain_a, test_cases)
        report_b = self.evaluator.evaluate(chain_b, test_cases)

        # Statistical test on exact match scores
        scores_a = [1 if r.exact_match else 0 for r in report_a.results]
        scores_b = [1 if r.exact_match else 0 for r in report_b.results]

        if len(scores_a) >= 2 and len(scores_b) >= 2:
            _, p_value = scipy_stats.mannwhitneyu(scores_a, scores_b, alternative="two-sided")
        else:
            p_value = 1.0

        mean_a = sum(scores_a) / max(len(scores_a), 1)
        mean_b = sum(scores_b) / max(len(scores_b), 1)
        effect_size = abs(mean_a - mean_b)

        if mean_a > mean_b:
            winner = getattr(chain_a, "name", "Chain A")
        elif mean_b > mean_a:
            winner = getattr(chain_b, "name", "Chain B")
        else:
            winner = "Tie"

        cost_savings = report_a.total_cost_usd - report_b.total_cost_usd

        rec = (
            f"Use {winner}" if p_value < 0.05
            else "No statistically significant difference. Consider cost as the deciding factor."
        )

        return ABTestReport(
            chain_a_name=getattr(chain_a, "name", "Chain A"),
            chain_b_name=getattr(chain_b, "name", "Chain B"),
            report_a=report_a,
            report_b=report_b,
            winner=winner,
            p_value=p_value,
            effect_size=effect_size,
            cost_savings_usd=cost_savings,
            recommendation=rec,
        )
