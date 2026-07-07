# Evaluation & A/B Testing

## Test cases

Test cases live in JSONL files — one JSON object per line:

```json
{"id": "tc-1", "input": {"input": "text to process"}, "expected_output": "expected result", "metadata": {"difficulty": "easy"}}
```

See [examples/test_cases.jsonl](../examples/test_cases.jsonl).

## Evaluating a chain

```python
from chainforge.evaluation.evaluator import Evaluator

test_cases = Evaluator.from_jsonl("examples/test_cases.jsonl")
evaluator = Evaluator()
report = evaluator.evaluate(chain, test_cases)
print(report.summary())
```

Or from the CLI:

```bash
forge eval examples/chains/summarization.yaml examples/test_cases.jsonl
```

Metrics reported:

| Metric | Meaning |
|---|---|
| `exact_match` | Fraction of cases where output == expected (case-insensitive) |
| `avg_completion_tokens` | Average tokens generated per case |
| `token_efficiency` | Accuracy per token — rewards being right *and* concise |

## A/B testing two chains

Is prompt B actually better than prompt A, or just lucky? `ABTest` answers
with statistics instead of vibes:

```python
from chainforge.evaluation.ab_test import ABTest

ab = ABTest(significance_level=0.05)
report = ab.compare(chain_a, chain_b, test_cases)
print(report.summary())
```

```bash
forge compare chain_a.yaml chain_b.yaml examples/test_cases.jsonl
```

- **Mann-Whitney U test** — non-parametric significance test (pass/fail
  scores are not normally distributed, so a t-test would be invalid)
- **p-value < 0.05** → the difference is statistically real
- **Cohen's d** — effect size: how *big* the difference is
  (0.2 small, 0.5 medium, 0.8 large)

The report includes accuracy and cost for both chains, so you can trade
quality against spend with evidence.
