# Quick Start

Get a chain running in under 5 minutes.

## Install

```bash
git clone https://github.com/Martian172/llm-chain-forge.git
cd llm-chain-forge
pip install -r requirements.txt
pip install -e .
```

## Your first chain (no API key needed)

The `MockProvider` simulates an LLM locally — free, offline, deterministic.

```python
from chainforge.core.chain import Chain
from chainforge.core.link import Link
from chainforge.providers.mock_provider import MockProvider

provider = MockProvider()

chain = Chain(name="hello-chain")
chain.add_link(Link(
    name="expand",
    prompt_template="Expand on this topic in 3 sentences: {{input}}",
    provider=provider,
))
chain.add_link(Link(
    name="summarize",
    prompt_template="Summarize in one sentence: {{expand.output}}",
    provider=provider,
))

result = chain.run({"input": "The future of AI in healthcare"})
print(result.output)
print(f"{result.token_usage.total_tokens} tokens, "
      f"${result.cost_usd:.5f}, {result.latency_ms:.0f}ms")
```

Each link's output is stored in the shared context under `<link_name>.output`,
so later templates can reference `{{expand.output}}`.

## Switch to a real LLM

```python
from chainforge.providers.openai_provider import OpenAIProvider
provider = OpenAIProvider(model="gpt-4o-mini")  # reads OPENAI_API_KEY env var
```

Everything else stays identical. See [providers.md](providers.md).

## Run a chain from YAML

```bash
forge run examples/chains/summarization.yaml --input '{"input": "Your text"}'
```

## Next steps

- [Provider configuration](providers.md)
- [Evaluation & A/B testing](evaluation.md)
- [Web playground](playground.md)
