# Provider Configuration

Providers abstract the LLM API behind a common interface (`BaseProvider`),
so chains never depend on a specific vendor.

## MockProvider (default — testing)

No network, no API key, no cost. Simulates latency, token counts, and
even failures for testing retry logic:

```python
from chainforge.providers.mock_provider import MockProvider

provider = MockProvider(
    response="Fixed response text",              # or:
    response_fn=lambda prompt: f"Echo: {prompt}",  # dynamic responses
    simulated_latency_ms=100,
    error_rate=0.0,          # raise errors randomly to test retries
)
```

## OpenAIProvider

```bash
export OPENAI_API_KEY=sk-...
```

```python
from chainforge.providers.openai_provider import OpenAIProvider

provider = OpenAIProvider(model="gpt-4o-mini")
```

- Supports `gpt-3.5-turbo`, `gpt-4`, `gpt-4o`, `gpt-4o-mini`
- Automatic retries with exponential backoff + jitter on 429/503
- Cost calculated per call from a built-in pricing table

## AnthropicProvider

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from chainforge.providers.anthropic_provider import AnthropicProvider

provider = AnthropicProvider(model="claude-3-haiku-20240307")
```

## Per-link providers

Each link can use a different provider — e.g. a cheap model for extraction
and a strong model for the final answer:

```python
chain.add_link(Link(name="extract", prompt_template="...", provider=cheap))
chain.add_link(Link(name="answer",  prompt_template="...", provider=strong))
```

## Writing your own provider

Subclass `BaseProvider` and implement `complete()` / `async_complete()`
returning a `ProviderResponse`. Nothing else in the framework changes.
