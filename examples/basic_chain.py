"""Basic chain usage example."""
from chainforge.core.chain import Chain
from chainforge.core.link import Link
from chainforge.providers.mock_provider import MockProvider

# Create a mock provider (swap for OpenAIProvider in production)
provider = MockProvider(response_fn=lambda prompt: f"[Processed]: {prompt}")

# Build a simple two-step chain
chain = Chain(name="basic-demo")

chain.add_link(Link(
    name="expand",
    prompt_template="Expand on this topic in 3 sentences: {{input}}",
    provider=provider,
    temperature=0.7,
))

chain.add_link(Link(
    name="summarize",
    prompt_template="Summarize in one sentence: {{expand.output}}",
    provider=provider,
    temperature=0.3,
))

# Run the chain
result = chain.run({"input": "The future of AI in healthcare"})

print("=== Chain Result ===")
print(f"Output: {result.output}")
print(f"Latency: {result.latency_ms:.0f}ms")
print(f"Total tokens: {result.token_usage.total_tokens}")
print(f"Cost: ${result.cost_usd:.5f}")

# Visualize the chain
print("\n=== Chain Visualization ===")
chain.visualize()
