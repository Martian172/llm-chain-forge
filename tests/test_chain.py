"""Tests for chainforge.core.chain."""
import pytest
from chainforge.core.chain import Chain
from chainforge.core.link import Link
from chainforge.providers.mock_provider import MockProvider


@pytest.fixture
def mock_provider():
    return MockProvider(response_template="Processed: {prompt}")


@pytest.fixture
def simple_chain(mock_provider):
    chain = Chain(name="test-chain")
    chain.add_link(Link(name="step1", prompt_template="Say hi to {{name}}", provider=mock_provider))
    chain.add_link(Link(name="step2", prompt_template="Expand: {{step1.output}}", provider=mock_provider))
    return chain


class TestChain:
    def test_init(self):
        chain = Chain(name="demo")
        assert chain.name == "demo"
        assert len(chain.links) == 0

    def test_add_link(self, mock_provider):
        chain = Chain(name="demo")
        link = Link(name="l1", prompt_template="Hello {{x}}", provider=mock_provider)
        chain.add_link(link)
        assert len(chain.links) == 1

    def test_run(self, simple_chain):
        result = simple_chain.run({"name": "Alice"})
        assert result is not None
        assert result.output is not None
        assert isinstance(result.output, str)

    def test_result_has_metadata(self, simple_chain):
        result = simple_chain.run({"name": "Bob"})
        assert hasattr(result, "token_usage")
        assert hasattr(result, "latency_ms")
        assert hasattr(result, "cost")

    def test_chain_visualization(self, simple_chain):
        # Should not raise
        simple_chain.visualize()

    def test_empty_chain_run(self, mock_provider):
        chain = Chain(name="empty")
        result = chain.run({"x": "test"})
        # Empty chain should return the input as-is or empty
        assert result is not None
