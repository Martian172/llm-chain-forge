"""
Mock provider for testing LLM chains without API calls.

The MockProvider allows you to write tests that verify chain logic
without making real API calls. It supports:
- Static responses
- Dynamic response functions
- Simulated latency
- Simulated token counts
- Controlled error injection
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, AsyncIterator, Callable, Iterator, Optional

from chainforge.providers.base import BaseProvider, ProviderResponse, TokenUsage


class MockProvider(BaseProvider):
    """
    A mock LLM provider for testing and development.

    Instead of calling a real API, MockProvider returns configured
    responses. This is invaluable for:
    - Unit testing chain logic
    - CI/CD pipelines
    - Demos without API keys
    - Cost-free development

    Example:
        >>> mock = MockProvider(response="This is a test response.")
        >>> response = mock.complete("Any prompt here")
        >>> assert response.text == "This is a test response."

    Dynamic responses:
        >>> def my_response(prompt: str) -> str:
        ...     if "hello" in prompt.lower():
        ...         return "Hello! How can I help?"
        ...     return "I don't understand."
        >>> mock = MockProvider(response_fn=my_response)

    Error simulation:
        >>> mock = MockProvider(error_rate=0.1)  # 10% chance of error
    """

    PRICING: dict[str, dict[str, float]] = {
        "mock-model": {"input": 0.0, "output": 0.0},
        "mock-fast": {"input": 0.0, "output": 0.0},
        "mock-smart": {"input": 0.0, "output": 0.0},
    }

    def __init__(
        self,
        response: str = "This is a mock response from LLM Chain Forge.",
        response_fn: Optional[Callable[[str], str]] = None,
        model: str = "mock-model",
        simulated_latency_ms: float = 100.0,
        latency_jitter_ms: float = 50.0,
        simulated_prompt_tokens: Optional[int] = None,
        simulated_completion_tokens: Optional[int] = None,
        error_rate: float = 0.0,
        error_message: str = "Simulated API error",
        stream_chunk_size: int = 5,
        responses: Optional[list[str]] = None,
        cache: Optional[Any] = None,
    ) -> None:
        """
        Initialize the MockProvider.

        Args:
            response: Static response text to return.
            response_fn: Optional function(prompt) -> str for dynamic responses.
            model: Mock model name.
            simulated_latency_ms: Base latency to simulate in milliseconds.
            latency_jitter_ms: Random jitter added to latency.
            simulated_prompt_tokens: Override prompt token count (else auto-estimated).
            simulated_completion_tokens: Override completion token count.
            error_rate: Probability of raising an error (0.0–1.0).
            error_message: Error message when simulated error occurs.
            stream_chunk_size: Number of characters per streamed chunk.
            responses: Optional list of responses to cycle through.
            cache: Optional CacheManager instance.
        """
        super().__init__(
            api_key="mock-key",
            default_model=model,
            cache=cache,
        )
        self._static_response = response
        self._response_fn = response_fn
        self._simulated_latency_ms = simulated_latency_ms
        self._latency_jitter_ms = latency_jitter_ms
        self._simulated_prompt_tokens = simulated_prompt_tokens
        self._simulated_completion_tokens = simulated_completion_tokens
        self._error_rate = error_rate
        self._error_message = error_message
        self._stream_chunk_size = stream_chunk_size
        self._responses = responses
        self._response_index = 0
        self._call_log: list[dict[str, Any]] = []

    def _get_response_text(self, prompt: str) -> str:
        """Determine what text to return for a given prompt."""
        if self._response_fn is not None:
            return self._response_fn(prompt)

        if self._responses is not None:
            text = self._responses[self._response_index % len(self._responses)]
            self._response_index += 1
            return text

        return self._static_response

    def _simulate_latency(self) -> float:
        """Simulate and return the latency in ms."""
        jitter = random.uniform(-self._latency_jitter_ms, self._latency_jitter_ms)
        latency = max(0.0, self._simulated_latency_ms + jitter)
        time.sleep(latency / 1000.0)
        return latency

    async def _simulate_latency_async(self) -> float:
        """Async version of latency simulation."""
        jitter = random.uniform(-self._latency_jitter_ms, self._latency_jitter_ms)
        latency = max(0.0, self._simulated_latency_ms + jitter)
        await asyncio.sleep(latency / 1000.0)
        return latency

    def _maybe_raise_error(self) -> None:
        """Raise a simulated error based on error_rate."""
        if self._error_rate > 0.0 and random.random() < self._error_rate:
            raise RuntimeError(self._error_message)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (~4 chars per token)."""
        return max(1, len(text) // 4)

    def _build_response(
        self, text: str, prompt: str, latency_ms: float, model: str
    ) -> ProviderResponse:
        """Build a ProviderResponse from raw text."""
        prompt_tokens = self._simulated_prompt_tokens or self._estimate_tokens(prompt)
        completion_tokens = self._simulated_completion_tokens or self._estimate_tokens(text)

        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # Log the call
        self._call_log.append({
            "prompt": prompt[:100],
            "response": text[:100],
            "latency_ms": latency_ms,
            "model": model,
        })

        return ProviderResponse(
            text=text,
            token_usage=token_usage,
            model=model,
            latency_ms=latency_ms,
            cost_usd=0.0,  # Mock is free!
            finish_reason="stop",
            raw=None,
        )

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> ProviderResponse:
        """Return a mock completion synchronously."""
        self._maybe_raise_error()
        resolved_model = model or self.default_model
        text = self._get_response_text(prompt)
        latency_ms = self._simulate_latency()
        response = self._build_response(text, prompt, latency_ms, resolved_model)
        self._update_stats(response)
        return response

    async def async_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> ProviderResponse:
        """Return a mock completion asynchronously."""
        self._maybe_raise_error()
        resolved_model = model or self.default_model
        text = self._get_response_text(prompt)
        latency_ms = await self._simulate_latency_async()
        response = self._build_response(text, prompt, latency_ms, resolved_model)
        self._update_stats(response)
        return response

    def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream mock response in chunks."""
        self._maybe_raise_error()
        text = self._get_response_text(prompt)
        chunk_size = self._stream_chunk_size

        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            time.sleep(0.01)  # Small delay between chunks
            yield chunk

    async def async_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream mock response asynchronously in chunks."""
        self._maybe_raise_error()
        text = self._get_response_text(prompt)
        chunk_size = self._stream_chunk_size

        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            await asyncio.sleep(0.01)
            yield chunk

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Get the log of all calls made to this provider."""
        return list(self._call_log)

    @property
    def call_count(self) -> int:
        """Number of times this provider has been called."""
        return len(self._call_log)

    def reset(self) -> None:
        """Reset call log and response index."""
        self._call_log.clear()
        self._response_index = 0
        self.reset_stats()

    def assert_called(self, times: Optional[int] = None) -> None:
        """
        Assert this provider was called (useful in tests).

        Args:
            times: If provided, assert exact number of calls.

        Raises:
            AssertionError: If assertion fails.
        """
        if times is not None:
            assert self.call_count == times, (
                f"Expected {times} calls, got {self.call_count}"
            )
        else:
            assert self.call_count > 0, "Expected at least one call, got zero"

    def assert_prompt_contains(self, text: str) -> None:
        """
        Assert that at least one call's prompt contained the given text.

        Args:
            text: Text to search for in call prompts.
        """
        prompts = [c["prompt"] for c in self._call_log]
        assert any(text in p for p in prompts), (
            f"Expected prompt containing {text!r}, but calls were: {prompts}"
        )
