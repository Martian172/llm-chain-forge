"""
Abstract base class for all LLM providers in Chain Forge.

All providers must implement complete() and async_complete().
The base class provides cost calculation, response wrapping, and
a consistent interface regardless of the underlying API.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Optional


@dataclass
class TokenUsage:
    """Token usage counts for a single provider call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ProviderResponse:
    """
    Standardized response from any LLM provider.

    Attributes:
        text: The generated text content.
        token_usage: Token usage breakdown.
        model: The model that generated the response.
        latency_ms: Response time in milliseconds.
        cost_usd: Estimated cost in USD.
        finish_reason: Why generation stopped (stop, length, etc.).
        raw: Raw provider-specific response object.
    """

    text: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    finish_reason: str = "stop"
    raw: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_usage": self.token_usage.to_dict(),
            "model": self.model,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "finish_reason": self.finish_reason,
        }


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.

    Subclasses must implement:
    - complete(): Synchronous completion
    - async_complete(): Asynchronous completion

    Optionally override:
    - stream(): Synchronous streaming
    - async_stream(): Asynchronous streaming
    - calculate_cost(): Custom cost calculation
    """

    # Subclasses should define their pricing table:
    # PRICING: dict[str, dict[str, float]] = {
    #   "model-name": {"input": price_per_1k, "output": price_per_1k}
    # }
    PRICING: dict[str, dict[str, float]] = {}

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "",
        default_temperature: float = 0.7,
        default_max_tokens: int = 1024,
        timeout: float = 60.0,
        cache: Optional[Any] = None,
    ) -> None:
        """
        Initialize the provider.

        Args:
            api_key: API authentication key.
            default_model: Default model to use when none specified.
            default_temperature: Default sampling temperature.
            default_max_tokens: Default max completion tokens.
            timeout: Request timeout in seconds.
            cache: Optional CacheManager instance.
        """
        self.api_key = api_key
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout
        self.cache = cache

        # Statistics tracking
        self._total_calls: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._total_latency_ms: float = 0.0

    @abstractmethod
    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a completion synchronously.

        Args:
            prompt: The user prompt to complete.
            model: Override the default model.
            temperature: Override the default temperature.
            max_tokens: Override the default max tokens.
            system_prompt: Optional system message.
            **kwargs: Provider-specific additional parameters.

        Returns:
            ProviderResponse with the generated text and metadata.
        """
        ...

    @abstractmethod
    async def async_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a completion asynchronously.

        Args:
            prompt: The user prompt to complete.
            model: Override the default model.
            temperature: Override the default temperature.
            max_tokens: Override the default max tokens.
            system_prompt: Optional system message.
            **kwargs: Provider-specific additional parameters.

        Returns:
            ProviderResponse with the generated text and metadata.
        """
        ...

    def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream completion tokens synchronously.

        Default implementation calls complete() and yields the full result.
        Override for true streaming support.
        """
        response = self.complete(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs,
        )
        yield response.text

    async def async_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens asynchronously.

        Default implementation calls async_complete() and yields the full result.
        Override for true streaming support.
        """
        response = await self.async_complete(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs,
        )
        yield response.text

    def calculate_cost(
        self,
        token_usage: TokenUsage,
        model: str,
    ) -> float:
        """
        Calculate the cost of a completion in USD.

        Uses the PRICING table defined by the subclass.

        Args:
            token_usage: Token usage for the completion.
            model: Model name used.

        Returns:
            Estimated cost in USD.
        """
        if model not in self.PRICING:
            # Try to find a partial match
            for key in self.PRICING:
                if key in model or model in key:
                    model = key
                    break
            else:
                return 0.0

        pricing = self.PRICING[model]
        input_cost = (token_usage.prompt_tokens / 1000) * pricing.get("input", 0.0)
        output_cost = (token_usage.completion_tokens / 1000) * pricing.get("output", 0.0)
        return input_cost + output_cost

    def _update_stats(self, response: ProviderResponse) -> None:
        """Update internal usage statistics after a call."""
        self._total_calls += 1
        self._total_tokens += response.token_usage.total_tokens
        self._total_cost += response.cost_usd
        self._total_latency_ms += response.latency_ms

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated usage statistics."""
        avg_latency = (
            self._total_latency_ms / self._total_calls
            if self._total_calls > 0
            else 0.0
        )
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "avg_latency_ms": avg_latency,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._total_calls = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_latency_ms = 0.0

    def _resolve_params(
        self,
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> tuple[str, float, int]:
        """Resolve parameters, falling back to defaults."""
        return (
            model or self.default_model,
            temperature if temperature is not None else self.default_temperature,
            max_tokens or self.default_max_tokens,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.default_model!r})"
