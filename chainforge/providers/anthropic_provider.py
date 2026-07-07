"""
Anthropic Claude provider for LLM Chain Forge.

Supports Claude 3 Haiku, Sonnet, and Opus with streaming
and accurate cost tracking.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Iterator, Optional

from chainforge.providers.base import BaseProvider, ProviderResponse, TokenUsage


class AnthropicProvider(BaseProvider):
    """
    Anthropic API provider for Claude models.

    Supports Claude 3 Haiku, Sonnet, and Opus with:
    - Full streaming support
    - System prompt handling
    - Accurate cost tracking

    Example:
        >>> provider = AnthropicProvider(api_key="sk-ant-...", model="claude-3-haiku-20240307")
        >>> response = provider.complete("What is the meaning of life?")
        >>> print(response.text)
    """

    # Pricing per 1,000 tokens (as of early 2024)
    PRICING: dict[str, dict[str, float]] = {
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
        # Aliases
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    }

    # Max tokens limits per model
    MAX_TOKENS: dict[str, int] = {
        "claude-3-haiku-20240307": 4096,
        "claude-3-sonnet-20240229": 4096,
        "claude-3-opus-20240229": 4096,
        "claude-3-5-sonnet-20240620": 8192,
        "claude-3-5-sonnet-20241022": 8192,
        "claude-3-5-haiku-20241022": 8192,
    }

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
        cache: Optional[Any] = None,
    ) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
            model: Default Claude model to use.
            temperature: Default sampling temperature.
            max_tokens: Default max completion tokens.
            timeout: Request timeout in seconds.
            max_retries: Max retry attempts.
            cache: Optional CacheManager instance.
        """
        import os
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

        super().__init__(
            api_key=api_key,
            default_model=model,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
            timeout=timeout,
            cache=cache,
        )
        self.max_retries = max_retries
        self._client: Optional[Any] = None
        self._async_client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazily initialize the Anthropic sync client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            except ImportError:
                raise ImportError(
                    "anthropic package is required for AnthropicProvider. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def _get_async_client(self) -> Any:
        """Lazily initialize the Anthropic async client."""
        if self._async_client is None:
            try:
                import anthropic
                self._async_client = anthropic.AsyncAnthropic(
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            except ImportError:
                raise ImportError(
                    "anthropic package is required for AnthropicProvider. "
                    "Install with: pip install anthropic"
                )
        return self._async_client

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion synchronously using Claude."""
        resolved_model, resolved_temp, resolved_max = self._resolve_params(
            model, temperature, max_tokens
        )

        # Check cache
        if self.cache:
            cache_key = self.cache.make_key(prompt, resolved_model, resolved_temp)
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        # Enforce max_tokens limit for the model
        model_max = self.MAX_TOKENS.get(resolved_model, 4096)
        resolved_max = min(resolved_max, model_max)

        client = self._get_client()
        start_time = time.monotonic()

        call_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": resolved_max,
            "temperature": resolved_temp,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt

        message = client.messages.create(**call_kwargs)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        token_usage = TokenUsage(
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        )

        output_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                output_text += block.text

        cost = self.calculate_cost(token_usage, resolved_model)

        response = ProviderResponse(
            text=output_text,
            token_usage=token_usage,
            model=resolved_model,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            finish_reason=message.stop_reason or "stop",
            raw=message,
        )

        if self.cache:
            self.cache.set(cache_key, response)

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
        """Generate a completion asynchronously using Claude."""
        resolved_model, resolved_temp, resolved_max = self._resolve_params(
            model, temperature, max_tokens
        )

        # Check cache
        if self.cache:
            cache_key = self.cache.make_key(prompt, resolved_model, resolved_temp)
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        model_max = self.MAX_TOKENS.get(resolved_model, 4096)
        resolved_max = min(resolved_max, model_max)

        client = self._get_async_client()
        start_time = time.monotonic()

        call_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": resolved_max,
            "temperature": resolved_temp,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt

        message = await client.messages.create(**call_kwargs)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        token_usage = TokenUsage(
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        )

        output_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                output_text += block.text

        cost = self.calculate_cost(token_usage, resolved_model)

        response = ProviderResponse(
            text=output_text,
            token_usage=token_usage,
            model=resolved_model,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            finish_reason=message.stop_reason or "stop",
            raw=message,
        )

        if self.cache:
            self.cache.set(cache_key, response)

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
        """Stream Claude responses token by token."""
        resolved_model, resolved_temp, resolved_max = self._resolve_params(
            model, temperature, max_tokens
        )
        model_max = self.MAX_TOKENS.get(resolved_model, 4096)
        resolved_max = min(resolved_max, model_max)

        client = self._get_client()
        call_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": resolved_max,
            "temperature": resolved_temp,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt

        with client.messages.stream(**call_kwargs) as stream:
            for text in stream.text_stream:
                yield text

    async def async_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream Claude responses asynchronously."""
        resolved_model, resolved_temp, resolved_max = self._resolve_params(
            model, temperature, max_tokens
        )
        model_max = self.MAX_TOKENS.get(resolved_model, 4096)
        resolved_max = min(resolved_max, model_max)

        client = self._get_async_client()
        call_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": resolved_max,
            "temperature": resolved_temp,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt

        async with client.messages.stream(**call_kwargs) as stream:
            async for text in stream.text_stream:
                yield text
