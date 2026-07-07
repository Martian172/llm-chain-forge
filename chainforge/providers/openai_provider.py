"""
OpenAI provider for LLM Chain Forge.

Supports GPT-3.5-turbo, GPT-4, GPT-4o, and GPT-4o-mini with:
- Chat completions
- Streaming
- Function/tool calling
- Automatic retry with exponential backoff
- Accurate cost tracking
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Callable, Iterator, Optional

from chainforge.providers.base import BaseProvider, ProviderResponse, TokenUsage


class OpenAIProvider(BaseProvider):
    """
    OpenAI API provider.

    Supports all OpenAI chat completion models with streaming,
    function calling, retry logic, and cost tracking.

    Example:
        >>> provider = OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")
        >>> response = provider.complete("Explain quantum computing in one sentence.")
        >>> print(response.text)
        >>> print(f"Cost: ${response.cost_usd:.4f}")
    """

    # Pricing per 1,000 tokens (as of early 2024)
    PRICING: dict[str, dict[str, float]] = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-2024-05-13": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
    }

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
        cache: Optional[Any] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var).
            model: Default model to use.
            temperature: Default sampling temperature.
            max_tokens: Default max completion tokens.
            timeout: Request timeout in seconds.
            max_retries: Max retry attempts on transient errors.
            cache: Optional CacheManager instance.
            organization: Optional OpenAI organization ID.
            base_url: Optional custom base URL (for Azure OpenAI, etc.).
        """
        import os
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")

        super().__init__(
            api_key=api_key,
            default_model=model,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
            timeout=timeout,
            cache=cache,
        )
        self.max_retries = max_retries
        self.organization = organization
        self.base_url = base_url
        self._client: Optional[Any] = None
        self._async_client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazily initialize the OpenAI sync client."""
        if self._client is None:
            try:
                from openai import OpenAI
                kwargs: dict[str, Any] = {
                    "api_key": self.api_key,
                    "timeout": self.timeout,
                    "max_retries": self.max_retries,
                }
                if self.organization:
                    kwargs["organization"] = self.organization
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAIProvider. "
                    "Install with: pip install openai"
                )
        return self._client

    def _get_async_client(self) -> Any:
        """Lazily initialize the OpenAI async client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
                kwargs: dict[str, Any] = {
                    "api_key": self.api_key,
                    "timeout": self.timeout,
                    "max_retries": self.max_retries,
                }
                if self.organization:
                    kwargs["organization"] = self.organization
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._async_client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAIProvider. "
                    "Install with: pip install openai"
                )
        return self._async_client

    def _build_messages(
        self, prompt: str, system_prompt: str = ""
    ) -> list[dict[str, str]]:
        """Build the messages list for the chat completions API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a completion synchronously.

        Args:
            prompt: The user prompt.
            model: Model override.
            temperature: Temperature override.
            max_tokens: Max tokens override.
            system_prompt: Optional system message.
            tools: Optional list of function/tool definitions.
            tool_choice: Tool choice strategy ("auto", "none", or specific).
            **kwargs: Additional parameters passed to the API.

        Returns:
            ProviderResponse with generated text and metadata.
        """
        resolved_model, resolved_temp, resolved_max = self._resolve_params(
            model, temperature, max_tokens
        )

        # Check cache
        if self.cache:
            cache_key = self.cache.make_key(prompt, resolved_model, resolved_temp)
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)

        start_time = time.monotonic()

        call_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": resolved_temp,
            "max_tokens": resolved_max,
            **kwargs,
        }
        if tools:
            call_kwargs["tools"] = tools
        if tool_choice:
            call_kwargs["tool_choice"] = tool_choice

        completion = client.chat.completions.create(**call_kwargs)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        usage = completion.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

        # Get output text (handle tool calls too)
        message = completion.choices[0].message
        if message.content:
            output_text = message.content
        elif message.tool_calls:
            import json
            output_text = json.dumps([
                {
                    "function": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ])
        else:
            output_text = ""

        cost = self.calculate_cost(token_usage, resolved_model)

        response = ProviderResponse(
            text=output_text,
            token_usage=token_usage,
            model=resolved_model,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            finish_reason=completion.choices[0].finish_reason or "stop",
            raw=completion,
        )

        # Store in cache
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
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a completion asynchronously.

        Same parameters as complete(). Returns ProviderResponse.
        """
        resolved_model, resolved_temp, resolved_max = self._resolve_params(
            model, temperature, max_tokens
        )

        # Check cache
        if self.cache:
            cache_key = self.cache.make_key(prompt, resolved_model, resolved_temp)
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        client = self._get_async_client()
        messages = self._build_messages(prompt, system_prompt)

        start_time = time.monotonic()

        call_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": resolved_temp,
            "max_tokens": resolved_max,
            **kwargs,
        }
        if tools:
            call_kwargs["tools"] = tools
        if tool_choice:
            call_kwargs["tool_choice"] = tool_choice

        completion = await client.chat.completions.create(**call_kwargs)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        usage = completion.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

        message = completion.choices[0].message
        if message.content:
            output_text = message.content
        elif message.tool_calls:
            import json
            output_text = json.dumps([
                {
                    "function": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ])
        else:
            output_text = ""

        cost = self.calculate_cost(token_usage, resolved_model)

        response = ProviderResponse(
            text=output_text,
            token_usage=token_usage,
            model=resolved_model,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            finish_reason=completion.choices[0].finish_reason or "stop",
            raw=completion,
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
        """Stream tokens synchronously using SSE."""
        resolved_model, resolved_temp, resolved_max = self._resolve_params(
            model, temperature, max_tokens
        )
        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)

        with client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=resolved_temp,
            max_tokens=resolved_max,
            stream=True,
            **kwargs,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    async def async_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens asynchronously using SSE."""
        resolved_model, resolved_temp, resolved_max = self._resolve_params(
            model, temperature, max_tokens
        )
        client = self._get_async_client()
        messages = self._build_messages(prompt, system_prompt)

        async with await client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=resolved_temp,
            max_tokens=resolved_max,
            stream=True,
            **kwargs,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for.
            model: Model to use for tokenization.

        Returns:
            Token count.
        """
        try:
            import tiktoken
            model_name = model or self.default_model
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Rough estimate: ~4 chars per token
            return len(text) // 4

    def estimate_cost(
        self,
        prompt: str,
        expected_output_tokens: int = 500,
        model: Optional[str] = None,
    ) -> float:
        """
        Estimate cost before making a call.

        Args:
            prompt: The prompt text.
            expected_output_tokens: Expected output token count.
            model: Model to price for.

        Returns:
            Estimated cost in USD.
        """
        resolved_model = model or self.default_model
        prompt_tokens = self.count_tokens(prompt, resolved_model)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=expected_output_tokens,
        )
        return self.calculate_cost(usage, resolved_model)
