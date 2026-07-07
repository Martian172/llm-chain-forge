"""
Link class — an individual step in a Chain.

Each Link wraps a PromptTemplate + Provider call. It handles
variable substitution, execution, and result packaging.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from chainforge.core.context import ChainContext
    from chainforge.providers.base import BaseProvider


class PromptTemplate:
    """
    A prompt template with variable substitution.

    Variables are specified using {variable_name} syntax. Variables
    can reference outputs from previous links as {link_name.output}.

    Example:
        >>> template = PromptTemplate("Summarize: {input_text}")
        >>> prompt = template.render({"input_text": "Hello world"})
        "Summarize: Hello world"
    """

    def __init__(self, template: str, name: str = "") -> None:
        """
        Initialize a PromptTemplate.

        Args:
            template: The template string with {variable} placeholders.
            name: Optional name for this template.
        """
        self.template = template.strip()
        self.name = name
        self._variable_pattern = re.compile(r"\{([^}]+)\}")

    @property
    def variables(self) -> list[str]:
        """Extract all variable names from the template."""
        return self._variable_pattern.findall(self.template)

    def render(self, context: dict[str, Any]) -> str:
        """
        Render the template with the provided context.

        Args:
            context: Dictionary of variable names to values.

        Returns:
            The rendered prompt string.

        Raises:
            KeyError: If a required variable is missing from context.
        """
        rendered = self.template
        for var in self.variables:
            if var in context:
                rendered = rendered.replace(f"{{{var}}}", str(context[var]))
            else:
                # Leave unreplaced variables as-is (warn but don't fail)
                pass
        return rendered

    def validate(self, context: dict[str, Any]) -> list[str]:
        """
        Validate that all required variables are present.

        Args:
            context: Dictionary of available variables.

        Returns:
            List of missing variable names (empty if all present).
        """
        return [var for var in self.variables if var not in context]

    def __repr__(self) -> str:
        preview = self.template[:60].replace("\n", " ")
        return f"PromptTemplate({preview!r}...)"


@dataclass
class TokenUsage:
    """Token usage for a single link execution."""

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
class LinkResult:
    """
    The result of executing a single Link.

    Attributes:
        output: The text output from the LLM.
        token_usage: Token usage broken down by prompt/completion.
        latency_ms: Execution time in milliseconds.
        cost_usd: Estimated cost in US dollars.
        link_name: Name of the link that produced this result.
        model: Model that was used.
        raw_response: The raw provider response object.
    """

    output: str
    token_usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    link_name: str = ""
    model: str = ""
    raw_response: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_name": self.link_name,
            "output": self.output,
            "token_usage": self.token_usage,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "model": self.model,
        }

    def __repr__(self) -> str:
        return (
            f"LinkResult(link={self.link_name!r}, "
            f"output={self.output[:40]!r}..., "
            f"cost=${self.cost_usd:.4f})"
        )


class Link:
    """
    A single executable step in a Chain.

    Each Link holds a prompt template and provider configuration.
    When executed, it renders the template using the current context,
    calls the provider, and returns a LinkResult.

    Attributes:
        name: Unique identifier for this link within the chain.
        prompt_template: The PromptTemplate to render and execute.
        provider: The LLM provider to call.
        model: Model name/identifier to use.
        temperature: Sampling temperature (0.0–2.0).
        max_tokens: Maximum tokens in the completion.
        system_prompt: Optional system message (for chat models).
        input_variables: Declared input variable names.
        output_variable: Name to store the output under.
        retry_on_failure: Whether to retry on provider errors.
        max_retries: Maximum number of retry attempts.
    """

    def __init__(
        self,
        name: str,
        prompt_template: PromptTemplate | str,
        provider: Optional["BaseProvider"] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str = "",
        input_variables: Optional[list[str]] = None,
        output_variable: Optional[str] = None,
        retry_on_failure: bool = True,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize a Link.

        Args:
            name: Unique name for this link (used as context key prefix).
            prompt_template: PromptTemplate or raw string template.
            provider: LLM provider instance. If None, must be set before running.
            model: Override model for this link (overrides provider default).
            temperature: Sampling temperature.
            max_tokens: Max completion tokens.
            system_prompt: Optional system message.
            input_variables: Declared input variable names for validation.
            output_variable: Where to store output (defaults to "{name}.output").
            retry_on_failure: Whether to retry on transient errors.
            max_retries: Max retry count.
        """
        self.name = name
        self.prompt_template = (
            PromptTemplate(prompt_template)
            if isinstance(prompt_template, str)
            else prompt_template
        )
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.input_variables = input_variables or self.prompt_template.variables
        self.output_variable = output_variable or f"{name}.output"
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries

    def run(self, context: "ChainContext") -> LinkResult:
        """
        Execute this link synchronously.

        Args:
            context: The current chain execution context.

        Returns:
            LinkResult with output and execution metadata.
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.run_async(context))

    async def run_async(self, context: "ChainContext") -> LinkResult:
        """
        Execute this link asynchronously.

        Args:
            context: The current chain execution context.

        Returns:
            LinkResult with output and execution metadata.
        """
        if self.provider is None:
            raise ValueError(
                f"Link '{self.name}' has no provider configured. "
                "Set link.provider before running."
            )

        # Render the prompt with current context variables
        context_vars = dict(context.variables)
        rendered_prompt = self.prompt_template.render(context_vars)

        # Execute with retry logic
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries if self.retry_on_failure else 1):
            try:
                start_time = time.monotonic()
                response = await self.provider.async_complete(
                    prompt=rendered_prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    system_prompt=self.system_prompt,
                )
                elapsed_ms = (time.monotonic() - start_time) * 1000

                return LinkResult(
                    output=response.text,
                    token_usage={
                        "prompt_tokens": response.token_usage.prompt_tokens,
                        "completion_tokens": response.token_usage.completion_tokens,
                        "total_tokens": response.token_usage.total_tokens,
                    },
                    latency_ms=elapsed_ms,
                    cost_usd=response.cost_usd,
                    link_name=self.name,
                    model=response.model or self.model or "",
                    raw_response=response,
                )

            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff
                    import asyncio
                    await asyncio.sleep(wait_time)
                continue

        # All retries exhausted
        raise RuntimeError(
            f"Link '{self.name}' failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    def to_dict(self) -> dict[str, Any]:
        """Serialize the link to a dictionary (for YAML export)."""
        return {
            "name": self.name,
            "prompt_template": self.prompt_template.template,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "input_variables": self.input_variables,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        providers_map: Optional[dict[str, "BaseProvider"]] = None,
    ) -> "Link":
        """
        Create a Link from a dictionary (for YAML import).

        Args:
            data: Dictionary with link configuration.
            providers_map: Mapping of provider name strings to provider instances.

        Returns:
            A fully configured Link instance.
        """
        providers_map = providers_map or {}
        provider_name = data.get("provider")
        provider = providers_map.get(provider_name) if provider_name else None

        return cls(
            name=data["name"],
            prompt_template=PromptTemplate(data["prompt_template"]),
            provider=provider,
            model=data.get("model"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 1024),
            system_prompt=data.get("system_prompt", ""),
            input_variables=data.get("input_variables"),
            retry_on_failure=data.get("retry_on_failure", True),
            max_retries=data.get("max_retries", 3),
        )

    def __repr__(self) -> str:
        return (
            f"Link(name={self.name!r}, model={self.model!r}, "
            f"temperature={self.temperature})"
        )
