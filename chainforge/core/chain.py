"""
Core Chain class for LLM Chain Forge.

The Chain class orchestrates multiple Links into a coherent pipeline,
managing data flow, branching logic, parallel execution, and result aggregation.
"""

from __future__ import annotations

import asyncio
import time
import yaml
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from pathlib import Path

from chainforge.core.context import ChainContext, ExecutionTrace
from chainforge.core.link import Link, LinkResult


@dataclass
class TokenUsage:
    """Aggregate token usage across all links in a chain run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ChainResult:
    """
    The result of running a Chain.

    Attributes:
        output: The final output from the last link in the chain.
        metadata: Dictionary of all link outputs and intermediate values.
        token_usage: Aggregated token usage across all links.
        latency_ms: Total wall-clock latency in milliseconds.
        cost_usd: Estimated total cost in US dollars.
        link_results: Ordered list of individual link results.
        traces: Execution traces for debugging.
        success: Whether the chain completed without errors.
        error: Error message if chain failed.
    """

    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    link_results: list[LinkResult] = field(default_factory=list)
    traces: list[ExecutionTrace] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a dictionary."""
        return {
            "output": self.output,
            "metadata": self.metadata,
            "token_usage": self.token_usage.to_dict(),
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "success": self.success,
            "error": self.error,
            "link_results": [lr.to_dict() for lr in self.link_results],
        }

    def __repr__(self) -> str:
        return (
            f"ChainResult(output={self.output[:50]!r}..., "
            f"cost=${self.cost_usd:.4f}, latency={self.latency_ms:.0f}ms, "
            f"tokens={self.token_usage.total_tokens})"
        )


@dataclass
class BranchConfig:
    """Configuration for a conditional branch in a chain."""

    condition: Callable[[ChainContext], bool]
    if_true: Link
    if_false: Optional[Link] = None
    name: str = "branch"


class Chain:
    """
    Orchestrates a sequence of LLM prompt links.

    A Chain manages the flow of data through multiple Links, handling
    context passing, parallel execution, conditional branching, and result
    aggregation. Chains can be serialized to/from YAML for version control.

    Example:
        >>> provider = OpenAIProvider(api_key="sk-...")
        >>> chain = Chain(name="my-chain")
        >>> chain.add_link(Link(name="step1", prompt_template=Prompt("Hello {name}"), provider=provider))
        >>> result = chain.run({"name": "World"})
        >>> print(result.output)
    """

    def __init__(
        self,
        name: str = "unnamed-chain",
        description: str = "",
        version: str = "1.0.0",
        tags: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize a Chain.

        Args:
            name: Human-readable name for this chain.
            description: Optional description of what the chain does.
            version: Semantic version of the chain configuration.
            tags: Optional tags for categorization.
        """
        self.name = name
        self.description = description
        self.version = version
        self.tags = tags or []
        self._links: list[Link | BranchConfig] = []
        self._parallel_groups: list[list[Link]] = []

    @property
    def links(self) -> list[Link | BranchConfig]:
        """All links and branch configs in the chain."""
        return self._links

    def add_link(self, link: Link) -> "Chain":
        """
        Add a link to the end of the chain.

        Args:
            link: The Link to append to the chain.

        Returns:
            self for method chaining.
        """
        self._links.append(link)
        return self

    def branch(
        self,
        condition: Callable[[ChainContext], bool],
        if_true: Link,
        if_false: Optional[Link] = None,
        name: str = "branch",
    ) -> "Chain":
        """
        Add a conditional branch to the chain.

        At runtime, the condition function is evaluated with the current
        ChainContext. If True, if_true link runs; otherwise if_false runs.

        Args:
            condition: A callable that takes ChainContext and returns bool.
            if_true: Link to execute when condition is True.
            if_false: Optional Link to execute when condition is False.
            name: Name for this branch point.

        Returns:
            self for method chaining.

        Example:
            >>> chain.branch(
            ...     condition=lambda ctx: ctx.get("complexity") == "simple",
            ...     if_true=Link(name="fast", model="gpt-4o-mini", ...),
            ...     if_false=Link(name="deep", model="gpt-4o", ...),
            ... )
        """
        branch_config = BranchConfig(
            condition=condition,
            if_true=if_true,
            if_false=if_false,
            name=name,
        )
        self._links.append(branch_config)
        return self

    def add_parallel(self, links: list[Link]) -> "Chain":
        """
        Add a group of links that execute in parallel.

        All links in the group receive the same context. Their outputs
        are stored in the context keyed by their names.

        Args:
            links: List of Links to execute concurrently.

        Returns:
            self for method chaining.
        """
        # Store as a special marker — we wrap in a sentinel object
        parallel_marker = _ParallelGroup(links=links)
        self._links.append(parallel_marker)
        return self

    def run(
        self,
        input_data: dict[str, Any],
        context: Optional[ChainContext] = None,
    ) -> ChainResult:
        """
        Execute the chain synchronously.

        Iterates through all links, passing context between them.
        Each link's output is stored in the context under its name.

        Args:
            input_data: Initial input variables for the chain.
            context: Optional pre-initialized context.

        Returns:
            ChainResult with final output and execution metadata.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.run_async(input_data, context)
        )

    async def run_async(
        self,
        input_data: dict[str, Any],
        context: Optional[ChainContext] = None,
    ) -> ChainResult:
        """
        Execute the chain asynchronously.

        Args:
            input_data: Initial input variables for the chain.
            context: Optional pre-initialized context.

        Returns:
            ChainResult with final output and execution metadata.
        """
        if context is None:
            context = ChainContext()

        # Seed context with all input data
        for key, value in input_data.items():
            context.set(key, value)

        start_time = time.monotonic()
        link_results: list[LinkResult] = []
        total_cost = 0.0
        total_tokens = TokenUsage()
        last_output = ""

        try:
            for step in self._links:
                if isinstance(step, _ParallelGroup):
                    # Run all links in this group concurrently
                    tasks = [link.run_async(context) for link in step.links]
                    results = await asyncio.gather(*tasks)
                    for link, result in zip(step.links, results):
                        context.set(f"{link.name}.output", result.output)
                        context.trace(
                            link_name=link.name,
                            input_vars=dict(context.variables),
                            output=result.output,
                            latency_ms=result.latency_ms,
                        )
                        link_results.append(result)
                        total_cost += result.cost_usd
                        total_tokens = total_tokens + TokenUsage(
                            prompt_tokens=result.token_usage.get("prompt_tokens", 0),
                            completion_tokens=result.token_usage.get("completion_tokens", 0),
                        )
                        last_output = result.output

                elif isinstance(step, BranchConfig):
                    # Evaluate condition and pick the right link
                    condition_met = step.condition(context)
                    chosen_link = step.if_true if condition_met else step.if_false

                    if chosen_link is not None:
                        result = await chosen_link.run_async(context)
                        context.set(f"{chosen_link.name}.output", result.output)
                        context.set(f"{step.name}.chosen", chosen_link.name)
                        context.trace(
                            link_name=f"{step.name}/{chosen_link.name}",
                            input_vars=dict(context.variables),
                            output=result.output,
                            latency_ms=result.latency_ms,
                        )
                        link_results.append(result)
                        total_cost += result.cost_usd
                        total_tokens = total_tokens + TokenUsage(
                            prompt_tokens=result.token_usage.get("prompt_tokens", 0),
                            completion_tokens=result.token_usage.get("completion_tokens", 0),
                        )
                        last_output = result.output

                elif isinstance(step, Link):
                    result = await step.run_async(context)
                    context.set(f"{step.name}.output", result.output)
                    context.trace(
                        link_name=step.name,
                        input_vars=dict(context.variables),
                        output=result.output,
                        latency_ms=result.latency_ms,
                    )
                    link_results.append(result)
                    total_cost += result.cost_usd
                    total_tokens = total_tokens + TokenUsage(
                        prompt_tokens=result.token_usage.get("prompt_tokens", 0),
                        completion_tokens=result.token_usage.get("completion_tokens", 0),
                    )
                    last_output = result.output

            elapsed_ms = (time.monotonic() - start_time) * 1000

            return ChainResult(
                output=last_output,
                metadata=dict(context.variables),
                token_usage=total_tokens,
                latency_ms=elapsed_ms,
                cost_usd=total_cost,
                link_results=link_results,
                traces=context.traces,
                success=True,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return ChainResult(
                output="",
                metadata=dict(context.variables),
                token_usage=total_tokens,
                latency_ms=elapsed_ms,
                cost_usd=total_cost,
                link_results=link_results,
                traces=context.traces,
                success=False,
                error=str(exc),
            )

    def parallel_run(
        self,
        inputs: list[dict[str, Any]],
        max_concurrency: int = 5,
    ) -> list[ChainResult]:
        """
        Run the chain over multiple inputs concurrently.

        Args:
            inputs: List of input dictionaries, one per chain run.
            max_concurrency: Maximum number of concurrent executions.

        Returns:
            List of ChainResults in the same order as inputs.
        """
        return asyncio.get_event_loop().run_until_complete(
            self._parallel_run_async(inputs, max_concurrency)
        )

    async def _parallel_run_async(
        self,
        inputs: list[dict[str, Any]],
        max_concurrency: int,
    ) -> list[ChainResult]:
        """Async implementation of parallel_run."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_with_semaphore(input_data: dict[str, Any]) -> ChainResult:
            async with semaphore:
                return await self.run_async(input_data)

        tasks = [run_with_semaphore(inp) for inp in inputs]
        return list(await asyncio.gather(*tasks))

    def visualize(self) -> str:
        """
        Render an ASCII flowchart of the chain structure.

        Returns:
            A multi-line string with the chain visualization.
        """
        lines = []
        width = 50
        border = "─" * (width - 2)

        lines.append(f"┌{border}┐")
        title = f" {self.name} "
        lines.append(f"│{title.center(width - 2)}│")
        lines.append(f"├{border}┤")
        lines.append(f"│{'':^{width - 2}}│")
        lines.append(f"│{'[INPUT]':^{width - 2}}│")

        for step in self._links:
            lines.append(f"│{'│':^{width - 2}}│")
            lines.append(f"│{'▼':^{width - 2}}│")

            if isinstance(step, _ParallelGroup):
                names = " ║ ".join(f"[{l.name}]" for l in step.links)
                lines.append(f"│{('  ╠══ ' + names + ' ══╣  ').center(width - 2)}│")
            elif isinstance(step, BranchConfig):
                lines.append(f"│{f'<{step.name}?>'.center(width - 2)}│")
                true_name = f"[{step.if_true.name}]"
                false_name = f"[{step.if_false.name}]" if step.if_false else "[skip]"
                lines.append(f"│{(true_name + ' / ' + false_name).center(width - 2)}│")
            elif isinstance(step, Link):
                model_info = f"{step.model}" if step.model else ""
                node_text = f"[{step.name}]"
                if model_info:
                    node_text += f" ({model_info})"
                lines.append(f"│{node_text.center(width - 2)}│")

        lines.append(f"│{'│':^{width - 2}}│")
        lines.append(f"│{'▼':^{width - 2}}│")
        lines.append(f"│{'[OUTPUT]':^{width - 2}}│")
        lines.append(f"│{'':^{width - 2}}│")

        # Stats section
        link_count = sum(
            1 if isinstance(s, Link) else len(s.links) if isinstance(s, _ParallelGroup) else 2
            for s in self._links
        )
        lines.append(f"├{border}┤")
        lines.append(f"│{f' Links: {link_count}  Version: {self.version}'.ljust(width - 2)}│")
        lines.append(f"└{border}┘")

        return "\n".join(lines)

    def to_yaml(self) -> str:
        """
        Serialize the chain to a YAML string.

        Returns:
            YAML representation of the chain configuration.

        Note:
            Branch conditions (callables) cannot be fully serialized.
            Use named condition functions and reference them by name.
        """
        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "links": [],
        }

        for step in self._links:
            if isinstance(step, Link):
                data["links"].append(step.to_dict())
            elif isinstance(step, _ParallelGroup):
                data["links"].append({
                    "type": "parallel",
                    "links": [l.to_dict() for l in step.links],
                })
            elif isinstance(step, BranchConfig):
                data["links"].append({
                    "type": "branch",
                    "name": step.name,
                    "condition": "<callable — define in code>",
                    "if_true": step.if_true.to_dict(),
                    "if_false": step.if_false.to_dict() if step.if_false else None,
                })

        return yaml.dump(data, default_flow_style=False, allow_unicode=True)

    def save_yaml(self, path: str | Path) -> None:
        """Save chain configuration to a YAML file."""
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        providers_map: Optional[dict[str, Any]] = None,
    ) -> "Chain":
        """
        Load a chain from a YAML file.

        Args:
            path: Path to the YAML configuration file.
            providers_map: Mapping of provider names to provider instances.
                           Required to instantiate links with their providers.

        Returns:
            A Chain instance configured from the YAML.

        Example:
            >>> providers_map = {"openai": OpenAIProvider(api_key="...")}
            >>> chain = Chain.from_yaml("my_chain.yaml", providers_map)
        """
        from chainforge.core.link import Link, PromptTemplate

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        chain = cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
        )

        providers_map = providers_map or {}

        for link_data in data.get("links", []):
            link_type = link_data.get("type", "sequential")

            if link_type == "parallel":
                links = [
                    Link.from_dict(ld, providers_map)
                    for ld in link_data["links"]
                ]
                chain.add_parallel(links)

            elif link_type == "branch":
                # Branches defined in YAML use a simple variable-equality condition
                condition_str = link_data.get("condition", "")
                # Parse simple "variable == value" conditions
                condition_fn = _parse_yaml_condition(condition_str)
                if_true = Link.from_dict(link_data["if_true"], providers_map)
                if_false = (
                    Link.from_dict(link_data["if_false"], providers_map)
                    if link_data.get("if_false")
                    else None
                )
                chain.branch(
                    condition=condition_fn,
                    if_true=if_true,
                    if_false=if_false,
                    name=link_data.get("name", "branch"),
                )

            else:
                link = Link.from_dict(link_data, providers_map)
                chain.add_link(link)

        return chain

    def __repr__(self) -> str:
        return f"Chain(name={self.name!r}, links={len(self._links)}, version={self.version!r})"

    def __len__(self) -> int:
        return len(self._links)


class _ParallelGroup:
    """Internal sentinel class to mark a group of parallel links."""

    def __init__(self, links: list[Link]) -> None:
        self.links = links


def _parse_yaml_condition(condition_str: str) -> Callable[[ChainContext], bool]:
    """
    Parse a simple YAML condition string into a callable.

    Supports conditions like: "step_name.output == value"

    Args:
        condition_str: A simple equality condition string.

    Returns:
        A callable that takes ChainContext and returns bool.
    """
    if "==" in condition_str:
        left, right = [s.strip() for s in condition_str.split("==", 1)]
        right = right.strip("'\"")

        def condition(ctx: ChainContext) -> bool:
            return str(ctx.get(left, "")) == right

        return condition

    # Default: always True
    return lambda ctx: True
