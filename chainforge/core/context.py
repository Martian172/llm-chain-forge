"""
Chain execution context — manages state across link executions.

The ChainContext is the shared mutable state that flows through all
links in a chain. It holds variables, conversation history, retrieved
documents, and execution traces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExecutionTrace:
    """
    A single step in the execution trace.

    Attributes:
        link_name: Name of the link that produced this trace entry.
        timestamp: Unix timestamp when this step executed.
        input_vars: Snapshot of context variables at time of execution.
        output: The output produced by the link.
        latency_ms: How long the link took to execute.
    """

    link_name: str
    timestamp: float
    input_vars: dict[str, Any]
    output: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_name": self.link_name,
            "timestamp": self.timestamp,
            "input_vars": self.input_vars,
            "output": self.output,
            "latency_ms": self.latency_ms,
        }


class ChainContext:
    """
    Shared execution state for a chain run.

    ChainContext acts as the blackboard that all links read from and
    write to. It provides:
    - A flat variable store (accessible by key)
    - Conversation history (for multi-turn chains)
    - Retrieved document store (for RAG chains)
    - Execution traces for debugging

    Variable naming convention:
    - Input variables: "input_text", "user_query", etc.
    - Link outputs: "{link_name}.output"
    - Branch metadata: "{branch_name}.chosen"

    Example:
        >>> ctx = ChainContext()
        >>> ctx.set("user_query", "What is ML?")
        >>> ctx.set("step1.output", "Machine learning is...")
        >>> ctx.get("step1.output")
        "Machine learning is..."
    """

    def __init__(
        self,
        initial_variables: Optional[dict[str, Any]] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """
        Initialize a ChainContext.

        Args:
            initial_variables: Optional pre-seeded variables.
            conversation_history: Optional conversation history for multi-turn chains.
        """
        self._variables: dict[str, Any] = initial_variables or {}
        self._history: list[dict[str, str]] = conversation_history or []
        self._documents: list[dict[str, Any]] = []
        self._traces: list[ExecutionTrace] = []
        self._created_at: float = time.time()
        self._metadata: dict[str, Any] = {}

    @property
    def variables(self) -> dict[str, Any]:
        """Read-only view of all context variables."""
        return dict(self._variables)

    @property
    def traces(self) -> list[ExecutionTrace]:
        """Ordered list of execution traces."""
        return list(self._traces)

    @property
    def conversation_history(self) -> list[dict[str, str]]:
        """Conversation history (list of {"role": ..., "content": ...} dicts)."""
        return list(self._history)

    @property
    def documents(self) -> list[dict[str, Any]]:
        """Retrieved documents stored in this context."""
        return list(self._documents)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a variable from the context.

        Args:
            key: Variable name (supports dot notation: "link_name.output").
            default: Default value if key not found.

        Returns:
            The variable value, or default if not found.
        """
        return self._variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a variable in the context.

        Args:
            key: Variable name.
            value: Value to store.
        """
        self._variables[key] = value

    def update(self, updates: dict[str, Any]) -> None:
        """
        Update multiple variables at once.

        Args:
            updates: Dictionary of key-value pairs to set.
        """
        self._variables.update(updates)

    def delete(self, key: str) -> None:
        """Remove a variable from the context."""
        self._variables.pop(key, None)

    def has(self, key: str) -> bool:
        """Check if a variable exists in the context."""
        return key in self._variables

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Message role ("user", "assistant", "system").
            content: Message content.
        """
        self._history.append({"role": role, "content": content})

    def add_document(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> None:
        """
        Add a retrieved document to the context.

        Args:
            content: Document text content.
            metadata: Optional document metadata (source, score, etc.).
            doc_id: Optional document identifier.
        """
        self._documents.append({
            "id": doc_id or f"doc_{len(self._documents)}",
            "content": content,
            "metadata": metadata or {},
        })

    def get_formatted_docs(self, separator: str = "\n\n---\n\n") -> str:
        """
        Get all documents formatted as a single string.

        Args:
            separator: String to join documents with.

        Returns:
            Concatenated document contents.
        """
        return separator.join(doc["content"] for doc in self._documents)

    def trace(
        self,
        link_name: str,
        input_vars: dict[str, Any],
        output: str,
        latency_ms: float,
    ) -> None:
        """
        Record an execution trace entry.

        Args:
            link_name: Name of the link that just executed.
            input_vars: Snapshot of variables used as input.
            output: Output produced by the link.
            latency_ms: Execution time in milliseconds.
        """
        entry = ExecutionTrace(
            link_name=link_name,
            timestamp=time.time(),
            input_vars=input_vars,
            output=output,
            latency_ms=latency_ms,
        )
        self._traces.append(entry)

    def clear_traces(self) -> None:
        """Clear all execution traces."""
        self._traces.clear()

    def snapshot(self) -> dict[str, Any]:
        """
        Take a snapshot of the current context state.

        Returns:
            Dictionary with all context data for debugging.
        """
        return {
            "variables": dict(self._variables),
            "conversation_history": list(self._history),
            "documents": list(self._documents),
            "traces": [t.to_dict() for t in self._traces],
            "created_at": self._created_at,
            "metadata": self._metadata,
        }

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value (for framework use)."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value."""
        return self._metadata.get(key, default)

    def __repr__(self) -> str:
        var_count = len(self._variables)
        trace_count = len(self._traces)
        return f"ChainContext(variables={var_count}, traces={trace_count})"
