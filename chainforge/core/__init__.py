"""
Core module for LLM Chain Forge.

Provides the fundamental building blocks:
- Chain: orchestrates multiple links
- Link: individual prompt execution step
- ChainContext: execution state management
"""

from chainforge.core.chain import Chain, ChainResult
from chainforge.core.link import Link, LinkResult, PromptTemplate
from chainforge.core.context import ChainContext, ExecutionTrace

__all__ = [
    "Chain",
    "ChainResult",
    "Link",
    "LinkResult",
    "PromptTemplate",
    "ChainContext",
    "ExecutionTrace",
]
