"""
LLM Chain Forge - Build, Test & Optimize LLM Prompt Chains.

This package provides a lightweight yet powerful framework for:
- Building multi-step LLM prompt chains
- Evaluating chain performance with built-in metrics
- A/B testing different chain configurations
- Caching responses for efficiency
- Interactive web playground for experimentation
"""

from chainforge.core.chain import Chain, ChainResult
from chainforge.core.link import Link, LinkResult, PromptTemplate as Prompt
from chainforge.core.context import ChainContext
from chainforge.evaluation.evaluator import Evaluator
from chainforge.evaluation.ab_test import ABTest
from chainforge.cache.cache_manager import CacheManager
from chainforge import providers

__version__ = "0.1.0"
__author__ = "LLM Chain Forge Contributors"
__license__ = "MIT"

__all__ = [
    "Chain",
    "ChainResult",
    "Link",
    "LinkResult",
    "Prompt",
    "ChainContext",
    "Evaluator",
    "ABTest",
    "CacheManager",
    "providers",
    "__version__",
]
