"""Caching system for LLM Chain Forge."""
from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    total_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class BaseCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class InMemoryCache(BaseCache):
    """Simple in-memory LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}
        self.max_size = max_size

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if len(self._store) >= self.max_size:
            # Evict oldest entry
            oldest = next(iter(self._store))
            del self._store[oldest]
        expires_at = time.time() + ttl if ttl else None
        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class DiskCache(BaseCache):
    """Persistent disk cache using JSON files."""

    def __init__(self, cache_dir: str = ".chainforge_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            if data.get("expires_at") and time.time() > data["expires_at"]:
                p.unlink()
                return None
            return data["value"]
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        data = {"value": value, "expires_at": time.time() + ttl if ttl else None}
        self._path(key).write_text(json.dumps(data))

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.json"):
            f.unlink()


class CacheManager:
    """High-level cache manager for provider responses."""

    def __init__(self, backend: BaseCache | None = None) -> None:
        self.backend = backend or InMemoryCache()
        self.stats = CacheStats()

    @staticmethod
    def make_key(prompt: str, model: str, temperature: float, **kwargs: Any) -> str:
        """Create a deterministic cache key."""
        payload = json.dumps(
            {"prompt": prompt, "model": model, "temperature": temperature, **kwargs},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        value = self.backend.get(key)
        if value is not None:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        return value

    def set(self, key: str, value: Any, ttl: int | None = 3600) -> None:
        self.backend.set(key, value, ttl=ttl)

    def cache_call(self, fn, prompt: str, model: str, temperature: float = 0.0, **kwargs):
        """Wrap a provider call with caching."""
        key = self.make_key(prompt, model, temperature, **kwargs)
        cached = self.get(key)
        if cached is not None:
            return cached
        result = fn(prompt, model=model, temperature=temperature, **kwargs)
        self.set(key, result)
        return result

    def clear(self) -> None:
        self.backend.clear()
        self.stats = CacheStats()
