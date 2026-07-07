"""Tests for chainforge.cache."""
import pytest
from chainforge.cache.cache_manager import CacheManager, InMemoryCache, DiskCache
import tempfile, os


class TestInMemoryCache:
    def test_set_get(self):
        cache = InMemoryCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss(self):
        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expired(self):
        import time
        cache = InMemoryCache()
        cache.set("exp", "val", ttl=0)
        time.sleep(0.01)
        assert cache.get("exp") is None

    def test_clear(self):
        cache = InMemoryCache()
        cache.set("a", 1); cache.set("b", 2)
        cache.clear()
        assert len(cache) == 0

    def test_max_size_eviction(self):
        cache = InMemoryCache(max_size=3)
        cache.set("k1", 1); cache.set("k2", 2); cache.set("k3", 3)
        cache.set("k4", 4)  # Should evict oldest
        assert len(cache) <= 3


class TestDiskCache:
    def test_set_get(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        cache.set("mykey", {"data": 42})
        result = cache.get("mykey")
        assert result == {"data": 42}

    def test_clear(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        cache.set("x", "y")
        cache.clear()
        assert cache.get("x") is None


class TestCacheManager:
    def test_hit_miss_stats(self):
        mgr = CacheManager()
        key = mgr.make_key("hello", "gpt-4", 0.5)
        mgr.set(key, "response")
        mgr.get(key)  # hit
        mgr.get("missing")  # miss
        assert mgr.stats.hits == 1
        assert mgr.stats.misses == 1
        assert mgr.stats.hit_rate == 0.5

    def test_make_key_deterministic(self):
        k1 = CacheManager.make_key("hi", "gpt-4", 0.5)
        k2 = CacheManager.make_key("hi", "gpt-4", 0.5)
        assert k1 == k2

    def test_different_keys(self):
        k1 = CacheManager.make_key("hi", "gpt-4", 0.5)
        k2 = CacheManager.make_key("hi", "gpt-4o", 0.5)
        assert k1 != k2
