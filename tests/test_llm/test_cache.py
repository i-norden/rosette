"""Tests for LLM response cache."""

from __future__ import annotations

import json
import time

from snoopy.llm.cache import LLMCache


class TestLLMCacheBasic:
    def test_cache_disabled_returns_none(self, tmp_path) -> None:
        """Disabled cache always returns None on get."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), enabled=False)
        cache.put("sha256", "prompt", "model", {"content": "response"})
        assert cache.get("sha256", "prompt", "model") is None

    def test_cache_put_and_get(self, tmp_path) -> None:
        """Put then get returns the stored response."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        response = {"content": "test response", "parsed": None}
        cache.put("sha256abc", "test prompt", "claude-3", response)
        result = cache.get("sha256abc", "test prompt", "claude-3")
        assert result is not None
        assert result["content"] == "test response"

    def test_cache_miss(self, tmp_path) -> None:
        """Get on missing key returns None."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        assert cache.get("nonexistent", "prompt", "model") is None

    def test_cache_expired_entry(self, tmp_path) -> None:
        """Expired entries return None."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), ttl=1)
        response = {"content": "old"}
        cache.put("sha256", "prompt", "model", response)

        # Manually backdate the entry
        key = cache._make_key("sha256", "prompt", "model")
        path = cache._entry_path(key)
        with open(path) as f:
            entry = json.load(f)
        entry["_cached_at"] = time.time() - 100
        with open(path, "w") as f:
            json.dump(entry, f)

        result = cache.get("sha256", "prompt", "model")
        assert result is None
        assert not path.exists()  # Entry should be cleaned up

    def test_cache_strips_internal_metadata(self, tmp_path) -> None:
        """Returned entries should not contain _-prefixed keys."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        cache.put("sha", "prompt", "model", {"content": "x", "parsed": None})
        result = cache.get("sha", "prompt", "model")
        assert result is not None
        for key in result:
            assert not key.startswith("_")

    def test_cache_different_keys(self, tmp_path) -> None:
        """Different inputs produce different cache entries."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        cache.put("sha1", "prompt1", "model", {"content": "r1"})
        cache.put("sha2", "prompt2", "model", {"content": "r2"})
        assert cache.get("sha1", "prompt1", "model")["content"] == "r1"
        assert cache.get("sha2", "prompt2", "model")["content"] == "r2"

    def test_cache_corrupt_json(self, tmp_path) -> None:
        """Corrupt JSON returns None gracefully."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        cache.put("sha", "prompt", "model", {"content": "x"})
        key = cache._make_key("sha", "prompt", "model")
        path = cache._entry_path(key)
        path.write_text("not json{{{")
        assert cache.get("sha", "prompt", "model") is None


class TestLLMCacheClear:
    def test_clear_removes_all(self, tmp_path) -> None:
        """clear() removes all cache entries."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        cache.put("sha1", "p1", "m", {"c": "1"})
        cache.put("sha2", "p2", "m", {"c": "2"})
        count = cache.clear()
        assert count == 2
        assert cache.get("sha1", "p1", "m") is None
        assert cache.get("sha2", "p2", "m") is None

    def test_clear_disabled(self, tmp_path) -> None:
        """clear() on disabled cache returns 0."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), enabled=False)
        assert cache.clear() == 0

    def test_clear_empty_cache(self, tmp_path) -> None:
        """clear() on empty cache returns 0."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        assert cache.clear() == 0


class TestLLMCacheEvictExpired:
    def test_evict_expired_removes_old_entries(self, tmp_path) -> None:
        """evict_expired() removes only expired entries."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), ttl=100)

        # Add a fresh entry
        cache.put("fresh", "p", "m", {"c": "fresh"})

        # Add an expired entry manually
        cache.put("old", "p", "m", {"c": "old"})
        key = cache._make_key("old", "p", "m")
        path = cache._entry_path(key)
        with open(path) as f:
            entry = json.load(f)
        entry["_cached_at"] = time.time() - 200
        with open(path, "w") as f:
            json.dump(entry, f)

        count = cache.evict_expired()
        assert count == 1
        assert cache.get("fresh", "p", "m") is not None
        assert cache.get("old", "p", "m") is None

    def test_evict_expired_disabled(self, tmp_path) -> None:
        """evict_expired() on disabled cache returns 0."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), enabled=False)
        assert cache.evict_expired() == 0

    def test_evict_expired_corrupt_entries(self, tmp_path) -> None:
        """evict_expired() removes corrupt JSON entries."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), ttl=3600)
        cache.put("sha", "p", "m", {"c": "x"})
        key = cache._make_key("sha", "p", "m")
        path = cache._entry_path(key)
        path.write_text("corrupt{{{")
        count = cache.evict_expired()
        assert count == 1


class TestLLMCacheMakeKey:
    def test_make_key_deterministic(self) -> None:
        """Same inputs produce same key."""
        key1 = LLMCache._make_key("sha", "prompt", "model")
        key2 = LLMCache._make_key("sha", "prompt", "model")
        assert key1 == key2

    def test_make_key_different_inputs(self) -> None:
        """Different inputs produce different keys."""
        key1 = LLMCache._make_key("sha1", "prompt", "model")
        key2 = LLMCache._make_key("sha2", "prompt", "model")
        assert key1 != key2


class TestLLMCachePutDisabled:
    def test_put_disabled_no_file(self, tmp_path) -> None:
        """put() on disabled cache creates no files."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"), enabled=False)
        cache.put("sha", "prompt", "model", {"c": "x"})
        cache_dir = tmp_path / "cache"
        assert not cache_dir.exists() or len(list(cache_dir.glob("*.json"))) == 0
