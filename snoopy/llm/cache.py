"""Disk-backed LLM response cache keyed on (image_sha256, prompt_hash, model).

Avoids redundant API calls when re-analyzing the same image with the same
prompt (e.g. during pipeline retries or re-runs).  File-based JSON storage
with configurable TTL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Default TTL: 7 days
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600


class LLMCache:
    """Simple disk-backed JSON cache for LLM responses.

    Each cache entry is stored as a JSON file under ``cache_dir`` with a
    filename derived from the cache key.  Entries include a timestamp and
    are considered stale after ``ttl`` seconds.

    Args:
        cache_dir: Directory for cache files.  Created if it does not exist.
        ttl: Time-to-live in seconds for cache entries.
        enabled: When ``False``, all cache operations are no-ops.
    """

    def __init__(
        self,
        cache_dir: str = "data/llm_cache",
        ttl: int = _DEFAULT_TTL_SECONDS,
        enabled: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_key(image_sha256: str, prompt: str, model: str) -> str:
        """Derive a stable cache key from image hash, prompt, and model."""
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return f"{image_sha256}_{prompt_hash}_{model}"

    def _entry_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, image_sha256: str, prompt: str, model: str) -> dict | None:
        """Look up a cached response.

        Returns:
            The cached response dict, or ``None`` on miss or stale entry.
        """
        if not self.enabled:
            return None

        key = self._make_key(image_sha256, prompt, model)
        path = self._entry_path(key)

        if not path.exists():
            return None

        try:
            with open(path) as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Cache read error for %s: %s", key, e)
            return None

        # Check TTL
        stored_at = entry.get("_cached_at", 0)
        if time.time() - stored_at > self.ttl:
            logger.debug("Cache entry expired for %s", key)
            path.unlink(missing_ok=True)
            return None

        # Strip internal metadata before returning
        response = {k: v for k, v in entry.items() if not k.startswith("_")}
        return response

    def put(self, image_sha256: str, prompt: str, model: str, response: dict) -> None:
        """Store a response in the cache."""
        if not self.enabled:
            return

        key = self._make_key(image_sha256, prompt, model)
        path = self._entry_path(key)

        entry = dict(response)
        entry["_cached_at"] = time.time()
        entry["_image_sha256"] = image_sha256
        entry["_model"] = model

        try:
            with open(path, "w") as f:
                json.dump(entry, f, default=str)
        except OSError as e:
            logger.warning("Cache write error for %s: %s", key, e)

    def clear(self) -> int:
        """Remove all cache entries. Returns the number of entries removed."""
        if not self.enabled or not self.cache_dir.exists():
            return 0

        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
        return count

    def evict_expired(self) -> int:
        """Remove expired cache entries. Returns the number evicted."""
        if not self.enabled or not self.cache_dir.exists():
            return 0

        now = time.time()
        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                with open(path) as f:
                    entry = json.load(f)
                if now - entry.get("_cached_at", 0) > self.ttl:
                    path.unlink()
                    count += 1
            except (json.JSONDecodeError, OSError):
                # Corrupt entry, remove it
                path.unlink(missing_ok=True)
                count += 1
        return count
