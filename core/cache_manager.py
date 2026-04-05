from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable


class CacheEntry:
    """Cache entry with metadata"""

    def __init__(self, key: str, value: Any, ttl: int | None = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def access(self) -> Any:
        """Access cache entry and update metadata"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value


class AdvancedCache:
    """Advanced caching system with TTL, LRU, and persistence"""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_memory_items: int = 1000,
        default_ttl: int | None = None,
        enable_disk: bool = True,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_items = max_memory_items
        self.default_ttl = default_ttl
        self.enable_disk = enable_disk
        
        self._memory: dict[str, CacheEntry] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "disk_reads": 0,
            "disk_writes": 0,
        }
        
        if self.cache_dir and self.enable_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, key: str | dict[str, Any]) -> str:
        """Generate cache key from string or dict"""
        if isinstance(key, dict):
            data = json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
            return hashlib.sha256(data.encode("utf-8")).hexdigest()
        return str(key)

    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._memory:
            return
        
        lru_key = min(self._memory.keys(), key=lambda k: self._memory[k].last_accessed)
        del self._memory[lru_key]
        self._stats["evictions"] += 1

    def _load_from_disk(self, key: str) -> Any | None:
        """Load value from disk cache"""
        if not self.cache_dir or not self.enable_disk:
            return None
        
        json_path = self.cache_dir / f"{key}.json"
        pkl_path = self.cache_dir / f"{key}.pkl"
        
        # Try JSON first
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                self._stats["disk_reads"] += 1
                return data
            except Exception:
                pass
        
        # Try pickle
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                self._stats["disk_reads"] += 1
                return data
            except Exception:
                pass
        
        return None

    def _save_to_disk(self, key: str, value: Any) -> None:
        """Save value to disk cache"""
        if not self.cache_dir or not self.enable_disk:
            return
        
        # Try JSON first (human-readable)
        try:
            json_path = self.cache_dir / f"{key}.json"
            json_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            self._stats["disk_writes"] += 1
            return
        except (TypeError, ValueError):
            pass
        
        # Fallback to pickle
        try:
            pkl_path = self.cache_dir / f"{key}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(value, f)
            self._stats["disk_writes"] += 1
        except Exception:
            pass

    def get(self, key: str | dict[str, Any], default: Any = None) -> Any:
        """Get value from cache"""
        cache_key = self._make_key(key)
        
        # Check memory cache
        if cache_key in self._memory:
            entry = self._memory[cache_key]
            if entry.is_expired():
                del self._memory[cache_key]
                self._stats["misses"] += 1
                return default
            
            self._stats["hits"] += 1
            return entry.access()
        
        # Check disk cache
        value = self._load_from_disk(cache_key)
        if value is not None:
            self.set(cache_key, value)
            self._stats["hits"] += 1
            return value
        
        self._stats["misses"] += 1
        return default

    def set(self, key: str | dict[str, Any], value: Any, ttl: int | None = None) -> None:
        """Set value in cache"""
        cache_key = self._make_key(key)
        
        # Evict if necessary
        if len(self._memory) >= self.max_memory_items:
            self._evict_lru()
        
        # Store in memory
        ttl = ttl if ttl is not None else self.default_ttl
        self._memory[cache_key] = CacheEntry(cache_key, value, ttl)
        
        # Store on disk
        self._save_to_disk(cache_key, value)

    def delete(self, key: str | dict[str, Any]) -> None:
        """Delete value from cache"""
        cache_key = self._make_key(key)
        
        # Remove from memory
        if cache_key in self._memory:
            del self._memory[cache_key]
        
        # Remove from disk
        if self.cache_dir and self.enable_disk:
            for path in [
                self.cache_dir / f"{cache_key}.json",
                self.cache_dir / f"{cache_key}.pkl",
            ]:
                if path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass

    def clear(self) -> None:
        """Clear all cache"""
        self._memory.clear()
        
        if self.cache_dir and self.enable_disk:
            for path in self.cache_dir.glob("*"):
                if path.is_file():
                    try:
                        path.unlink()
                    except Exception:
                        pass

    def get_or_compute(
        self,
        key: str | dict[str, Any],
        compute_fn: Callable[[], Any],
        ttl: int | None = None,
    ) -> Any:
        """Get value from cache or compute if not present"""
        value = self.get(key)
        if value is None:
            value = compute_fn()
            self.set(key, value, ttl)
        return value

    def stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self._stats,
            "memory_items": len(self._memory),
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    def cleanup_expired(self) -> int:
        """Remove expired entries from memory cache"""
        expired_keys = [k for k, v in self._memory.items() if v.is_expired()]
        for key in expired_keys:
            del self._memory[key]
        return len(expired_keys)
