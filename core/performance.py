from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable


class CacheEntry:
    """Cache entry with metadata"""

    def __init__(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        *,
        created_at: float | None = None,
    ):
        self.key = key
        self.value = value
        self.created_at = (
            float(created_at) if created_at is not None else time.time()
        )
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

        self._memory: OrderedDict[str, CacheEntry] = OrderedDict()
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
            data = json.dumps(
                key,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            return hashlib.sha256(data.encode("utf-8")).hexdigest()
        return str(key)

    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._memory:
            return
        self._memory.popitem(last=False)
        self._stats["evictions"] += 1

    def _touch_memory(self, cache_key: str) -> CacheEntry | None:
        entry = self._memory.get(cache_key)
        if entry is None:
            return None
        self._memory.move_to_end(cache_key)
        return entry

    def _metadata_path(self, cache_key: str) -> Path | None:
        if not self.cache_dir or not self.enable_disk:
            return None
        return self.cache_dir / f"{cache_key}.meta.json"

    def _delete_disk_files(self, cache_key: str) -> None:
        if not self.cache_dir or not self.enable_disk:
            return
        for path in [
            self.cache_dir / f"{cache_key}.json",
            self.cache_dir / f"{cache_key}.pkl",
            self.cache_dir / f"{cache_key}.meta.json",
        ]:
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

    def _load_metadata(self, cache_key: str) -> dict[str, Any] | None:
        meta_path = self._metadata_path(cache_key)
        if meta_path is None or not meta_path.exists():
            return None
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _save_metadata(
        self,
        cache_key: str,
        *,
        storage: str,
        ttl: int | None,
        created_at: float,
    ) -> None:
        meta_path = self._metadata_path(cache_key)
        if meta_path is None:
            return
        payload = {
            "storage": str(storage),
            "ttl": None if ttl is None else int(ttl),
            "created_at": float(created_at),
        }
        try:
            meta_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _memory_store(
        self,
        cache_key: str,
        value: Any,
        *,
        ttl: int | None,
        created_at: float | None = None,
    ) -> None:
        if cache_key in self._memory:
            del self._memory[cache_key]
        elif len(self._memory) >= self.max_memory_items:
            self._evict_lru()
        self._memory[cache_key] = CacheEntry(
            cache_key,
            value,
            ttl,
            created_at=created_at,
        )
        self._memory.move_to_end(cache_key)

    def _load_from_disk(
        self,
        cache_key: str,
    ) -> tuple[Any, int | None, float | None] | None:
        """Load value from disk cache"""
        if not self.cache_dir or not self.enable_disk:
            return None

        json_path = self.cache_dir / f"{cache_key}.json"
        pkl_path = self.cache_dir / f"{cache_key}.pkl"
        metadata = self._load_metadata(cache_key)
        ttl = metadata.get("ttl") if isinstance(metadata, dict) else None
        if ttl is not None:
            try:
                ttl = int(ttl)
            except Exception:
                ttl = None
        created_at = metadata.get("created_at") if isinstance(metadata, dict) else None
        if created_at is not None:
            try:
                created_at = float(created_at)
            except Exception:
                created_at = None
        if ttl is not None and created_at is not None:
            if time.time() - created_at > ttl:
                self._delete_disk_files(cache_key)
                return None

        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                self._stats["disk_reads"] += 1
                return data, ttl, created_at
            except Exception:
                pass

        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as handle:
                    data = pickle.load(handle)
                self._stats["disk_reads"] += 1
                return data, ttl, created_at
            except Exception:
                pass

        return None

    def _save_to_disk(
        self,
        cache_key: str,
        value: Any,
        *,
        ttl: int | None,
        created_at: float,
    ) -> None:
        """Save value to disk cache"""
        if not self.cache_dir or not self.enable_disk:
            return

        try:
            json_path = self.cache_dir / f"{cache_key}.json"
            json_path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._save_metadata(
                cache_key,
                storage="json",
                ttl=ttl,
                created_at=created_at,
            )
            self._stats["disk_writes"] += 1
            return
        except (TypeError, ValueError):
            pass

        try:
            pkl_path = self.cache_dir / f"{cache_key}.pkl"
            with open(pkl_path, "wb") as handle:
                pickle.dump(value, handle)
            self._save_metadata(
                cache_key,
                storage="pickle",
                ttl=ttl,
                created_at=created_at,
            )
            self._stats["disk_writes"] += 1
        except Exception:
            pass

    def get(self, key: str | dict[str, Any], default: Any = None) -> Any:
        """Get value from cache"""
        cache_key = self._make_key(key)

        entry = self._touch_memory(cache_key)
        if entry is not None:
            if entry.is_expired():
                self.delete(cache_key)
                self._stats["misses"] += 1
                return default
            self._stats["hits"] += 1
            return entry.access()

        loaded = self._load_from_disk(cache_key)
        if loaded is not None:
            value, ttl, created_at = loaded
            self._memory_store(
                cache_key,
                value,
                ttl=ttl,
                created_at=created_at,
            )
            self._stats["hits"] += 1
            return value

        self._stats["misses"] += 1
        return default

    def set(self, key: str | dict[str, Any], value: Any, ttl: int | None = None) -> None:
        """Set value in cache"""
        cache_key = self._make_key(key)
        ttl = ttl if ttl is not None else self.default_ttl
        created_at = time.time()
        self._memory_store(
            cache_key,
            value,
            ttl=ttl,
            created_at=created_at,
        )
        self._save_to_disk(
            cache_key,
            value,
            ttl=ttl,
            created_at=created_at,
        )

    def delete(self, key: str | dict[str, Any]) -> None:
        """Delete value from cache"""
        cache_key = self._make_key(key)
        if cache_key in self._memory:
            del self._memory[cache_key]
        self._delete_disk_files(cache_key)

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


class PerformanceMonitor:
    """Monitor and track performance metrics"""

    def __init__(self):
        self._timings: dict[str, list[float]] = {}
        self._counters: dict[str, int] = {}

    @contextmanager
    def measure(self, operation: str):
        """Context manager for measuring operation time"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            if operation not in self._timings:
                self._timings[operation] = []
            self._timings[operation].append(duration)

    def increment(self, counter: str, value: int = 1) -> None:
        """Increment a counter"""
        self._counters[counter] = self._counters.get(counter, 0) + value

    def get_stats(self, operation: str) -> dict[str, float] | None:
        """Get statistics for an operation"""
        if operation not in self._timings or not self._timings[operation]:
            return None
        
        timings = self._timings[operation]
        return {
            "count": len(timings),
            "total": sum(timings),
            "mean": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings),
        }

    def get_all_stats(self) -> dict[str, Any]:
        """Get all performance statistics"""
        return {
            "timings": {op: self.get_stats(op) for op in self._timings},
            "counters": dict(self._counters),
        }

    def reset(self) -> None:
        """Reset all metrics"""
        self._timings.clear()
        self._counters.clear()

    def report(self) -> str:
        """Generate performance report"""
        lines = ["Performance Report", "=" * 50]
        
        if self._timings:
            lines.append("\nTimings:")
            for operation, stats in sorted(
                ((op, self.get_stats(op)) for op in self._timings),
                key=lambda x: x[1]["total"] if x[1] else 0,
                reverse=True,
            ):
                if stats:
                    lines.append(
                        f"  {operation}:"
                        f" count={stats['count']}"
                        f" total={stats['total']:.3f}s"
                        f" mean={stats['mean']:.3f}s"
                        f" min={stats['min']:.3f}s"
                        f" max={stats['max']:.3f}s"
                    )
        
        if self._counters:
            lines.append("\nCounters:")
            for counter, value in sorted(self._counters.items()):
                lines.append(f"  {counter}: {value}")
        
        return "\n".join(lines)


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor"""
    return _global_monitor


def timed(operation: str | None = None):
    """Decorator for timing function execution"""
    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with _global_monitor.measure(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


@contextmanager
def measure(operation: str):
    """Context manager for measuring operation time"""
    with _global_monitor.measure(operation):
        yield


def increment(counter: str, value: int = 1) -> None:
    """Increment a counter"""
    _global_monitor.increment(counter, value)


class BatchProcessor:
    """Process items in batches for better performance"""

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size

    def process(
        self,
        items: list[Any],
        process_fn: Callable[[list[Any]], Any],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Any]:
        """Process items in batches"""
        results = []
        total = len(items)
        
        for i in range(0, total, self.batch_size):
            batch = items[i : i + self.batch_size]
            batch_results = process_fn(batch)
            
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
            
            if progress_callback:
                progress_callback(min(i + self.batch_size, total), total)
        
        return results


class LazyLoader:
    """Lazy load expensive resources"""

    def __init__(self, loader_fn: Callable[[], Any]):
        self._loader_fn = loader_fn
        self._value = None
        self._loaded = False

    def get(self) -> Any:
        """Get the loaded value"""
        if not self._loaded:
            self._value = self._loader_fn()
            self._loaded = True
        return self._value

    def reset(self) -> None:
        """Reset the lazy loader"""
        self._value = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if value is loaded"""
        return self._loaded
