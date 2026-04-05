from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable


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
