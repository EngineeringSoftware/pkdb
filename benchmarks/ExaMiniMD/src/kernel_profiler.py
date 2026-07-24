"""Lightweight per-kernel wall-clock timing and invocation counter."""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List


class _Entry:
    __slots__ = ("count", "total", "lo", "hi", "times")

    def __init__(self) -> None:
        self.count: int = 0
        self.total: float = 0.0
        self.lo: float = float("inf")
        self.hi: float = 0.0
        self.times: List[float] = []

    def record(self, t: float) -> None:
        self.count += 1
        self.total += t
        if t < self.lo:
            self.lo = t
        if t > self.hi:
            self.hi = t
        self.times.append(round(t, 7))


class KernelProfiler:
    def __init__(self) -> None:
        self._data: Dict[str, _Entry] = {}

    def reset(self) -> None:
        """Clear all accumulated data (call before each simulation run)."""
        self._data.clear()

    @contextmanager
    def record(self, name: str) -> Generator[None, None, None]:
        """Context manager: time the enclosed block and attribute it to *name*."""
        entry = self._data.get(name)
        if entry is None:
            entry = _Entry()
            self._data[name] = entry
        t0 = time.perf_counter()
        try:
            yield
        finally:
            entry.record(time.perf_counter() - t0)

    def to_dict(self) -> dict:
        kernels = []
        for name, e in self._data.items():
            avg = e.total / e.count if e.count else 0.0
            kernels.append(
                {
                    "name": name,
                    "count": e.count,
                    "total_time": round(e.total, 6),
                    "avg_time": round(avg, 6),
                    "min_time": round(e.lo, 6) if e.count else None,
                    "max_time": round(e.hi, 6) if e.count else None,
                    "times": e.times,
                }
            )
        return {"kernels": kernels}

    def write_json(self, path: str | Path, indent: int = 2) -> None:
        """Serialise profiler data to *path* as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=indent))

    def print_summary(self) -> None:
        """Print a compact table to stdout."""
        total_run = sum(e.total for e in self._data.values())
        print(f"\n{'Kernel':<45} {'Count':>6}  {'Total(s)':>10}  {'Avg(ms)':>9}  {'%':>5}")
        print("-" * 85)
        for name, e in sorted(self._data.items(), key=lambda kv: -kv[1].total):
            pct = 100.0 * e.total / total_run if total_run else 0.0
            avg_ms = 1000.0 * e.total / e.count if e.count else 0.0
            print(f"  {name:<43} {e.count:>6}  {e.total:>10.4f}  {avg_ms:>9.3f}  {pct:>5.1f}%")


# Module-level singleton. All source files share the same instance.
profiler = KernelProfiler()
