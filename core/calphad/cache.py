from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _hash_payload(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


class CalphadCache:
    def __init__(self, cache_dir: str | Path | None = None, policy: str = "hybrid") -> None:
        self.policy = str(policy or "hybrid").strip().lower()
        self._memory: dict[str, dict[str, Any]] = {}
        self.cache_dir = None if cache_dir in (None, "") else Path(cache_dir)
        if self.cache_dir is not None and self.policy in {"hybrid", "disk"}:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, payload: dict[str, Any]) -> str:
        return _hash_payload(payload)

    def get(self, key: str) -> dict[str, Any] | None:
        if self.policy in {"hybrid", "memory"} and key in self._memory:
            return dict(self._memory[key])
        if self.cache_dir is not None and self.policy in {"hybrid", "disk"}:
            path = self.cache_dir / f"{key}.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        if self.policy == "hybrid":
                            self._memory[key] = dict(data)
                        return dict(data)
                except Exception:
                    return None
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        payload = dict(value)
        if self.policy in {"hybrid", "memory"}:
            self._memory[key] = payload
        if self.cache_dir is not None and self.policy in {"hybrid", "disk"}:
            path = self.cache_dir / f"{key}.json"
            try:
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            except Exception:
                return

