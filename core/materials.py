from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any


try:
    sys.modules.setdefault("core.cache_manager", import_module("core.performance"))
except Exception:
    pass


@dataclass(slots=True)
class MaterialPreset:
    """Serializable preset that defines one generation scenario."""

    name: str
    material: str
    lab: str
    generator: str
    image_size: tuple[int, int] = (2048, 2048)
    seed: int = 42
    composition: dict[str, float] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    microscope: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MaterialPreset":
        size = payload.get("image_size", [2048, 2048])
        if len(size) != 2:
            raise ValueError("image_size must contain exactly 2 integers")
        return cls(
            name=payload["name"],
            material=payload.get("material", payload["name"]),
            lab=payload.get("lab", "unspecified"),
            generator=payload["generator"],
            image_size=(int(size[0]), int(size[1])),
            seed=int(payload.get("seed", 42)),
            composition={
                str(k): float(v)
                for k, v in dict(
                    payload.get("composition", payload.get("metadata", {}).get("composition_wt", {}))
                ).items()
            },
            generation=dict(payload.get("generation", {})),
            microscope=dict(payload.get("microscope", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "material": self.material,
            "lab": self.lab,
            "generator": self.generator,
            "image_size": [self.image_size[0], self.image_size[1]],
            "seed": self.seed,
            "composition": self.composition,
            "generation": self.generation,
            "microscope": self.microscope,
            "metadata": self.metadata,
        }


def load_preset(path: str | Path) -> MaterialPreset:
    preset_path = Path(path)
    cached = _load_preset_cached(*_path_signature(preset_path))
    return MaterialPreset.from_dict(deepcopy(cached))


def list_presets(directory: str | Path) -> list[Path]:
    base = Path(directory)
    if not base.exists():
        return []
    return list(_list_presets_cached(*_path_signature(base)))


def _path_signature(path: Path) -> tuple[str, int, int]:
    resolved = path.resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=128)
def _load_preset_cached(path_str: str, mtime_ns: int, file_size: int) -> dict[str, Any]:
    del mtime_ns, file_size
    with Path(path_str).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Preset payload must be object: {path_str}")
    return payload


@lru_cache(maxsize=32)
def _list_presets_cached(
    path_str: str,
    mtime_ns: int,
    file_size: int,
) -> tuple[Path, ...]:
    del mtime_ns, file_size
    base = Path(path_str)
    return tuple(sorted(p for p in base.glob("*.json") if p.is_file()))
