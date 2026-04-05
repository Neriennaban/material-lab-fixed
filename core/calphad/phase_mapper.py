from __future__ import annotations

import json
from pathlib import Path

_MAP_PATH = Path(__file__).resolve().parents[1] / "rulebook" / "phase_texture_map.json"


def _load_map() -> dict:
    try:
        return json.loads(_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"defaults": {"*": "grains"}, "systems": {}}


_MAP = _load_map()


def map_phase_to_texture(system: str, phase_name: str) -> str:
    sys_name = str(system or "").strip().lower()
    phase_upper = str(phase_name or "").strip().upper()
    systems = _MAP.get("systems", {}) if isinstance(_MAP.get("systems"), dict) else {}
    defaults = _MAP.get("defaults", {}) if isinstance(_MAP.get("defaults"), dict) else {"*": "grains"}

    mapping = systems.get(sys_name, {})
    if isinstance(mapping, dict):
        if phase_upper in mapping:
            return str(mapping[phase_upper])
        for key, value in mapping.items():
            if key.endswith("*") and phase_upper.startswith(key[:-1]):
                return str(value)
            if key.startswith("*") and phase_upper.endswith(key[1:]):
                return str(value)
            if "*" in key:
                pattern = key.replace("*", "")
                if pattern and pattern in phase_upper:
                    return str(value)

    if phase_upper in defaults:
        return str(defaults[phase_upper])
    for key, value in defaults.items():
        if key == "*":
            return str(value)
        if key.endswith("*") and phase_upper.startswith(key[:-1]):
            return str(value)
        if key.startswith("*") and phase_upper.endswith(key[1:]):
            return str(value)
    return "grains"

