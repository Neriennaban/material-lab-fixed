from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cooling_modes import canonicalize_cooling_mode


def _normalize_route_operations(process_route: dict[str, Any]) -> None:
    operations = process_route.get("operations")
    if not isinstance(operations, list):
        return
    for op in operations:
        if not isinstance(op, dict):
            continue
        raw_mode = op.get("cooling_mode", "auto")
        op["cooling_mode"] = canonicalize_cooling_mode(raw_mode)


def migrate_preset(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Preset is not a JSON object: {path}")

    changed = False
    original_generator = str(payload.get("generator", "")).strip().lower()

    generation = payload.get("generation")
    if not isinstance(generation, dict):
        generation = {}
        payload["generation"] = generation
        changed = True

    auto_hint = generation.get("auto_hint")
    if not isinstance(auto_hint, dict):
        auto_hint = {}
        generation["auto_hint"] = auto_hint
        changed = True

    if original_generator and original_generator != "auto" and not auto_hint.get("preferred_generator"):
        auto_hint["preferred_generator"] = original_generator
        changed = True

    if payload.get("generator") != "auto":
        payload["generator"] = "auto"
        changed = True

    if "cooling_mode" not in generation:
        generation["cooling_mode"] = "auto"
        changed = True
    else:
        normalized_mode = canonicalize_cooling_mode(generation.get("cooling_mode"))
        if generation.get("cooling_mode") != normalized_mode:
            generation["cooling_mode"] = normalized_mode
            changed = True

    route = generation.get("process_route")
    if isinstance(route, dict):
        before = json.dumps(route, ensure_ascii=False, sort_keys=True)
        _normalize_route_operations(route)
        after = json.dumps(route, ensure_ascii=False, sort_keys=True)
        if before != after:
            changed = True

    if changed:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate presets to generator=auto")
    parser.add_argument(
        "--presets-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "presets",
        help="Directory with preset json files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    presets_dir = args.presets_dir
    if not presets_dir.exists():
        raise FileNotFoundError(f"Presets directory not found: {presets_dir}")

    changed = 0
    total = 0
    for path in sorted(presets_dir.glob("*.json")):
        total += 1
        if migrate_preset(path):
            changed += 1

    print(f"Processed {total} preset(s), changed {changed}.")


if __name__ == "__main__":
    main()
