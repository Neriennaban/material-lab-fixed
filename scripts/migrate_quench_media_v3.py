from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.metallography_v3.quench_media_v3 import canonicalize_quench_medium_code


def _normalize_payload(payload: dict) -> bool:
    thermal = payload.get("thermal_program")
    if not isinstance(thermal, dict):
        return False
    quench = thermal.get("quench")
    if not isinstance(quench, dict):
        quench = {}
        thermal["quench"] = quench

    raw_medium = str(quench.get("medium_code", "")).strip()
    bath_temperature_c = float(quench.get("bath_temperature_c", 20.0))
    canonical = canonicalize_quench_medium_code(
        raw_medium, bath_temperature_c=bath_temperature_c
    )
    resolved = str(canonical.get("resolved_code", raw_medium)).strip()

    changed = False
    if resolved and resolved != raw_medium:
        quench["medium_code"] = resolved
        changed = True
    elif not raw_medium:
        quench["medium_code"] = resolved or "water_20"
        changed = True

    if "bath_temperature_c" not in quench:
        quench["bath_temperature_c"] = 20.0
        changed = True
    if "quench_time_s" not in quench:
        quench["quench_time_s"] = 30.0
        changed = True
    if "sample_temperature_c" not in quench:
        quench["sample_temperature_c"] = 840.0
        changed = True

    return changed


def migrate_presets(presets_dir: Path) -> tuple[int, int]:
    total = 0
    changed = 0
    for path in sorted(presets_dir.glob("*.json")):
        total += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if not _normalize_payload(payload):
            continue
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        changed += 1
    return total, changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize V3 quench medium codes to explicit medium profiles."
    )
    parser.add_argument(
        "--presets-dir",
        default="presets_v3",
        help="Directory with V3 preset JSON files.",
    )
    args = parser.parse_args()

    presets_dir = Path(args.presets_dir)
    if not presets_dir.exists():
        raise SystemExit(f"Presets directory not found: {presets_dir}")

    total, changed = migrate_presets(presets_dir)
    print(f"Processed: {total}, changed: {changed}, dir: {presets_dir}")


if __name__ == "__main__":
    main()
