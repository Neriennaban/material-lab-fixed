from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        synthesis = payload.get("synthesis_profile")
        if not isinstance(synthesis, dict):
            synthesis = {}
            payload["synthesis_profile"] = synthesis

        previous = str(synthesis.get("system_generator_mode", "")).strip()
        if previous == "system_auto":
            continue
        synthesis["system_generator_mode"] = "system_auto"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        changed += 1
    return total, changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Add system_generator_mode to V3 presets.")
    parser.add_argument("--presets-dir", default="presets_v3", help="Directory with V3 preset JSON files.")
    args = parser.parse_args()

    presets_dir = Path(args.presets_dir)
    if not presets_dir.exists():
        raise SystemExit(f"Presets directory not found: {presets_dir}")
    total, changed = migrate_presets(presets_dir)
    print(f"Processed: {total}, changed: {changed}, dir: {presets_dir}")


if __name__ == "__main__":
    main()
