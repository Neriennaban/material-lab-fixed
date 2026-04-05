from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PHASE_MODEL = {
    "engine": "explicit_rules_v3",
    "phase_control_mode": "auto_with_override",
    "manual_phase_fractions": {},
    "manual_override_weight": 0.35,
    "allow_custom_fallback": True,
    "phase_balance_tolerance_pct": 20.0,
}


def _merge_phase_model(existing: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(DEFAULT_PHASE_MODEL)
    if isinstance(existing, dict):
        data.update(existing)
    manual = data.get("manual_phase_fractions", {})
    if not isinstance(manual, dict):
        manual = {}
    clean_manual = {}
    for key, value in manual.items():
        try:
            v = float(value)
        except Exception:
            continue
        if v > 0.0:
            clean_manual[str(key)] = v
    data["manual_phase_fractions"] = clean_manual
    return data


def migrate_preset(path: Path, dry_run: bool = False) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return False

    changed = False
    merged = _merge_phase_model(payload.get("phase_model"))
    if payload.get("phase_model") != merged:
        payload["phase_model"] = merged
        changed = True

    for legacy_key in ("thermo", "process_route", "deprecated_v3", "legacy_note"):
        if legacy_key in payload:
            payload.pop(legacy_key, None)
            changed = True

    if changed and not dry_run:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate V3 presets to strict explicit phase format.")
    parser.add_argument("--presets-dir", type=Path, default=Path("presets_v3"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    presets_dir = args.presets_dir
    if not presets_dir.exists():
        raise FileNotFoundError(f"Preset directory not found: {presets_dir}")

    total = 0
    changed = 0
    for path in sorted(presets_dir.glob("*.json")):
        total += 1
        if migrate_preset(path, dry_run=args.dry_run):
            changed += 1
            print(f"[UPDATED] {path}")
        else:
            print(f"[SKIP] {path}")
    print(f"Done. Total={total}, changed={changed}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
