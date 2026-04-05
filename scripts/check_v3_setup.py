from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts_v3 import MetallographyRequestV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check V3 explicit-phase setup (strict format, no legacy fields)")
    parser.add_argument(
        "--profiles-v3",
        type=Path,
        default=ROOT / "profiles_v3",
        help="Path to profiles_v3 directory",
    )
    parser.add_argument(
        "--presets-v3",
        type=Path,
        default=ROOT / "presets_v3",
        help="Path to presets_v3 directory",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="",
        help="Optional preset file name to test (without path). If empty, the first preset is used.",
    )
    return parser.parse_args()


def _load_payload(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return raw


def main() -> None:
    args = parse_args()
    required_profiles = [
        args.profiles_v3 / "metallography_profiles.json",
        args.profiles_v3 / "etch_profiles.json",
        args.profiles_v3 / "prep_templates.json",
    ]
    missing = [p for p in required_profiles if not p.exists()]
    if missing:
        for p in missing:
            print(f"[FAIL] missing profile file: {p}")
        raise SystemExit(1)

    if not args.presets_v3.exists():
        print(f"[FAIL] presets dir not found: {args.presets_v3}")
        raise SystemExit(1)
    preset_paths = sorted(args.presets_v3.glob("*.json"))
    if not preset_paths:
        print(f"[FAIL] no presets in: {args.presets_v3}")
        raise SystemExit(1)

    preset_path = args.presets_v3 / args.preset if args.preset else preset_paths[0]
    if not preset_path.exists():
        print(f"[FAIL] preset not found: {preset_path}")
        raise SystemExit(1)

    payload = _load_payload(preset_path)
    req = MetallographyRequestV3.from_dict(payload)
    pipe = MetallographyPipelineV3(presets_dir=args.presets_v3, profiles_dir=args.profiles_v3)
    out = pipe.generate(req)
    meta = dict(out.metadata)

    print(f"[OK] preset: {preset_path.name}")
    print(f"[OK] inferred_system: {meta.get('inferred_system', '')}")
    print(f"[OK] final_stage: {meta.get('final_stage', '')}")
    print(f"[OK] system_generator: {dict(meta.get('system_generator', {})).get('resolved_mode', '')}")
    print(f"[OK] quality_passed: {bool(dict(meta.get('quality_metrics', {})).get('passed', False))}")


if __name__ == "__main__":
    main()

