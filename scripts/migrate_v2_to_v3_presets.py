from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _default_prep_template(system: str) -> dict[str, Any]:
    return {
        "steps": [
            {"method": "grinding_800", "duration_s": 90.0, "abrasive_um": 18.0, "load_n": 22.0, "rpm": 180.0},
            {"method": "polishing_3um", "duration_s": 110.0, "abrasive_um": 3.0, "load_n": 14.0, "rpm": 140.0},
            {"method": "polishing_1um", "duration_s": 90.0, "abrasive_um": 1.0, "load_n": 10.0, "rpm": 120.0},
        ],
        "roughness_target_um": 0.06 if system.startswith("al-") else 0.05,
        "relief_mode": "phase_coupled" if system.startswith("al-") else "hardness_coupled",
        "contamination_level": 0.02,
    }


def _infer_system_hint(composition: dict[str, Any], generation: dict[str, Any]) -> str:
    hinted = str(generation.get("system", "")).strip().lower()
    if hinted:
        return hinted
    keys = {str(k).upper() for k in composition.keys()}
    if {"FE", "C"} <= keys:
        return "fe-c"
    if {"FE", "SI"} <= keys:
        return "fe-si"
    if {"AL", "SI"} <= keys:
        return "al-si"
    if {"CU", "ZN"} <= keys:
        return "cu-zn"
    if "AL" in keys and ("CU" in keys or "MG" in keys):
        return "al-cu-mg"
    return "custom-multicomponent"


def _thermal_from_v2(generation: dict[str, Any]) -> dict[str, Any]:
    route = generation.get("process_route", {})
    operations = route.get("operations", []) if isinstance(route, dict) else []
    points: list[dict[str, Any]] = [{"time_s": 0.0, "temperature_c": 20.0, "label": "Start", "locked": True}]
    t_cursor = 0.0
    medium_code = "water_20"
    bath_temperature_c = 20.0
    sample_temperature_c = 840.0
    quench_time_s = 35.0

    if not isinstance(operations, list):
        operations = []
    if not operations:
        temp = _safe_float(generation.get("temperature_c"), 780.0)
        mode = str(generation.get("cooling_mode", "equilibrium")).strip().lower()
        method = str(generation.get("generator", "normalize")).strip().lower()
        operations = [
            {
                "method": method,
                "temperature_c": temp,
                "duration_min": _safe_float(generation.get("duration_min"), 30.0),
                "cooling_mode": mode,
            }
        ]

    for idx, op in enumerate(operations, start=1):
        if not isinstance(op, dict):
            continue
        target_t = _safe_float(op.get("temperature_c"), 20.0)
        dur_min = max(0.1, _safe_float(op.get("duration_min"), 30.0))
        dur_s = dur_min * 60.0
        method = str(op.get("method", "")).strip().lower()
        cooling_mode = str(op.get("cooling_mode", "")).strip().lower()

        t_cursor += max(5.0, dur_s * 0.3)
        points.append({"time_s": t_cursor, "temperature_c": target_t, "label": f"Step {idx}: heat", "locked": False})
        t_cursor += max(5.0, dur_s * 0.7)
        points.append({"time_s": t_cursor, "temperature_c": target_t, "label": f"Step {idx}: hold", "locked": False})

        is_water_quench = method == "quench_water" or cooling_mode in {"quenched", "quench", "water_quench"}
        is_oil_quench = method == "quench_oil"
        if is_water_quench:
            medium_code = "water_20"
            bath_temperature_c = 20.0
            sample_temperature_c = max(sample_temperature_c, target_t)
            quench_time_s = max(20.0, min(240.0, dur_s * 0.25))
            cool_to = bath_temperature_c
            cool_dur = quench_time_s
        elif is_oil_quench:
            medium_code = "oil_20_80"
            bath_temperature_c = 60.0
            sample_temperature_c = max(sample_temperature_c, target_t)
            quench_time_s = max(40.0, min(420.0, dur_s * 0.35))
            cool_to = bath_temperature_c
            cool_dur = quench_time_s
        else:
            cool_to = 20.0
            cool_dur = max(120.0, min(3600.0, dur_s * 0.75))

        t_cursor += cool_dur
        points.append({"time_s": t_cursor, "temperature_c": cool_to, "label": f"Step {idx}: cool", "locked": False})

    return {
        "points": points,
        "quench": {
            "medium_code": medium_code,
            "quench_time_s": quench_time_s,
            "bath_temperature_c": bath_temperature_c,
            "sample_temperature_c": sample_temperature_c,
            "custom_medium_name": "",
            "custom_severity_factor": 1.0,
        },
        "sampling_mode": "per_degree",
        "degree_step_c": 1.0,
        "max_frames": 320,
    }


def convert_v2_to_v3(payload: dict[str, Any], stem: str) -> dict[str, Any]:
    composition = payload.get("composition", {}) if isinstance(payload.get("composition"), dict) else {}
    generation = payload.get("generation", {}) if isinstance(payload.get("generation"), dict) else {}
    microscope = payload.get("microscope", {}) if isinstance(payload.get("microscope"), dict) else {}
    system_hint = _infer_system_hint(composition, generation)

    etch_map = {
        "fe-c": "nital_2",
        "fe-si": "nital_2",
        "al-si": "keller",
        "al-cu-mg": "keller",
        "cu-zn": "picral",
    }
    etch = etch_map.get(system_hint, "nital_2")
    synth_profile = "textbook_alsi_bw" if system_hint.startswith("al-") else "textbook_steel_bw"
    image_size = payload.get("image_size", [1024, 1024])
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        image_size = [1024, 1024]

    return {
        "sample_id": stem,
        "composition_wt": composition,
        "system_hint": system_hint,
        "thermal_program": _thermal_from_v2(generation),
        "prep_route": _default_prep_template(system_hint),
        "etch_profile": {
            "reagent": etch,
            "time_s": 8.0,
            "temperature_c": 22.0,
            "agitation": "gentle",
            "overetch_factor": 1.0,
            "concentration_value": 2.0,
            "concentration_unit": "wt_pct",
            "concentration_wt_pct": 2.0,
            "concentration_mol_l": 0.4,
        },
        "synthesis_profile": {
            "profile_id": synth_profile,
            "phase_topology_mode": "auto",
            "system_generator_mode": "system_auto",
            "contrast_target": 1.0,
            "boundary_sharpness": 1.1,
            "artifact_level": 0.3,
            "composition_sensitivity_mode": "realistic",
            "generation_mode": "edu_engineering",
            "phase_emphasis_style": "contrast_texture",
            "phase_fraction_tolerance_pct": 20.0,
        },
        "phase_model": {
            "engine": "explicit_rules_v3",
            "phase_control_mode": "auto_with_override",
            "manual_phase_fractions": {},
            "manual_override_weight": 0.35,
            "allow_custom_fallback": True,
            "phase_balance_tolerance_pct": 20.0,
        },
        "microscope_profile": {
            "simulate_preview": False,
            "magnification": _safe_int(microscope.get("magnification"), 200),
            "focus": _safe_float(microscope.get("focus"), 0.95),
            "brightness": _safe_float(microscope.get("brightness"), 1.0),
            "contrast": _safe_float(microscope.get("contrast"), 1.1),
        },
        "seed": _safe_int(payload.get("seed"), 42),
        "resolution": [int(image_size[0]), int(image_size[1])],
        "strict_validation": True,
        "reference_profile_id": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate V2 presets to strict V3 format")
    parser.add_argument("--src", type=Path, default=ROOT / "presets", help="V2 presets folder")
    parser.add_argument("--dst", type=Path, default=ROOT / "presets_v3", help="V3 presets folder")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in dst")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*.json"))
    if not files:
        raise SystemExit("No source presets found")

    converted = 0
    skipped = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            skipped += 1
            continue
        out_payload = convert_v2_to_v3(payload, path.stem)
        out_path = args.dst / f"{path.stem}_v3.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        converted += 1

    print(f"Converted: {converted}, skipped: {skipped}, dst={args.dst}")


if __name__ == "__main__":
    main()

