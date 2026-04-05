from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "sample_id": "sample_v3",
    "composition_wt": {"Fe": 99.5, "C": 0.5},
    "system_hint": None,
    "prep_route": {
        "steps": [],
        "roughness_target_um": 0.05,
        "relief_mode": "hardness_coupled",
        "contamination_level": 0.0,
    },
    "etch_profile": {
        "reagent": "nital_2",
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
        "profile_id": "textbook_steel_bw",
        "phase_topology_mode": "auto",
        "system_generator_mode": "system_auto",
        "contrast_target": 1.0,
        "boundary_sharpness": 1.0,
        "artifact_level": 0.25,
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
        "magnification": 200,
        "focus": 0.95,
        "brightness": 1.0,
        "contrast": 1.1,
    },
    "seed": 42,
    "resolution": [2048, 2048],
    "strict_validation": True,
    "reference_profile_id": None,
}


def _deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


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


def _default_thermal_program() -> dict[str, Any]:
    return {
        "points": [
            {"time_s": 0.0, "temperature_c": 20.0, "label": "Start", "locked": True},
            {"time_s": 300.0, "temperature_c": 840.0, "label": "Heat", "locked": False},
            {"time_s": 480.0, "temperature_c": 840.0, "label": "Hold", "locked": False},
            {"time_s": 900.0, "temperature_c": 20.0, "label": "Cool", "locked": False},
        ],
        "quench": {
            "medium_code": "water_20",
            "quench_time_s": 35.0,
            "bath_temperature_c": 20.0,
            "sample_temperature_c": 840.0,
            "custom_medium_name": "",
            "custom_severity_factor": 1.0,
        },
        "sampling_mode": "per_degree",
        "degree_step_c": 1.0,
        "max_frames": 320,
    }


def _canonical_medium(code: str, bath_t: float) -> str:
    token = str(code or "").strip().lower()
    if token == "water":
        return "water_100" if float(bath_t) >= 70.0 else "water_20"
    mapping = {
        "water_20": "water_20",
        "water_100": "water_100",
        "brine": "brine_20_30",
        "brine_20_30": "brine_20_30",
        "oil": "oil_20_80",
        "oil_20_80": "oil_20_80",
        "polymer": "polymer",
        "air": "air",
        "furnace": "furnace",
        "custom": "custom",
    }
    return mapping.get(token, "water_20")


def _ascii_label(label: Any, index: int) -> str:
    raw = str(label or "").strip()
    if raw and all(ord(ch) < 128 for ch in raw):
        return raw
    if index == 0:
        return "Start"
    return f"Point {index}"


def _route_to_thermal(route: Any) -> dict[str, Any]:
    if not isinstance(route, dict):
        return _default_thermal_program()
    operations = route.get("operations", [])
    if not isinstance(operations, list) or not operations:
        return _default_thermal_program()

    points: list[dict[str, Any]] = [{"time_s": 0.0, "temperature_c": 20.0, "label": "Start", "locked": True}]
    t_cursor = 0.0
    medium = "water_20"
    bath_t = 20.0
    quench_time_s = 35.0
    sample_t = 840.0

    for idx, op in enumerate(operations, start=1):
        if not isinstance(op, dict):
            continue
        target_t = _safe_float(op.get("temperature_c"), 20.0)
        dur_min = max(0.1, _safe_float(op.get("duration_min"), 30.0))
        dur_s = dur_min * 60.0

        t_cursor += max(5.0, dur_s * 0.3)
        points.append(
            {
                "time_s": float(t_cursor),
                "temperature_c": float(target_t),
                "label": f"Step {idx}: heat",
                "locked": False,
            }
        )
        t_cursor += max(5.0, dur_s * 0.7)
        points.append(
            {
                "time_s": float(t_cursor),
                "temperature_c": float(target_t),
                "label": f"Step {idx}: hold",
                "locked": False,
            }
        )

        method = str(op.get("method", "")).strip().lower()
        cool_mode = str(op.get("cooling_mode", "")).strip().lower()
        is_water_quench = method == "quench_water" or cool_mode in {"quench", "quenched", "water_quench"}
        is_oil_quench = method == "quench_oil"
        is_slow = cool_mode in {"equilibrium", "slow_cool", "normalized"} or method in {
            "normalize",
            "anneal_full",
            "anneal_recrystallization",
            "cast_slow",
            "cast_fast",
            "directional_solidification",
        }

        if is_water_quench:
            medium = "water_20"
            bath_t = 20.0
            sample_t = max(sample_t, target_t)
            quench_time_s = max(15.0, min(240.0, dur_s * 0.2))
            cool_to = bath_t
            cool_dur_s = quench_time_s
        elif is_oil_quench:
            medium = "oil_20_80"
            bath_t = 60.0
            sample_t = max(sample_t, target_t)
            quench_time_s = max(35.0, min(500.0, dur_s * 0.35))
            cool_to = bath_t
            cool_dur_s = quench_time_s
        elif is_slow:
            cool_to = 20.0
            cool_dur_s = max(180.0, min(3600.0, dur_s * 0.9))
        else:
            cool_to = 20.0
            cool_dur_s = max(90.0, min(2400.0, dur_s * 0.6))

        t_cursor += cool_dur_s
        points.append(
            {
                "time_s": float(t_cursor),
                "temperature_c": float(cool_to),
                "label": f"Step {idx}: cool",
                "locked": False,
            }
        )

    if len(points) < 2:
        return _default_thermal_program()

    points = sorted(points, key=lambda item: float(item.get("time_s", 0.0)))
    dedup: list[dict[str, Any]] = []
    for point in points:
        if dedup and abs(_safe_float(point.get("time_s"), 0.0) - _safe_float(dedup[-1].get("time_s"), 0.0)) < 1e-9:
            dedup[-1] = point
        else:
            dedup.append(point)
    points = dedup
    if len(points) < 2:
        return _default_thermal_program()

    return {
        "points": points,
        "quench": {
            "medium_code": medium,
            "quench_time_s": float(quench_time_s),
            "bath_temperature_c": float(bath_t),
            "sample_temperature_c": float(sample_t),
            "custom_medium_name": "",
            "custom_severity_factor": 1.0,
        },
        "sampling_mode": "per_degree",
        "degree_step_c": 1.0,
        "max_frames": 320,
    }


def _normalize_preset(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    changed: list[str] = []
    out = _deep_copy_json(payload if isinstance(payload, dict) else {})

    for key in ("thermo", "process_route", "deprecated_v3", "legacy_note"):
        if key in out:
            out.pop(key, None)
            changed.append(f"removed:{key}")

    if "thermal_program" not in out:
        route = payload.get("process_route") if isinstance(payload, dict) else None
        out["thermal_program"] = _route_to_thermal(route)
        changed.append("added:thermal_program")

    tp = out.get("thermal_program")
    if not isinstance(tp, dict):
        out["thermal_program"] = _default_thermal_program()
        changed.append("fixed:thermal_program_type")
    else:
        quench = tp.get("quench", {})
        if not isinstance(quench, dict):
            quench = {}
        bath_t = _safe_float(quench.get("bath_temperature_c"), 20.0)
        medium = _canonical_medium(str(quench.get("medium_code", "water_20")), bath_t)
        quench["medium_code"] = medium
        quench["quench_time_s"] = _safe_float(quench.get("quench_time_s"), 35.0)
        quench["bath_temperature_c"] = bath_t
        quench["sample_temperature_c"] = _safe_float(quench.get("sample_temperature_c"), 840.0)
        quench["custom_medium_name"] = str(quench.get("custom_medium_name", ""))
        quench["custom_severity_factor"] = _safe_float(quench.get("custom_severity_factor"), 1.0)
        tp["quench"] = quench
        tp["sampling_mode"] = str(tp.get("sampling_mode", "per_degree"))
        tp["degree_step_c"] = _safe_float(tp.get("degree_step_c"), 1.0)
        tp["max_frames"] = _safe_int(tp.get("max_frames"), 320)
        points = tp.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            tp["points"] = _default_thermal_program()["points"]
            changed.append("fixed:thermal_points")
        else:
            cleaned_points: list[dict[str, Any]] = []
            for i, point in enumerate(points):
                if not isinstance(point, dict):
                    continue
                cleaned = dict(point)
                cleaned["label"] = _ascii_label(cleaned.get("label", ""), i)
                cleaned["time_s"] = _safe_float(cleaned.get("time_s"), float(i))
                cleaned["temperature_c"] = _safe_float(cleaned.get("temperature_c"), 20.0)
                cleaned["locked"] = bool(cleaned.get("locked", False))
                cleaned_points.append(cleaned)
            if len(cleaned_points) >= 2:
                tp["points"] = cleaned_points
        out["thermal_program"] = tp

    synth = out.get("synthesis_profile")
    if not isinstance(synth, dict):
        synth = _deep_copy_json(DEFAULTS["synthesis_profile"])
        out["synthesis_profile"] = synth
        changed.append("fixed:synthesis_profile")
    mode = str(synth.get("phase_topology_mode", "auto")).strip().lower()
    if mode.startswith("v2_"):
        synth["phase_topology_mode"] = "auto"
        changed.append("fixed:phase_topology_mode")

    for key, default in DEFAULTS.items():
        if key not in out:
            out[key] = _deep_copy_json(default)
            changed.append(f"added:{key}")

    return out, changed


def _has_legacy_keys(payload: dict[str, Any]) -> bool:
    if "thermo" in payload or "process_route" in payload:
        return True
    synth = payload.get("synthesis_profile", {})
    if isinstance(synth, dict):
        topo = str(synth.get("phase_topology_mode", ""))
        if topo.startswith("v2_"):
            return True
    return "thermal_program" not in payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize presets_v3 to strict V3 format.")
    parser.add_argument("--presets-dir", default="presets_v3")
    parser.add_argument("--apply", action="store_true", help="Rewrite files in-place")
    parser.add_argument("--check", action="store_true", help="Only validate (default mode)")
    args = parser.parse_args()

    root = Path(args.presets_dir)
    if not root.exists():
        print(f"[ERROR] presets dir not found: {root}")
        return 2

    files = sorted(root.glob("*.json"))
    if not files:
        print(f"[WARN] no preset files in: {root}")
        return 0

    apply_mode = bool(args.apply)
    failures: list[str] = []
    changed_total = 0

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                failures.append(f"{path.name}: payload is not JSON object")
                continue
            normalized, changed = _normalize_preset(payload)
            legacy = _has_legacy_keys(normalized)
            if legacy:
                failures.append(f"{path.name}: legacy keys still present after normalize")
                continue

            if apply_mode and (changed or normalized != payload):
                path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                changed_total += 1
                print(f"[UPDATED] {path.name}: {', '.join(changed) if changed else 'normalized'}")
            else:
                print(f"[OK] {path.name}")
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")

    if failures:
        print("\n[FAIL]")
        for row in failures:
            print(f"- {row}")
        return 1

    if apply_mode:
        print(f"\n[DONE] updated files: {changed_total}")
    else:
        print("\n[DONE] strict check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
