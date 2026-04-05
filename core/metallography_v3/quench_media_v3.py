from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_RULES_PATH = Path(__file__).resolve().parents[1] / "rulebook" / "quench_media_rules_v3.json"


def _load_rules() -> dict[str, Any]:
    if not _RULES_PATH.exists():
        return {}
    try:
        return json.loads(_RULES_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


_RULES = _load_rules()
_MEDIA = _RULES.get("media", {}) if isinstance(_RULES.get("media"), dict) else {}
_DEFAULTS = _RULES.get("defaults", {}) if isinstance(_RULES.get("defaults"), dict) else {}
_LIMITS = _RULES.get("limits", {}) if isinstance(_RULES.get("limits"), dict) else {}
_ALIASES = {str(k).strip().lower(): str(v).strip().lower() for k, v in dict(_RULES.get("aliases", {})).items()}
_UI_ORDER = [str(x).strip().lower() for x in list(_RULES.get("ui_order", [])) if str(x).strip()]
_LEGACY_MAPPING = _RULES.get("legacy_mapping", {}) if isinstance(_RULES.get("legacy_mapping"), dict) else {}


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _limit(key: str, fallback: tuple[float, float]) -> tuple[float, float]:
    raw = _LIMITS.get(key, list(fallback))
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    return float(fallback[0]), float(fallback[1])


def _profile(code: str) -> dict[str, Any]:
    payload = _MEDIA.get(code)
    if isinstance(payload, dict):
        return payload
    return {}


def _legacy_water_split_c() -> float:
    payload = _LEGACY_MAPPING.get("water")
    if not isinstance(payload, dict):
        return 70.0
    try:
        return float(payload.get("bath_temperature_split_c", 70.0))
    except Exception:
        return 70.0


def canonicalize_quench_medium_code(medium_code: str, bath_temperature_c: float | None = None) -> dict[str, Any]:
    raw = str(medium_code or "").strip().lower()
    if not raw:
        raw = str(_DEFAULTS.get("medium_code", "water_20")).strip().lower()

    mapping_note = ""
    legacy_mapping_applied = False
    resolved = raw

    if raw in _ALIASES:
        resolved = str(_ALIASES.get(raw, raw)).strip().lower()

    if resolved in {"water", "water_auto"}:
        split = _legacy_water_split_c()
        bath = float(bath_temperature_c) if bath_temperature_c is not None else float(_DEFAULTS.get("bath_temperature_c", 20.0))
        resolved = "water_100" if bath >= split else "water_20"
        legacy_mapping_applied = True
        mapping_note = "legacy_water_mapped_by_bath_temperature"
    elif raw == "water":
        split = _legacy_water_split_c()
        bath = float(bath_temperature_c) if bath_temperature_c is not None else float(_DEFAULTS.get("bath_temperature_c", 20.0))
        resolved = "water_100" if bath >= split else "water_20"
        legacy_mapping_applied = True
        mapping_note = "legacy_water_mapped_by_bath_temperature"

    if resolved not in _MEDIA:
        fallback = str(_DEFAULTS.get("medium_code", "water_20")).strip().lower()
        if fallback not in _MEDIA:
            fallback = next(iter(_MEDIA.keys()), "custom")
        mapping_note = "unknown_medium_fallback" if not mapping_note else mapping_note
        legacy_mapping_applied = True
        resolved = fallback

    return {
        "input_code": raw,
        "resolved_code": resolved,
        "legacy_mapping_applied": bool(legacy_mapping_applied),
        "mapping_note": str(mapping_note),
    }


def list_quench_media() -> list[dict[str, Any]]:
    ordered_codes: list[str] = []
    seen: set[str] = set()
    for code in _UI_ORDER:
        if code in _MEDIA and code not in seen:
            ordered_codes.append(code)
            seen.add(code)
    for code in sorted(_MEDIA.keys()):
        if code not in seen:
            ordered_codes.append(code)
            seen.add(code)

    out: list[dict[str, Any]] = []
    for code in ordered_codes:
        payload = _profile(code)
        out.append(
            {
                "code": str(code),
                "label_ru": str(payload.get("label_ru", code)),
                "severity_base": float(payload.get("severity_base", 1.0)),
                "cooling_rate_ref_c_per_s": float(payload.get("cooling_rate_ref_c_per_s", 50.0)),
                "typical_bath_temperature_c": float(payload.get("typical_bath_temperature_c", _DEFAULTS.get("bath_temperature_c", 20.0))),
            }
        )
    return out


def defaults_quench() -> dict[str, Any]:
    medium_default = str(_DEFAULTS.get("medium_code", "water_20")).strip().lower() or "water_20"
    if medium_default not in _MEDIA:
        medium_default = "water_20" if "water_20" in _MEDIA else (next(iter(_MEDIA.keys()), "custom"))
    return {
        "medium_code": medium_default,
        "quench_time_s": float(_DEFAULTS.get("quench_time_s", 30.0)),
        "bath_temperature_c": float(_DEFAULTS.get("bath_temperature_c", 20.0)),
        "sample_temperature_c": float(_DEFAULTS.get("sample_temperature_c", 840.0)),
        "custom_medium_name": "",
        "custom_severity_factor": float(_DEFAULTS.get("custom_severity_factor", 1.0)),
    }


def _interp_range(min_v: float, max_v: float, index01: float, inverse: bool = False) -> float:
    t = _clamp(float(index01), 0.0, 1.0)
    if inverse:
        t = 1.0 - t
    return float(min_v + (max_v - min_v) * t)


def resolve_quench_medium(
    medium_code: str,
    *,
    quench_time_s: float,
    bath_temperature_c: float,
    sample_temperature_c: float,
    custom_medium_name: str = "",
    custom_severity_factor: float = 1.0,
) -> dict[str, Any]:
    defaults = defaults_quench()

    time_low, time_high = _limit("quench_time_s", (0.1, 36000.0))
    bath_low, bath_high = _limit("bath_temperature_c", (-50.0, 300.0))
    sample_low, sample_high = _limit("sample_temperature_c", (20.0, 1700.0))
    sev_low, sev_high = _limit("custom_severity_factor", (0.05, 5.0))

    t_quench = _clamp(float(quench_time_s), time_low, time_high)
    t_bath = _clamp(float(bath_temperature_c), bath_low, bath_high)
    t_sample = _clamp(float(sample_temperature_c), sample_low, sample_high)
    custom_factor = _clamp(float(custom_severity_factor), sev_low, sev_high)

    canonical = canonicalize_quench_medium_code(medium_code=medium_code, bath_temperature_c=t_bath)
    code = str(canonical.get("resolved_code", defaults["medium_code"]))
    profile = _profile(code)

    severity_base_raw = float(profile.get("severity_base", 1.0))
    # Custom severity factor acts as user-controlled multiplier for all media.
    severity_base = float(severity_base_raw * custom_factor)
    cooling_ref = float(profile.get("cooling_rate_ref_c_per_s", 50.0))

    delta_t = max(0.0, t_sample - t_bath)
    time_factor = _clamp((t_quench / 30.0) ** 0.22, 0.45, 1.7)
    delta_factor = _clamp((delta_t / 800.0) ** 0.5, 0.5, 1.5)
    typical_bath = float(profile.get("typical_bath_temperature_c", t_bath))
    bath_temp_factor = _clamp(1.0 - (t_bath - typical_bath) * 0.004, 0.55, 1.25)
    effective_severity = _clamp(severity_base * time_factor * delta_factor * bath_temp_factor, 0.05, 5.0)

    sev_ratio = effective_severity / max(1e-6, severity_base_raw)
    cooling_rate_effective = float(max(0.0, cooling_ref * sev_ratio))

    cool_min = float(profile.get("cooling_rate_800_400_min", cooling_ref * 0.75))
    cool_max = float(profile.get("cooling_rate_800_400_max", cooling_ref * 1.25))
    hard_min = float(profile.get("hardness_hrc_as_quenched_min", 45.0))
    hard_max = float(profile.get("hardness_hrc_as_quenched_max", 62.0))
    stress_min = float(profile.get("stress_mpa_min", 250.0))
    stress_max = float(profile.get("stress_mpa_max", 1000.0))
    depth_min = float(profile.get("harden_depth_mm_min", 10.0))
    depth_max = float(profile.get("harden_depth_mm_max", 60.0))
    ra_min = float(profile.get("retained_austenite_pct_min", 8.0))
    ra_max = float(profile.get("retained_austenite_pct_max", 40.0))

    cooling_index = _clamp((cooling_rate_effective - cool_min) / max(1e-6, cool_max - cool_min), 0.0, 1.0)

    medium_martensite_bias = {
        "brine_20_30": 0.1,
        "water_20": 0.06,
        "water_100": -0.08,
        "oil_20_80": -0.12,
        "polymer": -0.04,
        "air": -0.18,
        "furnace": -0.22,
    }
    martensite_completion = _clamp(cooling_index + float(medium_martensite_bias.get(code, 0.0)), 0.0, 1.0)

    retained_austenite_est = _interp_range(ra_min, ra_max, martensite_completion, inverse=True)
    stress_est = _interp_range(stress_min, stress_max, cooling_index)
    hardness_est = _interp_range(hard_min, hard_max, martensite_completion)
    depth_est = _interp_range(depth_min, depth_max, cooling_index)

    if martensite_completion >= 0.72:
        martensite_type = "martensite_tetragonal"
    elif martensite_completion >= 0.45:
        martensite_type = "martensite_cubic"
    elif martensite_completion >= 0.28:
        martensite_type = "troostite_quench"
    else:
        martensite_type = "sorbite_quench"

    temper_shift = {
        "low": float(profile.get("temper_shift_c_low", 0.0)),
        "medium": float(profile.get("temper_shift_c_medium", 0.0)),
        "high": float(profile.get("temper_shift_c_high", 0.0)),
    }

    warnings: list[str] = []
    if code == "custom":
        warnings.append("Пользовательская среда: прогноз оценочный, достоверность ниже справочных профилей.")
    if bool(canonical.get("legacy_mapping_applied", False)) and str(canonical.get("mapping_note", "")):
        warnings.append(f"Код среды нормализован: {canonical.get('mapping_note')}")

    return {
        "medium_code_input": str(canonical.get("input_code", "")),
        "medium_code_resolved": code,
        "medium_code": code,
        "label_ru": str(profile.get("label_ru", code)),
        "legacy_mapping_applied": bool(canonical.get("legacy_mapping_applied", False)),
        "mapping_note": str(canonical.get("mapping_note", "")),
        "quench_time_s": t_quench,
        "bath_temperature_c": t_bath,
        "sample_temperature_c": t_sample,
        "delta_t_c": float(delta_t),
        "severity_base": severity_base,
        "severity_effective": float(effective_severity),
        "cooling_rate_ref_c_per_s": float(cooling_ref),
        "cooling_rate_effective_c_per_s": float(cooling_rate_effective),
        "cooling_rate_800_400_min": float(cool_min),
        "cooling_rate_800_400_max": float(cool_max),
        "cooling_rate_band_800_400": [float(cool_min), float(cool_max)],
        "hardness_hrc_as_quenched_min": float(hard_min),
        "hardness_hrc_as_quenched_max": float(hard_max),
        "hardness_hrc_as_quenched_range": [float(hard_min), float(hard_max)],
        "stress_mpa_min": float(stress_min),
        "stress_mpa_max": float(stress_max),
        "stress_mpa_range": [float(stress_min), float(stress_max)],
        "harden_depth_mm_min": float(depth_min),
        "harden_depth_mm_max": float(depth_max),
        "harden_depth_mm_range": [float(depth_min), float(depth_max)],
        "retained_austenite_pct_min": float(ra_min),
        "retained_austenite_pct_max": float(ra_max),
        "retained_austenite_pct_range": [float(ra_min), float(ra_max)],
        "retained_austenite_est_pct": float(retained_austenite_est),
        "stress_est_mpa": float(stress_est),
        "hardness_est_hrc": float(hardness_est),
        "harden_depth_est_mm": float(depth_est),
        "martensite_completion_index": float(martensite_completion),
        "martensite_type": str(martensite_type),
        "defect_risk": str(profile.get("defect_risk", "не определен")),
        "low_temper_required": bool(profile.get("low_temper_required", False)),
        "recommended_temper_hold_s": float(profile.get("recommended_temper_hold_s", 3000.0)),
        "temper_shift_c": dict(temper_shift),
        "temper_shift_c_low": float(temper_shift["low"]),
        "temper_shift_c_medium": float(temper_shift["medium"]),
        "temper_shift_c_high": float(temper_shift["high"]),
        "custom_medium_name": str(custom_medium_name or ""),
        "custom_severity_factor": custom_factor,
        "as_quenched_prediction": {
            "martensite_fraction_est": float(_clamp(martensite_completion * 0.92 + 0.04, 0.0, 1.0)),
            "retained_austenite_fraction_est": float(_clamp(retained_austenite_est / 100.0, 0.0, 1.0)),
            "martensite_type": str(martensite_type),
        },
        "operation_guidance": {
            "low_temper_required": bool(profile.get("low_temper_required", False)),
            "recommended_hold_s": float(profile.get("recommended_temper_hold_s", 3000.0)),
        },
        "warnings": warnings,
    }
