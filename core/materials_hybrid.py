from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .heat_treatment_calculator import (
    SteelComposition,
    calculate_ac1,
    calculate_ac3,
    calculate_mf_temperature,
    calculate_ms_temperature,
    calculate_phase_fractions_fe_c,
    estimate_hardenability,
    get_quench_temperature,
    get_tempering_temperature,
)
from .mechanical_properties_calculator import (
    calculate_astm_grain_size_number,
    calculate_properties_from_microstructure,
    get_material_grade_properties,
)

_RULEBOOK_DIR = Path(__file__).resolve().parent / "rulebook"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_TEXTBOOK_DATA = _load_json(_RULEBOOK_DIR / "textbook_material_properties.json")
_MODEL_SOURCE = "hybrid_textbook_calculator_v1"
_REFERENCE_DATASET = "textbook_material_properties"

_TEMPERED_STAGE_PHASE = {
    "tempered_low": {"martensite_tempered_low": 1.0},
    "tempered_medium": {"martensite_tempered_medium": 1.0},
    "tempered_high": {"martensite_tempered_high": 1.0},
}

_DIRECT_STAGE_PHASE = {
    "martensite": {"martensite": 1.0},
    "martensite_tetragonal": {"martensite": 1.0},
    "martensite_cubic": {"martensite": 1.0},
    "bainite_upper": {"bainite_upper": 1.0},
    "bainite_lower": {"bainite_lower": 1.0},
    "ferrite": {"ferrite": 1.0},
    "austenite": {"austenite": 1.0},
    "cementite": {"cementite": 1.0},
    "cementite_network": {"cementite": 1.0},
    "ledeburite": {"ledeburite": 1.0},
    **_TEMPERED_STAGE_PHASE,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _midpoint(min_v: Any, max_v: Any) -> float:
    return 0.5 * (_safe_float(min_v) + _safe_float(max_v))


def _ductility_label_from_elongation(elongation_pct: float) -> str:
    value = _safe_float(elongation_pct)
    if value >= 25.0:
        return "high"
    if value >= 15.0:
        return "medium-high"
    if value >= 8.0:
        return "medium"
    if value >= 4.0:
        return "medium-low"
    return "low"


def _ductility_label_from_hv(
    hv: float, overlay_rules: dict[str, Any], default: str
) -> str:
    thresholds = overlay_rules.get("ductility_thresholds_hv", [])
    if not isinstance(thresholds, list):
        return str(default)
    for threshold in thresholds:
        if not isinstance(threshold, dict):
            continue
        if hv <= _safe_float(threshold.get("max_hv", 9999.0), 9999.0):
            return str(threshold.get("label", default))
    return str(default)


def supports_hybrid_properties(
    inferred_system: str, composition: dict[str, float]
) -> bool:
    system = str(inferred_system or "").strip().lower()
    if system in {"fe-c", "fe-c-cast-iron"}:
        return True
    fe = _safe_float((composition or {}).get("Fe", 0.0))
    c = _safe_float((composition or {}).get("C", 0.0))
    return (fe > 0.0 and c > 0.0) or c >= 2.0


def _extract_mechanical_properties(payload: dict[str, Any]) -> dict[str, float] | None:
    hardness = payload.get("hardness", {}) if isinstance(payload, dict) else {}
    tensile = payload.get("tensile", {}) if isinstance(payload, dict) else {}
    hb = hardness.get("hb")
    uts = tensile.get("uts_mpa")
    elongation = tensile.get("total_elongation_pct")
    if hb is None or uts is None or elongation is None:
        return None
    return {
        "hb": _safe_float(hb),
        "uts_mpa": _safe_float(uts),
        "elongation_pct": _safe_float(elongation),
    }


def _build_mechanical_mix(
    phase_fractions: dict[str, float], carbon_pct: float
) -> dict[str, float] | None:
    filtered = {
        str(k): _safe_float(v)
        for k, v in dict(phase_fractions or {}).items()
        if _safe_float(v) > 0.0
    }
    if not filtered:
        return None
    payload = calculate_properties_from_microstructure(
        filtered, carbon_pct=_safe_float(carbon_pct)
    )
    return _extract_mechanical_properties(payload)


def _interpolate_carbon_table(carbon_pct: float) -> dict[str, float] | None:
    rows = list(
        _TEXTBOOK_DATA.get("mechanical_properties_by_carbon", {}).get("data", [])
    )
    if not rows:
        return None
    rows = sorted(
        (row for row in rows if isinstance(row, dict)),
        key=lambda row: _safe_float(row.get("carbon_pct")),
    )
    if not rows:
        return None

    c = _safe_float(carbon_pct)
    if c <= _safe_float(rows[0].get("carbon_pct")):
        row = rows[0]
        return {
            "hb": _safe_float(row.get("hb")),
            "uts_mpa": _safe_float(row.get("uts_mpa")),
            "elongation_pct": _safe_float(row.get("elongation_pct")),
        }
    if c >= _safe_float(rows[-1].get("carbon_pct")):
        row = rows[-1]
        return {
            "hb": _safe_float(row.get("hb")),
            "uts_mpa": _safe_float(row.get("uts_mpa")),
            "elongation_pct": _safe_float(row.get("elongation_pct")),
        }

    for left, right in zip(rows, rows[1:]):
        c0 = _safe_float(left.get("carbon_pct"))
        c1 = _safe_float(right.get("carbon_pct"))
        if not (c0 <= c <= c1):
            continue
        span = max(1e-9, c1 - c0)
        t = (c - c0) / span
        return {
            "hb": _safe_float(left.get("hb"))
            + (_safe_float(right.get("hb")) - _safe_float(left.get("hb"))) * t,
            "uts_mpa": _safe_float(left.get("uts_mpa"))
            + (_safe_float(right.get("uts_mpa")) - _safe_float(left.get("uts_mpa")))
            * t,
            "elongation_pct": _safe_float(left.get("elongation_pct"))
            + (
                _safe_float(right.get("elongation_pct"))
                - _safe_float(left.get("elongation_pct"))
            )
            * t,
        }
    return None


def _cast_iron_reference(
    composition: dict[str, float], final_stage: str
) -> dict[str, float] | None:
    cast_iron = _TEXTBOOK_DATA.get("cast_iron", {})
    c = _safe_float(composition.get("C", 0.0))
    si = _safe_float(composition.get("Si", 0.0))
    mg = _safe_float(composition.get("Mg", 0.0))
    stage = str(final_stage or "").strip().lower()

    if mg >= 0.03:
        ductile = cast_iron.get("ductile", {})
        return {
            "hb": _midpoint(
                ductile.get("hardness_hb", {}).get("min", 150),
                ductile.get("hardness_hb", {}).get("max", 300),
            ),
            "uts_mpa": _midpoint(
                ductile.get("tensile_strength_mpa", {}).get("min", 400),
                ductile.get("tensile_strength_mpa", {}).get("max", 900),
            ),
            "elongation_pct": _midpoint(
                ductile.get("elongation_pct", {}).get("min", 2),
                ductile.get("elongation_pct", {}).get("max", 20),
            ),
        }

    if stage == "ledeburite" or c >= 3.5:
        mix = _build_mechanical_mix(
            calculate_phase_fractions_fe_c(c, 20.0), carbon_pct=c
        )
        if mix is not None:
            return mix
        white = cast_iron.get("white", {})
        return {
            "hb": _midpoint(
                white.get("hardness_hb", {}).get("min", 450),
                white.get("hardness_hb", {}).get("max", 600),
            ),
            "uts_mpa": 400.0,
            "elongation_pct": 1.0,
        }

    if si >= 1.0 and c >= 2.5:
        grey = cast_iron.get("grey", {})
        return {
            "hb": _midpoint(
                grey.get("hardness_hb", {}).get("min", 150),
                grey.get("hardness_hb", {}).get("max", 250),
            ),
            "uts_mpa": _midpoint(
                grey.get("tensile_strength_mpa", {}).get("min", 150),
                grey.get("tensile_strength_mpa", {}).get("max", 350),
            ),
            "elongation_pct": 0.7,
        }
    return None


def _primary_reference_properties(
    composition: dict[str, float], final_stage: str
) -> dict[str, float] | None:
    stage = str(final_stage or "").strip().lower()
    c = _safe_float(composition.get("C", 0.0))
    steel_comp = SteelComposition.from_dict(
        {str(k): _safe_float(v) for k, v in dict(composition or {}).items()}
    )
    _ = steel_comp

    if c >= 2.0:
        cast_iron = _cast_iron_reference(composition=composition, final_stage=stage)
        if cast_iron is not None:
            return cast_iron

    if stage in {"alpha_pearlite", "pearlite", "pearlite_cementite"}:
        table = _interpolate_carbon_table(c)
        if table is not None:
            return table
        return _build_mechanical_mix(
            calculate_phase_fractions_fe_c(c, 20.0), carbon_pct=c
        )

    if stage in _DIRECT_STAGE_PHASE:
        return _build_mechanical_mix(_DIRECT_STAGE_PHASE[stage], carbon_pct=c)

    if stage in {"equilibrium", "normalized", "annealed", "ferrite_pearlite"}:
        table = _interpolate_carbon_table(c)
        if table is not None:
            return table
        return _build_mechanical_mix(
            calculate_phase_fractions_fe_c(c, 20.0), carbon_pct=c
        )

    return None


def estimate_hybrid_properties(
    *,
    composition: dict[str, float],
    final_stage: str,
    effect: dict[str, float],
    overlay_rules: dict[str, Any],
) -> dict[str, Any] | None:
    base = _primary_reference_properties(
        composition=composition, final_stage=final_stage
    )
    if base is None:
        return None

    hv = _safe_float(base.get("hb"))
    uts = _safe_float(base.get("uts_mpa"))
    elongation = _safe_float(base.get("elongation_pct"))
    overlay_used = False

    by_system = overlay_rules.get("systems", {}).get("fe-c", {})
    coeff = by_system.get("hv_coeff", {})
    hv_delta = 0.0
    if isinstance(coeff, dict):
        for key, value in coeff.items():
            hv_delta += _safe_float(value) * _safe_float(effect.get(key, 0.0))
    if abs(hv_delta) > 1e-9:
        hv += hv_delta
        uts += hv_delta * 1.8
        overlay_used = True

    defaults = overlay_rules.get("defaults", {})
    stage_adjust = by_system.get("stage_adjust", {})
    stage_data = (
        stage_adjust.get(str(final_stage or "").strip().lower(), {})
        if isinstance(stage_adjust, dict)
        else {}
    )
    if isinstance(stage_data, dict) and stage_data.get("ductility") is not None:
        ductility = str(stage_data.get("ductility"))
        overlay_used = True
    else:
        base_ductility = _ductility_label_from_elongation(elongation)
        ductility = _ductility_label_from_hv(
            hv,
            overlay_rules=overlay_rules,
            default=base_ductility or str(defaults.get("ductility_base", "medium")),
        )

    hv = _clamp(hv, 40.0, 900.0)
    uts = _clamp(uts, 120.0, 2600.0)

    return {
        "hv_estimate": round(float(hv), 2),
        "hardness_hv_est": round(float(hv), 2),
        "uts_estimate_mpa": round(float(uts), 2),
        "uts_mpa_est": round(float(uts), 2),
        "ductility_class": ductility,
        "model_note": "Hybrid textbook/calculator estimate with compatibility overlay.",
        "property_model_source": _MODEL_SOURCE,
        "reference_dataset": _REFERENCE_DATASET,
        "compatibility_overlay_used": bool(overlay_used),
        "fallback_used": False,
    }


def _normalize_phase_fractions(
    phase_fractions: dict[str, float] | None,
) -> dict[str, float]:
    aliases = {
        "graphite_flakes": "graphite_flake",
        "graphite_flake": "graphite_flake",
        "graphite_spheres": "graphite_spheroidal",
        "graphite_spheroidal": "graphite_spheroidal",
        "martensite_tetragonal": "martensite",
        "martensite_cubic": "martensite",
        "troostite_quench": "martensite_tempered_medium",
        "troostite_temper": "martensite_tempered_medium",
        "sorbite_quench": "martensite_tempered_high",
        "sorbite_temper": "martensite_tempered_high",
        "tempered_low": "martensite_tempered_low",
        "tempered_medium": "martensite_tempered_medium",
        "tempered_high": "martensite_tempered_high",
        "bainite": "bainite_upper",
        "bainite_upper": "bainite_upper",
        "bainite_lower": "bainite_lower",
    }
    cleaned: dict[str, float] = {}
    for key, value in dict(phase_fractions or {}).items():
        amount = _safe_float(value)
        if amount <= 0.0:
            continue
        mapped = aliases.get(
            str(key or "").strip().lower(), str(key or "").strip().lower()
        )
        cleaned[mapped] = float(cleaned.get(mapped, 0.0) + amount)
    total = sum(cleaned.values())
    if total <= 1e-12:
        return {}
    return {key: float(value / total) for key, value in cleaned.items()}


def _build_microstructure_payload(
    *,
    composition: dict[str, float],
    final_stage: str,
    phase_fractions: dict[str, float] | None,
    material_grade: str | None,
) -> dict[str, float] | None:
    normalized = _normalize_phase_fractions(phase_fractions)
    if normalized:
        return _extract_mechanical_properties(
            calculate_properties_from_microstructure(
                normalized, carbon_pct=_safe_float(composition.get("C", 0.0))
            )
        )

    if material_grade:
        grade_props = get_material_grade_properties(material_grade)
        if "error" not in grade_props:
            props = dict(grade_props.get("properties", {}))
            if "hardness" in props:
                return _extract_mechanical_properties(props)
            return {
                "hb": _safe_float(props.get("hardness_hb", 0.0)),
                "uts_mpa": _safe_float(props.get("tensile_strength_mpa", 0.0)),
                "elongation_pct": _safe_float(props.get("elongation_pct", 0.0)),
            }

    return _primary_reference_properties(
        composition=composition, final_stage=final_stage
    )


def calculate_hybrid_heat_treatment(
    *,
    composition: dict[str, float],
    material_grade: str | None = None,
) -> dict[str, Any]:
    steel = SteelComposition.from_dict(
        {str(k): _safe_float(v) for k, v in dict(composition or {}).items()}
    )
    carbon = _safe_float(composition.get("C", 0.0))
    if carbon < 0.76:
        steel_type = "hypoeutectoid"
    elif carbon <= 0.84:
        steel_type = "eutectoid"
    else:
        steel_type = "hypereutectoid"
    return {
        "ac1_c": round(_safe_float(calculate_ac1(steel)), 2),
        "ac3_c": None
        if calculate_ac3(steel) is None
        else round(_safe_float(calculate_ac3(steel)), 2),
        "ms_c": round(_safe_float(calculate_ms_temperature(steel)), 2),
        "mf_c": round(_safe_float(calculate_mf_temperature(steel)), 2),
        "quench_recommendation": get_quench_temperature(steel, steel_type=steel_type),
        "tempering_low": get_tempering_temperature("high", steel),
        "tempering_medium": get_tempering_temperature("medium", steel),
        "tempering_high": get_tempering_temperature("low", steel),
        "hardenability": estimate_hardenability(steel),
        "material_grade": material_grade,
        "property_model_source": _MODEL_SOURCE,
        "reference_dataset": _REFERENCE_DATASET,
    }


def calculate_hybrid_properties(
    *,
    composition: dict[str, float],
    inferred_system: str,
    final_stage: str,
    phase_fractions: dict[str, float] | None = None,
    material_grade: str | None = None,
    material_class_ru: str | None = None,
    effect_vector: dict[str, float] | None = None,
    grain_size_um: float | None = None,
    overlay_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overlay = overlay_rules or {
        "systems": {},
        "defaults": {},
        "ductility_thresholds_hv": [],
    }
    if not supports_hybrid_properties(
        inferred_system=inferred_system, composition=composition
    ):
        return {
            "hv_estimate": 0.0,
            "hardness_hv_est": 0.0,
            "uts_estimate_mpa": 0.0,
            "uts_mpa_est": 0.0,
            "ductility_class": "unknown",
            "property_model_source": _MODEL_SOURCE,
            "reference_dataset": _REFERENCE_DATASET,
            "compatibility_overlay_used": False,
            "fallback_used": True,
            "material_grade": material_grade,
            "material_class_ru": material_class_ru,
        }

    base = _build_microstructure_payload(
        composition=composition,
        final_stage=final_stage,
        phase_fractions=phase_fractions,
        material_grade=material_grade,
    )
    if base is None:
        hybrid = estimate_hybrid_properties(
            composition=composition,
            final_stage=final_stage,
            effect=effect_vector or {},
            overlay_rules=overlay,
        )
        hybrid["material_grade"] = material_grade
        hybrid["material_class_ru"] = material_class_ru
        if grain_size_um and grain_size_um > 0.0:
            hybrid["mean_grain_diameter_um"] = round(float(grain_size_um), 2)
        return hybrid

    hv = _safe_float(base.get("hb"))
    uts = _safe_float(base.get("uts_mpa"))
    elongation = _safe_float(base.get("elongation_pct"))
    route_est = estimate_hybrid_properties(
        composition=composition,
        final_stage=final_stage,
        effect=effect_vector or {},
        overlay_rules=overlay,
    )
    if route_est is None:
        route_est = {
            "hv_estimate": hv,
            "uts_estimate_mpa": uts,
            "ductility_class": _ductility_label_from_elongation(elongation),
            "compatibility_overlay_used": False,
        }
    overlay_hv = _safe_float(route_est.get("hv_estimate"), hv) - hv
    overlay_uts = _safe_float(route_est.get("uts_estimate_mpa"), uts) - uts
    hv += overlay_hv
    uts += overlay_uts
    result = {
        "hv_estimate": round(float(hv), 2),
        "hardness_hv_est": round(float(hv), 2),
        "uts_estimate_mpa": round(float(uts), 2),
        "uts_mpa_est": round(float(uts), 2),
        "yield_strength_estimate_mpa": round(float(uts) * 0.62, 2),
        "elongation_estimate_pct": round(float(elongation), 2),
        "ductility_class": route_est.get(
            "ductility_class", _ductility_label_from_elongation(elongation)
        ),
        "property_model_source": _MODEL_SOURCE,
        "reference_dataset": _REFERENCE_DATASET,
        "compatibility_overlay_used": bool(
            route_est.get("compatibility_overlay_used", False)
        ),
        "fallback_used": False,
        "material_grade": material_grade,
        "material_class_ru": material_class_ru,
    }
    if grain_size_um and grain_size_um > 0.0:
        try:
            grains_per_mm2 = 1.0 / max((grain_size_um / 1000.0) ** 2, 1e-9)
            astm = calculate_astm_grain_size_number(grains_per_mm2)
            result["grain_size_astm_number_proxy"] = round(
                _safe_float(astm.get("astm_grain_size_number", 0.0)), 2
            )
            result["mean_grain_diameter_um"] = round(float(grain_size_um), 2)
        except Exception:
            result["mean_grain_diameter_um"] = round(float(grain_size_um), 2)
    return result


def validate_expected_properties(
    expected_properties: dict[str, Any] | None,
    calculated_properties: dict[str, Any],
) -> dict[str, Any]:
    expected = dict(expected_properties or {})
    mapping = {
        "hardness_hb": "hv_estimate",
        "tensile_strength_mpa": "uts_estimate_mpa",
        "yield_strength_mpa": "yield_strength_estimate_mpa",
        "elongation_pct": "elongation_estimate_pct",
        "astm_grain_size": "grain_size_astm_number_proxy",
        "mean_grain_diameter_um": "mean_grain_diameter_um",
    }
    checks: list[dict[str, Any]] = []
    for expected_key, actual_key in mapping.items():
        spec = expected.get(expected_key)
        if not isinstance(spec, dict):
            continue
        actual = _safe_float(calculated_properties.get(actual_key, 0.0), 0.0)
        min_v = spec.get("min")
        max_v = spec.get("max")
        passed = True
        if min_v is not None and actual < _safe_float(min_v, actual):
            passed = False
        if max_v is not None and actual > _safe_float(max_v, actual):
            passed = False
        checks.append(
            {
                "metric": expected_key,
                "actual": actual,
                "min": None if min_v is None else _safe_float(min_v, actual),
                "max": None if max_v is None else _safe_float(max_v, actual),
                "passed": passed,
            }
        )
    return {
        "pass": all(bool(item.get("passed", False)) for item in checks)
        if checks
        else True,
        "checks": checks,
        "expected_properties_source": _REFERENCE_DATASET,
        "property_model_source": _MODEL_SOURCE,
    }
