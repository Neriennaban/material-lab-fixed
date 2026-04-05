from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contracts_v2 import ProcessRoute, ProcessingOperation, ProcessingState
from .generator_phase_map import (
    resolve_al_cu_mg_stage,
    resolve_al_si_stage,
    resolve_cu_zn_stage,
    resolve_fe_c_stage,
    resolve_fe_si_stage,
)
from .materials_hybrid import estimate_hybrid_properties, supports_hybrid_properties
from .route_validation import RouteValidationResult, validate_process_route

_RULEBOOK_DIR = Path(__file__).resolve().parent / "rulebook"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_TECH_RULES = _load_json(_RULEBOOK_DIR / "technology_rules.json")
_PROP_RULES = _load_json(_RULEBOOK_DIR / "properties_rules.json")

_EFFECT_KEYS = (
    "grain_size_factor",
    "elongation_factor",
    "texture_strength",
    "dislocation_proxy",
    "precipitation_level",
    "segregation_level",
    "residual_stress",
    "porosity_factor",
)


@dataclass(slots=True)
class RouteSimulationResult:
    final_processing: ProcessingState
    final_effect_vector: dict[str, float]
    route_timeline: list[dict[str, Any]]
    resolved_stage_by_step: list[dict[str, Any]]
    technology_influence: dict[str, Any]
    property_indicators: dict[str, Any]
    generator_param_overrides: dict[str, Any]
    final_stage: str
    route_validation: RouteValidationResult

    def to_metadata(self) -> dict[str, Any]:
        return {
            "route_timeline": self.route_timeline,
            "resolved_stage_by_step": self.resolved_stage_by_step,
            "technology_influence": self.technology_influence,
            "property_indicators": self.property_indicators,
            "final_effect_vector": self.final_effect_vector,
            "final_stage": self.final_stage,
            "route_validation": self.route_validation.to_dict(),
        }


def _stable_step_seed(base_seed: int, route_name: str, step_index: int) -> int:
    token = f"{route_name}|{step_index}".encode("utf-8")
    digest = hashlib.sha1(token).hexdigest()[:8]
    return int(base_seed) + int(digest, 16)


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _infer_route_system_stage(
    inferred_system: str,
    composition: dict[str, float],
    processing: ProcessingState,
) -> str:
    key = inferred_system.strip().lower()
    if key == "fe-c":
        return resolve_fe_c_stage(
            c_wt=float(composition.get("C", 0.0)),
            temperature_c=processing.temperature_c,
            cooling_mode=processing.cooling_mode,
            requested_stage="auto",
        )
    if key == "al-si":
        return resolve_al_si_stage(
            si_wt=float(composition.get("Si", 0.0)),
            temperature_c=processing.temperature_c,
            cooling_mode=processing.cooling_mode,
            requested_stage="auto",
        )
    if key == "cu-zn":
        return resolve_cu_zn_stage(
            zn_wt=float(composition.get("Zn", 0.0)),
            temperature_c=processing.temperature_c,
            cooling_mode=processing.cooling_mode,
            requested_stage="auto",
            deformation_pct=processing.deformation_pct,
        )
    if key == "al-cu-mg":
        return resolve_al_cu_mg_stage(
            temperature_c=processing.temperature_c,
            cooling_mode=processing.cooling_mode,
            requested_stage="auto",
            aging_temperature_c=processing.aging_temperature_c,
            aging_hours=processing.aging_hours,
        )
    if key == "fe-si":
        return resolve_fe_si_stage(
            temperature_c=processing.temperature_c,
            cooling_mode=processing.cooling_mode,
            requested_stage="auto",
            deformation_pct=processing.deformation_pct,
            si_wt=float(composition.get("Si", 0.0)),
        )
    return "unknown"


def _operation_effect(method: str) -> dict[str, float]:
    op = _TECH_RULES.get("operations", {}).get(method, {})
    effect = op.get("effects", {})
    out: dict[str, float] = {}
    for key in _EFFECT_KEYS:
        out[key] = float(effect.get(key, 0.0))
    return out


def _initial_effect_vector(inferred_system: str, composition: dict[str, float]) -> dict[str, float]:
    out = {key: 0.0 for key in _EFFECT_KEYS}
    c = float(composition.get("C", 0.0))
    si = float(composition.get("Si", 0.0))
    zn = float(composition.get("Zn", 0.0))
    cu = float(composition.get("Cu", 0.0))
    mg = float(composition.get("Mg", 0.0))

    if inferred_system == "fe-c":
        out["dislocation_proxy"] += _clamp(c * 0.08, 0.0, 0.25)
        out["precipitation_level"] += _clamp(c * 0.05, 0.0, 0.2)
    elif inferred_system == "al-si":
        out["segregation_level"] += _clamp(si * 0.01, 0.0, 0.22)
    elif inferred_system == "cu-zn":
        out["texture_strength"] += _clamp((zn - 20.0) * 0.004, 0.0, 0.2)
    elif inferred_system == "al-cu-mg":
        out["precipitation_level"] += _clamp((cu + mg) * 0.01, 0.0, 0.25)
    elif inferred_system == "fe-si":
        out["texture_strength"] += _clamp(si * 0.03, 0.0, 0.25)
    return out


def _post_cooling_observation_temp(cooling_mode: str, default_temp: float = 20.0) -> float:
    mode = str(cooling_mode or "").strip().lower()
    if mode in {"quenched", "quench", "water_quench", "oil_quench"}:
        return 24.0
    if mode in {"normalized"}:
        return 30.0
    if mode in {"tempered"}:
        return 28.0
    if mode in {"cold_worked"}:
        return 20.0
    if mode in {"aged", "natural_aged", "overaged"}:
        return 26.0
    if mode in {"solutionized"}:
        return 30.0
    if mode in {"equilibrium", "slow_cool"}:
        return 22.0
    return float(default_temp)


def _apply_operation_to_processing(base: ProcessingState, operation: ProcessingOperation) -> ProcessingState:
    # In route mode temperature is interpreted as the resulting state after operation.
    # Quench operations are resolved close to room temperature for stage prediction.
    resulting_temperature = float(operation.temperature_c)
    method = operation.method.strip().lower()
    note = str(operation.note or "").strip().lower()
    hold_observation_temp = "hold_observation_temp=true" in note
    observation_after_cooling_methods = {
        "cast_slow",
        "cast_fast",
        "directional_solidification",
        "normalize",
        "anneal_full",
        "anneal_recrystallization",
        "solution_treat",
        "homogenize",
    }
    if method.startswith("quench"):
        resulting_temperature = min(resulting_temperature, 40.0)
    elif method.startswith("age"):
        if operation.aging_temperature_c > 0.0:
            resulting_temperature = float(operation.aging_temperature_c)
    elif (method in observation_after_cooling_methods or method.startswith("anneal_")) and not hold_observation_temp:
        resulting_temperature = _post_cooling_observation_temp(
            cooling_mode=str(operation.cooling_mode),
            default_temp=20.0,
        )

    return ProcessingState(
        temperature_c=resulting_temperature,
        cooling_mode=str(operation.cooling_mode),
        deformation_pct=float(operation.deformation_pct),
        aging_hours=float(operation.aging_hours),
        aging_temperature_c=float(operation.aging_temperature_c),
        pressure_mpa=operation.pressure_mpa,
        note=(base.note + "; " + operation.note).strip("; ").strip(),
    )


def _effect_to_generator_params(
    generator: str,
    inferred_system: str,
    effect: dict[str, float],
    stage: str,
    processing: ProcessingState,
) -> dict[str, Any]:
    e = effect
    params: dict[str, Any] = {}
    gen = generator.strip().lower()

    if gen == "grains":
        params["mean_grain_size_px"] = _clamp(52.0 * (1.0 + e["grain_size_factor"]), 18.0, 140.0)
        params["grain_size_jitter"] = _clamp(0.2 + 0.25 * abs(e["segregation_level"]), 0.08, 0.75)
        params["elongation"] = _clamp(1.0 + 1.8 * max(0.0, e["elongation_factor"]), 1.0, 3.5)
        params["orientation_deg"] = _clamp(10.0 + 70.0 * max(0.0, e["texture_strength"]), 0.0, 90.0)
        params["boundary_contrast"] = _clamp(0.35 + 0.4 * max(0.0, e["dislocation_proxy"]), 0.2, 0.9)
        params["pore_fraction"] = _clamp(0.001 + 0.02 * max(0.0, e["porosity_factor"]), 0.0, 0.08)
    elif gen == "pearlite":
        params["lamella_period_px"] = _clamp(7.0 - 1.8 * max(0.0, e["dislocation_proxy"]), 3.2, 9.4)
        params["pearlite_fraction"] = _clamp(0.55 + 0.25 * max(0.0, e["precipitation_level"]), 0.1, 0.98)
        params["colony_size_px"] = _clamp(95.0 * (1.0 + e["grain_size_factor"]), 40.0, 180.0)
    elif gen == "phase_map":
        params["system"] = inferred_system
        params["stage"] = stage
        params["temperature_c"] = float(processing.temperature_c)
        params["cooling_mode"] = str(processing.cooling_mode)
        params["deformation_pct"] = float(processing.deformation_pct)
        params["aging_hours"] = float(processing.aging_hours)
        params["aging_temperature_c"] = float(processing.aging_temperature_c)
    elif gen == "dendritic_cast":
        params["cooling_rate"] = _clamp(40.0 + 65.0 * max(0.0, e["dislocation_proxy"]), 2.0, 220.0)
        params["thermal_gradient"] = _clamp(0.5 + 0.6 * max(0.0, e["texture_strength"]), 0.0, 1.8)
        params["primary_arm_spacing"] = _clamp(34.0 * (1.0 + e["grain_size_factor"] - 0.4 * e["dislocation_proxy"]), 8.0, 170.0)
        params["interdendritic_fraction"] = _clamp(0.28 + 0.35 * max(0.0, e["segregation_level"]), 0.05, 0.85)
        params["porosity_fraction"] = _clamp(0.005 + 0.06 * max(0.0, e["porosity_factor"]), 0.0, 0.25)
        params["secondary_arm_factor"] = _clamp(0.4 + 0.4 * max(0.0, e["texture_strength"]), 0.12, 1.2)
    elif gen == "aged_al":
        params["precipitate_fraction"] = _clamp(0.04 + 0.32 * max(0.0, e["precipitation_level"]), 0.01, 0.35)
        params["precipitate_scale_px"] = _clamp(1.3 + 2.8 * max(0.0, e["grain_size_factor"] + 0.6 * e["precipitation_level"]), 0.8, 4.6)

    return params


def _estimate_properties_legacy(
    inferred_system: str,
    final_stage: str,
    effect: dict[str, float],
    *,
    fallback_used: bool = False,
) -> dict[str, Any]:
    defaults = _PROP_RULES.get("defaults", {})
    by_system = _PROP_RULES.get("systems", {}).get(inferred_system, _PROP_RULES.get("systems", {}).get("custom-multicomponent", {}))

    hv = float(by_system.get("hv_base", defaults.get("hv_base", 120.0)))
    uts = float(by_system.get("uts_base_mpa", defaults.get("uts_base_mpa", 420.0)))
    coeff = by_system.get("hv_coeff", {})
    if isinstance(coeff, dict):
        for key, val in coeff.items():
            hv += float(val) * float(effect.get(key, 0.0))

    # UTS follows HV trend as educational approximation.
    uts += (hv - float(by_system.get("hv_base", defaults.get("hv_base", 120.0)))) * 1.8

    stage_adjust = by_system.get("stage_adjust", {})
    stage_data = stage_adjust.get(final_stage, {}) if isinstance(stage_adjust, dict) else {}
    if isinstance(stage_data, dict):
        hv += float(stage_data.get("hv", 0.0))
        uts += float(stage_data.get("uts", 0.0))

    hv = _clamp(hv, 40.0, 900.0)
    uts = _clamp(uts, 120.0, 2600.0)

    ductility = str(defaults.get("ductility_base", "medium"))
    thresholds = _PROP_RULES.get("ductility_thresholds_hv", [])
    if isinstance(thresholds, list):
        for threshold in thresholds:
            if not isinstance(threshold, dict):
                continue
            max_hv = float(threshold.get("max_hv", 9999))
            if hv <= max_hv:
                ductility = str(threshold.get("label", ductility))
                break
    if isinstance(stage_data, dict) and "ductility" in stage_data:
        ductility = str(stage_data["ductility"])

    return {
        "hv_estimate": round(float(hv), 2),
        "uts_estimate_mpa": round(float(uts), 2),
        "ductility_class": ductility,
        "model_note": "Educational estimate, not engineering design data.",
        "property_model_source": "legacy_properties_rules",
        "reference_dataset": "properties_rules.json",
        "compatibility_overlay_used": False,
        "fallback_used": bool(fallback_used),
    }


def _estimate_properties(
    inferred_system: str,
    composition: dict[str, float],
    final_stage: str,
    effect: dict[str, float],
) -> dict[str, Any]:
    if supports_hybrid_properties(inferred_system=inferred_system, composition=composition):
        hybrid = estimate_hybrid_properties(
            composition=composition,
            final_stage=final_stage,
            effect=effect,
            overlay_rules=_PROP_RULES,
        )
        if hybrid is not None:
            return hybrid
        return _estimate_properties_legacy(
            inferred_system=inferred_system,
            final_stage=final_stage,
            effect=effect,
            fallback_used=True,
        )
    return _estimate_properties_legacy(
        inferred_system=inferred_system,
        final_stage=final_stage,
        effect=effect,
        fallback_used=False,
    )


def simulate_process_route(
    composition: dict[str, float],
    inferred_system: str,
    route: ProcessRoute | dict[str, Any] | None,
    initial_processing: ProcessingState,
    generator: str,
    base_seed: int,
    step_preview_index: int | None = None,
) -> RouteSimulationResult:
    route_obj = route if isinstance(route, ProcessRoute) else ProcessRoute.from_dict(route)
    validation = validate_process_route(route=route_obj, inferred_system=inferred_system, processing_context=initial_processing)

    effect = _initial_effect_vector(inferred_system=inferred_system, composition=composition)
    timeline: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []

    processing = initial_processing
    final_stage = _infer_route_system_stage(
        inferred_system=inferred_system,
        composition=composition,
        processing=processing,
    )

    selected_step = step_preview_index
    operations = validation.normalized_operations

    for idx, op in enumerate(operations):
        processing = _apply_operation_to_processing(processing, op)
        delta = _operation_effect(op.method)
        for key in _EFFECT_KEYS:
            effect[key] += delta[key]
            effect[key] = _clamp(effect[key], -0.95, 0.95)

        resolved_stage = _infer_route_system_stage(
            inferred_system=inferred_system,
            composition=composition,
            processing=processing,
        )
        final_stage = resolved_stage
        step_seed = _stable_step_seed(base_seed=base_seed, route_name=route_obj.route_name, step_index=idx)
        timeline.append(
            {
                "step_index": idx,
                "method": op.method,
                "operation": op.to_dict(),
                "processing_state": processing.to_dict(),
                "delta_effect": delta,
                "effect_after_step": dict(effect),
                "resolved_stage": resolved_stage,
                "step_seed": int(step_seed),
            }
        )
        stages.append({"step_index": idx, "stage": resolved_stage})

    if selected_step is not None and timeline:
        pick = int(np.clip(int(selected_step), 0, len(timeline) - 1))
        picked = timeline[pick]
        processing = ProcessingState.from_dict(picked["processing_state"])
        effect = dict(picked["effect_after_step"])
        final_stage = str(picked["resolved_stage"])

    gen_overrides = _effect_to_generator_params(
        generator=generator,
        inferred_system=inferred_system,
        effect=effect,
        stage=final_stage,
        processing=processing,
    )
    props = _estimate_properties(
        inferred_system=inferred_system,
        composition=composition,
        final_stage=final_stage,
        effect=effect,
    )

    influence = {
        "summary": (
            f"Stage={final_stage}, dislocation={effect['dislocation_proxy']:.2f}, "
            f"precipitation={effect['precipitation_level']:.2f}, texture={effect['texture_strength']:.2f}"
        ),
        "warnings": list(validation.warnings),
        "critical_errors": list(validation.errors),
    }

    return RouteSimulationResult(
        final_processing=processing,
        final_effect_vector=effect,
        route_timeline=timeline,
        resolved_stage_by_step=stages,
        technology_influence=influence,
        property_indicators=props,
        generator_param_overrides=gen_overrides,
        final_stage=final_stage,
        route_validation=validation,
    )
