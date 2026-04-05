from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cooling_modes import canonicalize_cooling_mode

_SUPPORTED_GENERATORS = {
    "calphad_phase",
    "grains",
    "pearlite",
    "eutectic",
    "dislocations",
    "martensite",
    "tempered",
    "aged_al",
    "phase_map",
    "crm_fe_c",
    "dendritic_cast",
}

_PHASE_SYSTEMS = {"fe-c", "al-si", "cu-zn", "fe-si", "al-cu-mg"}


@dataclass(slots=True)
class AutoGeneratorDecision:
    selected_generator: str
    resolved_params: dict[str, Any]
    selection_reason: str
    selection_confidence: float
    coverage_mode: str  # supported | fallback


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _nearest_phase_system(composition: dict[str, float], inferred_system: str) -> str:
    system = str(inferred_system or "").strip().lower()
    if system in _PHASE_SYSTEMS:
        return system

    si = float(composition.get("Si", 0.0))
    fe = float(composition.get("Fe", 0.0))
    al = float(composition.get("Al", 0.0))
    cu = float(composition.get("Cu", 0.0))
    zn = float(composition.get("Zn", 0.0))
    mg = float(composition.get("Mg", 0.0))
    c = float(composition.get("C", 0.0))

    scores = {
        "fe-c": fe + 2.2 * c,
        "fe-si": fe + 1.5 * si,
        "al-si": al + 1.4 * si,
        "cu-zn": cu + 1.4 * zn,
        "al-cu-mg": al + 1.1 * cu + 1.0 * mg,
    }
    return max(scores, key=scores.get)


def _merge_params(
    base_params: dict[str, Any],
    user_params: dict[str, Any] | None,
    route_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    out = dict(base_params)
    out.update(dict(user_params or {}))
    out.update(dict(route_overrides or {}))
    return out


def _route_has_cast_signature(route_sim: Any, user_params: dict[str, Any]) -> bool:
    if any(key in user_params for key in ("cooling_rate", "primary_arm_spacing", "interdendritic_fraction")):
        return True
    timeline = getattr(route_sim, "route_timeline", None)
    if isinstance(timeline, list):
        for step in timeline:
            if not isinstance(step, dict):
                continue
            method = str(step.get("method", "")).strip().lower()
            if method.startswith("cast_") or method == "directional_solidification":
                return True
    return False


def _base_params_for(
    generator: str,
    composition: dict[str, float],
    inferred_system: str,
    processing: Any,
    route_sim: Any,
    user_params: dict[str, Any],
) -> dict[str, Any]:
    c = float(composition.get("C", 0.0))
    si = float(composition.get("Si", 0.0))
    cu = float(composition.get("Cu", 0.0))
    mg = float(composition.get("Mg", 0.0))

    if generator == "phase_map":
        system = _nearest_phase_system(composition=composition, inferred_system=inferred_system)
        stage = "auto"
        route_stage = ""
        if route_sim is not None:
            route_stage = str(getattr(route_sim, "final_stage", "") or "").strip().lower()
        if route_stage:
            stage = route_stage
        elif str(user_params.get("stage", "")).strip():
            stage = str(user_params.get("stage", "auto"))

        return {
            "system": system,
            "stage": stage,
            "temperature_c": float(getattr(processing, "temperature_c", 20.0)),
            "cooling_mode": str(getattr(processing, "cooling_mode", "equilibrium")),
            "deformation_pct": float(getattr(processing, "deformation_pct", 0.0)),
            "aging_hours": float(getattr(processing, "aging_hours", 0.0)),
            "aging_temperature_c": float(getattr(processing, "aging_temperature_c", 20.0)),
        }

    if generator == "calphad_phase":
        system = _nearest_phase_system(composition=composition, inferred_system=inferred_system)
        return {"system": system, "top_n_phases": 6}

    if generator == "dendritic_cast":
        mode = canonicalize_cooling_mode(getattr(processing, "cooling_mode", "equilibrium"))
        default_rate = 18.0 if mode in {"equilibrium", "slow_cool"} else 80.0
        return {"cooling_rate": default_rate}

    if generator == "dislocations":
        magnification = int(user_params.get("magnification", 200))
        return {"magnification": max(100, magnification)}

    if generator == "eutectic":
        return {"si_phase_fraction": float(np.clip(0.06 + 0.022 * si, 0.1, 0.75))}

    if generator == "aged_al":
        return {"precipitate_fraction": float(np.clip(0.03 + 0.012 * cu + 0.009 * mg, 0.04, 0.22))}

    if generator == "pearlite":
        return {"pearlite_fraction": float(np.clip(c / 0.82, 0.08, 0.98))}

    return {}


def select_auto_generator(
    composition: dict[str, float],
    inferred_system: str,
    processing: Any,
    route_sim: Any = None,
    user_params: dict[str, Any] | None = None,
    route_overrides: dict[str, Any] | None = None,
    calphad_available: bool = False,
) -> AutoGeneratorDecision:
    params = dict(user_params or {})
    comp = {str(k): max(0.0, float(v)) for k, v in composition.items()}
    total = max(1e-9, float(sum(comp.values())))

    c = float(comp.get("C", 0.0))
    si = float(comp.get("Si", 0.0))

    mode = canonicalize_cooling_mode(getattr(processing, "cooling_mode", "equilibrium"))
    temperature = float(getattr(processing, "temperature_c", 20.0))

    has_route = bool(getattr(route_sim, "route_timeline", []))
    final_stage = str(getattr(route_sim, "final_stage", "") or "").strip().lower()
    stage_requested = str(params.get("stage", "")).strip().lower()
    has_stage_request = stage_requested not in {"", "auto"}
    has_cast = _route_has_cast_signature(route_sim=route_sim, user_params=params)

    system = str(inferred_system or "custom-multicomponent").strip().lower()

    hint_gen = ""
    auto_hint = params.get("auto_hint")
    if isinstance(auto_hint, dict):
        hint_gen = str(auto_hint.get("preferred_generator", "")).strip().lower()

    selected = ""
    reason = ""
    confidence = 0.72
    coverage_mode = "supported"

    if calphad_available and system in _PHASE_SYSTEMS:
        selected = "calphad_phase"
        reason = "CALPHAD backend доступен: выбран calphad_phase для термодинамически согласованной генерации."
        confidence = 0.96

    if not selected and hint_gen in _SUPPORTED_GENERATORS:
        if not (hint_gen == "phase_map" and str(inferred_system).strip().lower() not in _PHASE_SYSTEMS):
            selected = hint_gen
            reason = "Использована подсказка пресета (preferred_generator)."
            confidence = 0.95

    if not selected and si >= 95.0 and (total - si) <= 5.0:
        selected = "dislocations"
        reason = "Монокристалл Si: выбран генератор травильных ямок/дислокаций."
        confidence = 0.96

    if not selected and has_cast:
        selected = "dendritic_cast"
        reason = "Маршрут/параметры имеют литейную сигнатуру: выбран дендритный генератор."
        confidence = 0.9

    if not selected and system == "fe-c":
        pearlite_hint = any(key in params for key in ("lamella_period_px", "pearlite_fraction", "colony_size_px"))
        near_pure_fe = c <= 0.06
        if has_route or has_stage_request or final_stage:
            selected = "phase_map"
            reason = "Fe-C со стадийностью/маршрутом: выбран phase_map."
            confidence = 0.9
        elif pearlite_hint or (0.10 <= c <= 1.4 and mode in {"equilibrium", "slow_cool", "normalized"}):
            selected = "pearlite"
            reason = "Fe-C в перлитной области: выбран pearlite."
            confidence = 0.85
        elif near_pure_fe:
            selected = "grains"
            reason = "Почти чистое Fe без стадийного режима: выбран grains."
            confidence = 0.88
        else:
            selected = "phase_map"
            reason = "Fe-C по умолчанию: выбран phase_map для универсального покрытия стадий."
            confidence = 0.8

    if not selected and system == "al-si":
        eutectic_hint = any(key in params for key in ("si_phase_fraction", "eutectic_scale_px", "morphology"))
        if has_route and final_stage in {"aged", "supersaturated"}:
            selected = "phase_map"
            reason = "Al-Si со стадией aging/supersaturated: выбран phase_map."
            confidence = 0.82
        elif has_cast or eutectic_hint or 7.0 <= si <= 16.0:
            selected = "eutectic"
            reason = "Al-Si в эвтектической/литейной области: выбран eutectic."
            confidence = 0.87
        else:
            selected = "phase_map"
            reason = "Al-Si вне явной эвтектической сигнатуры: выбран phase_map."
            confidence = 0.78

    if not selected and system == "al-cu-mg":
        aged_hint = any(key in params for key in ("precipitate_fraction", "precipitate_scale_px"))
        aged_mode = mode in {"aged", "natural_aged", "overaged"}
        if aged_hint or aged_mode or final_stage in {"artificial_aged", "natural_aged", "overaged"}:
            selected = "aged_al"
            reason = "Al-Cu-Mg с режимом старения: выбран aged_al."
            confidence = 0.9
        else:
            selected = "phase_map"
            reason = "Al-Cu-Mg без явной сигнатуры старения: выбран phase_map."
            confidence = 0.8

    if not selected and system in {"cu-zn", "fe-si"}:
        if has_route or has_stage_request or final_stage:
            selected = "phase_map"
            reason = "Система со стадийностью/маршрутом: выбран phase_map."
            confidence = 0.84
        else:
            selected = "grains"
            reason = "Однофазный/деформированный режим: выбран grains."
            confidence = 0.83

    if not selected:
        cold_non_cast = (not has_cast) and temperature <= 250.0 and mode in {
            "equilibrium",
            "slow_cool",
            "cold_worked",
            "normalized",
        }
        if cold_non_cast:
            selected = "grains"
            reason = "Неизвестная система в холодном не-литейном режиме: fallback на grains."
            confidence = 0.62
        else:
            selected = "dendritic_cast"
            reason = "Неизвестная система: fallback на dendritic_cast."
            confidence = 0.55
        coverage_mode = "fallback"

    if selected not in _SUPPORTED_GENERATORS:
        selected = "dendritic_cast"
        reason = "Защитный fallback: выбран dendritic_cast."
        confidence = 0.4
        coverage_mode = "fallback"

    base = _base_params_for(
        generator=selected,
        composition=comp,
        inferred_system=system,
        processing=processing,
        route_sim=route_sim,
        user_params=params,
    )
    resolved = _merge_params(base_params=base, user_params=params, route_overrides=route_overrides)

    return AutoGeneratorDecision(
        selected_generator=selected,
        resolved_params=resolved,
        selection_reason=reason,
        selection_confidence=_clamp(confidence, 0.0, 1.0),
        coverage_mode=coverage_mode,
    )
