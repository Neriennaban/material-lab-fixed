from __future__ import annotations

import math
from typing import Any

from core.contracts_v2 import ProcessingState
from core.metallography_v3.realism_utils import clamp, cooling_index


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _comp(composition_wt: dict[str, float] | None, key: str) -> float:
    if not isinstance(composition_wt, dict):
        return 0.0
    return _safe_float(composition_wt.get(key, 0.0), 0.0)


def _op_summary(thermal_summary: dict[str, Any] | None) -> dict[str, Any]:
    payload = (thermal_summary or {}).get("operation_inference", {})
    return dict(payload) if isinstance(payload, dict) else {}


def _summary_temp(thermal_summary: dict[str, Any] | None, processing: ProcessingState, key: str, fallback: float) -> float:
    if isinstance(thermal_summary, dict) and key in thermal_summary:
        return _safe_float(thermal_summary.get(key), fallback)
    return float(fallback if key != "temperature_end_c" else processing.temperature_c)


def _fe_c_equilibrium(c_wt: float) -> tuple[float, float, float]:
    c = max(0.0, float(c_wt))
    if c <= 0.77:
        pearlite = clamp((c - 0.02) / max(1e-6, 0.77 - 0.02), 0.0, 1.0)
        ferrite = clamp(1.0 - pearlite, 0.0, 1.0)
        return ferrite, pearlite, 0.0
    if c <= 2.14:
        cementite = clamp((c - 0.77) / max(1e-6, 2.14 - 0.77), 0.0, 1.0) * 0.34
        pearlite = clamp(1.0 - cementite, 0.0, 1.0)
        return 0.0, pearlite, cementite
    ledeburitic = clamp((c - 2.14) / max(1e-6, 4.30 - 2.14), 0.0, 1.0)
    pearlite = clamp(1.0 - 0.72 * ledeburitic, 0.0, 1.0)
    cementite = clamp(0.10 + 0.42 * ledeburitic, 0.0, 1.0)
    ferrite = clamp(1.0 - pearlite - cementite, 0.0, 1.0)
    total = max(1e-6, ferrite + pearlite + cementite)
    return ferrite / total, pearlite / total, cementite / total


def _van_bohemen_ms(c_wt: float, composition_wt: dict[str, float] | None) -> float:
    c = max(0.0, float(c_wt))
    alloy_penalty = (
        30.0 * _comp(composition_wt, "Mn")
        + 17.0 * _comp(composition_wt, "Ni")
        + 12.0 * _comp(composition_wt, "Cr")
        + 7.0 * _comp(composition_wt, "Mo")
        + 8.0 * _comp(composition_wt, "Si")
    )
    return float(565.0 - 600.0 * (1.0 - math.exp(-1.15 * c)) - 0.1 * alloy_penalty)


def _van_bohemen_bs(c_wt: float, composition_wt: dict[str, float] | None) -> float:
    c = max(0.0, float(c_wt))
    alloy_penalty = (
        26.0 * _comp(composition_wt, "Mn")
        + 14.0 * _comp(composition_wt, "Ni")
        + 12.0 * _comp(composition_wt, "Cr")
        + 7.0 * _comp(composition_wt, "Mo")
        + 9.0 * _comp(composition_wt, "Si")
    )
    return float(830.0 - 360.0 * (1.0 - math.exp(-1.25 * c)) - 0.1 * alloy_penalty)


def _koistinen_marburger(ms_c: float, final_temp_c: float) -> float:
    undercool = max(0.0, float(ms_c) - float(final_temp_c))
    return float(1.0 - math.exp(-0.011 * undercool))


def _normalize_fraction_payload(payload: dict[str, float]) -> dict[str, float]:
    cleaned = {str(k): max(0.0, float(v)) for k, v in payload.items() if float(v) > 1e-9}
    total = float(sum(cleaned.values()))
    if total <= 1e-9:
        return {}
    return {k: float(v) / total for k, v in cleaned.items()}


def _fe_c_stage_override(stage_l: str, retained_austenite: float, tempering_level: float) -> tuple[dict[str, float] | None, float]:
    liquid_fraction = 0.0
    overrides: dict[str, float] | None = None
    if stage_l == "liquid":
        liquid_fraction = 1.0
        overrides = {}
    elif stage_l == "liquid_gamma":
        liquid_fraction = 0.62
        overrides = {"retained_austenite": 1.0}
    elif stage_l == "delta_ferrite":
        overrides = {"ferrite_fraction": 0.78, "retained_austenite": 0.22}
    elif stage_l == "austenite":
        overrides = {"retained_austenite": 1.0}
    elif stage_l == "ferrite":
        overrides = {"ferrite_fraction": 1.0}
    elif stage_l == "alpha_gamma":
        overrides = {"ferrite_fraction": 0.55, "retained_austenite": 0.45}
    elif stage_l == "gamma_cementite":
        overrides = {"retained_austenite": 0.72, "cementite_fraction": 0.28}
    elif stage_l == "alpha_pearlite":
        overrides = {"ferrite_fraction": 0.50, "pearlite_fraction": 0.50}
    elif stage_l == "pearlite":
        overrides = {"pearlite_fraction": 0.995, "cementite_fraction": 0.005}
    elif stage_l == "pearlite_cementite":
        overrides = {"pearlite_fraction": 0.82, "cementite_fraction": 0.18}
    elif stage_l == "ledeburite":
        overrides = {"pearlite_fraction": 0.62, "cementite_fraction": 0.38}
    elif stage_l in {"martensite", "martensite_tetragonal", "martensite_cubic"}:
        overrides = {"martensite_fraction": 0.82, "cementite_fraction": 0.08, "retained_austenite": max(retained_austenite, 0.10)}
    elif stage_l == "bainite":
        overrides = {"bainite_fraction": 0.82, "cementite_fraction": 0.18}
    elif stage_l == "troostite_quench":
        overrides = {"martensite_fraction": 0.42, "bainite_fraction": 0.26, "cementite_fraction": 0.20, "retained_austenite": max(retained_austenite, 0.12)}
    elif stage_l == "sorbite_quench":
        overrides = {"martensite_fraction": 0.26, "bainite_fraction": 0.30, "ferrite_fraction": 0.20, "cementite_fraction": 0.24}
    elif stage_l in {"tempered_low", "troostite_temper", "tempered_medium", "sorbite_temper", "tempered_high"}:
        if stage_l == "tempered_low":
            mart = clamp(0.46 - 0.18 * tempering_level, 0.22, 0.46)
            ferr = clamp(0.12 + 0.12 * tempering_level, 0.12, 0.24)
            overrides = {"martensite_fraction": mart, "bainite_fraction": 0.18, "cementite_fraction": 0.22, "ferrite_fraction": ferr}
        elif stage_l in {"troostite_temper", "tempered_medium"}:
            mart = clamp(0.34 - 0.26 * tempering_level, 0.10, 0.34)
            ferr = clamp(0.24 + 0.24 * tempering_level, 0.24, 0.48)
            overrides = {"martensite_fraction": mart, "bainite_fraction": 0.18, "cementite_fraction": 0.24, "ferrite_fraction": ferr}
        elif stage_l == "sorbite_temper":
            mart = clamp(0.18 - 0.12 * tempering_level, 0.04, 0.18)
            ferr = clamp(0.42 + 0.18 * tempering_level, 0.42, 0.60)
            overrides = {"martensite_fraction": mart, "bainite_fraction": 0.14, "cementite_fraction": 0.24, "ferrite_fraction": ferr}
        else:
            overrides = {"martensite_fraction": 0.04, "bainite_fraction": 0.10, "cementite_fraction": 0.24, "ferrite_fraction": 0.62}
    return overrides, float(liquid_fraction)


def _fe_c_stage_hint(trace: dict[str, float], tempering_level: float) -> str:
    if float(trace.get("martensite_fraction", 0.0)) > 0.55 and tempering_level < 0.18:
        return "martensite"
    if float(trace.get("bainite_fraction", 0.0)) > 0.40:
        return "bainite"
    if tempering_level >= 0.70:
        return "sorbite_temper"
    if tempering_level >= 0.42:
        return "troostite_temper"
    if float(trace.get("pearlite_fraction", 0.0)) > 0.55:
        return "pearlite"
    if float(trace.get("ferrite_fraction", 0.0)) > 0.55:
        return "ferrite"
    return "mixed_transition"


def _fe_c_state(
    *,
    stage: str,
    composition_wt: dict[str, float] | None,
    processing: ProcessingState,
    effect_vector: dict[str, float] | None,
    thermal_summary: dict[str, Any] | None,
    quench_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    stage_l = str(stage or "").strip().lower()
    effect = dict(effect_vector or {})
    op_summary = _op_summary(thermal_summary)
    c_wt = _comp(composition_wt, "C")
    temp_min = _summary_temp(thermal_summary, processing, "temperature_min_c", processing.temperature_c)
    temp_max = _summary_temp(thermal_summary, processing, "temperature_max_c", processing.temperature_c)
    temp_end = _summary_temp(thermal_summary, processing, "temperature_end_c", processing.temperature_c)
    cool_rate = abs(
        min(
            _safe_float((thermal_summary or {}).get("max_effective_cooling_rate_c_per_s", 0.0), 0.0),
            _safe_float((thermal_summary or {}).get("max_cooling_rate_c_per_s", 0.0), 0.0),
        )
    )
    cool_idx = clamp(max(cooling_index(getattr(processing, "cooling_mode", "equilibrium")), cool_rate / 120.0), 0.0, 1.0)
    has_quench = bool(op_summary.get("has_quench", False) or (quench_summary or {}).get("effect_applied", False))
    has_temper = bool(op_summary.get("has_temper", False))
    medium_code = str((quench_summary or {}).get("medium_code_resolved", (quench_summary or {}).get("medium_code", ""))).strip().lower()
    ra_as_quenched = _safe_float(
        ((quench_summary or {}).get("as_quenched_prediction", {}) or {}).get("retained_austenite_fraction_est", 0.0),
        _safe_float((quench_summary or {}).get("retained_austenite_est_pct", 0.0), 0.0) / 100.0,
    )

    ms_c = _van_bohemen_ms(c_wt, composition_wt)
    bs_c = _van_bohemen_bs(c_wt, composition_wt)
    ferrite_eq, pearlite_eq, cementite_eq = _fe_c_equilibrium(c_wt)
    temper_temp = max(temp_end, _safe_float(op_summary.get("temper_peak_temperature_c", processing.aging_temperature_c), processing.aging_temperature_c))
    temper_hours = max(float(getattr(processing, "aging_hours", 0.0) or 0.0), 0.0)
    tempering_level = clamp(
        max(0.0, (temper_temp - 120.0) / 560.0) * (math.log1p(max(0.0, temper_hours)) / math.log1p(48.0)),
        0.0,
        1.0,
    ) if has_temper or "temper" in stage_l or "troostite" in stage_l or "sorbite" in stage_l else 0.0

    quench_factor = clamp((0.35 if has_quench else 0.0) + 0.75 * cool_idx + (0.08 if medium_code in {"water_20", "water_100", "brine_20_30"} else 0.0), 0.0, 1.0)
    martensite_raw = _koistinen_marburger(ms_c, min(temp_min, temp_end))
    martensite_fraction = martensite_raw * clamp(quench_factor * (1.0 - 0.55 * tempering_level), 0.0, 1.0)
    bainite_window = clamp((bs_c - temp_end) / 220.0, 0.0, 1.0)
    bainite_fraction = clamp((1.0 - martensite_raw) * bainite_window * max(0.0, quench_factor - 0.22) * (1.0 - 0.45 * tempering_level), 0.0, 0.85)
    pearlite_fraction = max(0.0, pearlite_eq * (1.0 - 0.72 * quench_factor) * (1.0 - 0.25 * bainite_fraction))
    ferrite_fraction = max(0.0, ferrite_eq * (1.0 - 0.55 * quench_factor))
    cementite_fraction = max(0.0, cementite_eq + 0.10 * tempering_level + 0.04 * bainite_fraction)
    retained_austenite = clamp(
        max(ra_as_quenched, 0.20 * martensite_raw * clamp(c_wt / 1.1, 0.0, 1.0) * (1.0 - 0.75 * tempering_level)),
        0.0,
        0.35,
    )

    if "martensite" in stage_l:
        martensite_fraction = max(martensite_fraction, 0.58)
    if "bainite" in stage_l:
        bainite_fraction = max(bainite_fraction, 0.42)
    if "pearlite" in stage_l:
        pearlite_fraction = max(pearlite_fraction, 0.45)
    if stage_l == "ferrite":
        ferrite_fraction = max(ferrite_fraction, 0.72)
    if "troostite" in stage_l or "sorbite" in stage_l or "tempered_" in stage_l:
        martensite_fraction *= 0.72
        cementite_fraction += 0.08 * tempering_level + 0.04

    phase_like = _normalize_fraction_payload(
        {
            "ferrite_fraction": ferrite_fraction,
            "pearlite_fraction": pearlite_fraction,
            "bainite_fraction": bainite_fraction,
            "martensite_fraction": martensite_fraction,
            "cementite_fraction": cementite_fraction,
            "retained_austenite": retained_austenite,
        }
    )
    stage_override, liquid_fraction_trace = _fe_c_stage_override(stage_l, retained_austenite, tempering_level)
    if stage_override is not None:
        phase_like = _normalize_fraction_payload(stage_override)

    grain_factor = _safe_float(effect.get("grain_size_factor", 0.0), 0.0)
    prior_austenite_grain_size_px = clamp(92.0 + 54.0 * grain_factor + 0.05 * max(temp_max - 800.0, 0.0) - 18.0 * cool_idx, 28.0, 180.0)
    interlamellar_spacing_px = clamp(9.4 - 5.8 * cool_idx + 1.25 * tempering_level - 0.9 * min(c_wt, 1.0), 1.8, 12.0)
    colony_size_px = clamp(138.0 - 76.0 * cool_idx + 0.18 * prior_austenite_grain_size_px + 18.0 * (1.0 - tempering_level), 24.0, 176.0)
    packet_size_px = clamp(56.0 + 16.0 * max(c_wt - 0.4, 0.0) - 26.0 * cool_idx + 10.0 * (1.0 - tempering_level), 12.0, 92.0)
    martensite_style = "lath_dominant" if c_wt < 0.25 else ("mixed_lath_plate" if c_wt < 0.55 else "plate_dominant")
    lath_plate_ratio = {"lath_dominant": 0.82, "mixed_lath_plate": 0.52, "plate_dominant": 0.22}[martensite_style]
    bainite_sheaf_density = clamp(phase_like.get("bainite_fraction", 0.0) * (0.55 + 0.75 * cool_idx), 0.0, 1.0)
    recovery_level = clamp(0.08 + 0.88 * tempering_level, 0.0, 1.0)
    carbide_scale_px = clamp(1.0 + 4.4 * recovery_level + 0.55 * phase_like.get("cementite_fraction", 0.0), 1.0, 6.2)

    suggested_stage = str(stage_l) if stage_override is not None else _fe_c_stage_hint(phase_like, tempering_level)
    return {
        "system": "fe-c",
        "stage_input": str(stage),
        "transformation_trace": {
            "liquid_fraction": float(liquid_fraction_trace),
            "austenite_fraction": float(phase_like.get("retained_austenite", 0.0)),
            "martensite_fraction": float(phase_like.get("martensite_fraction", 0.0)),
            "bainite_fraction": float(phase_like.get("bainite_fraction", 0.0)),
            "pearlite_fraction": float(phase_like.get("pearlite_fraction", 0.0)),
            "cementite_fraction": float(phase_like.get("cementite_fraction", 0.0)),
            "ferrite_fraction": float(phase_like.get("ferrite_fraction", 0.0)),
            "retained_austenite": float(phase_like.get("retained_austenite", 0.0)),
            "grain_size_proxy": float(prior_austenite_grain_size_px / 180.0),
            "suggested_stage_label": str(suggested_stage),
        },
        "kinetics_model": {
            "family": "physics_guided_hybrid",
            "fe_c_model": "jmak_additivity_plus_koistinen_marburger",
            "martensite_model": "Koistinen-Marburger",
            "diffusional_model": "JMAK-additivity surrogate",
            "ms_c": float(ms_c),
            "bs_c": float(bs_c),
            "cooling_index": float(cool_idx),
            "quench_factor": float(quench_factor),
            "tempering_level": float(tempering_level),
        },
        "morphology_state": {
            "prior_austenite_grain_size_px": float(prior_austenite_grain_size_px),
            "interlamellar_spacing_px": float(interlamellar_spacing_px),
            "colony_size_px": float(colony_size_px),
            "packet_size_px": float(packet_size_px),
            "bainite_sheaf_density": float(bainite_sheaf_density),
            "martensite_style": str(martensite_style),
            "lath_plate_ratio": float(lath_plate_ratio),
        },
        "precipitation_state": {
            "family": "tempering_carbides",
            "carbide_scale_px": float(carbide_scale_px),
            "recovery_level": float(recovery_level),
            "tempering_parameter": float(tempering_level),
            "retained_austenite_decay": float(clamp(tempering_level * 0.82, 0.0, 1.0)),
        },
        "validation_against_rules": {
            "input_stage": str(stage),
            "suggested_stage_label": str(suggested_stage),
            "stage_alignment_score": float(1.0 if suggested_stage in stage_l else 0.55 if stage_l in suggested_stage else 0.25),
            "quench_context_detected": bool(has_quench),
            "temper_context_detected": bool(has_temper),
        },
    }


def _al_si_state(
    *,
    stage: str,
    composition_wt: dict[str, float] | None,
    processing: ProcessingState,
    effect_vector: dict[str, float] | None,
    thermal_summary: dict[str, Any] | None,
    quench_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    _ = effect_vector, thermal_summary, quench_summary
    stage_l = str(stage or "").strip().lower()
    si = _comp(composition_wt, "Si")
    sr = _comp(composition_wt, "Sr")
    na = _comp(composition_wt, "Na")
    temp = float(processing.temperature_c)
    cool_idx = cooling_index(getattr(processing, "cooling_mode", "equilibrium"))
    modifier_active = bool(sr >= 0.005 or na >= 0.002)
    liquid_fraction = 1.0 if "liquid" == stage_l else (0.55 if stage_l.startswith("liquid_") else 0.0)
    primary_si_count = clamp(max(si - 12.6, 0.0) / 14.0 * (0.55 + 0.65 * cool_idx), 0.0, 1.0)
    sdas_px = clamp(68.0 - 34.0 * cool_idx - 0.9 * max(si - 12.6, 0.0), 14.0, 84.0)
    eutectic_scale_px = clamp(10.2 - 4.8 * cool_idx - (1.25 if modifier_active else 0.0), 2.0, 11.0)
    primary_si_size_px = clamp(4.0 + 0.55 * max(si - 12.6, 0.0) + 2.4 * (1.0 - cool_idx), 2.0, 18.0)
    precipitate_scale = clamp(1.2 + 1.8 * max(temp - 120.0, 0.0) / 220.0, 1.0, 4.0) if stage_l in {"supersaturated", "aged"} else 0.0
    return {
        "system": "al-si",
        "stage_input": str(stage),
        "transformation_trace": {
            "liquid_fraction": float(liquid_fraction),
            "solid_fraction": float(max(0.0, 1.0 - liquid_fraction)),
            "primary_si_count_proxy": float(primary_si_count),
            "grain_size_proxy": float(clamp(sdas_px / 84.0, 0.0, 1.0)),
        },
        "kinetics_model": {
            "family": "solidification_surrogate",
            "al_si_model": "sdas_eutectic_modifier_surrogate",
            "cooling_index": float(cool_idx),
            "modifier_active": bool(modifier_active),
        },
        "morphology_state": {
            "sdas_px": float(sdas_px),
            "dendrite_arm_spacing_px": float(sdas_px),
            "eutectic_scale_px": float(eutectic_scale_px),
            "primary_si_size_px": float(primary_si_size_px),
            "primary_si_count_proxy": float(primary_si_count),
            "eutectic_si_modifier": ("fibrous_modified" if modifier_active else "acicular_unmodified"),
        },
        "precipitation_state": {
            "family": "al_si_precipitation",
            "precipitate_scale_px": float(precipitate_scale),
            "supersaturation_level": float(0.65 if stage_l == "supersaturated" else (0.85 if stage_l == "aged" else 0.15)),
        },
        "validation_against_rules": {
            "input_stage": str(stage),
            "modifier_source": ("composition.Sr/Na" if modifier_active else "none"),
            "hyper_eutectic_expected": bool(si >= 12.6),
        },
    }


def _al_cu_mg_state(
    *,
    stage: str,
    composition_wt: dict[str, float] | None,
    processing: ProcessingState,
    effect_vector: dict[str, float] | None,
    thermal_summary: dict[str, Any] | None,
    quench_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    _ = effect_vector, thermal_summary
    stage_l = str(stage or "").strip().lower()
    cu = _comp(composition_wt, "Cu")
    mg = _comp(composition_wt, "Mg")
    quench_factor = 1.0 if str((quench_summary or {}).get("medium_code_resolved", "")).strip().lower() in {"water_20", "water_100", "brine_20_30"} else 0.72
    age_temp = float(max(processing.aging_temperature_c, processing.temperature_c))
    age_hours = float(max(processing.aging_hours, 0.0))
    aging_drive = clamp(max(0.0, (age_temp - 20.0) / 220.0) * (math.log1p(age_hours) / math.log1p(48.0)), 0.0, 1.0)
    if stage_l == "natural_aged":
        aging_drive = max(aging_drive, 0.30)
    if stage_l == "artificial_aged":
        aging_drive = max(aging_drive, 0.72)
    overaged = clamp(max(0.0, (age_temp - 185.0) / 55.0) + max(0.0, age_hours - 18.0) / 30.0, 0.0, 1.0)
    if stage_l == "overaged":
        overaged = max(overaged, 0.55)
    peak_strength = clamp(1.0 - abs(aging_drive - 0.72) / 0.72 - 0.35 * overaged, 0.0, 1.0)
    precipitate_scale_px = clamp(0.8 + 2.1 * aging_drive + 1.9 * overaged + 0.04 * cu + 0.03 * mg, 0.8, 7.2)
    if stage_l in {"solutionized", "quenched"} and aging_drive < 0.18:
        precipitate_scale_px = min(1.35, precipitate_scale_px)
    pfz_width_px = clamp(0.5 + 4.8 * overaged + 1.4 * (1.0 - quench_factor), 0.0, 8.0)
    precip_sequence = {
        "cluster_fraction": float(clamp(1.0 - aging_drive, 0.0, 1.0)),
        "gpb_s2_fraction": float(clamp(aging_drive * (1.0 - overaged), 0.0, 1.0)),
        "s_phase_fraction": float(clamp(0.45 * aging_drive + 0.35 * overaged, 0.0, 1.0)),
        "theta_fraction": float(clamp(0.22 + 0.24 * aging_drive, 0.0, 1.0)),
    }
    return {
        "system": "al-cu-mg",
        "stage_input": str(stage),
        "transformation_trace": {
            "precipitate_state": float(aging_drive),
            "grain_size_proxy": float(clamp(0.42 + 0.22 * (1.0 - overaged), 0.0, 1.0)),
            "peak_strength_fraction": float(peak_strength),
        },
        "kinetics_model": {
            "family": "precipitation_surrogate",
            "al_cu_mg_model": "cluster_gpb_sprime_s_surrogate",
            "aging_drive": float(aging_drive),
            "overaging_drive": float(overaged),
            "quench_factor": float(quench_factor),
        },
        "morphology_state": {
            "grain_size_px": float(clamp(88.0 - 16.0 * aging_drive + 18.0 * overaged, 24.0, 120.0)),
            "precipitate_density_proxy": float(clamp(peak_strength + 0.15, 0.0, 1.0)),
        },
        "precipitation_state": {
            "family": "al_cu_mg_precipitation",
            "precipitate_scale_px": float(precipitate_scale_px),
            "pfz_width_px": float(pfz_width_px),
            "peak_strength_fraction": float(peak_strength),
            "quench_sensitivity": float(clamp((cu + 0.75 * mg) / 7.0, 0.0, 1.0)),
            "precipitation_sequence": precip_sequence,
        },
        "validation_against_rules": {
            "input_stage": str(stage),
            "peak_aged_proxy": bool(peak_strength >= 0.65 and overaged < 0.35),
            "overaged_proxy": bool(overaged >= 0.5),
        },
    }


def _cu_zn_state(
    *,
    stage: str,
    composition_wt: dict[str, float] | None,
    processing: ProcessingState,
    effect_vector: dict[str, float] | None,
    thermal_summary: dict[str, Any] | None,
    quench_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    _ = effect_vector, thermal_summary, quench_summary
    stage_l = str(stage or "").strip().lower()
    zn = _comp(composition_wt, "Zn")
    def_pct = float(max(getattr(processing, "deformation_pct", 0.0), 0.0))
    temp = float(processing.temperature_c)
    cool_idx = cooling_index(getattr(processing, "cooling_mode", "equilibrium"))
    recrystallized = clamp((temp - 350.0) / 350.0 + 0.28 - def_pct * 0.002, 0.0, 1.0)
    if stage_l == "cold_worked":
        recrystallized *= 0.35
    recovery = clamp(0.18 + 0.55 * recrystallized + max(temp - 250.0, 0.0) / 900.0, 0.0, 1.0)
    twin_density = clamp(0.12 + 0.42 * recrystallized + 0.08 * max(zn - 30.0, 0.0) / 10.0, 0.0, 1.0)
    band_density = clamp(def_pct / 60.0 * (1.0 - 0.75 * recrystallized), 0.0, 1.0)
    ordering_factor = clamp(max(zn - 35.0, 0.0) / 10.0 + 0.22 * recovery + (0.35 if stage_l == "beta_prime" else 0.0), 0.0, 1.0)
    return {
        "system": "cu-zn",
        "stage_input": str(stage),
        "transformation_trace": {
            "alpha_fraction": float(clamp(1.0 - max(zn - 34.0, 0.0) / 18.0, 0.0, 1.0)),
            "beta_fraction": float(clamp(max(zn - 32.0, 0.0) / 18.0, 0.0, 1.0)),
            "grain_size_proxy": float(clamp(0.3 + 0.45 * recrystallized, 0.0, 1.0)),
            "recovery_fraction": float(recovery),
        },
        "kinetics_model": {
            "family": "recovery_recrystallization_surrogate",
            "cu_zn_model": "stacking_fault_twinning_surrogate",
            "cooling_index": float(cool_idx),
            "recrystallized_fraction": float(recrystallized),
        },
        "morphology_state": {
            "twin_density": float(twin_density),
            "recrystallized_fraction": float(recrystallized),
            "recovery_fraction": float(recovery),
            "ordering_factor": float(ordering_factor),
            "deformation_band_density": float(band_density),
            "grain_size_px": float(clamp(92.0 - 0.65 * def_pct - 18.0 * cool_idx + 30.0 * recrystallized, 20.0, 128.0)),
            "beta_boundary_bias": float(clamp(0.45 + 0.28 * ordering_factor + 0.12 * cool_idx, 0.0, 1.0)),
        },
        "precipitation_state": {
            "family": "ordering_and_boundary_segmentation",
            "beta_prime_ordering_fraction": float(ordering_factor),
        },
        "validation_against_rules": {
            "input_stage": str(stage),
            "cold_work_signature": bool(band_density >= 0.25),
            "annealed_signature": bool(recrystallized >= 0.45),
        },
    }


def _fe_si_state(
    *,
    stage: str,
    composition_wt: dict[str, float] | None,
    processing: ProcessingState,
    effect_vector: dict[str, float] | None,
    thermal_summary: dict[str, Any] | None,
    quench_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    _ = quench_summary
    stage_l = str(stage or "").strip().lower()
    si = _comp(composition_wt, "Si")
    def_pct = float(max(getattr(processing, "deformation_pct", 0.0), 0.0))
    temp = _summary_temp(thermal_summary, processing, "temperature_end_c", processing.temperature_c)
    cool_idx = cooling_index(getattr(processing, "cooling_mode", "equilibrium"))
    grain_bias = _safe_float((effect_vector or {}).get("grain_size_factor", 0.0), 0.0)
    recrystallized = clamp((temp - 500.0) / 350.0 + 0.20 + 0.25 * max(si - 1.5, 0.0) / 4.0 - def_pct * 0.0015, 0.0, 1.0)
    if "cold_worked" in stage_l:
        recrystallized *= 0.28
    band_density = clamp(def_pct / 55.0 * (1.0 - 0.70 * recrystallized), 0.0, 1.0)
    texture_sharpness = clamp(0.32 + 0.48 * recrystallized + 0.10 * max(si, 0.0) / 4.0, 0.0, 1.0)
    grain_size_px = clamp(46.0 + 58.0 * recrystallized + 24.0 * grain_bias - 18.0 * cool_idx, 16.0, 128.0)
    return {
        "system": "fe-si",
        "stage_input": str(stage),
        "transformation_trace": {
            "ferrite_fraction": 1.0,
            "grain_size_proxy": float(clamp(grain_size_px / 128.0, 0.0, 1.0)),
            "recrystallized_fraction": float(recrystallized),
        },
        "kinetics_model": {
            "family": "recovery_recrystallization_surrogate",
            "fe_si_model": "bcc_recrystallization_texture_surrogate",
            "cooling_index": float(cool_idx),
            "recrystallized_fraction": float(recrystallized),
        },
        "morphology_state": {
            "cold_work_band_density": float(band_density),
            "recrystallized_fraction": float(recrystallized),
            "texture_sharpness": float(texture_sharpness),
            "grain_size_px": float(grain_size_px),
        },
        "precipitation_state": {
            "family": "none",
            "stored_energy_proxy": float(clamp(def_pct / 100.0, 0.0, 1.0)),
        },
        "validation_against_rules": {
            "input_stage": str(stage),
            "cold_work_signature": bool(band_density >= 0.18),
            "recrystallized_signature": bool(recrystallized >= 0.45),
        },
    }


def build_transformation_state(
    *,
    inferred_system: str,
    stage: str,
    composition_wt: dict[str, float] | None,
    processing: ProcessingState,
    effect_vector: dict[str, float] | None = None,
    thermal_summary: dict[str, Any] | None = None,
    quench_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system = str(inferred_system or "").strip().lower()
    if system == "fe-c":
        return _fe_c_state(
            stage=stage,
            composition_wt=composition_wt,
            processing=processing,
            effect_vector=effect_vector,
            thermal_summary=thermal_summary,
            quench_summary=quench_summary,
        )
    if system == "al-si":
        return _al_si_state(
            stage=stage,
            composition_wt=composition_wt,
            processing=processing,
            effect_vector=effect_vector,
            thermal_summary=thermal_summary,
            quench_summary=quench_summary,
        )
    if system == "al-cu-mg":
        return _al_cu_mg_state(
            stage=stage,
            composition_wt=composition_wt,
            processing=processing,
            effect_vector=effect_vector,
            thermal_summary=thermal_summary,
            quench_summary=quench_summary,
        )
    if system == "cu-zn":
        return _cu_zn_state(
            stage=stage,
            composition_wt=composition_wt,
            processing=processing,
            effect_vector=effect_vector,
            thermal_summary=thermal_summary,
            quench_summary=quench_summary,
        )
    if system == "fe-si":
        return _fe_si_state(
            stage=stage,
            composition_wt=composition_wt,
            processing=processing,
            effect_vector=effect_vector,
            thermal_summary=thermal_summary,
            quench_summary=quench_summary,
        )
    return {
        "system": str(system),
        "stage_input": str(stage),
        "transformation_trace": {"grain_size_proxy": 0.5},
        "kinetics_model": {"family": "fallback"},
        "morphology_state": {},
        "precipitation_state": {},
        "validation_against_rules": {"input_stage": str(stage)},
    }


def metadata_blocks_from_transformation_state(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    state = dict(payload or {})
    return {
        "transformation_trace": dict(state.get("transformation_trace", {})),
        "kinetics_model": dict(state.get("kinetics_model", {})),
        "morphology_state": dict(state.get("morphology_state", {})),
        "precipitation_state": dict(state.get("precipitation_state", {})),
        "validation_against_rules": dict(state.get("validation_against_rules", {})),
    }
