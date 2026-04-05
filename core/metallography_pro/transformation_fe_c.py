from __future__ import annotations

import math
from typing import Any

from core.contracts_v2 import ProcessingState
from core.metallography_v3.realism_utils import clamp, cooling_index
from core.metallography_v3.system_generators.base import normalize_phase_fractions

from .contracts import ContinuousTransformationState


def _phase_value(phase_fractions: dict[str, float], *names: str) -> float:
    return float(sum(float(phase_fractions.get(name, 0.0)) for name in names))


def _tempering_level(processing: ProcessingState, thermal_summary: dict[str, Any] | None) -> float:
    payload = dict(thermal_summary or {})
    op_summary = dict(payload.get("operation_inference", {})) if isinstance(payload.get("operation_inference", {}), dict) else {}
    has_temper_context = bool(op_summary.get("has_temper", False)) or "temper" in str(getattr(processing, "cooling_mode", "")).strip().lower()
    if not has_temper_context:
        return 0.0
    peak_t = float(
        op_summary.get(
            "temper_peak_temperature_c",
            payload.get("temperature_end_c", getattr(processing, "temperature_c", 20.0)),
        )
    )
    aging_h = float(max(getattr(processing, "aging_hours", 0.0) or 0.0, 0.0))
    base = max(0.0, (peak_t - 120.0) / 560.0)
    time_factor = math.log1p(aging_h) / math.log1p(24.0) if aging_h > 0.0 else 0.0
    return float(clamp(max(base, base * time_factor), 0.0, 1.0))


def _ae1_temperature_c() -> float:
    return 727.0


def _ae3_temperature_c(carbon_wt: float) -> float:
    c = float(max(0.0, carbon_wt))
    if c >= 0.77:
        return _ae1_temperature_c()
    return float(clamp(910.0 - 203.0 * math.sqrt(max(c, 1e-9)) + 44.7 * c - 15.2 * c * c, 727.0, 910.0))


def _t0_temperature_c(carbon_wt: float) -> float:
    c = float(max(0.0, carbon_wt))
    return float(clamp(835.0 - 198.0 * c, 420.0, 835.0))


def _segment_time_in_temperature_range(
    temp0_c: float,
    temp1_c: float,
    dt_s: float,
    low_c: float,
    high_c: float,
) -> float:
    low = float(min(low_c, high_c))
    high = float(max(low_c, high_c))
    if dt_s <= 1e-9:
        return 0.0
    if abs(temp1_c - temp0_c) <= 1e-9:
        return float(dt_s if low <= temp0_c <= high else 0.0)

    t_min = min(temp0_c, temp1_c)
    t_max = max(temp0_c, temp1_c)
    overlap_low = max(low, t_min)
    overlap_high = min(high, t_max)
    if overlap_high <= overlap_low:
        return 0.0
    frac = (overlap_high - overlap_low) / max(t_max - t_min, 1e-9)
    return float(dt_s * frac)


def _window_exposure_counters(thermal_summary: dict[str, Any], *, ae1_c: float, ae3_c: float, bs_c: float, ms_c: float) -> dict[str, float]:
    segments = list(thermal_summary.get("segments", [])) if isinstance(thermal_summary.get("segments", []), list) else []
    out = {
        "austenitization_hold_s": 0.0,
        "time_in_upper_c_window_s": 0.0,
        "time_in_lower_c_window_s": 0.0,
        "time_below_ms_s": 0.0,
        "time_in_bainite_hold_s": 0.0,
    }
    if not segments:
        hold_s = float(max(0.0, thermal_summary.get("hold_time_s", 0.0) or 0.0))
        t_max = float(thermal_summary.get("temperature_max_c", thermal_summary.get("temperature_end_c", 20.0)) or 20.0)
        t_end = float(thermal_summary.get("temperature_end_c", t_max) or t_max)
        if t_max >= ae3_c:
            out["austenitization_hold_s"] = hold_s
        if bs_c < t_end < ae3_c:
            out["time_in_upper_c_window_s"] = hold_s
        if ms_c < t_end < bs_c:
            out["time_in_lower_c_window_s"] = hold_s
            out["time_in_bainite_hold_s"] = hold_s
        if t_end < ms_c:
            out["time_below_ms_s"] = hold_s
        return {k: float(v) for k, v in out.items()}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        dt_s = float(seg.get("dt_s", seg.get("duration_s", 0.0)) or 0.0)
        temp0_c = float(seg.get("temp0_c", 0.0) or 0.0)
        temp1_c = float(seg.get("temp1_c", 0.0) or 0.0)
        kind = str(seg.get("kind", "") or "")
        out["time_in_upper_c_window_s"] += _segment_time_in_temperature_range(temp0_c, temp1_c, dt_s, bs_c, ae3_c)
        out["time_in_lower_c_window_s"] += _segment_time_in_temperature_range(temp0_c, temp1_c, dt_s, ms_c, bs_c)
        out["time_below_ms_s"] += _segment_time_in_temperature_range(temp0_c, temp1_c, dt_s, -273.15, ms_c)
        if kind == "hold":
            mean_temp = 0.5 * (temp0_c + temp1_c)
            if mean_temp >= ae3_c:
                out["austenitization_hold_s"] += dt_s
            if ms_c < mean_temp < bs_c:
                out["time_in_bainite_hold_s"] += dt_s
    return {k: float(v) for k, v in out.items()}


def _triangular_temperature_weight(temp_c: float, low_c: float, center_c: float, high_c: float) -> float:
    low = float(min(low_c, high_c))
    high = float(max(low_c, high_c))
    center = float(clamp(center_c, low, high))
    temp = float(temp_c)
    if temp <= low or temp >= high:
        return 0.0
    if temp <= center:
        return float(clamp((temp - low) / max(center - low, 1e-9), 0.0, 1.0))
    return float(clamp((high - temp) / max(high - center, 1e-9), 0.0, 1.0))


def _family_effective_exposures(
    thermal_summary: dict[str, Any],
    *,
    ae1_c: float,
    ae3_c: float,
    bs_c: float,
    ms_c: float,
    t0_c: float,
) -> dict[str, float]:
    segments = list(thermal_summary.get("segments", [])) if isinstance(thermal_summary.get("segments", []), list) else []
    ferrite_low = ae1_c + 8.0
    ferrite_high = ae3_c
    ferrite_center = ae1_c + 0.62 * max(ae3_c - ae1_c, 1e-6)
    pearlite_low = max(ms_c + 15.0, ae1_c - 150.0)
    pearlite_high = ae1_c + 30.0
    pearlite_center = float(clamp(ae1_c - 35.0, pearlite_low, pearlite_high))
    bainite_low = ms_c + 12.0
    bainite_high = min(bs_c, t0_c + 35.0)
    bainite_center = bainite_low + 0.58 * max(bainite_high - bainite_low, 1e-6)
    out = {
        "ferrite_effective_exposure_s": 0.0,
        "pearlite_effective_exposure_s": 0.0,
        "bainite_effective_exposure_s": 0.0,
        "martensite_effective_exposure_s": 0.0,
    }
    if not segments:
        hold_s = float(max(0.0, thermal_summary.get("hold_time_s", 0.0) or 0.0))
        t_max = float(thermal_summary.get("temperature_max_c", thermal_summary.get("temperature_end_c", 20.0)) or 20.0)
        t_end = float(thermal_summary.get("temperature_end_c", t_max) or t_max)
        out["ferrite_effective_exposure_s"] = hold_s * _triangular_temperature_weight(t_end, ferrite_low, ferrite_center, ferrite_high)
        out["pearlite_effective_exposure_s"] = hold_s * _triangular_temperature_weight(t_end, pearlite_low, pearlite_center, pearlite_high)
        out["bainite_effective_exposure_s"] = hold_s * _triangular_temperature_weight(t_end, bainite_low, bainite_center, bainite_high)
        undercool_frac = clamp((ms_c - t_end) / max(ms_c - 20.0, 1e-6), 0.0, 1.0) if t_end < ms_c else 0.0
        out["martensite_effective_exposure_s"] = hold_s * undercool_frac
        return {k: float(v) for k, v in out.items()}

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        dt_s = float(seg.get("dt_s", seg.get("duration_s", 0.0)) or 0.0)
        if dt_s <= 1e-9:
            continue
        temp0_c = float(seg.get("temp0_c", 0.0) or 0.0)
        temp1_c = float(seg.get("temp1_c", 0.0) or 0.0)
        kind = str(seg.get("kind", "") or "").strip().lower()
        mean_temp = 0.5 * (temp0_c + temp1_c)
        hold_gain = 1.15 if kind == "hold" else 0.72
        cool_gain = 0.88 if kind == "cool" else 1.0

        ferrite_dt = _segment_time_in_temperature_range(temp0_c, temp1_c, dt_s, ferrite_low, ferrite_high)
        pearlite_dt = _segment_time_in_temperature_range(temp0_c, temp1_c, dt_s, pearlite_low, pearlite_high)
        bainite_dt = _segment_time_in_temperature_range(temp0_c, temp1_c, dt_s, bainite_low, bainite_high)
        martensite_dt = _segment_time_in_temperature_range(temp0_c, temp1_c, dt_s, -273.15, ms_c)

        out["ferrite_effective_exposure_s"] += ferrite_dt * _triangular_temperature_weight(mean_temp, ferrite_low, ferrite_center, ferrite_high) * hold_gain
        out["pearlite_effective_exposure_s"] += pearlite_dt * _triangular_temperature_weight(mean_temp, pearlite_low, pearlite_center, pearlite_high) * hold_gain
        out["bainite_effective_exposure_s"] += bainite_dt * _triangular_temperature_weight(mean_temp, bainite_low, bainite_center, bainite_high) * hold_gain
        if martensite_dt > 0.0:
            undercool_frac = clamp((ms_c - mean_temp) / max(ms_c - 20.0, 1e-6), 0.0, 1.0)
            out["martensite_effective_exposure_s"] += martensite_dt * (0.35 + 0.65 * undercool_frac) * cool_gain
    return {k: float(v) for k, v in out.items()}


def _saturating_progress(exposure_s: float, tau_s: float) -> float:
    exposure = float(max(0.0, exposure_s))
    tau = float(max(1e-6, tau_s))
    return float(clamp(1.0 - math.exp(-exposure / tau), 0.0, 1.0))


def _max_cooling_rate_c_per_s(thermal_summary: dict[str, Any]) -> float:
    explicit = float(max(0.0, thermal_summary.get("max_cooling_rate_c_per_s", 0.0) or 0.0))
    if explicit > 0.0:
        return explicit
    segments = list(thermal_summary.get("segments", [])) if isinstance(thermal_summary.get("segments", []), list) else []
    max_rate = 0.0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        dt_s = float(seg.get("dt_s", seg.get("duration_s", 0.0)) or 0.0)
        temp0_c = float(seg.get("temp0_c", 0.0) or 0.0)
        temp1_c = float(seg.get("temp1_c", 0.0) or 0.0)
        if dt_s > 1e-9 and temp1_c < temp0_c:
            max_rate = max(max_rate, abs(temp1_c - temp0_c) / dt_s)
    return float(max_rate)


def _porter_hardenability_factor(
    *,
    carbon_wt: float,
    cool_idx: float,
    has_quench: bool,
    max_cooling_rate_c_per_s: float,
) -> float:
    carbon_factor = min(max(carbon_wt, 0.0) / 0.85, 1.0)
    cooling_factor = min(max(max_cooling_rate_c_per_s, 0.0) / 80.0, 1.0)
    return float(
        clamp(
            0.18 + 0.44 * carbon_factor + 0.20 * cool_idx + 0.10 * cooling_factor + (0.08 if has_quench else 0.0),
            0.0,
            1.0,
        )
    )


def build_continuous_transformation_state(
    *,
    composition_wt: dict[str, float],
    stage: str,
    phase_fractions: dict[str, float],
    processing: ProcessingState,
    thermal_summary: dict[str, Any] | None = None,
    quench_summary: dict[str, Any] | None = None,
) -> ContinuousTransformationState:
    phases_in = normalize_phase_fractions(dict(phase_fractions))
    stage_l = str(stage or "").strip().lower()
    carbon = float(max(0.0, composition_wt.get("C", 0.0)))
    cool_idx = float(cooling_index(str(getattr(processing, "cooling_mode", ""))))
    thermal = dict(thermal_summary or {})
    t_max = float(thermal.get("temperature_max_c", getattr(processing, "temperature_c", 20.0)))
    t_end = float(thermal.get("temperature_end_c", getattr(processing, "temperature_c", 20.0)))
    hold_s = float(max(0.0, thermal.get("hold_time_s", 0.0)))
    max_cooling_rate = _max_cooling_rate_c_per_s(thermal)
    hold_factor = math.log1p(max(0.0, hold_s) / 60.0) / math.log1p(120.0)
    tempering_level = _tempering_level(processing=processing, thermal_summary=thermal)
    op_summary = dict(thermal.get("operation_inference", {})) if isinstance(thermal.get("operation_inference", {}), dict) else {}
    has_quench = bool(op_summary.get("has_quench", False) or dict(quench_summary or {}).get("effect_applied", False))
    martensite_input_fraction = _phase_value(phases_in, "MARTENSITE", "MARTENSITE_TETRAGONAL", "MARTENSITE_CUBIC")

    ae1_c = _ae1_temperature_c()
    ae3_c = _ae3_temperature_c(carbon)
    ms_c = float(clamp(565.0 - 600.0 * (1.0 - math.exp(-1.15 * carbon)), 50.0, 565.0))
    bs_c = float(clamp(830.0 - 360.0 * (1.0 - math.exp(-1.25 * carbon)), 180.0, 830.0))
    t0_c = _t0_temperature_c(carbon)
    counters = _window_exposure_counters(thermal, ae1_c=ae1_c, ae3_c=ae3_c, bs_c=bs_c, ms_c=ms_c)
    effective_exposures = _family_effective_exposures(thermal, ae1_c=ae1_c, ae3_c=ae3_c, bs_c=bs_c, ms_c=ms_c, t0_c=t0_c)
    undercool_ae3 = max(0.0, ae3_c - t_end)
    bainite_window = clamp((bs_c - t_end) / max(bs_c - ms_c, 1e-6), 0.0, 1.0) if t_end < bs_c else 0.0
    martensite_window = clamp((ms_c - t_end) / max(ms_c, 1e-6), 0.0, 1.0) if t_end < ms_c else 0.0
    ferrite_window = clamp((ae3_c - t_end) / max(ae3_c - ae1_c, 1e-6), 0.0, 1.0) if t_end < ae3_c else 0.0
    pearlite_window = clamp((ae1_c - t_end + 220.0) / 220.0, 0.0, 1.0)
    explicit_temper_stage = stage_l in {"tempered_low", "tempered_medium", "tempered_high", "troostite_temper", "sorbite_temper"}
    martensitic_context = bool(martensite_input_fraction > 0.05 or explicit_temper_stage)
    if bainite_window > 0.15 and not explicit_temper_stage and martensite_input_fraction < 0.05:
        tempering_level = 0.0
    has_temper = bool((op_summary.get("has_temper", False) or tempering_level > 0.05 or explicit_temper_stage) and martensitic_context)
    if stage_l == "alpha_pearlite":
        boundary_bias_context = clamp(0.72 - 0.22 * abs(carbon - 0.45), 0.45, 0.88)
    elif stage_l == "pearlite_cementite":
        boundary_bias_context = clamp(0.78 + 0.10 * min(1.0, max(0.0, carbon - 0.77) / 0.5), 0.68, 0.94)
    else:
        boundary_bias_context = 0.18 if ferrite_window > 0.20 else 0.0

    continuous_cooling_shift_factor = float(
        clamp(
            0.30
            + 0.42 * cool_idx
            + 0.18 * min(max_cooling_rate / 80.0, 1.0)
            + (0.10 if has_quench else 0.0),
            0.0,
            1.0,
        )
    )
    hardenability_factor = _porter_hardenability_factor(
        carbon_wt=carbon,
        cool_idx=cool_idx,
        has_quench=has_quench,
        max_cooling_rate_c_per_s=max_cooling_rate,
    )
    ferrite_nucleation_drive = float(
        clamp(
            (0.22 + 0.78 * _saturating_progress(effective_exposures["ferrite_effective_exposure_s"], 220.0))
            * ferrite_window
            * (0.32 + 0.68 * max(0.0, boundary_bias_context))
            * (1.0 - 0.72 * hardenability_factor * continuous_cooling_shift_factor),
            0.0,
            1.0,
        )
    )
    pearlite_nucleation_drive = float(
        clamp(
            (0.24 + 0.76 * _saturating_progress(effective_exposures["pearlite_effective_exposure_s"], 180.0))
            * pearlite_window
            * (1.0 - 0.56 * hardenability_factor * continuous_cooling_shift_factor),
            0.0,
            1.0,
        )
    )
    bainite_nucleation_drive = float(
        clamp(
            (0.26 + 0.74 * _saturating_progress(effective_exposures["bainite_effective_exposure_s"], 145.0))
            * bainite_window
            * (0.42 + 0.58 * hardenability_factor)
            * (0.30 + 0.70 * min(1.0, counters["time_in_bainite_hold_s"] / 240.0)),
            0.0,
            1.0,
        )
    )
    diffusional_equivalent_time_s = float(
        max(
            0.0,
            (
                effective_exposures["ferrite_effective_exposure_s"]
                + effective_exposures["pearlite_effective_exposure_s"]
                + 0.85 * effective_exposures["bainite_effective_exposure_s"]
            )
            * (1.0 - 0.45 * hardenability_factor * continuous_cooling_shift_factor),
        )
    )

    ferrite_progress = float(
        clamp(
            _saturating_progress(effective_exposures["ferrite_effective_exposure_s"], 230.0)
            * ferrite_nucleation_drive
            * (0.35 + 0.65 * ferrite_window)
            * (1.0 - bainite_window * 0.50)
            * (1.0 - martensite_window * (0.70 + 0.15 * hardenability_factor)),
            0.0,
            1.0,
        )
    )
    pearlite_progress = float(
        clamp(
            _saturating_progress(effective_exposures["pearlite_effective_exposure_s"], 190.0)
            * pearlite_nucleation_drive
            * (0.30 + 0.70 * pearlite_window)
            * (1.0 - bainite_window * (0.35 + 0.15 * hardenability_factor))
            * (1.0 - martensite_window * (0.50 + 0.15 * hardenability_factor)),
            0.0,
            1.0,
        )
    )
    bainite_activation_progress = float(
        clamp(
            _saturating_progress(effective_exposures["bainite_effective_exposure_s"], 150.0)
            * bainite_nucleation_drive
            * (0.35 + 0.65 * bainite_window)
            * (0.30 + 0.70 * min(1.0, counters["time_in_bainite_hold_s"] / 240.0)),
            0.0,
            1.0,
        )
    )
    km_progress = float(clamp(1.0 - math.exp(-0.011 * max(0.0, ms_c - t_end)), 0.0, 1.0)) if t_end < ms_c else 0.0
    martensite_conversion_progress = float(
        clamp(
            km_progress
            * _saturating_progress(effective_exposures["martensite_effective_exposure_s"], 45.0)
            * (0.72 + 0.28 * hardenability_factor)
            * (1.0 if has_quench else 0.55),
            0.0,
            1.0,
        )
    )

    family_weights = {
        "ferritic_family": float(
            clamp(
                (1.0 - cool_idx)
                * ferrite_progress,
                0.0,
                1.0,
            )
        ),
        "pearlitic_family": float(
            clamp(
                (1.0 - cool_idx * 0.85)
                * pearlite_progress,
                0.0,
                1.0,
            )
        ),
        "bainitic_family": float(
            clamp(
                (0.20 + 0.80 * cool_idx)
                * bainite_activation_progress
                * (1.0 - martensite_conversion_progress * 0.30),
                0.0,
                1.0,
            )
        ),
        "martensitic_family": float(
            clamp(
                (0.35 + 0.65 * cool_idx)
                * martensite_conversion_progress,
                0.0,
                1.0,
            )
        ),
    }
    if has_temper:
        family_weights["tempered_martensitic_family"] = float(clamp(max(family_weights["martensitic_family"], 0.25) * tempering_level, 0.0, 1.0))
    total_fw = float(sum(family_weights.values()))
    if total_fw > 1e-9:
        family_weights = {k: float(v / total_fw) for k, v in family_weights.items()}

    ferritic_weight = float(family_weights.get("ferritic_family", 0.0))
    pearlitic_weight = float(family_weights.get("pearlitic_family", 0.0))

    phases = dict(phases_in)
    bainite_target = 0.0
    if (
        has_quench
        and not has_temper
        and t_end > ms_c + 5.0
        and t_end < bs_c + 25.0
        and t_end < t0_c + 80.0
    ):
        bainite_target = float(clamp(0.18 + 0.62 * bainite_activation_progress, 0.0, 0.88))
    if bainite_target > float(phases.get("BAINITE", 0.0)):
        shift_needed = bainite_target - float(phases.get("BAINITE", 0.0))
        donor_order = ["PEARLITE", "FERRITE", "MARTENSITE", "AUSTENITE"]
        for donor in donor_order:
            if shift_needed <= 1e-9:
                break
            available = float(phases.get(donor, 0.0))
            if available <= 1e-9:
                continue
            delta = min(available, shift_needed)
            phases[donor] = available - delta
            phases["BAINITE"] = float(phases.get("BAINITE", 0.0) + delta)
            shift_needed -= delta
        phases = normalize_phase_fractions(phases)

    ferrite_fraction = _phase_value(phases, "FERRITE")
    tempered_product_fraction = _phase_value(phases, "SORBITE", "TROOSTITE")
    pearlite_fraction = _phase_value(phases, "PEARLITE")
    cementite_fraction = _phase_value(phases, "CEMENTITE")
    martensite_fraction = _phase_value(phases, "MARTENSITE", "MARTENSITE_TETRAGONAL", "MARTENSITE_CUBIC")
    bainite_fraction = _phase_value(phases, "BAINITE")
    retained_austenite_fraction = _phase_value(phases, "AUSTENITE")
    ferrite_pearlite_numerator = pearlite_fraction * max(pearlitic_weight, 1e-9) * max(pearlite_progress, 1e-9)
    ferrite_pearlite_denominator = max(
        ferrite_pearlite_numerator + ferrite_fraction * max(ferritic_weight, 1e-9) * max(ferrite_progress, 1e-9),
        1e-9,
    )
    ferrite_pearlite_competition_index = float(clamp(ferrite_pearlite_numerator / ferrite_pearlite_denominator, 0.0, 1.0))

    austenitize_excess = max(0.0, t_max - 760.0)
    prior_austenite_grain_size_um = clamp(14.0 + 26.0 * min(1.4, austenitize_excess / 220.0) + 10.0 * hold_factor - 7.0 * cool_idx, 6.0, 96.0)
    colony_size_um_mean = clamp(prior_austenite_grain_size_um * (0.55 + 0.25 * (1.0 - cool_idx)), 2.5, 60.0)
    colony_size_um_std = clamp(colony_size_um_mean * 0.24, 0.5, 18.0)
    spacing_base = 1.18 - 0.72 * cool_idx - 0.20 * min(carbon, 1.2) + 0.30 * tempering_level
    interlamellar_spacing_um_mean = clamp(spacing_base, 0.12, 1.60)
    interlamellar_spacing_um_std = clamp(interlamellar_spacing_um_mean * 0.18, 0.02, 0.32)

    proeutectoid_boundary_bias = float(boundary_bias_context if stage_l in {"alpha_pearlite", "pearlite_cementite"} else 0.0)

    martensite_packet_size_um = clamp(prior_austenite_grain_size_um * (0.24 + 0.20 * (1.0 - cool_idx)) + 6.0 * max(0.0, carbon - 0.35), 0.8, 24.0)
    bainite_sheaf_length_um = clamp(prior_austenite_grain_size_um * (0.18 + 0.22 * cool_idx) + 2.0 * max(0.0, bs_c - t_end) / 100.0, 2.0, 28.0)
    bainite_sheaf_thickness_um = clamp(0.15 + 0.65 * max(0.0, 1.0 - cool_idx) + 0.10 * max(0.0, carbon - 0.2), 0.12, 1.40)
    bainite_sheaf_density = clamp((0.55 + 0.75 * cool_idx) * max(bainite_fraction, 0.0), 0.0, 1.0)
    carbide_size_um = clamp(0.05 + 0.42 * tempering_level + 0.18 * cementite_fraction, 0.03, 0.95)
    source_phase = str(dict(quench_summary or {}).get("effect_applied", False)).lower()
    martensite_style = "lath_dominant" if carbon < 0.25 else ("mixed_lath_plate" if carbon < 0.55 else "plate_dominant")

    if stage_l in {"bainite", "troostite_quench", "sorbite_quench"} or bainite_fraction > max(0.10, martensite_fraction * 0.75):
        bainite_family = "upper_bainite_sheaves" if t_end >= 350.0 else "lower_bainite_sheaves"
    else:
        bainite_family = "none"
    ferrite_family = "none"
    if ferrite_fraction > 0.05:
        ferrite_boundary_strength = float(
            clamp(
                0.40 * min(1.0, effective_exposures["ferrite_effective_exposure_s"] / 220.0)
                + 0.30 * min(1.0, counters["time_in_upper_c_window_s"] / 240.0)
                + 0.30 * max(0.0, boundary_bias_context),
                0.0,
                1.0,
            )
        )
        widmanstatten_propensity = float(
            clamp(
                (0.45 * cool_idx + 0.30 * continuous_cooling_shift_factor + 0.25 * clamp(undercool_ae3 / 80.0, 0.0, 1.0))
                * (1.0 - ferrite_boundary_strength)
                * (1.0 - 0.45 * pearlite_fraction),
                0.0,
                1.0,
            )
        )
        if widmanstatten_propensity >= 0.34:
            ferrite_family = "widmanstatten"
        else:
            ferrite_family = "allotriomorphic"
    pearlite_family = "lamellar_colonies" if pearlite_fraction > 0.10 else "none"
    if stage_l in {"troostite_temper", "sorbite_temper", "tempered_low", "tempered_medium", "tempered_high"} or (has_temper and (martensite_fraction > 0.05 or tempered_product_fraction > 0.10)):
        transformation_family = "tempered_martensitic_family"
    elif martensite_fraction > 0.35 and has_quench:
        transformation_family = "martensitic_family"
    elif bainite_fraction > 0.15 or stage_l in {"bainite", "troostite_quench", "sorbite_quench"}:
        transformation_family = "bainitic_family"
    elif ferrite_fraction > 0.20 and pearlite_fraction > 0.20:
        if ferrite_pearlite_competition_index <= 0.42:
            transformation_family = "ferritic_family"
        elif ferrite_pearlite_competition_index >= 0.58:
            transformation_family = "pearlitic_family"
        else:
            transformation_family = "mixed_family"
    elif pearlite_fraction > 0.20:
        transformation_family = "pearlitic_family"
    elif ferrite_fraction > 0.20:
        transformation_family = "ferritic_family"
    else:
        transformation_family = "mixed_family"
    if transformation_family in {"martensitic_family", "tempered_martensitic_family", "bainitic_family", "ferritic_family"}:
        growth_mode = "displacive"
    else:
        growth_mode = "reconstructive"
    if transformation_family == "martensitic_family":
        partitioning_mode = "diffusionless"
    elif transformation_family == "bainitic_family":
        partitioning_mode = "displacive_with_carbon_rejection"
    elif transformation_family == "pearlitic_family":
        partitioning_mode = "eutectoid_diffusional"
    else:
        partitioning_mode = "partitioning_diffusional"
    incomplete_transformation_limit_active = bool(transformation_family == "bainitic_family" and t_end < t0_c + 25.0)
    return ContinuousTransformationState(
        system="fe-c",
        resolved_stage=str(stage),
        transformation_family=str(transformation_family),
        growth_mode=str(growth_mode),
        partitioning_mode=str(partitioning_mode),
        incomplete_transformation_limit_active=bool(incomplete_transformation_limit_active),
        ferrite_morphology_family=str(ferrite_family),
        bainite_morphology_family=str(bainite_family),
        martensite_morphology_family=str(martensite_style),
        pearlite_morphology_family=str(pearlite_family),
        family_weights=family_weights,
        phase_fractions=phases,
        ferrite_fraction=float(ferrite_fraction),
        pearlite_fraction=float(pearlite_fraction),
        cementite_fraction=float(cementite_fraction),
        martensite_fraction=float(martensite_fraction),
        bainite_fraction=float(bainite_fraction),
        retained_austenite_fraction=float(retained_austenite_fraction),
        ae1_temperature_c=float(ae1_c),
        ae3_temperature_c=float(ae3_c),
        bs_temperature_c=float(bs_c),
        ms_temperature_c=float(ms_c),
        t0_temperature_c=float(t0_c),
        austenitization_hold_s=float(counters["austenitization_hold_s"]),
        time_in_upper_c_window_s=float(counters["time_in_upper_c_window_s"]),
        time_in_lower_c_window_s=float(counters["time_in_lower_c_window_s"]),
        time_below_ms_s=float(counters["time_below_ms_s"]),
        time_in_bainite_hold_s=float(counters["time_in_bainite_hold_s"]),
        ferrite_effective_exposure_s=float(effective_exposures["ferrite_effective_exposure_s"]),
        pearlite_effective_exposure_s=float(effective_exposures["pearlite_effective_exposure_s"]),
        bainite_effective_exposure_s=float(effective_exposures["bainite_effective_exposure_s"]),
        martensite_effective_exposure_s=float(effective_exposures["martensite_effective_exposure_s"]),
        diffusional_equivalent_time_s=float(diffusional_equivalent_time_s),
        hardenability_factor=float(hardenability_factor),
        continuous_cooling_shift_factor=float(continuous_cooling_shift_factor),
        ferrite_nucleation_drive=float(ferrite_nucleation_drive),
        pearlite_nucleation_drive=float(pearlite_nucleation_drive),
        bainite_nucleation_drive=float(bainite_nucleation_drive),
        ferrite_progress=float(ferrite_progress),
        pearlite_progress=float(pearlite_progress),
        ferrite_pearlite_competition_index=float(ferrite_pearlite_competition_index),
        bainite_activation_progress=float(bainite_activation_progress),
        martensite_conversion_progress=float(martensite_conversion_progress),
        prior_austenite_grain_size_um=float(prior_austenite_grain_size_um),
        colony_size_um_mean=float(colony_size_um_mean),
        colony_size_um_std=float(colony_size_um_std),
        interlamellar_spacing_um_mean=float(interlamellar_spacing_um_mean),
        interlamellar_spacing_um_std=float(interlamellar_spacing_um_std),
        proeutectoid_boundary_bias=float(proeutectoid_boundary_bias),
        martensite_packet_size_um=float(martensite_packet_size_um),
        bainite_sheaf_length_um=float(bainite_sheaf_length_um),
        bainite_sheaf_thickness_um=float(bainite_sheaf_thickness_um),
        bainite_sheaf_density=float(bainite_sheaf_density),
        carbide_size_um=float(carbide_size_um),
        recovery_level=float(tempering_level),
        confidence={
            "phase_fractions": 0.88,
            "grain_scale": 0.70,
            "spacing": 0.66,
            "family_gating": 0.67,
            "surface_link": 0.62,
        },
        provenance={
            "phase_fractions": "phase_bundle",
            "grain_scale": "S4/S5/Steels engineering fit",
            "spacing": "S8/S9/Steels engineering fit",
            "thermodynamics": "Porter/Easterling ch.1 equilibrium, driving-force, lever-rule surrogate",
            "diffusion": "Porter/Easterling ch.2 Arrhenius diffusion weighting surrogate",
            "interface_growth": "Porter/Easterling ch.3 interface-controlled vs diffusion-controlled growth framing",
            "diffusional_transformations": "Porter/Easterling ch.5 TTT/CCT/additivity/hardenability surrogate",
            "diffusionless_transformations": "Porter/Easterling ch.6 martensite/tempering surrogate",
            "martensite": "S11 + Bhadeshia martensite",
            "bainite": "Bhadeshia bainite family gating",
            "bainite_split": "upper/lower bainite used as morphology family only",
            "ferrite": "Bhadeshia ferrite family gating",
            "ferrite_split": "allotriomorphic/widmanstatten split from cooling and undercooling context",
            "pearlite": "Bhadeshia pearlite family gating",
            "derived_labels": "troostite/sorbite retained as engineering labels over existing families",
            "scheduler": "family-specific effective exposure scheduler",
            "quench_context": f"quench_summary.effect_applied={source_phase}",
        },
    )
