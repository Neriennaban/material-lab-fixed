from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts_v2 import ProcessingState
from core.contracts_v3 import ThermalPointV3, ThermalProgramV3, ThermalTransitionV3
from core.metallography_v3.quench_media_v3 import defaults_quench, resolve_quench_medium

_QUENCH_MEDIUM_CODES: set[str] = {
    "water_20",
    "water_100",
    "brine_20_30",
    "oil_20_80",
    "polymer",
    "custom",
}

_DEFAULT_TRANSITION_MODELS: set[str] = {"linear", "sigmoid", "power", "cosine"}
_RULES_PATH = Path(__file__).resolve().parents[1] / "rulebook" / "thermal_transition_rules_v3.json"


def _load_transition_rules() -> dict[str, Any]:
    if not _RULES_PATH.exists():
        return {}
    try:
        return json.loads(_RULES_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


_TRANSITION_RULES = _load_transition_rules()

@dataclass(slots=True)
class ThermalSegmentV3:
    index: int
    t0_s: float
    t1_s: float
    temp0_c: float
    temp1_c: float
    dt_s: float
    dtemp_c: float
    slope_c_per_s: float
    kind: str  # heat | cool | hold

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "t0_s": float(self.t0_s),
            "t1_s": float(self.t1_s),
            "temp0_c": float(self.temp0_c),
            "temp1_c": float(self.temp1_c),
            "dt_s": float(self.dt_s),
            "dtemp_c": float(self.dtemp_c),
            "slope_c_per_s": float(self.slope_c_per_s),
            "kind": str(self.kind),
        }


def _segment_to_dict(seg: ThermalSegmentV3) -> dict[str, Any]:
    return {
        "segment_index": int(seg.index),
        "t0_s": float(seg.t0_s),
        "t1_s": float(seg.t1_s),
        "temp0_c": float(seg.temp0_c),
        "temp1_c": float(seg.temp1_c),
        "duration_s": float(seg.dt_s),
        "slope_c_per_s": float(seg.slope_c_per_s),
        "kind": str(seg.kind),
    }


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _rules_medium_defaults() -> dict[str, Any]:
    payload = _TRANSITION_RULES.get("medium_defaults", {})
    return payload if isinstance(payload, dict) else {}


def _rules_fallback() -> dict[str, Any]:
    payload = _TRANSITION_RULES.get("fallback", {})
    return payload if isinstance(payload, dict) else {}


def _sanitize_transition(raw: ThermalTransitionV3 | None) -> ThermalTransitionV3:
    if not isinstance(raw, ThermalTransitionV3):
        raw = ThermalTransitionV3()
    mode = str(raw.model or "linear").strip().lower()
    if mode not in _DEFAULT_TRANSITION_MODELS:
        mode = "linear"
    curv = _clamp(float(raw.curvature), 0.15, 12.0)
    medium = str(raw.segment_medium_code or "inherit").strip().lower() or "inherit"
    if medium not in {
        "inherit",
        "water_20",
        "water_100",
        "brine_20_30",
        "oil_20_80",
        "polymer",
        "air",
        "furnace",
        "custom",
    }:
        medium = "inherit"
    factor = raw.segment_medium_factor
    medium_factor = None if factor is None else _clamp(float(factor), 0.2, 2.8)
    return ThermalTransitionV3(
        model=mode,
        curvature=curv,
        segment_medium_code=medium,
        segment_medium_factor=medium_factor,
        notes=str(raw.notes or ""),
    )


def _resolve_segment_medium_code(
    segment_kind: str,
    transition_medium_code: str,
    global_quench: dict[str, Any],
) -> str:
    code = str(transition_medium_code or "inherit").strip().lower()
    if code != "inherit":
        return code
    if str(segment_kind) == "cool":
        quench_code = str(global_quench.get("medium_code_resolved", global_quench.get("medium_code", "air"))).strip().lower()
        if quench_code:
            return quench_code
    return "air"


def _resolve_transition_for_segment(
    prev_point: ThermalPointV3,
    next_point: ThermalPointV3,
    segment_kind: str,
    global_quench: dict[str, Any],
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ = next_point
    transition = _sanitize_transition(getattr(prev_point, "transition_to_next", None))
    rules = rules if isinstance(rules, dict) else {}
    by_medium = _rules_medium_defaults()
    fallback = _rules_fallback()
    medium_code = _resolve_segment_medium_code(segment_kind, transition.segment_medium_code, global_quench)
    medium_payload = by_medium.get(medium_code, {}) if isinstance(by_medium.get(medium_code), dict) else {}

    model = str(transition.model or "").strip().lower()
    if model not in _DEFAULT_TRANSITION_MODELS:
        model = str(medium_payload.get("model", "")).strip().lower()
    if model not in _DEFAULT_TRANSITION_MODELS:
        model = str(fallback.get("model", "linear")).strip().lower()
    if model not in _DEFAULT_TRANSITION_MODELS:
        model = "linear"

    curvature = float(transition.curvature)
    if curvature <= 0.0:
        curvature = float(medium_payload.get("curvature", 1.0))
    curvature = _clamp(curvature, 0.15, 12.0)
    curv_low = float(medium_payload.get("curvature_min", fallback.get("curvature_min", 0.2)))
    curv_high = float(medium_payload.get("curvature_max", fallback.get("curvature_max", 8.0)))
    if curv_high < curv_low:
        curv_low, curv_high = curv_high, curv_low
    curvature = _clamp(curvature, curv_low, curv_high)

    if transition.segment_medium_factor is None:
        medium_factor = float(medium_payload.get("rate_factor", fallback.get("rate_factor", 1.0)))
    else:
        medium_factor = float(transition.segment_medium_factor)
    medium_factor = _clamp(medium_factor, 0.2, 2.8)

    return {
        "model": model,
        "curvature": float(curvature),
        "segment_medium_code": str(medium_code),
        "segment_medium_factor": float(medium_factor),
        "notes": str(transition.notes or ""),
    }


def _transition_fn(u: np.ndarray, model: str, curvature: float) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    mode = str(model or "linear").strip().lower()
    c = max(0.15, float(curvature))
    if mode == "sigmoid":
        k = 2.0 + c * 3.0
        num = 1.0 / (1.0 + np.exp(-k * (u - 0.5)))
        lo = 1.0 / (1.0 + math.exp(-k * (-0.5)))
        hi = 1.0 / (1.0 + math.exp(-k * 0.5))
        den = max(1e-9, hi - lo)
        return np.clip((num - lo) / den, 0.0, 1.0)
    if mode == "power":
        k = max(0.25, c)
        out = np.empty_like(u)
        left = u <= 0.5
        out[left] = 0.5 * np.power(np.clip(u[left] * 2.0, 0.0, 1.0), k)
        out[~left] = 1.0 - 0.5 * np.power(np.clip((1.0 - u[~left]) * 2.0, 0.0, 1.0), k)
        return np.clip(out, 0.0, 1.0)
    if mode == "cosine":
        return np.clip(0.5 - 0.5 * np.cos(np.pi * u), 0.0, 1.0)
    return u


def _segment_transition_report(
    seg: ThermalSegmentV3,
    transition: dict[str, Any],
) -> dict[str, Any]:
    return {
        "segment_index": int(seg.index),
        "kind": str(seg.kind),
        "model": str(transition.get("model", "linear")),
        "curvature": float(transition.get("curvature", 1.0)),
        "segment_medium_code": str(transition.get("segment_medium_code", "air")),
        "effective_rate_factor": float(transition.get("segment_medium_factor", 1.0)),
    }


def _sample_segment_rows(
    seg: ThermalSegmentV3,
    transition: dict[str, Any],
    degree_step: float,
    max_frames: int,
) -> list[dict[str, Any]]:
    model = str(transition.get("model", "linear"))
    curvature = float(transition.get("curvature", 1.0))
    medium = str(transition.get("segment_medium_code", "air"))
    medium_factor = float(transition.get("segment_medium_factor", 1.0))

    if seg.kind == "hold":
        count = max(2, min(max_frames, int(round(seg.dt_s / 30.0)) + 1))
    else:
        dtemp = abs(seg.temp1_c - seg.temp0_c)
        count = max(2, int(np.ceil(dtemp / degree_step)) + 1)
        count = min(count, max_frames)

    u = np.linspace(0.0, 1.0, num=count)
    eased = _transition_fn(u, model=model, curvature=curvature)
    ts = seg.t0_s + (seg.t1_s - seg.t0_s) * u
    if seg.kind == "hold":
        temps = np.full_like(ts, seg.temp0_c, dtype=np.float64)
    else:
        temps = seg.temp0_c + (seg.temp1_c - seg.temp0_c) * eased

    out: list[dict[str, Any]] = []
    for uu, tt, tc in zip(u, ts, temps, strict=True):
        out.append(
            {
                "segment": int(seg.index),
                "time_s": float(tt),
                "temperature_c": float(tc),
                "kind": str(seg.kind),
                "u": float(uu),
                "model": model,
                "curvature": float(curvature),
                "segment_medium_code": medium,
                "segment_medium_factor": float(medium_factor),
            }
        )
    return out


def infer_operations_from_thermal_program(
    program: ThermalProgramV3,
    *,
    summary: dict[str, Any] | None = None,
    quench_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer educational technological operations from thermal curve segments."""
    points = normalize_thermal_points(program.points)
    segments = build_segments(points)
    if summary is None:
        summary = summarize_thermal_program(program)
    qsum = dict(quench_summary or {})
    transition_rules = _TRANSITION_RULES
    ops: list[dict[str, Any]] = []

    t_max = float(summary.get("temperature_max_c", max(float(p.temperature_c) for p in points)))
    medium_code = str(qsum.get("medium_code_resolved", qsum.get("medium_code", ""))).strip().lower()
    # Quench presence is strictly inferred from curve cooling segment(s), independent of medium settings.
    quench_threshold = -8.0
    transition_by_segment: dict[int, dict[str, Any]] = {}
    for seg in segments:
        p0 = points[int(seg.index)]
        p1 = points[int(seg.index) + 1]
        transition_by_segment[int(seg.index)] = _resolve_transition_for_segment(
            prev_point=p0,
            next_point=p1,
            segment_kind=str(seg.kind),
            global_quench=qsum,
            rules=transition_rules,
        )

    high_temp_holds: list[ThermalSegmentV3] = []
    post_quench_heats: list[ThermalSegmentV3] = []
    post_quench_holds: list[ThermalSegmentV3] = []
    had_quench = False
    quench_detected_by_curve = False
    first_quench_segment_index: int | None = None
    temper_mean_temps: list[float] = []
    temper_hold_total_s = 0.0

    # Pass 1: core operations from direct segments.
    for seg in segments:
        transition = transition_by_segment.get(int(seg.index), {})
        seg_rows = _sample_segment_rows(
            seg=seg,
            transition=transition,
            degree_step=max(0.1, float(program.degree_step_c)),
            max_frames=max(24, min(600, int(program.max_frames))),
        )
        medium_factor = float(transition.get("segment_medium_factor", 1.0))
        local_rates: list[float] = []
        for idx in range(1, len(seg_rows)):
            dt = float(seg_rows[idx]["time_s"]) - float(seg_rows[idx - 1]["time_s"])
            if abs(dt) < 1e-9:
                continue
            dtemp = float(seg_rows[idx]["temperature_c"]) - float(seg_rows[idx - 1]["temperature_c"])
            local_rates.append(float(dtemp / dt) * max(0.2, medium_factor))
        if local_rates:
            if str(seg.kind) == "cool":
                effective_slope = float(min(local_rates))
            elif str(seg.kind) == "heat":
                effective_slope = float(max(local_rates))
            else:
                effective_slope = 0.0
        else:
            effective_slope = float(seg.slope_c_per_s) * max(0.2, medium_factor)
        mean_temp = 0.5 * (float(seg.temp0_c) + float(seg.temp1_c))
        if seg.kind == "heat" and float(seg.temp1_c) >= 700.0:
            ops.append(
                {
                    "code": "austenitization_heat",
                    "label_ru": "Нагрев под аустенизацию",
                    "confidence": 0.92 if float(seg.temp1_c) >= 760.0 else 0.78,
                    "reason": "Нагрев до высокой температуры (>=700°C).",
                    "segment": _segment_to_dict(seg),
                }
            )
        if seg.kind == "hold" and mean_temp >= 680.0 and float(seg.dt_s) >= 45.0:
            high_temp_holds.append(seg)
            ops.append(
                {
                    "code": "austenitization_hold",
                    "label_ru": "Выдержка при аустенизации",
                    "confidence": 0.90 if mean_temp >= 760.0 else 0.76,
                    "reason": "Изотермическая выдержка в аустенитной области.",
                    "segment": _segment_to_dict(seg),
                }
            )
        if seg.kind == "cool":
            raw_slope = float(seg.slope_c_per_s)
            slope = float(effective_slope)
            if raw_slope <= quench_threshold:
                had_quench = True
                quench_detected_by_curve = True
                if first_quench_segment_index is None:
                    first_quench_segment_index = int(seg.index)
                ops.append(
                    {
                        "code": "quench_cooling",
                        "label_ru": "Закалочное охлаждение",
                        "confidence": 0.95 if slope <= -20.0 else 0.82,
                        "reason": "Высокая скорость охлаждения на участке кривой.",
                        "segment": {
                            **_segment_to_dict(seg),
                            "slope_c_per_s_effective": float(effective_slope),
                            "medium_code_used": str(transition.get("segment_medium_code", "air")),
                        },
                    }
                )
            elif slope <= -2.0:
                ops.append(
                    {
                        "code": "accelerated_cooling",
                        "label_ru": "Ускоренное охлаждение",
                        "confidence": 0.72,
                        "reason": "Эффективное охлаждение быстрее спокойного воздушного.",
                        "segment": {
                            **_segment_to_dict(seg),
                            "slope_c_per_s_effective": float(effective_slope),
                            "medium_code_used": str(transition.get("segment_medium_code", "air")),
                        },
                    }
                )
            else:
                ops.append(
                    {
                        "code": "slow_cooling",
                        "label_ru": "Медленное охлаждение",
                        "confidence": 0.70,
                        "reason": "Низкая эффективная скорость охлаждения.",
                        "segment": {
                            **_segment_to_dict(seg),
                            "slope_c_per_s_effective": float(effective_slope),
                            "medium_code_used": str(transition.get("segment_medium_code", "air")),
                        },
                    }
                )

    temper_low_min = 150.0
    temper_low_max = 250.0
    temper_medium_max = 450.0
    temper_high_max = 650.0

    # Pass 2: temper operations after quench.
    if had_quench:
        for seg in segments:
            if first_quench_segment_index is not None and first_quench_segment_index >= 0 and int(seg.index) <= first_quench_segment_index:
                continue
            if seg.kind == "heat" and float(seg.temp1_c) >= temper_low_min and float(seg.temp1_c) <= temper_high_max:
                post_quench_heats.append(seg)
            if seg.kind == "hold":
                mean_temp = 0.5 * (float(seg.temp0_c) + float(seg.temp1_c))
                if temper_low_min <= mean_temp <= temper_high_max and float(seg.dt_s) >= 30.0:
                    post_quench_holds.append(seg)

        temper_candidates = post_quench_holds or post_quench_heats
        for seg in temper_candidates:
            mean_temp = 0.5 * (float(seg.temp0_c) + float(seg.temp1_c))
            temper_mean_temps.append(float(mean_temp))
            if seg.kind == "hold":
                temper_hold_total_s += float(seg.dt_s)
            if mean_temp <= temper_low_max:
                code = "temper_low"
                label = "Низкий отпуск"
            elif mean_temp <= temper_medium_max:
                code = "temper_medium"
                label = "Средний отпуск"
            else:
                code = "temper_high"
                label = "Высокий отпуск"
            ops.append(
                {
                    "code": code,
                    "label_ru": label,
                    "confidence": 0.80 if seg.kind == "hold" else 0.66,
                    "reason": "Нагрев/выдержка после закалки в диапазоне отпускных температур.",
                    "segment": _segment_to_dict(seg),
                }
            )

    # Deduplicate near-identical operations for cleaner report.
    dedup: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for op in ops:
        seg_payload = op.get("segment", {})
        if isinstance(seg_payload, dict):
            seg_idx = int(seg_payload.get("segment_index", -1))
        else:
            seg_idx = -1
        key = (str(op.get("code", "")), seg_idx)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(op)

    has_temper = bool(any(str(op.get("code", "")).startswith("temper_") for op in dedup))
    has_quench = bool(any(str(op.get("code", "")) == "quench_cooling" for op in dedup))
    temper_peak_t = float(max(temper_mean_temps)) if temper_mean_temps else 0.0
    temper_mean_t = float(sum(temper_mean_temps) / len(temper_mean_temps)) if temper_mean_temps else 0.0
    temper_code = ""
    temper_band_detected = ""
    temper_band_confidence = 0.0
    if has_temper:
        if temper_peak_t <= temper_low_max:
            temper_code = "temper_low"
            temper_band_detected = "low"
        elif temper_peak_t <= temper_medium_max:
            temper_code = "temper_medium"
            temper_band_detected = "medium"
        else:
            temper_code = "temper_high"
            temper_band_detected = "high"

        band_center = {
            "low": 200.0,
            "medium": 350.0,
            "high": 550.0,
        }
        center = float(band_center.get(temper_band_detected, temper_peak_t))
        spread = {
            "low": 50.0,
            "medium": 100.0,
            "high": 100.0,
        }
        scale = max(1.0, float(spread.get(temper_band_detected, 120.0)))
        temp_fit = max(0.0, 1.0 - abs(float(temper_peak_t) - center) / scale)
        hold_fit = _clamp(float(temper_hold_total_s) / 600.0, 0.0, 1.0)
        temper_band_confidence = _clamp(0.45 + 0.35 * temp_fit + 0.2 * hold_fit, 0.0, 0.99)

    temper_shift_map = dict(qsum.get("temper_shift_c", {})) if isinstance(qsum.get("temper_shift_c", {}), dict) else {}
    shift_lookup = {
        "temper_low": float(temper_shift_map.get("low", 0.0)),
        "temper_medium": float(temper_shift_map.get("medium", 0.0)),
        "temper_high": float(temper_shift_map.get("high", 0.0)),
    }
    recommended_temper_shift_c = float(shift_lookup.get(temper_code, 0.0)) if temper_code else 0.0

    summary_ops = {
        "count": int(len(dedup)),
        "has_austenitization": bool(any(str(op.get("code", "")).startswith("austenitization") for op in dedup)),
        "has_quench": has_quench,
        "has_temper": has_temper,
        "quench_detected_by_curve": bool(quench_detected_by_curve),
        "quench_presence_rule": "curve_only",
        "max_temperature_c": float(t_max),
        "temper_peak_temperature_c": float(temper_peak_t),
        "temper_mean_temperature_c": float(temper_mean_t),
        "temper_total_hold_s": float(temper_hold_total_s),
        "temper_code": str(temper_code),
        "temper_band_detected": str(temper_band_detected),
        "temper_band_confidence": float(temper_band_confidence),
        "recommended_temper_shift_c": float(recommended_temper_shift_c),
        "recommended_temper_shift_map_c": shift_lookup,
        "medium_code_resolved": str(medium_code),
        "segment_transition_report": [
            _segment_transition_report(seg, transition_by_segment.get(int(seg.index), {}))
            for seg in segments
        ],
        "recommended_cooling_mode": str(temper_code if has_temper else ("quenched" if has_quench else "equilibrium")),
        "stage_inference_profile": "fe_c_temper_curve_v2",
    }
    return {
        "operations": dedup,
        "summary": summary_ops,
        "source": "thermal_curve_inference_v1",
    }

def normalize_thermal_points(points: list[ThermalPointV3] | list[dict[str, Any]], fallback_temp_c: float = 20.0) -> list[ThermalPointV3]:
    if not isinstance(points, list):
        points = []
    out: list[ThermalPointV3] = []
    for item in points:
        if isinstance(item, ThermalPointV3):
            point = item
        elif isinstance(item, dict):
            point = ThermalPointV3.from_dict(item)
        else:
            continue
        out.append(
            ThermalPointV3(
                time_s=max(0.0, float(point.time_s)),
                temperature_c=float(point.temperature_c),
                label=str(point.label),
                locked=bool(point.locked),
                transition_to_next=_sanitize_transition(getattr(point, "transition_to_next", None)),
            )
        )

    if len(out) < 2:
        out = [
            ThermalPointV3(time_s=0.0, temperature_c=float(fallback_temp_c), label="Старт", locked=True),
            ThermalPointV3(time_s=600.0, temperature_c=float(fallback_temp_c), label="Финиш", locked=False),
        ]
    out.sort(key=lambda p: float(p.time_s))

    dedup: list[ThermalPointV3] = []
    for point in out:
        if dedup and abs(float(point.time_s) - float(dedup[-1].time_s)) < 1e-9:
            dedup[-1] = point
        else:
            dedup.append(point)
    if len(dedup) < 2:
        dedup.append(ThermalPointV3(time_s=float(dedup[0].time_s) + 1.0, temperature_c=float(dedup[0].temperature_c)))
    return dedup


def build_segments(points: list[ThermalPointV3]) -> list[ThermalSegmentV3]:
    norm = normalize_thermal_points(points)
    out: list[ThermalSegmentV3] = []
    for idx in range(len(norm) - 1):
        p0 = norm[idx]
        p1 = norm[idx + 1]
        dt = max(1e-9, float(p1.time_s) - float(p0.time_s))
        dtemp = float(p1.temperature_c) - float(p0.temperature_c)
        slope = dtemp / dt
        if abs(dtemp) <= 0.5:
            kind = "hold"
        elif dtemp > 0.0:
            kind = "heat"
        else:
            kind = "cool"
        out.append(
            ThermalSegmentV3(
                index=idx,
                t0_s=float(p0.time_s),
                t1_s=float(p1.time_s),
                temp0_c=float(p0.temperature_c),
                temp1_c=float(p1.temperature_c),
                dt_s=float(dt),
                dtemp_c=float(dtemp),
                slope_c_per_s=float(slope),
                kind=kind,
            )
        )
    return out


def validate_thermal_program(program: ThermalProgramV3) -> dict[str, Any]:
    points = normalize_thermal_points(program.points)
    errors: list[str] = []
    warnings: list[str] = []
    if len(points) < 2:
        errors.append("Нужно минимум 2 точки термопрограммы.")
    for idx in range(1, len(points)):
        if points[idx].time_s <= points[idx - 1].time_s:
            errors.append(f"Точки {idx} и {idx+1}: время должно строго возрастать.")
    if max(float(p.temperature_c) for p in points) < 200.0:
        warnings.append("Максимальная температура < 200°C: термообработка может быть слабо выраженной.")
    segments = build_segments(points)
    if not any(seg.kind == "cool" for seg in segments):
        warnings.append("В термопрограмме нет явного участка охлаждения.")
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "point_count": len(points),
        "segments": [seg.to_dict() for seg in segments],
    }


def summarize_thermal_program(program: ThermalProgramV3) -> dict[str, Any]:
    points = normalize_thermal_points(program.points)
    segments = build_segments(points)
    q_defaults = defaults_quench()
    q_payload = program.quench.to_dict() if hasattr(program, "quench") else {}
    qsum = resolve_quench_medium(
        str(q_payload.get("medium_code", q_defaults["medium_code"])),
        quench_time_s=float(q_payload.get("quench_time_s", q_defaults["quench_time_s"])),
        bath_temperature_c=float(q_payload.get("bath_temperature_c", q_defaults["bath_temperature_c"])),
        sample_temperature_c=float(q_payload.get("sample_temperature_c", max(float(p.temperature_c) for p in points))),
        custom_medium_name=str(q_payload.get("custom_medium_name", "")),
        custom_severity_factor=float(q_payload.get("custom_severity_factor", q_defaults["custom_severity_factor"])),
    )
    temps = [float(p.temperature_c) for p in points]
    slopes = [float(seg.slope_c_per_s) for seg in segments]
    effective_slopes: list[float] = []
    transition_report: list[dict[str, Any]] = []
    nonlinear_count = 0
    for seg in segments:
        p0 = points[int(seg.index)]
        p1 = points[int(seg.index) + 1]
        tr = _resolve_transition_for_segment(
            prev_point=p0,
            next_point=p1,
            segment_kind=str(seg.kind),
            global_quench=qsum,
            rules=_TRANSITION_RULES,
        )
        medium_factor = float(tr.get("segment_medium_factor", 1.0))
        seg_rows = _sample_segment_rows(
            seg=seg,
            transition=tr,
            degree_step=max(0.1, float(program.degree_step_c)),
            max_frames=max(24, min(800, int(program.max_frames))),
        )
        local_rates: list[float] = []
        for idx in range(1, len(seg_rows)):
            dt = float(seg_rows[idx]["time_s"]) - float(seg_rows[idx - 1]["time_s"])
            if abs(dt) < 1e-9:
                continue
            dtemp = float(seg_rows[idx]["temperature_c"]) - float(seg_rows[idx - 1]["temperature_c"])
            local_rates.append(float(dtemp / dt) * max(0.2, medium_factor))
        if local_rates:
            if str(seg.kind) == "cool":
                effective_slopes.append(float(min(local_rates)))
            elif str(seg.kind) == "heat":
                effective_slopes.append(float(max(local_rates)))
            else:
                effective_slopes.append(0.0)
        else:
            factor = float(tr.get("segment_medium_factor", 1.0))
            effective_slopes.append(float(seg.slope_c_per_s) * max(0.2, factor))
        transition_report.append(_segment_transition_report(seg, tr))
        if str(tr.get("model", "linear")).strip().lower() != "linear":
            nonlinear_count += 1
    hold_s = float(sum(seg.dt_s for seg in segments if seg.kind == "hold"))
    heat_s = float(sum(seg.dt_s for seg in segments if seg.kind == "heat"))
    cool_s = float(sum(seg.dt_s for seg in segments if seg.kind == "cool"))

    cycles = 0
    for idx in range(1, len(segments)):
        prev = segments[idx - 1].kind
        cur = segments[idx].kind
        if prev == "cool" and cur == "heat":
            cycles += 1

    duration_s = float(points[-1].time_s - points[0].time_s)
    if duration_s > 1e-9 and segments:
        avg_temp = 0.0
        for seg in segments:
            avg_temp += float(seg.dt_s) * (float(seg.temp0_c) + float(seg.temp1_c)) * 0.5
        temperature_avg_c = float(avg_temp / duration_s)
    else:
        temperature_avg_c = float(sum(temps) / max(1, len(temps)))

    return {
        "point_count": len(points),
        "duration_s": duration_s,
        "temperature_min_c": float(min(temps)),
        "temperature_max_c": float(max(temps)),
        "temperature_avg_c": float(temperature_avg_c),
        "temperature_start_c": float(points[0].temperature_c),
        "temperature_end_c": float(points[-1].temperature_c),
        "hold_time_s": hold_s,
        "heat_time_s": heat_s,
        "cool_time_s": cool_s,
        "max_heating_rate_c_per_s": float(max([s for s in slopes if s > 0.0] or [0.0])),
        "max_cooling_rate_c_per_s": float(min([s for s in slopes if s < 0.0] or [0.0])),
        "max_effective_heating_rate_c_per_s": float(max([s for s in effective_slopes if s > 0.0] or [0.0])),
        "max_effective_cooling_rate_c_per_s": float(min([s for s in effective_slopes if s < 0.0] or [0.0])),
        "thermal_cycles": int(cycles),
        "segments": [seg.to_dict() for seg in segments],
        "curve_interpolation_mode": "mixed_per_segment",
        "nonlinear_enabled": bool(nonlinear_count > 0),
        "segment_count_with_nonlinear": int(nonlinear_count),
        "segment_transition_report": transition_report,
    }


def sample_thermal_program(program: ThermalProgramV3) -> list[dict[str, Any]]:
    points = normalize_thermal_points(program.points)
    q_defaults = defaults_quench()
    q_payload = program.quench.to_dict() if hasattr(program, "quench") else {}
    qsum = resolve_quench_medium(
        str(q_payload.get("medium_code", q_defaults["medium_code"])),
        quench_time_s=float(q_payload.get("quench_time_s", q_defaults["quench_time_s"])),
        bath_temperature_c=float(q_payload.get("bath_temperature_c", q_defaults["bath_temperature_c"])),
        sample_temperature_c=float(q_payload.get("sample_temperature_c", max(float(p.temperature_c) for p in points))),
        custom_medium_name=str(q_payload.get("custom_medium_name", "")),
        custom_severity_factor=float(q_payload.get("custom_severity_factor", q_defaults["custom_severity_factor"])),
    )
    mode = str(program.sampling_mode or "per_degree").strip().lower()
    if mode not in {"per_degree", "points"}:
        mode = "per_degree"
    if mode == "points":
        return [
            {
                "index": idx,
                "time_s": float(p.time_s),
                "temperature_c": float(p.temperature_c),
                "label": str(p.label),
                "model": str(getattr(p.transition_to_next, "model", "linear")),
                "curvature": float(getattr(p.transition_to_next, "curvature", 1.0)),
                "segment_medium_code": str(getattr(p.transition_to_next, "segment_medium_code", "inherit")),
                "segment_medium_factor": getattr(p.transition_to_next, "segment_medium_factor", None),
            }
            for idx, p in enumerate(points)
        ]

    degree_step = max(0.1, float(program.degree_step_c))
    max_frames = max(2, int(program.max_frames))
    segments = build_segments(points)
    samples: list[dict[str, Any]] = []
    for seg in segments:
        p0 = points[int(seg.index)]
        p1 = points[int(seg.index) + 1]
        transition = _resolve_transition_for_segment(
            prev_point=p0,
            next_point=p1,
            segment_kind=str(seg.kind),
            global_quench=qsum,
            rules=_TRANSITION_RULES,
        )
        rows = _sample_segment_rows(
            seg=seg,
            transition=transition,
            degree_step=degree_step,
            max_frames=max_frames,
        )
        samples.extend(rows)

    compact: list[dict[str, Any]] = []
    for row in samples:
        if compact:
            prev = compact[-1]
            if abs(float(prev["time_s"]) - float(row["time_s"])) < 1e-6 and abs(float(prev["temperature_c"]) - float(row["temperature_c"])) < 1e-6:
                continue
        compact.append(row)
    return compact[: max_frames]


def effective_processing_from_thermal(program: ThermalProgramV3) -> tuple[ProcessingState, dict[str, Any], dict[str, Any]]:
    points = normalize_thermal_points(program.points)
    summary = summarize_thermal_program(program)
    quench_defaults = defaults_quench()
    q_payload = program.quench.to_dict() if hasattr(program, "quench") else {}
    q = resolve_quench_medium(
        str(q_payload.get("medium_code", quench_defaults["medium_code"])),
        quench_time_s=float(q_payload.get("quench_time_s", quench_defaults["quench_time_s"])),
        bath_temperature_c=float(q_payload.get("bath_temperature_c", quench_defaults["bath_temperature_c"])),
        sample_temperature_c=float(q_payload.get("sample_temperature_c", summary["temperature_max_c"])),
        custom_medium_name=str(q_payload.get("custom_medium_name", "")),
        custom_severity_factor=float(q_payload.get("custom_severity_factor", quench_defaults["custom_severity_factor"])),
    )

    op_payload = infer_operations_from_thermal_program(
        program,
        summary=summary,
        quench_summary=q,
    )
    op_summary = dict(op_payload.get("summary", {}))
    effect_applied = bool(op_summary.get("has_quench", False))
    effect_reason = "curve_quench_detected" if effect_applied else "no_quench_segment"

    medium_code_resolved = str(q.get("medium_code_resolved", q.get("medium_code", ""))).strip().lower()
    cooling_mode = "equilibrium"
    if bool(op_summary.get("has_temper", False)):
        cooling_mode = str(op_summary.get("temper_code", "")).strip().lower() or "tempered"
    elif effect_applied:
        cooling_mode = "quenched"
    elif medium_code_resolved in {"air", "furnace"}:
        cooling_mode = "slow_cool"

    observed_temp = float(points[-1].temperature_c)
    quench_observed_estimate = float(observed_temp)
    if effect_applied and float(q["quench_time_s"]) > 0.0:
        delta = float(q["sample_temperature_c"]) - float(q["bath_temperature_c"])
        tau = max(0.2, 22.0 / max(0.1, float(q["severity_effective"])))
        quench_observed_estimate = float(q["bath_temperature_c"]) + delta * math.exp(-float(q["quench_time_s"]) / tau)

    max_cool_rate = max(
        abs(float(summary.get("max_cooling_rate_c_per_s", 0.0))),
        abs(float(summary.get("max_effective_cooling_rate_c_per_s", 0.0))),
    )
    if effect_applied and float(q["quench_time_s"]) > 0.0:
        span = max(1e-6, float(q["quench_time_s"]))
        quench_rate = abs(float(q["sample_temperature_c"]) - float(quench_observed_estimate)) / span
        max_cool_rate = max(max_cool_rate, quench_rate)

    processing = ProcessingState(
        temperature_c=float(observed_temp),
        cooling_mode=str(cooling_mode),
        deformation_pct=0.0,
        aging_hours=0.0,
        aging_temperature_c=float(observed_temp),
        note=f"thermal_program:{len(points)}",
    )
    summary["operation_inference"] = dict(op_summary)
    summary["observed_temperature_c"] = float(observed_temp)
    summary["quench_observed_temperature_estimate_c"] = float(quench_observed_estimate)
    summary["max_effective_cooling_rate_c_per_s"] = float(max_cool_rate)
    summary["has_temper"] = bool(op_summary.get("has_temper", False))
    summary["has_quench"] = bool(op_summary.get("has_quench", False))
    if "temper_peak_temperature_c" in op_summary:
        summary["temper_peak_temperature_c"] = float(op_summary.get("temper_peak_temperature_c", 0.0))
    summary["recommended_temper_shift_c"] = float(op_summary.get("recommended_temper_shift_c", 0.0))
    summary["recommended_temper_shift_map_c"] = dict(op_summary.get("recommended_temper_shift_map_c", {}))
    summary["stage_inference_profile"] = str(op_summary.get("stage_inference_profile", "fe_c_temper_curve_v2"))
    summary["quench_medium_code_resolved"] = str(medium_code_resolved)
    summary["quench_effect_applied"] = bool(effect_applied)
    summary["quench_effect_reason"] = str(effect_reason)
    summary["as_quenched_prediction"] = dict(q.get("as_quenched_prediction", {}))
    summary["operation_guidance"] = dict(q.get("operation_guidance", {}))
    if not effect_applied and medium_code_resolved in _QUENCH_MEDIUM_CODES:
        q_warnings = list(q.get("warnings", [])) if isinstance(q.get("warnings", []), list) else []
        note = "Среда закалки выбрана, но на кривой нет закалочного сегмента; влияние среды не применяется."
        if note not in q_warnings:
            q_warnings.append(note)
        q["warnings"] = q_warnings
    q["effect_applied"] = bool(effect_applied)
    q["effect_reason"] = str(effect_reason)
    q["observed_temperature_c"] = float(observed_temp)
    q["quench_observed_temperature_estimate_c"] = float(quench_observed_estimate)
    q["max_effective_cooling_rate_c_per_s"] = float(max_cool_rate)
    return processing, summary, q
