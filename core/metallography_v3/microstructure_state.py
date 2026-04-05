from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.contracts_v2 import ProcessingState


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _infer_stage_from_operations(summary: dict[str, Any], fallback: str = "equilibrium") -> str:
    has_quench = bool(summary.get("has_quench", False))
    has_temper = bool(summary.get("has_temper", False))
    if has_temper:
        if bool(summary.get("has_temper_high", False)):
            return "sorbite_temper"
        if bool(summary.get("has_temper_medium", False)):
            return "troostite_temper"
        if bool(summary.get("has_temper_low", False)):
            return "tempered_low"
    if has_quench:
        return "martensite"
    return str(fallback or "equilibrium")


def _property_indicators(stage: str, effect_vector: dict[str, float]) -> dict[str, float]:
    stage_l = str(stage).strip().lower()
    hard = 180.0
    if "martensite" in stage_l:
        hard = 620.0
    elif "tempered_low" in stage_l:
        hard = 560.0
    elif "troostite" in stage_l:
        hard = 430.0
    elif "sorbite" in stage_l:
        hard = 300.0
    elif "pearlite" in stage_l:
        hard = 250.0
    elif "bainite" in stage_l:
        hard = 360.0
    hard = hard + 40.0 * float(effect_vector.get("dislocation_proxy", 0.0))
    uts = hard * 3.15
    return {
        "hardness_hv_est": float(max(80.0, hard)),
        "uts_mpa_est": float(max(250.0, uts)),
    }


@dataclass(slots=True)
class MicrostructureStateV3:
    final_processing: ProcessingState
    process_timeline: list[dict[str, Any]]
    resolved_stage_by_step: list[dict[str, Any]]
    effect_vector: dict[str, float]
    final_stage: str
    property_indicators: dict[str, float]
    technology_influence: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "process_timeline": list(self.process_timeline),
            "resolved_stage_by_step": list(self.resolved_stage_by_step),
            "final_effect_vector": dict(self.effect_vector),
            "final_stage": str(self.final_stage),
            "property_indicators": dict(self.property_indicators),
            "technology_influence": dict(self.technology_influence),
        }


def build_microstructure_state(
    *,
    composition: dict[str, float],
    inferred_system: str,
    processing: ProcessingState,
    thermal_summary: dict[str, Any],
    operations_from_curve: dict[str, Any],
    quench_summary: dict[str, Any] | None,
    seed: int,
) -> MicrostructureStateV3:
    _ = int(seed)
    composition_norm = {str(k): _safe_float(v) for k, v in dict(composition or {}).items()}
    op_summary = dict(operations_from_curve.get("summary", {})) if isinstance(operations_from_curve, dict) else {}
    op_rows = list(operations_from_curve.get("operations", [])) if isinstance(operations_from_curve, dict) else []
    qsum = dict(quench_summary or {})

    has_quench = bool(op_summary.get("has_quench", False))
    has_temper = bool(op_summary.get("has_temper", False))
    max_cooling = _safe_float(thermal_summary.get("max_cooling_rate_c_per_s", 0.0))
    avg_temp = _safe_float(thermal_summary.get("temperature_avg_c", processing.temperature_c), processing.temperature_c)
    c_wt = _safe_float(composition_norm.get("C", 0.0))

    dislocation = 0.08 + (0.42 if has_quench else 0.12) + max(0.0, min(0.35, abs(max_cooling) / 120.0))
    if has_temper:
        dislocation *= 0.7
    grain_size_factor = -0.10 if has_quench else 0.05
    if has_temper and bool(op_summary.get("has_temper_high", False)):
        grain_size_factor += 0.06
    segregation = max(0.0, min(0.45, 0.03 + c_wt * 0.08))

    effect_vector = {
        "dislocation_proxy": float(max(0.0, min(1.0, dislocation))),
        "grain_size_factor": float(max(-0.4, min(0.4, grain_size_factor))),
        "segregation_level": float(segregation),
        "elongation_factor": float(max(0.0, min(0.8, 0.15 if has_temper else 0.05))),
        "thermal_intensity": float(max(0.0, min(1.0, avg_temp / 1200.0))),
    }

    final_stage = _infer_stage_from_operations(op_summary, fallback=str(inferred_system or "equilibrium"))
    process_timeline: list[dict[str, Any]] = []
    resolved_stage_by_step: list[dict[str, Any]] = []
    for idx, op in enumerate(op_rows):
        if not isinstance(op, dict):
            continue
        op_type = str(op.get("type", "segment"))
        start_s = _safe_float(op.get("start_s", op.get("time_start_s", 0.0)))
        end_s = _safe_float(op.get("end_s", op.get("time_end_s", start_s)))
        label = str(op.get("label", op_type))
        stage = final_stage
        if op_type == "quench_cooling":
            stage = "martensite"
        elif op_type == "temper_low":
            stage = "tempered_low"
        elif op_type == "temper_medium":
            stage = "troostite_temper"
        elif op_type == "temper_high":
            stage = "sorbite_temper"
        process_timeline.append(
            {
                "index": int(idx),
                "operation": op_type,
                "label": label,
                "start_s": float(start_s),
                "end_s": float(end_s),
            }
        )
        resolved_stage_by_step.append(
            {
                "index": int(idx),
                "operation": op_type,
                "stage": stage,
            }
        )

    if not process_timeline:
        process_timeline.append(
            {
                "index": 0,
                "operation": "thermal_program",
                "label": "thermal_program",
                "start_s": 0.0,
                "end_s": _safe_float(thermal_summary.get("duration_s", 0.0)),
            }
        )
        resolved_stage_by_step.append({"index": 0, "operation": "thermal_program", "stage": final_stage})

    technology_influence = {
        "source": "thermal_program_v3",
        "has_quench": bool(has_quench),
        "has_temper": bool(has_temper),
        "quench_medium": str(qsum.get("medium_code_resolved", qsum.get("medium_code", ""))),
        "quench_effect_applied": bool(qsum.get("effect_applied", False)),
    }

    return MicrostructureStateV3(
        final_processing=processing,
        process_timeline=process_timeline,
        resolved_stage_by_step=resolved_stage_by_step,
        effect_vector=effect_vector,
        final_stage=str(final_stage),
        property_indicators=_property_indicators(final_stage, effect_vector),
        technology_influence=technology_influence,
    )

