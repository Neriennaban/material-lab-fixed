from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.contracts_v2 import ProcessingState
from core.contracts_v3 import PhaseModelConfigV3
from core.generator_phase_map import (
    normalize_system,
    resolve_al_cu_mg_stage,
    resolve_al_si_stage,
    resolve_cu_zn_stage,
    resolve_fe_c_stage,
    resolve_fe_si_stage,
)

_RULES_PATH = Path(__file__).resolve().parents[1] / "rulebook" / "explicit_phase_rules_v3.json"
_FE_C_STAGE_RULES_PATH = Path(__file__).resolve().parents[1] / "rulebook" / "fe_c_stage_rules_v3.json"
_FE_C_TABLES_PATH = Path(__file__).resolve().parents[1] / "rulebook" / "fe_c_tempering_fraction_tables_v3.json"


def _load_rules() -> dict[str, Any]:
    if not _RULES_PATH.exists():
        return {}
    with _RULES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_RULES = _load_rules()
_FE_C_STAGE_RULES = {}
if _FE_C_STAGE_RULES_PATH.exists():
    try:
        _FE_C_STAGE_RULES = json.loads(_FE_C_STAGE_RULES_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        _FE_C_STAGE_RULES = {}
_FE_C_TABLES = {}
if _FE_C_TABLES_PATH.exists():
    try:
        _FE_C_TABLES = json.loads(_FE_C_TABLES_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        _FE_C_TABLES = {}
_KNOWN_SYSTEMS = set(_RULES.get("known_systems", ["fe-c", "fe-si", "al-si", "cu-zn", "al-cu-mg"]))
_MARKERS: dict[str, list[str]] = {
    str(k): [str(vv) for vv in vv_list]
    for k, vv_list in dict(_RULES.get("inference_markers", {})).items()
    if isinstance(vv_list, list)
}
_PHASE_ALIASES: dict[str, str] = {
    "L": "LIQUID",
    "FE3C": "CEMENTITE",
    "CARBIDE": "CEMENTITE",
    "MARTENSITE_T": "MARTENSITE_TETRAGONAL",
    "MARTENSITE_C": "MARTENSITE_CUBIC",
}

def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _norm_dict(values: dict[str, float]) -> dict[str, float]:
    cleaned = {str(k): max(0.0, float(v)) for k, v in values.items() if float(v) > 1e-9}
    total = float(sum(cleaned.values()))
    if total <= 1e-12:
        return {}
    return {k: float(v / total) for k, v in cleaned.items()}


def _norm_phase_dict(values: dict[str, float]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for k, v in values.items():
        val = float(v)
        if val <= 1e-9:
            continue
        name = str(k).strip().upper().replace("-", "_").replace(" ", "_")
        name = _PHASE_ALIASES.get(name, name)
        cleaned[name] = float(cleaned.get(name, 0.0) + max(0.0, val))
    total = float(sum(cleaned.values()))
    if total <= 1e-12:
        return {}
    return {k: float(v / total) for k, v in cleaned.items()}


def _composition_norm(composition: dict[str, float]) -> dict[str, float]:
    return _norm_dict({str(k): max(0.0, float(v)) for k, v in composition.items()})


def _liquid_fraction_fe_c(c_wt: float, temp_c: float) -> float:
    c = max(0.0, float(c_wt))
    t = float(temp_c)
    liquidus = 1538.0 - 83.0 * min(c, 4.3)
    solidus = 1493.0 - 58.0 * c if c <= 2.1 else 1147.0 + (4.3 - min(c, 4.3)) * 38.0
    if liquidus <= solidus + 1e-9:
        return 0.0
    return _clamp((t - solidus) / (liquidus - solidus), 0.0, 1.0)


def _liquid_fraction_al_si(si_wt: float, temp_c: float) -> float:
    si = max(0.0, float(si_wt))
    t = float(temp_c)
    solidus = 577.0
    if si <= 12.6:
        liquidus = 660.0 - 6.6 * si
    else:
        liquidus = 577.0 + 2.4 * (si - 12.6)
    if liquidus <= solidus + 1e-9:
        return 0.0
    return _clamp((t - solidus) / (liquidus - solidus), 0.0, 1.0)


def _ferrite_vs_pearlite(
    c_wt: float,
    *,
    ferrite_solubility_c: float = 0.02,
    eutectoid_c: float = 0.77,
) -> tuple[float, float]:
    pearlite = _clamp(
        (float(c_wt) - float(ferrite_solubility_c))
        / max(1e-9, float(eutectoid_c) - float(ferrite_solubility_c)),
        0.0,
        1.0,
    )
    ferrite = _clamp(1.0 - pearlite, 0.0, 1.0)
    return ferrite, pearlite


def _pearlite_equilibrium_constituents(
    *,
    c_wt: float,
    ferrite_solubility_c: float,
    eutectoid_c: float,
    cementite_c: float,
) -> dict[str, float]:
    c = float(max(0.0, c_wt))
    if c <= float(eutectoid_c):
        ferrite, pearlite = _ferrite_vs_pearlite(
            c,
            ferrite_solubility_c=float(ferrite_solubility_c),
            eutectoid_c=float(eutectoid_c),
        )
        return _norm_dict({"FERRITE": ferrite, "PEARLITE": pearlite, "CEMENTITE": 0.0})

    cementite = _clamp(
        (c - float(eutectoid_c)) / max(1e-9, float(cementite_c) - float(eutectoid_c)),
        0.0,
        1.0,
    )
    pearlite = _clamp(1.0 - cementite, 0.0, 1.0)
    return _norm_dict({"FERRITE": 0.0, "PEARLITE": pearlite, "CEMENTITE": cementite})


def _pearlite_internal_true_phases(
    *,
    ferrite_solubility_c: float,
    eutectoid_c: float,
    cementite_c: float,
) -> dict[str, float]:
    ferrite = _clamp(
        (float(cementite_c) - float(eutectoid_c))
        / max(1e-9, float(cementite_c) - float(ferrite_solubility_c)),
        0.0,
        1.0,
    )
    return _norm_phase_dict({"FERRITE": ferrite, "CEMENTITE": 1.0 - ferrite})


def _true_phases_from_steel_microconstituents(
    *,
    microconstituents: dict[str, float],
    pearlite_internal: dict[str, float],
) -> dict[str, float]:
    ferrite = float(microconstituents.get("FERRITE", 0.0))
    pearlite = float(microconstituents.get("PEARLITE", 0.0))
    cementite = float(microconstituents.get("CEMENTITE", 0.0))
    return _norm_phase_dict(
        {
            "FERRITE": ferrite
            + pearlite * float(pearlite_internal.get("FERRITE", 0.0)),
            "CEMENTITE": cementite
            + pearlite * float(pearlite_internal.get("CEMENTITE", 0.0)),
        }
    )


def _steel_room_temperature_equilibrium(
    *,
    c_wt: float,
    ferrite_solubility_c: float,
    eutectoid_c: float,
    steel_limit_c: float,
    cementite_c: float,
) -> dict[str, Any]:
    c = float(max(0.0, c_wt))
    ferrite_limit = float(ferrite_solubility_c)
    eutectoid = float(eutectoid_c)
    steel_limit = float(steel_limit_c)
    pearlite_internal = _pearlite_internal_true_phases(
        ferrite_solubility_c=ferrite_limit,
        eutectoid_c=eutectoid,
        cementite_c=float(cementite_c),
    )

    if c <= ferrite_limit + 1e-9:
        micro = {"FERRITE": 1.0}
    elif c < eutectoid - 1e-9:
        ferrite, pearlite = _ferrite_vs_pearlite(
            c,
            ferrite_solubility_c=ferrite_limit,
            eutectoid_c=eutectoid,
        )
        micro = {"FERRITE": ferrite, "PEARLITE": pearlite}
    elif c <= steel_limit + 1e-9:
        proeutectoid_cementite = _clamp(
            (c - eutectoid) / max(1e-9, float(cementite_c) - eutectoid),
            0.0,
            1.0,
        )
        micro = {
            "PEARLITE": _clamp(1.0 - proeutectoid_cementite, 0.0, 1.0),
            "CEMENTITE": proeutectoid_cementite,
        }
    else:
        micro = {}

    micro_norm = _norm_phase_dict(micro)
    return {
        "microconstituents": micro_norm,
        "true_phases": _true_phases_from_steel_microconstituents(
            microconstituents=micro_norm,
            pearlite_internal=pearlite_internal,
        ),
        "pearlite_internal_true_phases": pearlite_internal,
        "thresholds_wt_pct": {
            "ferrite_max_c": float(ferrite_limit),
            "eutectoid_c": float(eutectoid),
            "steel_limit_c": float(steel_limit),
            "cementite_c": float(cementite_c),
        },
    }


def _rule_float(payload: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except Exception:
        return float(default)


def _fe_c_table_phase_map() -> dict[str, str]:
    payload = dict(_FE_C_TABLES.get("phase_key_map", {})) if isinstance(_FE_C_TABLES, dict) else {}
    out: dict[str, str] = {}
    for k, v in payload.items():
        key = str(k).strip().lower()
        if not key:
            continue
        phase = str(v).strip().upper().replace("-", "_").replace(" ", "_")
        if phase:
            out[key] = _PHASE_ALIASES.get(phase, phase)
    return out


def _table_stage_id_for_fe_c(stage: str) -> str:
    stage_l = str(stage).strip().lower()
    mapping = {
        "martensite": "quench_water_20",
        "martensite_tetragonal": "quench_water_20",
        "martensite_cubic": "quench_water_20",
        "tempered_low": "temper_low_150_250",
        "troostite_temper": "temper_medium_250_450",
        "tempered_medium": "temper_medium_250_450",
        "sorbite_temper": "temper_high_450_650",
        "tempered_high": "temper_high_450_650",
    }
    return str(mapping.get(stage_l, ""))


def _interp_table_row_by_c(table_id: str, c_wt: float) -> dict[str, Any] | None:
    if not isinstance(_FE_C_TABLES, dict):
        return None
    tables = dict(_FE_C_TABLES.get("tables", {}))
    rows_raw = tables.get(str(table_id), [])
    if not isinstance(rows_raw, list) or not rows_raw:
        return None

    parsed_rows: list[dict[str, float]] = []
    for row in rows_raw:
        if not isinstance(row, dict):
            continue
        try:
            c_row = float(row.get("c_wt", 0.0))
        except Exception:
            continue
        values: dict[str, float] = {"c_wt": c_row}
        for k, v in row.items():
            if str(k) == "c_wt":
                continue
            try:
                values[str(k)] = float(v)
            except Exception:
                continue
        parsed_rows.append(values)
    if not parsed_rows:
        return None
    parsed_rows.sort(key=lambda x: float(x.get("c_wt", 0.0)))

    cfg = dict(_FE_C_TABLES.get("range_c_wt", {})) if isinstance(_FE_C_TABLES.get("range_c_wt", {}), dict) else {}
    c_min = float(cfg.get("min", parsed_rows[0]["c_wt"]))
    c_max = float(cfg.get("max", parsed_rows[-1]["c_wt"]))
    margin = max(0.0, float(cfg.get("soft_extrapolation_margin", 0.05)))

    c_req = float(c_wt)
    clamped = False
    if c_req < c_min:
        if c_req < c_min - margin:
            return None
        c_req = c_min
        clamped = True
    elif c_req > c_max:
        if c_req > c_max + margin:
            return None
        c_req = c_max
        clamped = True

    left = parsed_rows[0]
    right = parsed_rows[-1]
    for idx in range(len(parsed_rows) - 1):
        c0 = float(parsed_rows[idx]["c_wt"])
        c1 = float(parsed_rows[idx + 1]["c_wt"])
        if c0 <= c_req <= c1:
            left = parsed_rows[idx]
            right = parsed_rows[idx + 1]
            break
    c0 = float(left["c_wt"])
    c1 = float(right["c_wt"])
    ratio = 0.0 if abs(c1 - c0) < 1e-12 else _clamp((c_req - c0) / (c1 - c0), 0.0, 1.0)
    keys = sorted(set(left.keys()) | set(right.keys()))
    out: dict[str, float] = {}
    for key in keys:
        if key == "c_wt":
            continue
        v0 = float(left.get(key, 0.0))
        v1 = float(right.get(key, v0))
        out[key] = float(v0 + (v1 - v0) * ratio)
    # Accept both percentage and fraction tables.
    max_v = max([float(v) for v in out.values()] or [0.0])
    if max_v > 1.5:
        out = {k: float(v) / 100.0 for k, v in out.items()}
    out = _norm_dict(out)
    return {
        "table_id": str(table_id),
        "constituents": out,
        "c_wt_requested": float(c_wt),
        "c_wt_used": float(c_req),
        "clamped": bool(clamped),
    }


def _apply_medium_correction_to_constituents(
    *,
    table_id: str,
    constituents: dict[str, float],
    medium_code: str,
    quench_effect_applied: bool,
) -> tuple[dict[str, float], bool]:
    if not quench_effect_applied:
        return dict(constituents), False
    medium = str(medium_code or "").strip().lower()
    if not medium or medium == "water_20":
        return dict(constituents), False
    if not isinstance(_FE_C_TABLES, dict):
        return dict(constituents), False
    corr_all = dict(_FE_C_TABLES.get("medium_corrections", {}))
    medium_corr = corr_all.get(medium, {})
    if not isinstance(medium_corr, dict):
        return dict(constituents), False
    table_corr = medium_corr.get(str(table_id), {})
    if not isinstance(table_corr, dict) or not table_corr:
        return dict(constituents), False

    out = {str(k): float(v) for k, v in constituents.items()}
    changed = False
    for key, delta in table_corr.items():
        name = str(key).strip().lower()
        try:
            dv = float(delta)
        except Exception:
            continue
        if abs(dv) <= 1e-12:
            continue
        changed = True
        out[name] = float(out.get(name, 0.0) + dv / 100.0)
    out = {k: max(0.0, float(v)) for k, v in out.items()}
    return _norm_dict(out), bool(changed)


def _build_fe_c_constituents_from_table(
    *,
    table_id: str,
    c_wt: float,
    medium_code: str,
    quench_effect_applied: bool,
) -> dict[str, Any] | None:
    row = _interp_table_row_by_c(table_id=table_id, c_wt=c_wt)
    if not isinstance(row, dict):
        return None
    base_constituents = dict(row.get("constituents", {})) if isinstance(row.get("constituents", {}), dict) else {}
    corrected, correction_applied = _apply_medium_correction_to_constituents(
        table_id=table_id,
        constituents=base_constituents,
        medium_code=medium_code,
        quench_effect_applied=quench_effect_applied,
    )
    return {
        "table_id": str(table_id),
        "c_wt_requested": float(row.get("c_wt_requested", c_wt)),
        "c_wt_used": float(row.get("c_wt_used", c_wt)),
        "clamped": bool(row.get("clamped", False)),
        "medium_code": str(medium_code or ""),
        "medium_correction_applied": bool(correction_applied),
        "constituents_base": base_constituents,
        "constituents_corrected": corrected,
    }


def _constituents_to_phase_fractions(constituents_map: dict[str, float]) -> dict[str, float]:
    phase_map = _fe_c_table_phase_map()
    merged: dict[str, float] = {}
    for name, value in dict(constituents_map).items():
        constituent = str(name).strip().lower()
        phase = str(phase_map.get(constituent, constituent.upper())).strip().upper().replace("-", "_").replace(" ", "_")
        phase = _PHASE_ALIASES.get(phase, phase)
        merged[phase] = float(merged.get(phase, 0.0) + max(0.0, float(value)))
    return _norm_phase_dict(merged)


def _resolve_fe_c_stage_from_thermal(
    *,
    c_wt: float,
    processing: ProcessingState,
    thermal_summary: dict[str, Any] | None,
    quench_summary: dict[str, Any] | None,
) -> str:
    """Resolve Fe-C teaching stage using editable rulebook and thermal/quench context."""
    rules = dict(_FE_C_STAGE_RULES.get("stages", {})) if isinstance(_FE_C_STAGE_RULES, dict) else {}
    defaults = dict(_FE_C_STAGE_RULES.get("defaults", {})) if isinstance(_FE_C_STAGE_RULES, dict) else {}

    cooling_rate = 0.0
    if isinstance(thermal_summary, dict):
        cooling_rate = max(
            cooling_rate,
            abs(_rule_float(thermal_summary, "max_effective_cooling_rate_c_per_s", 0.0)),
            abs(_rule_float(thermal_summary, "max_cooling_rate_c_per_s", 0.0)),
        )
    quench_severity = _rule_float(quench_summary or {}, "severity_effective", 0.0)
    observed_t = _rule_float(thermal_summary or {}, "observed_temperature_c", float(processing.temperature_c))

    temper_shift = {}
    if isinstance(quench_summary, dict) and isinstance(quench_summary.get("temper_shift_c"), dict):
        temper_shift = dict(quench_summary.get("temper_shift_c", {}))
    shift_low = _rule_float(temper_shift, "low", _rule_float(quench_summary or {}, "temper_shift_c_low", 0.0))
    shift_medium = _rule_float(temper_shift, "medium", _rule_float(quench_summary or {}, "temper_shift_c_medium", 0.0))
    shift_high = _rule_float(temper_shift, "high", _rule_float(quench_summary or {}, "temper_shift_c_high", 0.0))

    temper_low_t = _rule_float(defaults, "temper_low_max_c", 250.0) + shift_low
    temper_medium_t = _rule_float(defaults, "temper_medium_max_c", 450.0) + shift_medium
    temper_high_t = _rule_float(defaults, "temper_high_max_c", 650.0) + shift_high
    temper_min_t = _rule_float(defaults, "temper_min_c", 150.0)
    temper_hold_min = _rule_float(defaults, "temper_hold_min_s", 120.0)
    austenitize_min = _rule_float(defaults, "austenitize_min_c", 780.0)
    t_max = _rule_float(thermal_summary or {}, "temperature_max_c", float(processing.temperature_c))
    hold_s = _rule_float(thermal_summary or {}, "hold_time_s", 0.0)
    cooling_mode_l = str(getattr(processing, "cooling_mode", "")).strip().lower()
    quench_medium = str(
        (quench_summary or {}).get("medium_code_resolved", (quench_summary or {}).get("medium_code", ""))
    ).strip().lower()
    quench_effect_applied = bool((quench_summary or {}).get("effect_applied", False))
    op_summary = {}
    if isinstance(thermal_summary, dict):
        op_payload = thermal_summary.get("operation_inference", {})
        if isinstance(op_payload, dict):
            op_summary = op_payload

    stage_base = resolve_fe_c_stage(
        c_wt=float(c_wt),
        temperature_c=float(observed_t),
        cooling_mode=str(processing.cooling_mode),
        requested_stage="auto",
    )

    has_quench = (
        cooling_mode_l in {"quench", "quenched"}
        or bool(op_summary.get("has_quench", False))
        or quench_effect_applied
    )
    has_temper = (
        cooling_mode_l.startswith("temper")
        or bool(op_summary.get("has_temper", False))
        or bool((thermal_summary or {}).get("has_temper", False))
    )
    temper_peak_t = _rule_float(
        op_summary,
        "temper_peak_temperature_c",
        _rule_float(thermal_summary or {}, "temper_peak_temperature_c", float(observed_t)),
    )
    temper_band_detected = str(op_summary.get("temper_band_detected", "")).strip().lower()
    temper_hold_s = _rule_float(op_summary, "temper_total_hold_s", 0.0)
    if temper_hold_s <= 0.0:
        temper_hold_s = max(0.0, hold_s)

    # Tempered variants first if curve contains explicit temper region.
    if has_temper and has_quench and t_max >= austenitize_min and (
        temper_hold_s >= temper_hold_min or temper_peak_t >= temper_min_t
    ):
        if temper_band_detected == "low":
            return "tempered_low"
        if temper_band_detected == "medium":
            return "troostite_temper"
        if temper_band_detected == "high":
            return "sorbite_temper"
        if temper_peak_t <= temper_low_t:
            return "tempered_low"
        if temper_peak_t <= temper_medium_t:
            return "troostite_temper"
        if temper_peak_t <= temper_high_t:
            return "sorbite_temper"

    # Quench-derived variants.
    m_tetra = dict(rules.get("martensite_tetragonal", {}))
    m_cubic = dict(rules.get("martensite_cubic", {}))
    troostite_q = dict(rules.get("troostite_quench", {}))
    sorbite_q = dict(rules.get("sorbite_quench", {}))

    medium_bias = {
        "brine_20_30": 0.12,
        "water_20": 0.06,
        "water_100": -0.08,
        "oil_20_80": -0.12,
        "polymer": -0.04,
        "custom": 0.0,
    }
    effective_severity = float(quench_severity)
    if has_quench or quench_effect_applied:
        effective_severity = float(quench_severity + medium_bias.get(quench_medium, 0.0))
    if t_max >= austenitize_min and (effective_severity > 0.0 or has_quench) and has_quench:
        if (
            float(c_wt) >= _rule_float(m_tetra, "c_min_wt", 0.6)
            and cooling_rate >= _rule_float(m_tetra, "cooling_rate_min_c_per_s", 20.0)
            and effective_severity >= _rule_float(m_tetra, "severity_min", 0.7)
        ):
            return "martensite_tetragonal"
        if (
            float(c_wt) <= _rule_float(m_cubic, "c_max_wt", 0.6)
            and cooling_rate >= _rule_float(m_cubic, "cooling_rate_min_c_per_s", 20.0)
            and effective_severity >= _rule_float(m_cubic, "severity_min", 0.55)
        ):
            return "martensite_cubic"
        if (
            effective_severity >= _rule_float(troostite_q, "severity_min", 0.42)
            and effective_severity <= _rule_float(troostite_q, "severity_max", 0.75)
            and cooling_rate >= _rule_float(troostite_q, "cooling_rate_min_c_per_s", 10.0)
            and cooling_rate <= _rule_float(troostite_q, "cooling_rate_max_c_per_s", 28.0)
        ):
            return "troostite_quench"
        if (
            effective_severity >= _rule_float(sorbite_q, "severity_min", 0.2)
            and effective_severity <= _rule_float(sorbite_q, "severity_max", 0.55)
            and cooling_rate >= _rule_float(sorbite_q, "cooling_rate_min_c_per_s", 4.0)
            and cooling_rate <= _rule_float(sorbite_q, "cooling_rate_max_c_per_s", 16.0)
        ):
            return "sorbite_quench"

    return stage_base


def infer_training_system(
    composition: dict[str, float],
    system_hint: str | None = None,
) -> tuple[str, float, bool]:
    hinted = normalize_system(system_hint or "")
    if hinted in _KNOWN_SYSTEMS:
        return hinted, 1.0, False

    comp = _composition_norm(composition)
    if not comp:
        return "custom-multicomponent", 0.25, True

    scores: list[tuple[str, float]] = []
    for system in sorted(_KNOWN_SYSTEMS):
        markers = _MARKERS.get(system, [])
        score = sum(float(comp.get(m, 0.0)) for m in markers)
        scores.append((system, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    if not scores:
        return "custom-multicomponent", 0.25, True

    best_system, best_score = scores[0]
    if best_score < 0.35:
        return "custom-multicomponent", _clamp(0.2 + best_score, 0.2, 0.6), True
    return best_system, _clamp(0.4 + 0.6 * best_score, 0.4, 0.98), False


def resolve_stage(
    system: str,
    composition: dict[str, float],
    processing: ProcessingState,
    thermal_summary: dict[str, Any] | None = None,
    quench_summary: dict[str, Any] | None = None,
) -> str:
    sys_name = normalize_system(system)
    if sys_name == "fe-c":
        return _resolve_fe_c_stage_from_thermal(
            c_wt=float(composition.get("C", 0.0)),
            processing=processing,
            thermal_summary=thermal_summary,
            quench_summary=quench_summary,
        )
    if sys_name == "al-si":
        return resolve_al_si_stage(
            si_wt=float(composition.get("Si", 0.0)),
            temperature_c=float(processing.temperature_c),
            cooling_mode=str(processing.cooling_mode),
            requested_stage="auto",
        )
    if sys_name == "cu-zn":
        return resolve_cu_zn_stage(
            zn_wt=float(composition.get("Zn", 0.0)),
            temperature_c=float(processing.temperature_c),
            cooling_mode=str(processing.cooling_mode),
            requested_stage="auto",
            deformation_pct=float(processing.deformation_pct),
        )
    if sys_name == "al-cu-mg":
        return resolve_al_cu_mg_stage(
            temperature_c=float(processing.temperature_c),
            cooling_mode=str(processing.cooling_mode),
            requested_stage="auto",
            aging_temperature_c=float(processing.aging_temperature_c),
            aging_hours=float(processing.aging_hours),
        )
    if sys_name == "fe-si":
        return resolve_fe_si_stage(
            temperature_c=float(processing.temperature_c),
            cooling_mode=str(processing.cooling_mode),
            requested_stage="auto",
            deformation_pct=float(processing.deformation_pct),
            si_wt=float(composition.get("Si", 0.0)),
        )
    if float(processing.temperature_c) >= 1350.0:
        return "liquid_custom"
    if float(processing.deformation_pct) >= 8.0:
        return "deformed_custom"
    if float(processing.aging_hours) > 0.5:
        return "aged_custom"
    return "custom_equilibrium"


def estimate_auto_phase_fractions(
    system: str,
    stage: str,
    composition: dict[str, float],
    processing: ProcessingState,
    thermal_summary: dict[str, Any] | None = None,
    quench_summary: dict[str, Any] | None = None,
    calibration_trace: dict[str, Any] | None = None,
) -> dict[str, float]:
    sys_name = normalize_system(system)
    stage_l = str(stage).strip().lower()
    comp = _composition_norm(composition)

    if sys_name == "fe-c":
        c = float(comp.get("C", 0.0) * 100.0)
        fe_c_rules = dict(_RULES.get("fe_c", {})) if isinstance(_RULES, dict) else {}
        eutectoid_c = _rule_float(fe_c_rules, "eutectoid_c", 0.77)
        ferrite_solubility_c = _rule_float(fe_c_rules, "ferrite_solubility_c", 0.02)
        cementite_c = _rule_float(fe_c_rules, "cementite_c", 6.67)
        steel_limit_c = _rule_float(fe_c_rules, "max_c_hypereutectoid", 2.14)
        qsum = dict(quench_summary or {})
        retained_austenite_fraction = _clamp(
            _rule_float(
                qsum.get("as_quenched_prediction", {}) if isinstance(qsum.get("as_quenched_prediction", {}), dict) else {},
                "retained_austenite_fraction_est",
                _rule_float(qsum, "retained_austenite_est_pct", 0.0) / 100.0,
            ),
            0.0,
            0.65,
        )
        medium_code = str(qsum.get("medium_code_resolved", qsum.get("medium_code", ""))).strip().lower()
        quench_effect_applied = bool(qsum.get("effect_applied", False))
        thermal_has_context = isinstance(thermal_summary, dict) and bool(thermal_summary)
        op_summary = {}
        if isinstance(thermal_summary, dict):
            op_payload = thermal_summary.get("operation_inference", {})
            if isinstance(op_payload, dict):
                op_summary = op_payload
        has_quench_from_curve = bool(op_summary.get("has_quench", False) or quench_effect_applied)
        has_temper_from_curve = bool(op_summary.get("has_temper", False) or bool((thermal_summary or {}).get("has_temper", False)))
        medium_bias = {
            "brine_20_30": 0.06,
            "water_20": 0.03,
            "water_100": -0.03,
            "oil_20_80": -0.05,
        }
        if quench_effect_applied:
            retained_austenite_fraction = _clamp(
                retained_austenite_fraction - float(medium_bias.get(medium_code, 0.0)),
                0.0,
                0.65,
            )

        table_id = _table_stage_id_for_fe_c(stage_l)
        table_trace: dict[str, Any] = {}
        if table_id:
            if table_id == "quench_water_20":
                curve_ok = has_quench_from_curve or not thermal_has_context
            else:
                curve_ok = (has_quench_from_curve and has_temper_from_curve) or not thermal_has_context
            if curve_ok:
                table_trace = (
                    _build_fe_c_constituents_from_table(
                        table_id=table_id,
                        c_wt=c,
                        medium_code=medium_code,
                        quench_effect_applied=has_quench_from_curve,
                    )
                    or {}
                )
                if table_trace:
                    constituents_corrected = dict(table_trace.get("constituents_corrected", {}))
                    phase_from_table = _constituents_to_phase_fractions(constituents_corrected)
                    if stage_l == "martensite_tetragonal" and "MARTENSITE" in phase_from_table:
                        phase_from_table["MARTENSITE_TETRAGONAL"] = float(phase_from_table.pop("MARTENSITE"))
                    elif stage_l == "martensite_cubic" and "MARTENSITE" in phase_from_table:
                        phase_from_table["MARTENSITE_CUBIC"] = float(phase_from_table.pop("MARTENSITE"))
                    phase_from_table = _norm_phase_dict(phase_from_table)
                    if calibration_trace is not None and isinstance(calibration_trace, dict):
                        calibration_trace.clear()
                        calibration_trace.update(
                            {
                                "fraction_source": "table_interpolated",
                                "table_id": str(table_id),
                                "table_row_clamped": bool(table_trace.get("clamped", False)),
                                "table_c_wt_requested": float(table_trace.get("c_wt_requested", c)),
                                "table_c_wt_used": float(table_trace.get("c_wt_used", c)),
                                "medium_code_resolved": str(medium_code),
                                "medium_correction_applied": bool(table_trace.get("medium_correction_applied", False)),
                                "target_constituents_from_table": dict(constituents_corrected),
                                "target_phase_fractions_from_table": dict(phase_from_table),
                                "calibration_profile": str(_FE_C_TABLES.get("profile_id", "fe_c_tempering_tables_v1"))
                                if isinstance(_FE_C_TABLES, dict)
                                else "fe_c_tempering_tables_v1",
                                "calibration_source": str(_FE_C_TABLES.get("calibration_source", "user_tables_2026_03_04"))
                                if isinstance(_FE_C_TABLES, dict)
                                else "user_tables_2026_03_04",
                            }
                        )
                    return phase_from_table
        if calibration_trace is not None and isinstance(calibration_trace, dict) and not calibration_trace:
            calibration_trace.update({"fraction_source": "default_formula"})
        steel_equilibrium = _steel_room_temperature_equilibrium(
            c_wt=c,
            ferrite_solubility_c=ferrite_solubility_c,
            eutectoid_c=eutectoid_c,
            steel_limit_c=steel_limit_c,
            cementite_c=cementite_c,
        )
        if calibration_trace is not None and isinstance(calibration_trace, dict):
            calibration_trace.update(
                {
                    "steel_microconstituents_auto": dict(
                        steel_equilibrium.get("microconstituents", {})
                    ),
                    "steel_true_phases_auto": dict(
                        steel_equilibrium.get("true_phases", {})
                    ),
                    "pearlite_internal_true_phases": dict(
                        steel_equilibrium.get("pearlite_internal_true_phases", {})
                    ),
                    "steel_equilibrium_thresholds_wt_pct": dict(
                        steel_equilibrium.get("thresholds_wt_pct", {})
                    ),
                }
            )
        if stage_l == "liquid":
            return {"LIQUID": 1.0}
        if stage_l == "liquid_gamma":
            lf = _liquid_fraction_fe_c(c, float(processing.temperature_c))
            return {"LIQUID": lf, "AUSTENITE": 1.0 - lf}
        if stage_l == "delta_ferrite":
            return {"DELTA_FERRITE": 0.82, "AUSTENITE": 0.18}
        if stage_l == "austenite":
            return {"AUSTENITE": 1.0}
        if stage_l == "alpha_gamma":
            ferrite = _clamp(
                (float(eutectoid_c) - c) / max(1e-9, float(eutectoid_c)),
                0.25,
                0.95,
            )
            return {"FERRITE": ferrite, "AUSTENITE": 1.0 - ferrite}
        if stage_l == "gamma_cementite":
            cementite = _clamp(
                (c - float(eutectoid_c))
                / max(1e-9, float(steel_limit_c) - float(eutectoid_c))
                * 0.45
                + 0.05,
                0.05,
                0.5,
            )
            return {"AUSTENITE": 1.0 - cementite, "CEMENTITE": cementite}
        if stage_l == "alpha_pearlite":
            return _pearlite_equilibrium_constituents(
                c_wt=c,
                ferrite_solubility_c=ferrite_solubility_c,
                eutectoid_c=eutectoid_c,
                cementite_c=cementite_c,
            )
        if stage_l == "pearlite":
            return _pearlite_equilibrium_constituents(
                c_wt=c,
                ferrite_solubility_c=ferrite_solubility_c,
                eutectoid_c=eutectoid_c,
                cementite_c=cementite_c,
            )
        if stage_l == "pearlite_cementite":
            return _pearlite_equilibrium_constituents(
                c_wt=c,
                ferrite_solubility_c=ferrite_solubility_c,
                eutectoid_c=eutectoid_c,
                cementite_c=cementite_c,
            )
        if stage_l == "martensite":
            carbide = _clamp(c / 2.1 * 0.22, 0.03, 0.24)
            ra = _clamp(retained_austenite_fraction, 0.0, 0.35)
            return _norm_dict({"MARTENSITE": 1.0 - carbide - ra, "CEMENTITE": carbide, "AUSTENITE": ra})
        if stage_l == "martensite_tetragonal":
            carbide = _clamp(0.06 + c / 2.1 * 0.2, 0.06, 0.3)
            ra = _clamp(retained_austenite_fraction, 0.01, 0.32)
            return _norm_dict({"MARTENSITE_TETRAGONAL": 1.0 - carbide - ra, "CEMENTITE": carbide, "AUSTENITE": ra})
        if stage_l == "martensite_cubic":
            carbide = _clamp(0.03 + c / 2.1 * 0.1, 0.03, 0.16)
            ra = _clamp(retained_austenite_fraction, 0.02, 0.4)
            return _norm_dict({"MARTENSITE_CUBIC": 1.0 - carbide - ra, "CEMENTITE": carbide, "AUSTENITE": ra})
        if stage_l == "bainite":
            carbide = _clamp(c / 2.1 * 0.18, 0.04, 0.2)
            return {"BAINITE": 1.0 - carbide, "CEMENTITE": carbide}
        if stage_l == "troostite_quench":
            carb = _clamp(0.08 + c / 2.1 * 0.14, 0.08, 0.25)
            ra = _clamp(retained_austenite_fraction * 0.8, 0.0, 0.2)
            return _norm_dict({"TROOSTITE": 1.0 - carb - ra, "CEMENTITE": carb, "AUSTENITE": ra})
        if stage_l == "sorbite_quench":
            carb = _clamp(0.06 + c / 2.1 * 0.1, 0.06, 0.2)
            ra = _clamp(retained_austenite_fraction * 0.65, 0.0, 0.18)
            return _norm_dict({"SORBITE": 1.0 - carb - ra, "CEMENTITE": carb, "AUSTENITE": ra})
        if stage_l == "troostite_temper":
            carb = _clamp(0.12 + c / 2.1 * 0.16, 0.12, 0.3)
            ferr = _clamp(0.2 + c / 2.1 * 0.1, 0.2, 0.45)
            mart = _clamp(retained_austenite_fraction * 0.35, 0.0, 0.08)
            return _norm_dict({"TROOSTITE": 1.0 - carb - ferr - mart, "CEMENTITE": carb, "FERRITE": ferr, "MARTENSITE": mart})
        if stage_l == "sorbite_temper":
            carb = _clamp(0.14 + c / 2.1 * 0.14, 0.14, 0.28)
            ferr = _clamp(0.35 + c / 2.1 * 0.1, 0.35, 0.58)
            mart = _clamp(retained_austenite_fraction * 0.08, 0.0, 0.03)
            return _norm_dict({"SORBITE": 1.0 - carb - ferr - mart, "CEMENTITE": carb, "FERRITE": ferr, "MARTENSITE": mart})
        if stage_l == "tempered_low":
            carb = _clamp(0.08 + c / 2.1 * 0.18, 0.08, 0.3)
            ra = _clamp(retained_austenite_fraction * 0.55, 0.0, 0.12)
            mart = _clamp(0.76 - carb * 0.2 - ra, 0.25, 0.8)
            ferr = max(0.0, 1.0 - mart - carb - ra)
            return _norm_dict({"MARTENSITE": mart, "CEMENTITE": carb, "FERRITE": ferr, "AUSTENITE": ra})
        if stage_l == "tempered_medium":
            carb = _clamp(0.14 + c / 2.1 * 0.22, 0.12, 0.34)
            ferrite = _clamp(0.35 + c / 2.1 * 0.1, 0.28, 0.5)
            mart = _clamp(0.22 - retained_austenite_fraction * 0.15, 0.05, 0.22)
            return _norm_dict({"TROOSTITE": 0.45, "MARTENSITE": mart, "CEMENTITE": carb, "FERRITE": ferrite})
        if stage_l == "tempered_high":
            carb = _clamp(0.2 + c / 2.1 * 0.25, 0.16, 0.42)
            ferrite = _clamp(0.45 + c / 2.1 * 0.12, 0.4, 0.6)
            return _norm_dict({"SORBITE": 0.42, "FERRITE": ferrite, "CEMENTITE": carb, "MARTENSITE": 0.01})
        if stage_l == "ledeburite":
            return {"CEMENTITE": 0.45, "PEARLITE": 0.3, "AUSTENITE": 0.25}
        return {"FERRITE": 1.0}

    if sys_name == "al-si":
        si = float(comp.get("Si", 0.0) * 100.0)
        if stage_l == "liquid":
            return {"LIQUID": 1.0}
        if stage_l == "liquid_alpha":
            lf = _liquid_fraction_al_si(si, float(processing.temperature_c))
            return {"LIQUID": lf, "FCC_A1": 1.0 - lf}
        if stage_l == "liquid_si":
            lf = _liquid_fraction_al_si(si, float(processing.temperature_c))
            solid = 1.0 - lf
            si_solid = _clamp(0.35 + (si - 12.6) * 0.02, 0.25, 0.7) * solid
            return _norm_dict({"LIQUID": lf, "FCC_A1": solid - si_solid, "SI": si_solid})
        if stage_l == "alpha_eutectic":
            eut = _clamp(si / 12.6 * 0.42, 0.08, 0.5)
            return {"FCC_A1": 1.0 - eut, "EUTECTIC_ALSI": eut}
        if stage_l == "eutectic":
            return {"EUTECTIC_ALSI": 0.68, "FCC_A1": 0.2, "SI": 0.12}
        if stage_l == "primary_si_eutectic":
            primary_si = _clamp((si - 12.6) / 15.0 * 0.55 + 0.15, 0.15, 0.62)
            return _norm_dict({"EUTECTIC_ALSI": 0.55, "SI": primary_si, "FCC_A1": 0.25})
        if stage_l == "supersaturated":
            return {"FCC_A1": 0.94, "PRECIPITATE": 0.06}
        if stage_l == "aged":
            age_factor = _clamp(float(processing.aging_hours) / 12.0, 0.0, 1.0)
            precip = 0.08 + 0.16 * age_factor
            return {"FCC_A1": 1.0 - precip, "PRECIPITATE": precip}
        return {"FCC_A1": 0.88, "SI": 0.12}

    if sys_name == "cu-zn":
        zn = float(comp.get("Zn", 0.0) * 100.0)
        if stage_l in {"liquid", "liquid_alpha", "liquid_beta"}:
            lf = _clamp((float(processing.temperature_c) - 830.0) / 250.0, 0.0, 1.0)
            solid = 1.0 - lf
            if stage_l == "liquid_beta":
                return _norm_dict({"LIQUID": lf, "BETA": solid * 0.8, "ALPHA": solid * 0.2})
            return _norm_dict({"LIQUID": lf, "ALPHA": solid * 0.8, "BETA": solid * 0.2})
        if stage_l == "alpha":
            return {"ALPHA": 1.0}
        if stage_l == "alpha_beta":
            beta = _clamp((zn - 35.0) / 11.0, 0.12, 0.55)
            return {"ALPHA": 1.0 - beta, "BETA": beta}
        if stage_l == "beta":
            return {"BETA": 0.86, "ALPHA": 0.14}
        if stage_l == "beta_prime":
            return {"BETA_PRIME": 0.76, "BETA": 0.24}
        if stage_l == "cold_worked":
            beta = _clamp((zn - 35.0) / 11.0, 0.1, 0.5)
            return _norm_dict({"ALPHA": 1.0 - beta, "BETA": beta, "DEFORMATION_BANDS": 0.1})
        return {"ALPHA": 0.8, "BETA": 0.2}

    if sys_name == "fe-si":
        si = float(comp.get("Si", 0.0) * 100.0)
        if stage_l == "liquid":
            return {"LIQUID": 1.0}
        if stage_l == "liquid_ferrite":
            lf = _clamp((float(processing.temperature_c) - 1410.0) / 170.0, 0.0, 1.0)
            return {"LIQUID": lf, "BCC_B2": 1.0 - lf}
        inter = _clamp((si - 1.0) / 5.0 * 0.28, 0.0, 0.28)
        if stage_l == "cold_worked_ferrite":
            return _norm_dict({"BCC_B2": 0.8 - inter * 0.5, "FESI_INTERMETALLIC": inter, "DEFORMATION_BANDS": 0.2})
        return _norm_dict({"BCC_B2": 1.0 - inter, "FESI_INTERMETALLIC": inter})

    if sys_name == "al-cu-mg":
        cu = float(comp.get("Cu", 0.0) * 100.0)
        mg = float(comp.get("Mg", 0.0) * 100.0)
        precip_index = _clamp((cu + 0.7 * mg) / 8.0, 0.0, 1.0)
        if stage_l == "solutionized":
            p = _clamp(0.02 + 0.04 * precip_index, 0.02, 0.08)
            return {"FCC_A1": 1.0 - p, "THETA": p}
        if stage_l == "quenched":
            return {"FCC_A1": 0.98, "THETA": 0.02}
        if stage_l == "natural_aged":
            p = _clamp(0.06 + 0.12 * precip_index, 0.05, 0.2)
            return {"FCC_A1": 1.0 - p, "THETA": p}
        if stage_l == "artificial_aged":
            p = _clamp(0.12 + 0.28 * precip_index, 0.12, 0.4)
            return _norm_dict({"FCC_A1": 1.0 - p, "THETA": p * 0.6, "S_PHASE": p * 0.4})
        if stage_l == "overaged":
            p = _clamp(0.18 + 0.34 * precip_index, 0.18, 0.52)
            return _norm_dict({"FCC_A1": 1.0 - p, "QPHASE": p * 0.45, "THETA": p * 0.3, "S_PHASE": p * 0.25})
        return {"FCC_A1": 0.92, "THETA": 0.08}

    # custom-multicomponent fallback
    fallback = dict(_RULES.get("fallback", {}))
    defaults = dict(fallback.get("default_phases", {"MATRIX": 0.72, "SECONDARY": 0.18, "PRECIPITATES": 0.1}))
    priority = [str(x) for x in fallback.get("matrix_element_priority", ["Fe", "Al", "Cu", "Ni", "Ti", "Mg"])]
    matrix_element = ""
    for el in priority:
        if float(comp.get(el, 0.0)) > 0.0:
            matrix_element = el
            break
    if not matrix_element and comp:
        matrix_element = max(comp, key=comp.get)
    matrix_share = float(comp.get(matrix_element, 0.0)) if matrix_element else 0.0
    alloying = _clamp(1.0 - matrix_share, 0.0, 1.0)
    scale = float(fallback.get("alloying_index_scale", 35.0))
    ai = _clamp(alloying * 100.0 / max(1e-6, scale), 0.0, 1.0)

    liquid_fraction = 0.0
    if stage_l == "liquid_custom":
        liquid_fraction = _clamp((float(processing.temperature_c) - 1200.0) / 250.0, 0.0, 1.0)
    base_matrix = _clamp(float(defaults.get("MATRIX", 0.72)) - ai * 0.12, 0.45, 0.85)
    secondary = _clamp(float(defaults.get("SECONDARY", 0.18)) + ai * 0.08, 0.08, 0.35)
    precip = _clamp(float(defaults.get("PRECIPITATES", 0.1)) + ai * 0.06, 0.04, 0.24)
    if stage_l == "deformed_custom":
        return _norm_dict({"MATRIX": base_matrix, "SECONDARY": secondary, "DEFORMATION_BANDS": 0.14, "PRECIPITATE": precip})
    if stage_l == "aged_custom":
        return _norm_dict({"MATRIX": base_matrix - 0.05, "SECONDARY": secondary, "PRECIPITATE": precip + 0.1})
    if liquid_fraction > 0.0:
        return _norm_dict({"LIQUID": liquid_fraction, "MATRIX": base_matrix * (1.0 - liquid_fraction), "SECONDARY": secondary * (1.0 - liquid_fraction), "PRECIPITATE": precip * (1.0 - liquid_fraction)})
    return _norm_dict({"MATRIX": base_matrix, "SECONDARY": secondary, "PRECIPITATE": precip})


def blend_phase_fractions(
    auto_phase_fractions: dict[str, float],
    manual_phase_fractions: dict[str, float],
    mode: str,
    weight: float,
) -> dict[str, float]:
    auto_norm = _norm_phase_dict(auto_phase_fractions)
    manual_norm = _norm_phase_dict(manual_phase_fractions)
    policy = str(mode or "auto_with_override").strip().lower()
    w = _clamp(float(weight), 0.0, 1.0)

    if policy == "auto_only":
        return auto_norm or manual_norm or {"MATRIX": 1.0}
    if policy == "manual_only":
        return manual_norm or auto_norm or {"MATRIX": 1.0}

    # auto_with_override
    if not manual_norm:
        return auto_norm or {"MATRIX": 1.0}
    if not auto_norm:
        return manual_norm
    keys = sorted(set(auto_norm) | set(manual_norm))
    mixed: dict[str, float] = {}
    for key in keys:
        mixed[key] = (1.0 - w) * float(auto_norm.get(key, 0.0)) + w * float(manual_norm.get(key, 0.0))
    return _norm_phase_dict(mixed) or {"MATRIX": 1.0}


@dataclass(slots=True)
class PhaseBundleV3:
    system: str
    stage: str
    phase_fractions: dict[str, float]
    phase_model_report: dict[str, Any]
    confidence: float


def build_phase_bundle(
    *,
    composition: dict[str, float],
    processing: ProcessingState,
    system_hint: str | None,
    phase_model: PhaseModelConfigV3,
    thermal_summary: dict[str, Any] | None = None,
    quench_summary: dict[str, Any] | None = None,
) -> PhaseBundleV3:
    system, confidence, fallback_used = infer_training_system(composition=composition, system_hint=system_hint)
    if system == "custom-multicomponent" and not bool(phase_model.allow_custom_fallback):
        raise ValueError("SYSTEM_UNSUPPORTED: custom-multicomponent disabled by phase model policy")

    stage = resolve_stage(
        system=system,
        composition=composition,
        processing=processing,
        thermal_summary=thermal_summary,
        quench_summary=quench_summary,
    )
    calibration_trace: dict[str, Any] = {"fraction_source": "default_formula"}
    auto_fractions = estimate_auto_phase_fractions(
        system=system,
        stage=stage,
        composition=composition,
        processing=processing,
        thermal_summary=thermal_summary,
        quench_summary=quench_summary,
        calibration_trace=calibration_trace,
    )
    blended = blend_phase_fractions(
        auto_phase_fractions=auto_fractions,
        manual_phase_fractions=dict(phase_model.manual_phase_fractions),
        mode=str(phase_model.phase_control_mode),
        weight=float(phase_model.manual_override_weight),
    )
    manual_norm = _norm_phase_dict(dict(phase_model.manual_phase_fractions))
    blend_applied = bool(str(phase_model.phase_control_mode).strip().lower() == "auto_with_override" and manual_norm)
    tol = _clamp(float(phase_model.phase_balance_tolerance_pct), 0.0, 100.0)
    keys = sorted(set(auto_fractions) | set(blended))
    error_pct: dict[str, float] = {}
    for key in keys:
        base = max(float(auto_fractions.get(key, 0.0)), 1e-6)
        err = abs(float(blended.get(key, 0.0)) - float(auto_fractions.get(key, 0.0))) / base * 100.0
        error_pct[key] = float(err)
    within_tolerance = bool(all(float(v) <= tol + 1e-9 for v in error_pct.values()))

    table_target_phases = (
        dict(calibration_trace.get("target_phase_fractions_from_table", {}))
        if isinstance(calibration_trace.get("target_phase_fractions_from_table", {}), dict)
        else {}
    )
    pearlite_internal_true_phases = (
        dict(calibration_trace.get("pearlite_internal_true_phases", {}))
        if isinstance(calibration_trace.get("pearlite_internal_true_phases", {}), dict)
        else {}
    )
    steel_microconstituents_auto = (
        dict(calibration_trace.get("steel_microconstituents_auto", {}))
        if isinstance(calibration_trace.get("steel_microconstituents_auto", {}), dict)
        else {}
    )
    steel_true_phases_auto = (
        dict(calibration_trace.get("steel_true_phases_auto", {}))
        if isinstance(calibration_trace.get("steel_true_phases_auto", {}), dict)
        else {}
    )
    steel_thresholds = (
        dict(calibration_trace.get("steel_equilibrium_thresholds_wt_pct", {}))
        if isinstance(calibration_trace.get("steel_equilibrium_thresholds_wt_pct", {}), dict)
        else {}
    )
    table_match_error_by_phase_pct: dict[str, float] = {}
    table_match_error_pct = 0.0
    if table_target_phases:
        for phase_name, target_val in table_target_phases.items():
            tv = max(0.0, float(target_val))
            av = max(0.0, float(blended.get(str(phase_name), 0.0)))
            table_match_error_by_phase_pct[str(phase_name)] = float(abs(av - tv) * 100.0)
        table_match_error_pct = float(max(table_match_error_by_phase_pct.values() or [0.0]))
    steel_true_phases_blended = (
        _true_phases_from_steel_microconstituents(
            microconstituents=blended,
            pearlite_internal=pearlite_internal_true_phases,
        )
        if pearlite_internal_true_phases
        else {}
    )

    report = {
        "auto_phase_fractions": auto_fractions,
        "manual_phase_fractions": manual_norm,
        "blended_phase_fractions": blended,
        "microconstituent_fractions_auto": steel_microconstituents_auto,
        "microconstituent_fractions_after_blend": (
            dict(blended) if steel_microconstituents_auto else {}
        ),
        "true_phase_fractions_auto": steel_true_phases_auto,
        "true_phase_fractions_after_blend": steel_true_phases_blended,
        "pearlite_internal_true_phases": pearlite_internal_true_phases,
        "steel_equilibrium_thresholds_wt_pct": steel_thresholds,
        "blend_applied": blend_applied,
        "phase_balance_tolerance_pct": tol,
        "fraction_error_pct": error_pct,
        "within_tolerance": within_tolerance,
        "fallback_used": bool(fallback_used or system == "custom-multicomponent"),
        "fallback_reason": (
            "no_confident_known_system"
            if bool(fallback_used or system == "custom-multicomponent")
            else ""
        ),
        "quench_medium_code_resolved": str((quench_summary or {}).get("medium_code_resolved", "")),
        "temper_shift_c": dict((quench_summary or {}).get("temper_shift_c", {}))
        if isinstance((quench_summary or {}).get("temper_shift_c", {}), dict)
        else {},
        "stage_rule_source": "fe_c_stage_rules_v3.json" if system == "fe-c" else "explicit_phase_rules_v3.json",
        "fraction_source": str(calibration_trace.get("fraction_source", "default_formula")),
        "calibration_profile": str(calibration_trace.get("calibration_profile", "")),
        "calibration_mode": "table_interpolated"
        if str(calibration_trace.get("fraction_source", "")) == "table_interpolated"
        else "default_formula",
        "calibration_source": str(calibration_trace.get("calibration_source", "")),
        "calibration_table_id": str(calibration_trace.get("table_id", "")),
        "calibration_table_c_wt_requested": calibration_trace.get("table_c_wt_requested", None),
        "calibration_table_c_wt_used": calibration_trace.get("table_c_wt_used", None),
        "calibration_table_clamped": bool(calibration_trace.get("table_row_clamped", False)),
        "calibration_medium_correction_applied": bool(calibration_trace.get("medium_correction_applied", False)),
        "target_constituents_from_table": dict(calibration_trace.get("target_constituents_from_table", {}))
        if isinstance(calibration_trace.get("target_constituents_from_table", {}), dict)
        else {},
        "target_phase_fractions_from_table": table_target_phases,
        "actual_phase_fractions_after_blend": dict(blended),
        "table_match_error_by_phase_pct": table_match_error_by_phase_pct,
        "table_match_error_pct": float(table_match_error_pct),
    }
    return PhaseBundleV3(
        system=system,
        stage=stage,
        phase_fractions=blended,
        phase_model_report=report,
        confidence=float(confidence),
    )
