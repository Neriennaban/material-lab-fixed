from __future__ import annotations

import math
from typing import Any


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def hbw_from_indent(load_kgf: float, ball_d_mm: float, indent_d_mm: float) -> dict[str, Any]:
    """Brinell hardness (HBW) direct calculation.

    Formula (ISO style): HBW = 2P / (pi * D * (D - sqrt(D^2 - d^2)))
    where P in kgf, D and d in mm.
    """
    p = float(load_kgf)
    d_ball = float(ball_d_mm)
    d_indent = float(indent_d_mm)

    if p <= 0.0:
        raise ValueError("P (нагрузка) должна быть > 0")
    if d_ball <= 0.0:
        raise ValueError("D (диаметр шарика) должен быть > 0")
    if d_indent <= 0.0 or d_indent >= d_ball:
        raise ValueError("d (диаметр отпечатка) должен быть в диапазоне (0, D)")

    root = d_ball * d_ball - d_indent * d_indent
    if root <= 0.0:
        raise ValueError("Некорректные параметры: D^2 - d^2 <= 0")

    denom = math.pi * d_ball * (d_ball - math.sqrt(root))
    if denom <= 0.0:
        raise ValueError("Деление на ноль при расчете HBW")

    hbw = (2.0 * p) / denom
    return {
        "mode": "direct",
        "P_kgf": p,
        "D_mm": d_ball,
        "d_mm": d_indent,
        "HBW": float(hbw),
    }


def hbw_estimate_from_microstructure(
    *,
    system: str,
    stage: str,
    phase_fractions: dict[str, float] | None,
    effect_vector: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Educational estimate of HBW from phase/stage.

    This is an instructional estimator and not a certified engineering calculation.
    """
    sys_name = str(system or "").strip().lower()
    st = str(stage or "").strip().lower()
    phases = {str(k).upper(): float(v) for k, v in dict(phase_fractions or {}).items()}
    fx = {str(k): float(v) for k, v in dict(effect_vector or {}).items()}

    # stage baseline
    base = 120.0
    if sys_name == "fe-c":
        if "martensite" in st:
            base = 560.0
        elif "troostite" in st:
            base = 430.0
        elif "sorbite" in st:
            base = 320.0
        elif "bainite" in st:
            base = 410.0
        elif "tempered_high" in st:
            base = 240.0
        elif "tempered_medium" in st:
            base = 300.0
        elif "tempered_low" in st:
            base = 380.0
        elif "pearlite" in st and "cementite" in st:
            base = 260.0
        elif "pearlite" in st:
            base = 210.0
        elif "ferrite" in st:
            base = 120.0
    elif sys_name == "al-si":
        base = 85.0 if "eutectic" in st else 70.0
    elif sys_name == "cu-zn":
        base = 130.0 if "beta" in st else 95.0
    elif sys_name == "fe-si":
        base = 140.0
    elif sys_name == "al-cu-mg":
        if "overaged" in st:
            base = 120.0
        elif "aged" in st:
            base = 145.0
        elif "quenched" in st:
            base = 105.0
        else:
            base = 90.0

    # phase contribution
    phase_delta = 0.0
    phase_delta += phases.get("MARTENSITE", 0.0) * 180.0
    phase_delta += phases.get("CEMENTITE", 0.0) * 70.0
    phase_delta += phases.get("PEARLITE", 0.0) * 30.0
    phase_delta += phases.get("BAINITE", 0.0) * 110.0
    phase_delta -= phases.get("FERRITE", 0.0) * 20.0
    phase_delta += phases.get("BETA", 0.0) * 25.0
    phase_delta += phases.get("BETA_PRIME", 0.0) * 40.0
    phase_delta += phases.get("SI", 0.0) * 30.0
    phase_delta += phases.get("THETA", 0.0) * 26.0
    phase_delta += phases.get("S_PHASE", 0.0) * 22.0

    disloc = max(0.0, fx.get("dislocation_proxy", 0.0))
    precip = max(0.0, fx.get("precipitation_level", 0.0))
    residual = max(0.0, fx.get("residual_stress", 0.0))
    effects = disloc * 38.0 + precip * 26.0 + residual * 14.0

    hbw = _clamp(base + phase_delta + effects, 45.0, 780.0)

    return {
        "mode": "estimated",
        "system": sys_name,
        "stage": st,
        "HBW": float(hbw),
        "note": "Оценочный расчет по структуре (учебный).",
    }
