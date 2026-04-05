from __future__ import annotations

from typing import Any

import numpy as np

from .cooling_modes import canonicalize_cooling_mode


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_cooling_curve_points(
    points: list[dict[str, Any]] | list[tuple[float, float]] | None,
    *,
    fallback_temperature_c: float,
) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []
    for item in points or []:
        if isinstance(item, dict):
            t = _safe_float(item.get("time_min", item.get("time", 0.0)), 0.0)
            temp = _safe_float(item.get("temperature_c", item.get("temperature", fallback_temperature_c)), fallback_temperature_c)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            t = _safe_float(item[0], 0.0)
            temp = _safe_float(item[1], fallback_temperature_c)
        else:
            continue
        normalized.append({"time_min": float(max(0.0, t)), "temperature_c": float(temp)})

    normalized.sort(key=lambda x: x["time_min"])

    dedup: list[dict[str, float]] = []
    for point in normalized:
        if dedup and abs(point["time_min"] - dedup[-1]["time_min"]) < 1e-9:
            dedup[-1] = point
        else:
            dedup.append(point)

    if len(dedup) < 2:
        start_temp = float(fallback_temperature_c)
        end_temp = 20.0 if start_temp >= 20.0 else start_temp
        dedup = [
            {"time_min": 0.0, "temperature_c": start_temp},
            {"time_min": 10.0, "temperature_c": end_temp},
        ]

    return dedup


def infer_cooling_mode_from_slope(
    slope_c_per_min: float,
    *,
    base_mode: str,
) -> str:
    base = canonicalize_cooling_mode(base_mode)
    slope = float(slope_c_per_min)
    if slope <= -25.0:
        return "quenched"
    if slope <= -8.0:
        return "normalized"
    if slope <= -2.0:
        return "slow_cool"

    if abs(slope) <= 1.5 and base in {
        "aged",
        "overaged",
        "natural_aged",
        "tempered",
        "solutionized",
        "cold_worked",
    }:
        return base

    return "equilibrium"


def _sample_segment_points(
    t0: float,
    temp0: float,
    t1: float,
    temp1: float,
    *,
    mode: str,
    degree_step: float,
) -> list[tuple[float, float]]:
    dt = float(t1 - t0)
    dtemp = float(temp1 - temp0)
    if dt <= 0.0:
        return [(t1, temp1)]

    if mode != "per_degree" or abs(dtemp) < 1e-9:
        return [(t1, temp1)]

    step = max(0.1, float(degree_step))
    direction = 1.0 if dtemp > 0 else -1.0
    samples = int(abs(dtemp) / step)
    if samples <= 0:
        return [(t1, temp1)]

    out: list[tuple[float, float]] = []
    for idx in range(1, samples + 1):
        target_temp = temp0 + direction * step * idx
        if (direction > 0 and target_temp >= temp1) or (direction < 0 and target_temp <= temp1):
            break
        alpha = (target_temp - temp0) / dtemp
        target_time = t0 + alpha * dt
        out.append((float(target_time), float(target_temp)))
    out.append((float(t1), float(temp1)))
    return out


def sample_cooling_curve(
    points: list[dict[str, float]],
    *,
    mode: str = "per_degree",
    degree_step: float = 1.0,
    max_points: int = 220,
    base_mode: str = "equilibrium",
) -> list[dict[str, float | str]]:
    if len(points) < 2:
        return []

    sample_mode = str(mode or "per_degree").strip().lower()
    if sample_mode not in {"per_degree", "points"}:
        sample_mode = "per_degree"

    sequence: list[tuple[float, float]] = [(float(points[0]["time_min"]), float(points[0]["temperature_c"]))]
    for idx in range(1, len(points)):
        prev = points[idx - 1]
        curr = points[idx]
        seg = _sample_segment_points(
            prev["time_min"],
            prev["temperature_c"],
            curr["time_min"],
            curr["temperature_c"],
            mode=sample_mode,
            degree_step=degree_step,
        )
        sequence.extend(seg)

    if max_points > 2 and len(sequence) > max_points:
        keep_idx = np.linspace(0, len(sequence) - 1, num=int(max_points), dtype=int)
        sequence = [sequence[int(i)] for i in keep_idx]

    out: list[dict[str, float | str]] = []
    for idx, (time_min, temperature_c) in enumerate(sequence):
        if idx == 0:
            next_time, next_temp = sequence[min(1, len(sequence) - 1)]
            dt = max(1e-9, float(next_time - time_min))
            slope = (float(next_temp) - float(temperature_c)) / dt
        else:
            prev_time, prev_temp = sequence[idx - 1]
            dt = max(1e-9, float(time_min - prev_time))
            slope = (float(temperature_c) - float(prev_temp)) / dt

        mode_code = infer_cooling_mode_from_slope(slope, base_mode=base_mode)
        out.append(
            {
                "index": idx,
                "time_min": float(time_min),
                "temperature_c": float(temperature_c),
                "slope_c_per_min": float(slope),
                "cooling_mode": mode_code,
            }
        )

    return out
