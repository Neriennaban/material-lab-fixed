from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any


COMPARE_MODE_MAP: dict[str, str] = {
    "single": "single",
    "before_after": "before_after",
    "step_by_step": "step_by_step",
    "diff_map": "diff_map",
    "phase_transition_curve": "phase_transition_curve",
    "Один кадр": "single",
    "До/После": "before_after",
    "Пошагово": "step_by_step",
    "Карта отличий": "diff_map",
    "Фазовый переход (кривая)": "phase_transition_curve",
    "один кадр": "single",
    "до/после": "before_after",
    "пошагово": "step_by_step",
    "карта отличий": "diff_map",
    "фазовый переход (кривая)": "phase_transition_curve",
}


def normalize_compare_mode(value: str | None) -> str:
    if value is None:
        return "single"
    key = str(value).strip()
    if key in COMPARE_MODE_MAP:
        return COMPARE_MODE_MAP[key]
    low = key.lower()
    return COMPARE_MODE_MAP.get(low, "single")


def estimate_um_per_px(
    *,
    um_per_px_100x: float | None,
    crop_size_px: tuple[int, int] | list[int] | None,
    output_size_px: tuple[int, int] | list[int] | None,
) -> float:
    base = 1.0 if um_per_px_100x is None else float(um_per_px_100x)
    if base <= 0.0:
        base = 1.0

    if crop_size_px is None or output_size_px is None:
        return base

    try:
        crop_h = max(1, int(crop_size_px[0]))
        crop_w = max(1, int(crop_size_px[1]))
        out_h = max(1, int(output_size_px[0]))
        out_w = max(1, int(output_size_px[1]))
    except Exception:
        return base

    _ = crop_h, out_h
    return float(base) * (float(crop_w) / float(out_w))


def choose_scale_bar(
    um_per_px: float,
    *,
    min_px: int = 90,
    max_px: int = 220,
    target_px: int = 140,
) -> dict[str, float | bool]:
    um = float(max(1e-6, um_per_px))
    candidates_um = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

    best_um = candidates_um[0]
    best_score = float("inf")
    for candidate in candidates_um:
        px = float(candidate) / um
        in_range = min_px <= px <= max_px
        score = abs(px - target_px) if in_range else abs(px - target_px) + 1000.0
        if score < best_score:
            best_score = score
            best_um = candidate

    bar_px = float(best_um) / um
    return {
        "enabled": True,
        "um_per_px": float(um),
        "bar_um": float(best_um),
        "bar_nm": float(best_um * 1000.0),
        "bar_px": float(round(bar_px, 3)),
    }


def default_session_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_capture_metadata(
    *,
    source_image: str,
    source_metadata: str,
    microscope_params: dict[str, Any],
    view_meta: dict[str, Any],
    route_summary: dict[str, Any],
    session_id: str,
    capture_index: int,
    reticle_enabled: bool,
    scale_bar: dict[str, Any],
    controls_state: dict[str, Any],
    source_generator_version: str = "",
    prep_signature: dict[str, Any] | None = None,
    etch_signature: dict[str, Any] | None = None,
    quality_metrics: dict[str, Any] | None = None,
    mask_rendering: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_image": source_image,
        "source_metadata": source_metadata,
        "microscope_params": microscope_params,
        "view_meta": view_meta,
        "route_summary": route_summary,
        "session_id": session_id,
        "capture_index": int(capture_index),
        "reticle_enabled": bool(reticle_enabled),
        "scale_bar": dict(scale_bar),
        "controls_state": dict(controls_state),
        "source_generator_version": str(source_generator_version or ""),
        "prep_signature": dict(prep_signature or {}),
        "etch_signature": dict(etch_signature or {}),
        "quality_metrics": dict(quality_metrics or {}),
        "mask_rendering": dict(mask_rendering or {}),
    }


def normalize_capture_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(payload or {})
    data.setdefault("source_image", "")
    data.setdefault("source_metadata", "")
    data.setdefault("microscope_params", {})
    data.setdefault("view_meta", {})
    data.setdefault("route_summary", {})
    data.setdefault("session_id", "")
    data.setdefault("capture_index", 0)
    data.setdefault("reticle_enabled", True)
    data.setdefault(
        "scale_bar",
        {
            "enabled": False,
            "um_per_px": 1.0,
            "bar_um": 100.0,
            "bar_nm": 100000.0,
            "bar_px": 100.0,
        },
    )
    data.setdefault(
        "controls_state",
        {
            "objective": 200,
            "focus_distance_mm": 18.0,
            "focus_coarse": 0.95,
            "focus_fine": 0.0,
            "stage_x": 0.5,
            "stage_y": 0.5,
        },
    )
    data.setdefault("source_generator_version", "")
    data.setdefault("prep_signature", {})
    data.setdefault("etch_signature", {})
    data.setdefault("quality_metrics", {})
    data.setdefault("mask_rendering", {})
    return data
