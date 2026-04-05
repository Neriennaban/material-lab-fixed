from __future__ import annotations

import math
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_tuple2(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None
    return None


def format_metric_length_um(length_um: float) -> str:
    value = max(0.0, float(length_um))
    if value >= 1_000.0:
        return f"{value / 1000.0:.3f} мм"
    if value >= 1.0:
        return f"{value:.2f} мкм"
    return f"{value * 1000.0:.0f} нм"


def format_metric_area_um2(area_um2: float) -> str:
    value = max(0.0, float(area_um2))
    if value >= 1_000_000.0:
        return f"{value / 1_000_000.0:.4f} мм²"
    return f"{value:.2f} мкм²"


def line_measurement(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    um_per_px: float,
) -> dict[str, float | bool | str]:
    x0, y0 = float(start_xy[0]), float(start_xy[1])
    x1, y1 = float(end_xy[0]), float(end_xy[1])
    dx = x1 - x0
    dy = y1 - y0
    length_px = math.hypot(dx, dy)
    um_per_px_safe = max(1e-9, float(um_per_px))
    length_um = length_px * um_per_px_safe
    angle_deg = math.degrees(math.atan2(dy, dx)) if length_px > 0.0 else 0.0
    return {
        "valid": bool(length_px > 0.0),
        "kind": "line",
        "x0_px": float(x0),
        "y0_px": float(y0),
        "x1_px": float(x1),
        "y1_px": float(y1),
        "dx_px": float(dx),
        "dy_px": float(dy),
        "length_px": float(length_px),
        "length_um": float(length_um),
        "angle_deg": float(angle_deg),
        "um_per_px": float(um_per_px_safe),
        "label": format_metric_length_um(length_um),
    }


def polygon_area_measurement(
    vertices_xy: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    um_per_px: float,
) -> dict[str, Any]:
    points: list[tuple[float, float]] = []
    for value in vertices_xy:
        pt = _safe_tuple2(value)
        if pt is not None:
            points.append(pt)
    um_per_px_safe = max(1e-9, float(um_per_px))
    if len(points) < 3:
        return {
            "valid": False,
            "kind": "polygon_area",
            "vertex_count": int(len(points)),
            "area_px2": 0.0,
            "area_um2": 0.0,
            "perimeter_px": 0.0,
            "perimeter_um": 0.0,
            "um_per_px": float(um_per_px_safe),
            "vertices_px": [[float(x), float(y)] for x, y in points],
            "label": format_metric_area_um2(0.0),
        }

    twice_area = 0.0
    perimeter_px = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        twice_area += (x0 * y1) - (x1 * y0)
        perimeter_px += math.hypot(x1 - x0, y1 - y0)
    area_px2 = abs(twice_area) * 0.5
    area_um2 = area_px2 * (um_per_px_safe**2)
    perimeter_um = perimeter_px * um_per_px_safe
    return {
        "valid": bool(area_px2 > 0.0),
        "kind": "polygon_area",
        "vertex_count": int(len(points)),
        "area_px2": float(area_px2),
        "area_um2": float(area_um2),
        "perimeter_px": float(perimeter_px),
        "perimeter_um": float(perimeter_um),
        "um_per_px": float(um_per_px_safe),
        "vertices_px": [[float(x), float(y)] for x, y in points],
        "label": format_metric_area_um2(area_um2),
    }


def estimate_um_per_px_from_geometry(
    *,
    um_per_px_100x: float,
    crop_size_px: tuple[int, int] | list[int] | None,
    output_size_px: tuple[int, int] | list[int] | None,
) -> float:
    base = max(1e-9, float(um_per_px_100x))
    if crop_size_px is None or output_size_px is None:
        return float(base)
    try:
        crop_w = max(1, int(crop_size_px[1]))
        out_w = max(1, int(output_size_px[1]))
    except Exception:
        return float(base)
    return float(base) * (float(crop_w) / float(out_w))


def derive_um_per_px_100x(metadata: dict[str, Any] | None, default: float = 1.0) -> tuple[float, str]:
    meta = dict(metadata or {})
    microscope_ready = meta.get("microscope_ready")
    if isinstance(microscope_ready, dict):
        val = microscope_ready.get("um_per_px_100x")
        if isinstance(val, (int, float)) and float(val) > 0.0:
            return float(val), "metadata.microscope_ready.um_per_px_100x"

        native_um_per_px = microscope_ready.get("native_um_per_px")
        if isinstance(native_um_per_px, (int, float)) and float(native_um_per_px) > 0.0:
            source_mag = _find_source_magnification(meta)
            if source_mag is not None:
                return float(native_um_per_px) * float(source_mag) / 100.0, (
                    "metadata.microscope_ready.native_um_per_px"
                )

        rendered_um_per_px = microscope_ready.get("um_per_px")
        render_scale = microscope_ready.get("render_scale")
        if (
            isinstance(rendered_um_per_px, (int, float))
            and float(rendered_um_per_px) > 0.0
            and isinstance(render_scale, (int, float))
            and float(render_scale) > 0.0
        ):
            source_mag = _find_source_magnification(meta)
            if source_mag is not None:
                native_um_per_px = float(rendered_um_per_px) * float(render_scale)
                return native_um_per_px * float(source_mag) / 100.0, (
                    "metadata.microscope_ready.um_per_px * render_scale"
                )

        val_any = microscope_ready.get("um_per_px")
        if isinstance(val_any, (int, float)) and float(val_any) > 0.0:
            source_mag = _find_source_magnification(meta)
            if source_mag is not None:
                return float(val_any) * float(source_mag) / 100.0, "metadata.microscope_ready.um_per_px"

    microscope_params = meta.get("microscope_params")
    if isinstance(microscope_params, dict):
        val = microscope_params.get("um_per_px_100x")
        if isinstance(val, (int, float)) and float(val) > 0.0:
            return float(val), "metadata.microscope_params.um_per_px_100x"
        val_any = microscope_params.get("um_per_px")
        if isinstance(val_any, (int, float)) and float(val_any) > 0.0:
            source_mag = _find_source_magnification(meta)
            if source_mag is not None:
                return float(val_any) * float(source_mag) / 100.0, "metadata.microscope_params.um_per_px"

    val2 = meta.get("um_per_px_100x")
    if isinstance(val2, (int, float)) and float(val2) > 0.0:
        return float(val2), "metadata.um_per_px_100x"

    return float(max(1e-9, default)), "default.assumption"


def _find_source_magnification(metadata: dict[str, Any]) -> float | None:
    req_v3 = metadata.get("request_v3")
    if isinstance(req_v3, dict):
        microscope_profile = req_v3.get("microscope_profile")
        if isinstance(microscope_profile, dict):
            mag = microscope_profile.get("magnification")
            if isinstance(mag, (int, float)) and float(mag) > 0.0:
                return float(mag)

    microscope_params = metadata.get("microscope_params")
    if isinstance(microscope_params, dict):
        mag = microscope_params.get("magnification")
        if isinstance(mag, (int, float)) and float(mag) > 0.0:
            return float(mag)

    microscope_ready = metadata.get("microscope_ready")
    if isinstance(microscope_ready, dict):
        mags = microscope_ready.get("recommended_magnifications")
        if isinstance(mags, list) and mags:
            first = mags[0]
            if isinstance(first, (int, float)) and float(first) > 0.0:
                return float(first)
    return None


def scale_audit_report(
    *,
    objective: int,
    source_size_px: tuple[int, int] | list[int] | None,
    crop_size_px: tuple[int, int] | list[int] | None,
    output_size_px: tuple[int, int] | list[int] | None,
    um_per_px_100x: float,
    actual_um_per_px: float,
    reference_magnification: int = 100,
    crop_tolerance_px: float = 1.0,
    scale_rel_tolerance: float = 1e-6,
) -> dict[str, float | bool | str | None]:
    objective_safe = max(1, int(objective))
    expected_um_per_px = estimate_um_per_px_from_geometry(
        um_per_px_100x=um_per_px_100x,
        crop_size_px=crop_size_px,
        output_size_px=output_size_px,
    )
    delta_um_per_px = float(actual_um_per_px) - float(expected_um_per_px)
    rel_error = abs(delta_um_per_px) / max(abs(expected_um_per_px), 1e-12)

    crop_ok = None
    expected_crop_h = None
    expected_crop_w = None
    crop_delta_h = None
    crop_delta_w = None
    if source_size_px is not None and crop_size_px is not None:
        try:
            src_h = max(1, int(source_size_px[0]))
            src_w = max(1, int(source_size_px[1]))
            crop_h = max(1, int(crop_size_px[0]))
            crop_w = max(1, int(crop_size_px[1]))
            ratio = float(reference_magnification) / float(objective_safe)
            expected_crop_h = min(src_h, max(32, int(round(src_h * ratio))))
            expected_crop_w = min(src_w, max(32, int(round(src_w * ratio))))
            crop_delta_h = float(crop_h - expected_crop_h)
            crop_delta_w = float(crop_w - expected_crop_w)
            crop_ok = bool(
                abs(crop_delta_h) <= float(crop_tolerance_px)
                and abs(crop_delta_w) <= float(crop_tolerance_px)
            )
        except Exception:
            crop_ok = None

    scale_ok = bool(rel_error <= float(scale_rel_tolerance))
    overall_ok = bool(scale_ok and (crop_ok is not False))
    return {
        "ok": overall_ok,
        "scale_ok": scale_ok,
        "crop_ok": crop_ok,
        "expected_um_per_px": float(expected_um_per_px),
        "actual_um_per_px": float(actual_um_per_px),
        "delta_um_per_px": float(delta_um_per_px),
        "relative_error": float(rel_error),
        "expected_crop_h_px": expected_crop_h,
        "expected_crop_w_px": expected_crop_w,
        "crop_delta_h_px": crop_delta_h,
        "crop_delta_w_px": crop_delta_w,
        "reference_magnification": int(reference_magnification),
        "objective": int(objective_safe),
    }


__all__ = [
    "derive_um_per_px_100x",
    "estimate_um_per_px_from_geometry",
    "format_metric_area_um2",
    "format_metric_length_um",
    "line_measurement",
    "polygon_area_measurement",
    "scale_audit_report",
]
