from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .contracts_v2 import ProcessingState

_RULEBOOK_DIR = Path(__file__).resolve().parent / "rulebook"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_DIAGRAM_RULES = _load_json(_RULEBOOK_DIR / "diagram_lines.json")
_SYSTEM_RULES = _load_json(_RULEBOOK_DIR / "systems_rules.json")
_TEXTBOOK_FE_C_RULES = _load_json(_RULEBOOK_DIR / "diagram_textbook_fe_c_v3.json")

DEFAULT_LAYER_FLAGS: dict[str, bool] = {
    "axes": True,
    "grid": True,
    "lines": True,
    "invariants": True,
    "regions": True,
    "marker": True,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _linmap(value: float, src0: float, src1: float, dst0: float, dst1: float) -> float:
    if abs(src1 - src0) < 1e-12:
        return dst0
    return dst0 + (value - src0) / (src1 - src0) * (dst1 - dst0)


def _prettify_phase_text(raw: str | Any) -> str:
    text = str(raw or "")
    if not text:
        return ""
    text = text.replace("Fe3C", "Fe₃C").replace("FE3C", "Fe₃C")
    greek_map = {
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
        "delta": "δ",
        "epsilon": "ε",
    }
    for token, symbol in greek_map.items():
        text = re.sub(rf"(?<![A-Za-z]){token}(?![A-Za-z])", symbol, text, flags=re.IGNORECASE)
    return text


@lru_cache(maxsize=32)
def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    sz = max(9, int(size))
    candidates: list[str] = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/timesbd.ttf",
            ]
        )
    candidates.extend(
        [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/times.ttf",
        ]
    )
    try:
        pil_font = (Path(ImageFont.__file__).resolve().parent / "Fonts" / "DejaVuSans.ttf")
        candidates.append(str(pil_font))
    except Exception:
        pass

    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def available_diagram_systems() -> list[str]:
    return sorted(_DIAGRAM_RULES.get("systems", {}).keys())


def resolve_diagram_system(
    requested_system: str | None,
    composition: dict[str, float],
    inferred_system: str | None = None,
) -> tuple[str, float, bool]:
    systems = set(available_diagram_systems())
    if requested_system:
        requested = requested_system.strip().lower()
        if requested in systems:
            return requested, 1.0, False

    if inferred_system:
        inferred = inferred_system.strip().lower()
        if inferred in systems:
            return inferred, 1.0, False

    total = max(1e-9, float(sum(max(0.0, float(v)) for v in composition.values())))
    markers = _SYSTEM_RULES.get("system_inference", {})
    scored: list[tuple[str, float]] = []
    for system_name in available_diagram_systems():
        mm = markers.get(system_name, [])
        marker_sum = sum(float(composition.get(el, 0.0)) for el in mm)
        score = marker_sum / total
        scored.append((system_name, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    if scored and scored[0][1] > 0.0:
        top = scored[0]
        confidence = float(_clamp(0.25 + top[1] * 0.75, 0.25, 0.95))
        return top[0], confidence, True
    return "fe-c", 0.2, True


def _point_for_system(
    system: str,
    composition: dict[str, float],
    processing: ProcessingState,
) -> tuple[float, float]:
    key = system.strip().lower()
    if key == "fe-c":
        return float(composition.get("C", 0.0)), float(processing.temperature_c)
    if key == "al-si":
        return float(composition.get("Si", 0.0)), float(processing.temperature_c)
    if key == "cu-zn":
        return float(composition.get("Zn", 0.0)), float(processing.temperature_c)
    if key == "fe-si":
        return float(composition.get("Si", 0.0)), float(processing.temperature_c)
    if key == "al-cu-mg":
        return float(processing.aging_hours), float(processing.aging_temperature_c)
    return float(composition.get("C", 0.0)), float(processing.temperature_c)


def estimate_phase_region(system: str, x_value: float, y_value: float) -> str:
    spec = _DIAGRAM_RULES.get("systems", {}).get(system)
    if not isinstance(spec, dict):
        return "unknown"
    x_axis = spec.get("x_axis", {})
    y_axis = spec.get("y_axis", {})
    x0, x1 = float(x_axis.get("min", 0.0)), float(x_axis.get("max", 1.0))
    y0, y1 = float(y_axis.get("min", 0.0)), float(y_axis.get("max", 1.0))
    regions = spec.get("regions", [])
    if not regions:
        return "unknown"

    best_name = "unknown"
    best_dist = float("inf")
    for region in regions:
        anchor = region.get("anchor", [])
        if not isinstance(anchor, list) or len(anchor) != 2:
            continue
        ax = float(anchor[0])
        ay = float(anchor[1])
        dx = (x_value - ax) / max(1e-9, abs(x1 - x0))
        dy = (y_value - ay) / max(1e-9, abs(y1 - y0))
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_name = str(region.get("name", "unknown"))
    return best_name


def _draw_polyline(draw: ImageDraw.ImageDraw, curve: list[tuple[int, int]], *, color: tuple[int, int, int], width: int = 2) -> None:
    if len(curve) < 2:
        return
    draw.line(curve, fill=color, width=max(1, int(width)))


def _render_fe_c_textbook_snapshot(
    *,
    composition: dict[str, float],
    processing: ProcessingState,
    spec: dict[str, Any],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    layer_flags: dict[str, bool],
    system_confidence: float,
    is_fallback: bool,
    phase_region: str,
    current_x: float,
    current_y: float,
) -> Image.Image:
    image = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(image)
    font_small = _load_font(14)
    font_axis = _load_font(18)
    font_title = _load_font(20, bold=True)

    left = 88
    right = width - 34
    top = 28
    bottom = height - 84

    def px(xv: float) -> int:
        return int(_linmap(xv, x_min, x_max, left, right))

    def py(yv: float) -> int:
        return int(_linmap(yv, y_min, y_max, bottom, top))

    # Canvas and border
    draw.rectangle((left, top, right, bottom), fill=(255, 255, 255), outline=(20, 20, 20), width=2)

    rules = _TEXTBOOK_FE_C_RULES if isinstance(_TEXTBOOK_FE_C_RULES, dict) else {}
    x_ticks = [float(x) for x in list(rules.get("x_ticks", []))]
    y_ticks = [float(y) for y in list(rules.get("y_ticks", []))]
    ref_verticals = [float(x) for x in list(rules.get("reference_verticals", []))]
    ref_isotherms = [float(y) for y in list(rules.get("reference_isotherms", []))]
    line_styles = rules.get("line_styles", {}) if isinstance(rules.get("line_styles", {}), dict) else {}

    if layer_flags.get("grid", True):
        for xv in x_ticks:
            x = px(xv)
            draw.line((x, top, x, bottom), fill=(214, 214, 214), width=1)
        for yv in y_ticks:
            y = py(yv)
            draw.line((left, y, right, y), fill=(214, 214, 214), width=1)

    if layer_flags.get("axes", True):
        draw.line((left, top, left, bottom), fill=(16, 16, 16), width=2)
        draw.line((left, bottom, right, bottom), fill=(16, 16, 16), width=2)
        x_axis_label = str(rules.get("x_axis_label_ru", "Содержание углерода C, % (по массе)"))
        y_axis_label = str(rules.get("y_axis_label_ru", "Температура T, °C"))
        draw.text((left + 120, bottom + 38), x_axis_label, fill=(24, 24, 24), font=font_axis)
        draw.text((10, top + 120), y_axis_label, fill=(24, 24, 24), font=font_axis)

        for xv in x_ticks:
            x = px(xv)
            label = f"{xv:g}"
            draw.text((x - 10, bottom + 8), label, fill=(20, 20, 20), font=font_small)
        for yv in y_ticks:
            y = py(yv)
            label = f"{yv:.0f}"
            draw.text((left - 46, y - 8), label, fill=(20, 20, 20), font=font_small)

    if layer_flags.get("invariants", True):
        for xv in ref_verticals:
            x = px(xv)
            draw.line((x, py(727.0), x, bottom), fill=(88, 88, 88), width=1)
        for yv in ref_isotherms:
            y = py(yv)
            draw.line((left, y, right, y), fill=(48, 48, 48), width=2)

    if layer_flags.get("lines", True):
        # Canonical lines
        for line in spec.get("lines", []):
            pts = line.get("points", [])
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            curve: list[tuple[int, int]] = []
            for point in pts:
                if not isinstance(point, list) or len(point) != 2:
                    continue
                curve.append((px(float(point[0])), py(float(point[1]))))
            name = str(line.get("name", "")).strip().lower()
            if name == "a3":
                style = line_styles.get("a3", {})
            elif name == "acm":
                style = line_styles.get("acm", {})
            elif name in {"liquidus_left", "liquidus_right", "solidus", "a1"}:
                style = line_styles.get("default", {})
            else:
                style = line_styles.get("default", {})
            raw_color = style.get("color", [20, 20, 20]) if isinstance(style, dict) else [20, 20, 20]
            raw_width = style.get("width", 2) if isinstance(style, dict) else 2
            color = tuple(int(v) for v in list(raw_color)[:3]) if isinstance(raw_color, list) else (20, 20, 20)
            _draw_polyline(draw, curve, color=color, width=int(raw_width))

        # Ensure A3 and Acm are visible in textbook style even if absent in source rulebook.
        x_a3_0, y_a3_0 = 0.0, 911.0
        x_a3_1, y_a3_1 = 0.77, 727.0
        style_a3 = line_styles.get("a3", {})
        color_a3 = tuple(int(v) for v in list(style_a3.get("color", [45, 90, 230]))[:3])
        width_a3 = int(style_a3.get("width", 2))
        _draw_polyline(draw, [(px(x_a3_0), py(y_a3_0)), (px(x_a3_1), py(y_a3_1))], color=color_a3, width=width_a3)

        x_acm_0, y_acm_0 = 0.77, 727.0
        x_acm_1, y_acm_1 = 2.14, 1147.0
        style_acm = line_styles.get("acm", {})
        color_acm = tuple(int(v) for v in list(style_acm.get("color", [210, 48, 48]))[:3])
        width_acm = int(style_acm.get("width", 2))
        _draw_polyline(draw, [(px(x_acm_0), py(y_acm_0)), (px(x_acm_1), py(y_acm_1))], color=color_acm, width=width_acm)

    if layer_flags.get("regions", True):
        region_labels = rules.get("region_labels", [])
        if isinstance(region_labels, list):
            for item in region_labels:
                if not isinstance(item, dict):
                    continue
                txt = _prettify_phase_text(item.get("text", ""))
                xv = float(item.get("x", 0.0))
                yv = float(item.get("y", 0.0))
                draw.text((px(xv), py(yv)), txt, fill=(20, 20, 20), font=font_small)

    current_px = px(_clamp(current_x, x_min, x_max))
    current_py = py(_clamp(current_y, y_min, y_max))
    if layer_flags.get("marker", True):
        draw.line((current_px, top, current_px, bottom), fill=(180, 0, 0), width=1)
        draw.line((left, current_py, right, current_py), fill=(180, 0, 0), width=1)
        draw.ellipse((current_px - 5, current_py - 5, current_px + 5, current_py + 5), fill=(255, 255, 255), outline=(180, 0, 0), width=2)

    draw.text((left, 6), str(rules.get("title", "Диаграмма состояния Fe-C")), fill=(18, 18, 18), font=font_title)
    draw.text((right - 240, 8), f"Область: {_prettify_phase_text(phase_region)}", fill=(18, 18, 18), font=font_small)
    draw.text((right - 240, 26), f"Уверенность: {system_confidence:.2f}", fill=(70, 70, 70), font=font_small)
    if is_fallback:
        draw.text((left + 260, 8), "Fallback для custom состава", fill=(145, 80, 0), font=font_small)

    legend = rules.get("legend", [])
    if isinstance(legend, list) and legend:
        draw.text((left + 180, bottom + 56), "Условные обозначения:", fill=(20, 20, 20), font=font_small)
        x_pos = left + 16
        y_pos = bottom + 38
        for token in legend:
            draw.text((x_pos, y_pos), _prettify_phase_text(token), fill=(20, 20, 20), font=font_small)
            y_pos += 16
            if y_pos > height - 12:
                y_pos = bottom + 38
                x_pos += 230
    return image


def render_diagram_snapshot(
    composition: dict[str, float],
    processing: ProcessingState,
    requested_system: str | None = None,
    inferred_system: str | None = None,
    confidence: float | None = None,
    layers: dict[str, bool] | None = None,
    style_profile: str | None = None,
    size: tuple[int, int] = (900, 460),
) -> dict[str, Any]:
    layer_flags = dict(DEFAULT_LAYER_FLAGS)
    if layers:
        layer_flags.update({k: bool(v) for k, v in layers.items()})

    system, fallback_confidence, is_fallback = resolve_diagram_system(
        requested_system=requested_system,
        composition=composition,
        inferred_system=inferred_system,
    )
    system_confidence = float(confidence if confidence is not None else fallback_confidence)

    spec = _DIAGRAM_RULES.get("systems", {}).get(system, {})
    x_axis = spec.get("x_axis", {"label": "x", "min": 0.0, "max": 1.0})
    y_axis = spec.get("y_axis", {"label": "y", "min": 0.0, "max": 1.0})

    x_min = float(x_axis.get("min", 0.0))
    x_max = float(x_axis.get("max", 1.0))
    y_min = float(y_axis.get("min", 0.0))
    y_max = float(y_axis.get("max", 1.0))

    width, height = int(size[0]), int(size[1])
    style_code = str(style_profile or "default_dark").strip().lower()

    current_x, current_y = _point_for_system(system=system, composition=composition, processing=processing)
    current_x = _clamp(current_x, x_min, x_max)
    current_y = _clamp(current_y, y_min, y_max)
    phase_region = estimate_phase_region(system=system, x_value=current_x, y_value=current_y)

    textbook_enabled = style_code == "textbook_fe_c" and system == "fe-c"
    if textbook_enabled:
        image = _render_fe_c_textbook_snapshot(
            composition=composition,
            processing=processing,
            spec=spec,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            width=width,
            height=height,
            layer_flags=layer_flags,
            system_confidence=system_confidence,
            is_fallback=is_fallback,
            phase_region=phase_region,
            current_x=current_x,
            current_y=current_y,
        )
        return {
            "image": image,
            "used_system": system,
            "confidence": system_confidence,
            "is_fallback": is_fallback,
            "phase_region": phase_region,
            "current_point": {"x": current_x, "y": current_y},
            "layers": layer_flags,
            "diagram_style": {"profile_id": "textbook_fe_c", "applied": True, "system": system},
            "diagram_style_report": {
                "has_reference_isotherms": True,
                "has_reference_verticals": True,
                "label_mode_ru": True,
            },
        }

    image = Image.new("RGB", (width, height), color=(16, 22, 32))
    draw = ImageDraw.Draw(image)
    font_small = _load_font(13)
    font_axis = _load_font(14)
    font_title = _load_font(16, bold=True)

    # Background gradient
    for yi in range(height):
        tone = int(_linmap(yi, 0, max(1, height - 1), 34, 16))
        draw.line((0, yi, width, yi), fill=(tone, tone + 8, tone + 16))

    left = 78
    right = width - 28
    top = 26
    bottom = height - 56

    def px(xv: float) -> int:
        return int(_linmap(xv, x_min, x_max, left, right))

    def py(yv: float) -> int:
        return int(_linmap(yv, y_min, y_max, bottom, top))

    # Grid and axes
    if layer_flags["grid"]:
        x_ticks = 8
        y_ticks = 8
        for i in range(x_ticks + 1):
            t = i / x_ticks
            x = int(left + (right - left) * t)
            draw.line((x, top, x, bottom), fill=(56, 72, 94), width=1)
            xv = _linmap(t, 0.0, 1.0, x_min, x_max)
            draw.text((x - 16, bottom + 8), f"{xv:.2g}", fill=(218, 224, 231), font=font_small)
        for i in range(y_ticks + 1):
            t = i / y_ticks
            y = int(bottom - (bottom - top) * t)
            draw.line((left, y, right, y), fill=(56, 72, 94), width=1)
            yv = _linmap(t, 0.0, 1.0, y_min, y_max)
            draw.text((12, y - 8), f"{yv:.0f}", fill=(218, 224, 231), font=font_small)

    if layer_flags["axes"]:
        draw.line((left, top, left, bottom), fill=(226, 231, 237), width=2)
        draw.line((left, bottom, right, bottom), fill=(226, 231, 237), width=2)
        draw.text((left + 8, bottom + 30), str(x_axis.get("label", "x")), fill=(229, 233, 239), font=font_axis)
        draw.text((8, top + 2), str(y_axis.get("label", "y")), fill=(229, 233, 239), font=font_axis)

    if layer_flags["lines"]:
        for line in spec.get("lines", []):
            pts = line.get("points", [])
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            curve = []
            for point in pts:
                if not isinstance(point, list) or len(point) != 2:
                    continue
                curve.append((px(float(point[0])), py(float(point[1]))))
            if len(curve) >= 2:
                draw.line(curve, fill=(248, 191, 74), width=2)

    if layer_flags["invariants"]:
        for item in spec.get("invariants", []):
            point = item.get("point", [])
            if not isinstance(point, list) or len(point) != 2:
                continue
            x = px(float(point[0]))
            y = py(float(point[1]))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(140, 232, 160), outline=(198, 240, 209), width=1)
            draw.text((x + 6, y - 8), _prettify_phase_text(item.get("name", "")), fill=(198, 240, 209), font=font_small)

    if layer_flags["regions"]:
        for region in spec.get("regions", []):
            anchor = region.get("anchor", [])
            if not isinstance(anchor, list) or len(anchor) != 2:
                continue
            x = px(float(anchor[0]))
            y = py(float(anchor[1]))
            draw.text((x, y), _prettify_phase_text(region.get("name", "")), fill=(213, 223, 236), font=font_small)

    current_px = px(current_x)
    current_py = py(current_y)

    if layer_flags["marker"]:
        draw.line((current_px, top, current_px, bottom), fill=(255, 86, 86), width=1)
        draw.line((left, current_py, right, current_py), fill=(255, 86, 86), width=1)
        draw.ellipse(
            (current_px - 6, current_py - 6, current_px + 6, current_py + 6),
            fill=(255, 210, 210),
            outline=(255, 86, 86),
            width=2,
        )

    draw.text((left, 6), f"Diagram: {system}", fill=(239, 242, 247), font=font_title)
    draw.text((right - 255, 6), f"Region: {_prettify_phase_text(phase_region)}", fill=(239, 242, 247), font=font_small)
    draw.text((right - 255, 22), f"Confidence: {system_confidence:.2f}", fill=(196, 212, 228), font=font_small)
    if is_fallback:
        draw.text((left + 170, 6), "Fallback mode for custom composition", fill=(255, 208, 134), font=font_small)

    return {
        "image": image,
        "used_system": system,
        "confidence": system_confidence,
        "is_fallback": is_fallback,
        "phase_region": phase_region,
        "current_point": {"x": current_x, "y": current_y},
        "layers": layer_flags,
        "diagram_style": {"profile_id": "default_dark", "applied": False, "system": system},
        "diagram_style_report": {
            "has_reference_isotherms": False,
            "has_reference_verticals": False,
            "label_mode_ru": False,
        },
    }


def diagram_snapshot_params(
    composition: dict[str, float],
    processing: ProcessingState,
    requested_system: str | None = None,
    inferred_system: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    system, fallback_confidence, is_fallback = resolve_diagram_system(
        requested_system=requested_system,
        composition=composition,
        inferred_system=inferred_system,
    )
    system_confidence = float(confidence if confidence is not None else fallback_confidence)
    x, y = _point_for_system(system=system, composition=composition, processing=processing)
    region = estimate_phase_region(system=system, x_value=x, y_value=y)
    return {
        "used_system": system,
        "confidence": system_confidence,
        "is_fallback": is_fallback,
        "current_point": {"x": float(x), "y": float(y)},
        "phase_region": region,
    }


def save_diagram_png(snapshot: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = snapshot.get("image")
    if not isinstance(image, Image.Image):
        raise ValueError("snapshot does not contain a PIL image")
    image.save(output)
    return output
