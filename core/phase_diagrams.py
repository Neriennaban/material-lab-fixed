from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageDraw


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _axes(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    draw.line((x0, y0, x0, y1), fill=color, width=2)
    draw.line((x0, y1, x1, y1), fill=color, width=2)


def _map(value: float, src0: float, src1: float, dst0: float, dst1: float) -> float:
    if abs(src1 - src0) < 1e-9:
        return dst0
    t = (value - src0) / (src1 - src0)
    return dst0 + t * (dst1 - dst0)


def render_fe_c_phase_diagram(
    width: int,
    height: int,
    carbon_pct: float,
    temperature_c: float,
    fractions: dict[str, float] | None = None,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (18, 23, 29))
    draw = ImageDraw.Draw(img)

    left = 72
    right = width - 30
    top = 26
    bottom = height - 48

    axis = (220, 224, 228)
    grid = (68, 80, 95)
    line = (244, 185, 66)
    marker = (255, 84, 84)
    text = (232, 236, 241)

    _axes(draw, left, top, right, bottom, axis)

    for c_tick in [0, 0.8, 2.14, 4.3, 6.67]:
        x = int(_map(c_tick, 0.0, 6.67, left, right))
        draw.line((x, top, x, bottom), fill=grid, width=1)
        draw.text((x - 14, bottom + 8), f"{c_tick:g}", fill=text)

    for t_tick in [20, 200, 400, 600, 727, 900, 1147, 1400, 1600]:
        y = int(_map(t_tick, 20, 1600, bottom, top))
        draw.line((left, y, right, y), fill=grid, width=1)
        draw.text((12, y - 7), f"{t_tick:g}", fill=text)

    def xy(c: float, t: float) -> tuple[int, int]:
        return int(_map(c, 0.0, 6.67, left, right)), int(_map(t, 20, 1600, bottom, top))

    # Key boundaries of Fe-C (educational approximation).
    draw.line([xy(0.0, 1538), xy(4.3, 1147)], fill=line, width=2)   # liquidus (left)
    draw.line([xy(4.3, 1147), xy(6.67, 1250)], fill=line, width=2)  # liquidus (right)
    draw.line([xy(0.0, 1493), xy(2.14, 1147)], fill=(122, 197, 255), width=2)  # solidus
    draw.line([xy(0.76, 727), xy(2.14, 1147)], fill=(122, 197, 255), width=2)   # Acm

    # A3 curve.
    points = []
    for ci in [i * 0.04 for i in range(0, 20)]:
        a3 = 912 - 203 * math.sqrt(max(ci, 0.0))
        points.append(xy(ci, a3))
    draw.line(points, fill=(122, 197, 255), width=2)

    # Eutectoid/eutectic isotherms.
    draw.line([xy(0.0, 727), xy(6.67, 727)], fill=(173, 214, 149), width=2)
    draw.line([xy(2.14, 1147), xy(6.67, 1147)], fill=(173, 214, 149), width=2)

    draw.text((left + 8, top + 6), "Fe-C phase diagram", fill=text)
    draw.text((left + 8, top + 24), "L", fill=text)
    draw.text((left + 80, top + 84), "L + gamma", fill=text)
    draw.text((left + 28, top + 152), "gamma", fill=text)
    draw.text((left + 12, bottom - 78), "alpha + pearlite", fill=text)
    draw.text((left + 250, bottom - 78), "pearlite + Fe3C", fill=text)
    draw.text((left + 315, top + 156), "ledeburite", fill=text)

    cx = _clamp(carbon_pct, 0.0, 6.67)
    tc = _clamp(temperature_c, 20.0, 1600.0)
    px, py = xy(cx, tc)
    draw.line((px, bottom, px, top), fill=(255, 84, 84), width=1)
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=marker, width=2, fill=(255, 190, 190))

    draw.text((right - 150, top + 6), f"C={cx:.3f} wt.%", fill=text)
    draw.text((right - 150, top + 24), f"T={tc:.1f} C", fill=text)

    if fractions:
        ytxt = top + 52
        for key in ("ferrite", "pearlite", "cementite", "ledeburite", "graphite"):
            value = fractions.get(key, 0.0)
            if value > 0.001:
                draw.text((right - 150, ytxt), f"{key}: {value:.2f}", fill=text)
                ytxt += 16

    draw.text((left + 8, bottom + 24), "Carbon, wt.%", fill=text)
    draw.text((8, top + 4), "Temperature, C", fill=text)
    return img


def render_al_si_phase_diagram(width: int, height: int, si_pct: float, temperature_c: float) -> Image.Image:
    img = Image.new("RGB", (width, height), (18, 23, 29))
    draw = ImageDraw.Draw(img)
    left, right, top, bottom = 72, width - 30, 26, height - 48
    axis = (220, 224, 228)
    grid = (68, 80, 95)
    line = (244, 185, 66)
    text = (232, 236, 241)
    marker = (255, 84, 84)

    _axes(draw, left, top, right, bottom, axis)
    for xval in [0, 5, 12.6, 20, 25]:
        x = int(_map(xval, 0.0, 25.0, left, right))
        draw.line((x, top, x, bottom), fill=grid, width=1)
        draw.text((x - 10, bottom + 8), f"{xval:g}", fill=text)
    for tval in [20, 200, 400, 577, 660, 700]:
        y = int(_map(tval, 20, 700, bottom, top))
        draw.line((left, y, right, y), fill=grid, width=1)
        draw.text((12, y - 7), f"{tval:g}", fill=text)

    def xy(s: float, t: float) -> tuple[int, int]:
        return int(_map(s, 0.0, 25.0, left, right)), int(_map(t, 20, 700, bottom, top))

    draw.line([xy(0, 660), xy(12.6, 577)], fill=line, width=2)
    draw.line([xy(25, 700), xy(12.6, 577)], fill=line, width=2)
    draw.line([xy(0, 577), xy(25, 577)], fill=(173, 214, 149), width=2)
    draw.text((left + 8, top + 6), "Al-Si phase diagram", fill=text)
    draw.text((left + 28, top + 96), "L + alpha", fill=text)
    draw.text((left + 240, top + 88), "L + Si", fill=text)
    draw.text((left + 110, bottom - 72), "alpha + eutectic", fill=text)
    draw.text((left + 350, bottom - 72), "Si + eutectic", fill=text)

    px, py = xy(_clamp(si_pct, 0.0, 25.0), _clamp(temperature_c, 20.0, 700.0))
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=marker, fill=(255, 190, 190), width=2)
    draw.text((right - 130, top + 8), f"Si={si_pct:.2f}%", fill=text)
    draw.text((right - 130, top + 26), f"T={temperature_c:.1f} C", fill=text)
    draw.text((left + 8, bottom + 24), "Si, wt.%", fill=text)
    draw.text((8, top + 4), "Temperature, C", fill=text)
    return img


def render_cu_zn_phase_diagram(width: int, height: int, zn_pct: float, temperature_c: float) -> Image.Image:
    img = Image.new("RGB", (width, height), (18, 23, 29))
    draw = ImageDraw.Draw(img)
    left, right, top, bottom = 72, width - 30, 26, height - 48
    axis = (220, 224, 228)
    grid = (68, 80, 95)
    text = (232, 236, 241)
    marker = (255, 84, 84)
    _axes(draw, left, top, right, bottom, axis)

    for xval in [0, 20, 35, 46, 50, 60]:
        x = int(_map(xval, 0.0, 60.0, left, right))
        draw.line((x, top, x, bottom), fill=grid, width=1)
        draw.text((x - 10, bottom + 8), f"{xval:g}", fill=text)
    for tval in [20, 200, 400, 600, 800, 920, 1000]:
        y = int(_map(tval, 20, 1000, bottom, top))
        draw.line((left, y, right, y), fill=grid, width=1)
        draw.text((12, y - 7), f"{tval:g}", fill=text)

    def x(z: float) -> int:
        return int(_map(z, 0.0, 60.0, left, right))

    draw.rectangle((x(0), top + 18, x(35), bottom - 2), outline=(122, 197, 255), width=2)
    draw.rectangle((x(35), top + 18, x(46), bottom - 2), outline=(244, 185, 66), width=2)
    draw.rectangle((x(46), top + 18, x(60), bottom - 2), outline=(173, 214, 149), width=2)
    draw.text((x(10), top + 24), "alpha", fill=text)
    draw.text((x(37), top + 24), "alpha+beta", fill=text)
    draw.text((x(50), top + 24), "beta", fill=text)
    draw.text((left + 8, top + 6), "Cu-Zn phase map", fill=text)

    px = x(_clamp(zn_pct, 0.0, 60.0))
    py = int(_map(_clamp(temperature_c, 20.0, 1000.0), 20, 1000, bottom, top))
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=marker, fill=(255, 190, 190), width=2)
    draw.text((right - 140, top + 8), f"Zn={zn_pct:.2f}%", fill=text)
    draw.text((right - 140, top + 26), f"T={temperature_c:.1f} C", fill=text)
    draw.text((left + 8, bottom + 24), "Zn, wt.%", fill=text)
    draw.text((8, top + 4), "Temperature, C", fill=text)
    return img


def render_fe_si_phase_diagram(width: int, height: int, si_pct: float, temperature_c: float) -> Image.Image:
    img = Image.new("RGB", (width, height), (18, 23, 29))
    draw = ImageDraw.Draw(img)
    left, right, top, bottom = 72, width - 30, 26, height - 48
    axis = (220, 224, 228)
    grid = (68, 80, 95)
    text = (232, 236, 241)
    marker = (255, 84, 84)
    _axes(draw, left, top, right, bottom, axis)

    for xval in [0, 1, 2, 3, 4, 6]:
        x = int(_map(xval, 0.0, 6.0, left, right))
        draw.line((x, top, x, bottom), fill=grid, width=1)
        draw.text((x - 8, bottom + 8), f"{xval:g}", fill=text)
    for tval in [20, 200, 400, 700, 900, 1200, 1500]:
        y = int(_map(tval, 20, 1500, bottom, top))
        draw.line((left, y, right, y), fill=grid, width=1)
        draw.text((12, y - 7), f"{tval:g}", fill=text)

    draw.line((left, int(_map(900, 20, 1500, bottom, top)), right, int(_map(900, 20, 1500, bottom, top))), fill=(244, 185, 66), width=2)
    draw.text((left + 8, top + 6), "Fe-Si process-state map", fill=text)
    draw.text((left + 40, top + 54), "hot ferrite", fill=text)
    draw.text((left + 40, bottom - 110), "recrystallized ferrite", fill=text)
    draw.text((left + 320, bottom - 110), "cold worked ferrite", fill=text)

    px = int(_map(_clamp(si_pct, 0.0, 6.0), 0.0, 6.0, left, right))
    py = int(_map(_clamp(temperature_c, 20.0, 1500.0), 20, 1500, bottom, top))
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=marker, fill=(255, 190, 190), width=2)
    draw.text((right - 140, top + 8), f"Si={si_pct:.2f}%", fill=text)
    draw.text((right - 140, top + 26), f"T={temperature_c:.1f} C", fill=text)
    draw.text((left + 8, bottom + 24), "Si, wt.%", fill=text)
    draw.text((8, top + 4), "Temperature, C", fill=text)
    return img


def render_al_cu_mg_aging_diagram(width: int, height: int, temperature_c: float, aging_hours: float) -> Image.Image:
    img = Image.new("RGB", (width, height), (18, 23, 29))
    draw = ImageDraw.Draw(img)
    left, right, top, bottom = 72, width - 30, 26, height - 48
    axis = (220, 224, 228)
    grid = (68, 80, 95)
    text = (232, 236, 241)
    marker = (255, 84, 84)
    _axes(draw, left, top, right, bottom, axis)

    for xval in [0, 1, 2, 4, 8, 16, 24]:
        x = int(_map(xval, 0.0, 24.0, left, right))
        draw.line((x, top, x, bottom), fill=grid, width=1)
        draw.text((x - 10, bottom + 8), f"{xval:g}", fill=text)
    for tval in [20, 100, 150, 180, 220, 300, 500]:
        y = int(_map(tval, 20, 500, bottom, top))
        draw.line((left, y, right, y), fill=grid, width=1)
        draw.text((12, y - 7), f"{tval:g}", fill=text)

    draw.text((left + 8, top + 6), "Al-Cu-Mg aging map (T-time)", fill=text)
    draw.rectangle((left + 18, top + 42, right - 18, top + 82), outline=(122, 197, 255), width=2)
    draw.text((left + 24, top + 52), "solutionized / quenched", fill=text)
    draw.rectangle((left + 18, top + 130, right - 18, top + 170), outline=(173, 214, 149), width=2)
    draw.text((left + 24, top + 140), "artificial aging window", fill=text)
    draw.rectangle((left + 18, top + 222, right - 18, top + 262), outline=(244, 185, 66), width=2)
    draw.text((left + 24, top + 232), "overaging region", fill=text)

    px = int(_map(_clamp(aging_hours, 0.0, 24.0), 0.0, 24.0, left, right))
    py = int(_map(_clamp(temperature_c, 20.0, 500.0), 20, 500, bottom, top))
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=marker, fill=(255, 190, 190), width=2)
    draw.text((right - 150, top + 8), f"t={aging_hours:.1f} h", fill=text)
    draw.text((right - 150, top + 26), f"T={temperature_c:.1f} C", fill=text)
    draw.text((left + 8, bottom + 24), "Aging time, h", fill=text)
    draw.text((8, top + 4), "Temperature, C", fill=text)
    return img


def render_detailed_diagram(
    system: str,
    composition: dict[str, float],
    temperature_c: float,
    aging_hours: float = 8.0,
    fractions: dict[str, float] | None = None,
    size: tuple[int, int] = (780, 420),
) -> Image.Image:
    width, height = size
    key = system.strip().lower()
    if key == "fe-c":
        return render_fe_c_phase_diagram(
            width=width,
            height=height,
            carbon_pct=float(composition.get("C", 0.0)),
            temperature_c=temperature_c,
            fractions=fractions,
        )
    if key == "al-si":
        return render_al_si_phase_diagram(
            width=width,
            height=height,
            si_pct=float(composition.get("Si", 0.0)),
            temperature_c=temperature_c,
        )
    if key == "cu-zn":
        return render_cu_zn_phase_diagram(
            width=width,
            height=height,
            zn_pct=float(composition.get("Zn", 0.0)),
            temperature_c=temperature_c,
        )
    if key == "fe-si":
        return render_fe_si_phase_diagram(
            width=width,
            height=height,
            si_pct=float(composition.get("Si", 0.0)),
            temperature_c=temperature_c,
        )
    return render_al_cu_mg_aging_diagram(
        width=width,
        height=height,
        temperature_c=temperature_c,
        aging_hours=aging_hours,
    )


def infer_system_from_context(
    material: str,
    generator: str,
    composition: dict[str, Any],
) -> str:
    comp = {k: float(v) for k, v in composition.items()} if composition else {}
    name = material.lower()
    gen = generator.lower()

    if "fe-c" in name or ("steel" in name and comp.get("C", 0.0) > 0):
        return "fe-c"
    if "al-si" in name or (comp.get("Al", 0.0) > 0 and comp.get("Si", 0.0) > 0):
        return "al-si"
    if "brass" in name or (comp.get("Cu", 0.0) > 0 and comp.get("Zn", 0.0) > 0):
        return "cu-zn"
    if "fe-si" in name or (comp.get("Fe", 0.0) > 0 and comp.get("Si", 0.0) > 0 and comp.get("C", 0.0) <= 0.05):
        return "fe-si"
    if "dural" in name or "aged_al" in gen or ("al" in name and comp.get("Cu", 0.0) > 0):
        return "al-cu-mg"
    return "fe-c"

