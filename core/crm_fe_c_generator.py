from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Grain:
    x: float
    y: float
    phase: str
    angle: float
    spacing: float


@dataclass(slots=True)
class RenderSettings:
    boundary_strength: int = 56
    distortion_level: float = 0.6


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_fractions(raw: dict[str, float]) -> dict[str, float]:
    clipped = {k: max(0.0, v) for k, v in raw.items()}
    total = sum(clipped.values())
    if total <= 1e-9:
        return {"ferrite": 1.0, "pearlite": 0.0, "cementite": 0.0, "ledeburite": 0.0, "graphite": 0.0}
    return {k: v / total for k, v in clipped.items()}


def phase_fractions_fe_c(carbon_pct: float, iron_type: str = "auto") -> dict[str, float]:
    """
    Educational room-temperature phase estimation for Fe-C.
    carbon_pct: 0..6.67 wt.%
    iron_type: auto | white_cast_iron | gray_cast_iron
    """

    c = clamp(carbon_pct, 0.0, 6.67)

    if c <= 2.14:
        if c <= 0.02:
            return normalize_fractions(
                {"ferrite": 0.98, "pearlite": 0.02, "cementite": 0.0, "ledeburite": 0.0, "graphite": 0.0}
            )
        if c <= 0.76:
            pearlite = c / 0.76
            return normalize_fractions(
                {
                    "ferrite": 1.0 - pearlite,
                    "pearlite": pearlite,
                    "cementite": 0.0,
                    "ledeburite": 0.0,
                    "graphite": 0.0,
                }
            )
        if c <= 0.80:
            pearlite = 0.95 + (c - 0.76) / 0.04 * 0.05
            return normalize_fractions(
                {
                    "ferrite": 0.0,
                    "pearlite": pearlite,
                    "cementite": 1.0 - pearlite,
                    "ledeburite": 0.0,
                    "graphite": 0.0,
                }
            )
        pearlite = clamp((6.67 - c) / (6.67 - 0.8), 0.0, 1.0)
        return normalize_fractions(
            {
                "ferrite": 0.0,
                "pearlite": pearlite,
                "cementite": 1.0 - pearlite,
                "ledeburite": 0.0,
                "graphite": 0.0,
            }
        )

    white = iron_type == "white_cast_iron"
    gray = iron_type == "gray_cast_iron"
    if iron_type == "auto":
        white = True
        gray = False

    if c <= 4.3:
        ledeburite = clamp((c - 2.14) / (4.3 - 2.14), 0.0, 1.0)
        pearlite = 1.0 - ledeburite
        cementite = 0.12 * ledeburite if white else 0.05 * ledeburite
        graphite = 0.0
        if gray:
            graphite = 0.20 * ledeburite + 0.05
            pearlite *= 1.0 - graphite
            ledeburite *= 1.0 - graphite
    else:
        primary = clamp((c - 4.3) / (6.67 - 4.3), 0.0, 1.0)
        ledeburite = 1.0 - 0.55 * primary
        pearlite = 0.20 * (1.0 - primary)
        cementite = 0.80 * primary if white else 0.35 * primary
        graphite = 0.0
        if gray:
            graphite = 0.35 * primary + 0.10
            cementite *= 1.0 - graphite
            ledeburite *= 1.0 - graphite
            pearlite *= 1.0 - graphite

    return normalize_fractions(
        {
            "ferrite": 0.0,
            "pearlite": pearlite,
            "cementite": cementite,
            "ledeburite": ledeburite,
            "graphite": graphite,
        }
    )


def _pick_phase(fractions: dict[str, float], p: float) -> str:
    cur = 0.0
    for phase in ("ferrite", "pearlite", "cementite", "ledeburite", "graphite"):
        cur += fractions.get(phase, 0.0)
        if p <= cur:
            return phase
    return "ferrite"


def _generate_grains(width: int, height: int, count: int, fractions: dict[str, float], rng: random.Random) -> list[Grain]:
    grains: list[Grain] = []
    for _ in range(max(2, count)):
        grains.append(
            Grain(
                x=rng.uniform(0, width),
                y=rng.uniform(0, height),
                phase=_pick_phase(fractions, rng.random()),
                angle=rng.uniform(0, math.pi),
                spacing=rng.uniform(3.0, 9.0),
            )
        )
    return grains


def _voronoi_nearest(x: int, y: int, grains: list[Grain]) -> tuple[Grain, float, float]:
    d1 = float("inf")
    d2 = float("inf")
    nearest = grains[0]
    for grain in grains:
        dx = x - grain.x
        dy = y - grain.y
        d = dx * dx + dy * dy
        if d < d1:
            d2 = d1
            d1 = d
            nearest = grain
        elif d < d2:
            d2 = d
    return nearest, d1, d2


def _phase_color(phase: str, grain: Grain, x: int, y: int, rng: random.Random) -> tuple[int, int, int]:
    if phase == "ferrite":
        base = 214 + rng.randint(-10, 10)
        return int(clamp(base + rng.randint(-3, 3), 0, 255)), int(clamp(base + rng.randint(-4, 4), 0, 255)), int(
            clamp(base, 0, 255)
        )

    if phase == "pearlite":
        proj = (x * math.cos(grain.angle) + y * math.sin(grain.angle)) / grain.spacing
        lamella = math.sin(2.0 * math.pi * proj)
        base = 88 if lamella > 0 else 146
        gray = int(clamp(base + rng.randint(-8, 8), 0, 255))
        return gray, gray, gray

    if phase == "cementite":
        proj = (x * math.cos(grain.angle * 1.2) - y * math.sin(grain.angle * 0.8)) / max(2.5, grain.spacing * 0.65)
        needle = abs(math.sin(3.0 * proj))
        base = int(175 + 60 * needle)
        c = int(clamp(base + rng.randint(-8, 8), 0, 255))
        return c, c, c

    if phase == "ledeburite":
        radial = math.sin(0.14 * math.hypot(x - grain.x, y - grain.y) + grain.angle * 4)
        swirl = math.sin(0.09 * x + 0.12 * y + grain.angle * 3.3)
        mix = 0.5 * radial + 0.5 * swirl
        base = 105 if mix > 0 else 170
        c = int(clamp(base + rng.randint(-14, 14), 0, 255))
        return c, c, c

    shade = int(clamp(38 + rng.randint(-8, 6), 0, 255))
    return shade, shade, shade


def _apply_microscope_distortions(
    data: bytearray,
    width: int,
    height: int,
    rng: random.Random,
    distortion_level: float,
) -> None:
    cx = width * 0.5
    cy = height * 0.5
    max_r = math.hypot(cx, cy)

    lx = rng.uniform(0.35, 0.65) * width
    ly = rng.uniform(0.35, 0.65) * height

    dust_count = int(45 * distortion_level)
    scratch_count = int(5 * distortion_level)
    dust = [(rng.uniform(0, width), rng.uniform(0, height), rng.uniform(1.0, 2.8)) for _ in range(dust_count)]
    scratches = [
        (
            rng.uniform(0, width),
            rng.uniform(0, height),
            rng.uniform(0, math.pi),
            rng.uniform(width * 0.2, width * 0.8),
            rng.uniform(0.8, 1.7),
        )
        for _ in range(scratch_count)
    ]

    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 3
            r = data[idx]
            g = data[idx + 1]
            b = data[idx + 2]

            rr = math.hypot(x - cx, y - cy) / max_r
            vignette = 1.0 - 0.22 * distortion_level * rr * rr
            light = 1.0 + 0.14 * distortion_level * math.cos(0.008 * (x - lx)) * math.cos(0.007 * (y - ly))
            factor = vignette * light
            r = int(clamp(r * factor, 0, 255))
            g = int(clamp(g * factor, 0, 255))
            b = int(clamp(b * factor, 0, 255))

            noise = rng.randint(-int(10 * distortion_level), int(10 * distortion_level))
            r = int(clamp(r + noise, 0, 255))
            g = int(clamp(g + noise, 0, 255))
            b = int(clamp(b + noise, 0, 255))

            for px, py, pr in dust:
                dx = x - px
                dy = y - py
                if dx * dx + dy * dy <= pr * pr:
                    r = int(clamp(r + 40, 0, 255))
                    g = int(clamp(g + 40, 0, 255))
                    b = int(clamp(b + 40, 0, 255))
                    break

            for sx, sy, ang, length, th in scratches:
                tx = x - sx
                ty = y - sy
                u = tx * math.cos(ang) + ty * math.sin(ang)
                v = -tx * math.sin(ang) + ty * math.cos(ang)
                if 0 <= u <= length and abs(v) <= th:
                    r = int(clamp(r - 42, 0, 255))
                    g = int(clamp(g - 42, 0, 255))
                    b = int(clamp(b - 42, 0, 255))
                    break

            data[idx] = r
            data[idx + 1] = g
            data[idx + 2] = b


def _render_microstructure(
    width: int,
    height: int,
    grains: list[Grain],
    fractions: dict[str, float],
    rng: random.Random,
    settings: RenderSettings,
) -> bytearray:
    data = bytearray(width * height * 3)
    graphite_count = int(120 * fractions.get("graphite", 0.0))
    graphite_flakes = [
        (rng.uniform(0, width), rng.uniform(0, height), rng.uniform(5.0, 15.0), rng.uniform(0, math.pi))
        for _ in range(graphite_count)
    ]

    for y in range(height):
        for x in range(width):
            nearest, d1, d2 = _voronoi_nearest(x, y, grains)
            r, g, b = _phase_color(nearest.phase, nearest, x, y, rng)

            delta = math.sqrt(max(0.0, d2)) - math.sqrt(max(0.0, d1))
            if delta < 2.2:
                shade = int((2.2 - delta) / 2.2 * settings.boundary_strength)
                r = max(0, r - shade)
                g = max(0, g - shade)
                b = max(0, b - shade)

            cementite_net = fractions.get("cementite", 0.0) + 0.6 * fractions.get("ledeburite", 0.0)
            net_w = 1.3 + 1.8 * cementite_net
            if cementite_net > 0.01 and delta < net_w:
                boost = int(120 * cementite_net * (1 - delta / max(net_w, 1e-6)))
                r = int(clamp(r + boost, 0, 255))
                g = int(clamp(g + boost, 0, 255))
                b = int(clamp(b + boost, 0, 255))

            if fractions.get("graphite", 0.0) > 0.001:
                for gx, gy, gl, ga in graphite_flakes:
                    tx = x - gx
                    ty = y - gy
                    u = tx * math.cos(ga) + ty * math.sin(ga)
                    v = -tx * math.sin(ga) + ty * math.cos(ga)
                    if abs(v) < 1.1 and -gl * 0.5 <= u <= gl * 0.5:
                        r, g, b = 22, 22, 22
                        break

            idx = (y * width + x) * 3
            data[idx] = r
            data[idx + 1] = g
            data[idx + 2] = b

    _apply_microscope_distortions(data, width, height, rng, settings.distortion_level)
    return data


def generate_crm_fe_c_rgb(
    width: int,
    height: int,
    carbon_pct: float,
    grains_count: int,
    seed: int,
    iron_type: str = "auto",
    distortion_level: float = 0.6,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Generate RGB microstructure image in style of crm-ai-main example.
    Returns image array (H, W, 3) and phase fractions.
    """

    rng = random.Random(int(seed))
    fractions = phase_fractions_fe_c(carbon_pct, iron_type)
    grains = _generate_grains(width, height, grains_count, fractions, rng)
    blob = _render_microstructure(
        width=width,
        height=height,
        grains=grains,
        fractions=fractions,
        rng=rng,
        settings=RenderSettings(distortion_level=clamp(distortion_level, 0.0, 1.0)),
    )
    arr = np.frombuffer(blob, dtype=np.uint8).reshape((height, width, 3)).copy()
    return arr, fractions


def format_fraction_summary(fractions: dict[str, float]) -> str:
    parts: list[str] = []
    for key in ("ferrite", "pearlite", "cementite", "ledeburite", "graphite"):
        value = fractions.get(key, 0.0)
        if value > 0.001:
            parts.append(f"{key}={value:.2f}")
    return ", ".join(parts)

