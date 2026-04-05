from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from .generator_grains import generate_grain_structure


def _smooth_random_field(rng: np.random.Generator, size: tuple[int, int], sigma: float) -> np.ndarray:
    field = rng.normal(0.0, 1.0, size=size).astype(np.float32)
    if ndimage is not None:
        field = ndimage.gaussian_filter(field, sigma=max(0.2, float(sigma)))
    return field


def _draw_irregular_particles(
    image: np.ndarray,
    rng: np.random.Generator,
    count: int,
    radius_range: tuple[float, float],
    tone_range: tuple[int, int],
) -> np.ndarray:
    height, width = image.shape
    canvas = Image.fromarray(image.astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(canvas)
    for _ in range(max(1, int(count))):
        cx = float(rng.uniform(0.0, max(1.0, width - 1.0)))
        cy = float(rng.uniform(0.0, max(1.0, height - 1.0)))
        base_r = float(rng.uniform(radius_range[0], radius_range[1]))
        tone = int(rng.integers(tone_range[0], tone_range[1] + 1))
        if rng.random() < 0.4:
            ax = base_r * float(rng.uniform(0.75, 1.8))
            ay = base_r * float(rng.uniform(0.65, 1.5))
            draw.ellipse((cx - ax, cy - ay, cx + ax, cy + ay), fill=tone)
            continue
        vertex_count = int(rng.integers(5, 9))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        points: list[tuple[float, float]] = []
        for idx in range(vertex_count):
            angle = phase + 2.0 * math.pi * float(idx) / float(vertex_count) + float(rng.normal(0.0, 0.18))
            rad = base_r * float(rng.uniform(0.55, 1.35))
            points.append((cx + rad * math.cos(angle), cy + rad * math.sin(angle)))
        draw.polygon(points, fill=tone)
    return np.asarray(canvas, dtype=np.uint8)


def generate_pearlite_structure(
    size: tuple[int, int],
    seed: int,
    pearlite_fraction: float = 0.75,
    lamella_period_px: float = 7.0,
    colony_size_px: float = 120.0,
    ferrite_brightness: int = 95,
    cementite_brightness: int = 175,
) -> dict[str, Any]:
    """Generate educational ferrite + pearlite structure."""

    rng = np.random.default_rng(seed)
    height, width = size
    colony = generate_grain_structure(
        size=size,
        seed=seed + 101,
        mean_grain_size_px=max(32.0, colony_size_px),
        grain_size_jitter=0.2,
        equiaxed=1.0,
        elongation=1.0,
        orientation_deg=0.0,
        boundary_width_px=1,
        boundary_contrast=0.0,
    )
    labels = colony["labels"]
    colony_count = int(labels.max()) + 1
    theta = rng.uniform(0.0, math.pi, size=colony_count)
    is_pearlite = rng.random(colony_count) < float(np.clip(pearlite_fraction, 0.0, 1.0))

    yy, xx = np.mgrid[0:height, 0:width]
    theta_field = theta[labels]
    projection = xx * np.cos(theta_field) + yy * np.sin(theta_field)

    period = max(2.0, float(lamella_period_px))
    lamella = np.sin((2.0 * math.pi / period) * projection)
    cementite_mask = (lamella > 0) & is_pearlite[labels]
    pearlite_mask = is_pearlite[labels]

    ferrite = np.full((height, width), int(ferrite_brightness), dtype=np.uint8)
    image = ferrite.copy()
    image[pearlite_mask] = int((ferrite_brightness + cementite_brightness) * 0.5)
    image[cementite_mask] = int(cementite_brightness)

    boundaries = np.zeros_like(labels, dtype=bool)
    boundaries[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundaries[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    image[boundaries] = np.clip(image[boundaries].astype(np.int16) - 30, 0, 255).astype(np.uint8)

    if ndimage is not None:
        image = ndimage.gaussian_filter(image.astype(np.float32), sigma=0.6).clip(0, 255).astype(np.uint8)

    return {
        "image": image,
        "labels": labels,
        "metadata": {
            "colony_count": colony_count,
            "pearlite_fraction": float(pearlite_mask.mean()),
            "lamella_period_px": float(period),
        },
    }


def generate_martensite_structure(
    size: tuple[int, int],
    seed: int,
    needle_count: int = 3800,
    needle_length_px: tuple[int, int] = (10, 90),
    packet_spread_deg: float = 14.0,
) -> dict[str, Any]:
    """Generate stylized lath/needle martensite."""

    rng = np.random.default_rng(seed)
    height, width = size
    canvas = Image.new("L", (width, height), color=132)
    draw = ImageDraw.Draw(canvas)

    for _ in range(max(100, int(needle_count))):
        cx = float(rng.uniform(0.0, max(1.0, width - 1.0)))
        cy = float(rng.uniform(0.0, max(1.0, height - 1.0)))
        angle = float(rng.uniform(0.0, math.pi))
        length = float(rng.uniform(needle_length_px[0], needle_length_px[1]))
        dx = math.cos(angle) * length * 0.5
        dy = math.sin(angle) * length * 0.5
        p0 = (cx - dx, cy - dy)
        p1 = (cx + dx, cy + dy)
        tone = int(rng.integers(42, 208))
        width_px = 1 if rng.random() < 0.68 else 2
        draw.line((p0, p1), fill=tone, width=width_px)

    image = np.asarray(canvas, dtype=np.uint8)
    if ndimage is not None:
        image = ndimage.gaussian_filter(image.astype(np.float32), sigma=0.72).clip(0, 255).astype(np.uint8)

    return {
        "image": image,
        "metadata": {
            "needle_count": int(needle_count),
            "packet_spread_deg": float(packet_spread_deg),
        },
    }


def generate_tempered_steel_structure(
    size: tuple[int, int],
    seed: int,
    temper_temperature_c: int,
) -> dict[str, Any]:
    """Generate educational tempered-steel textures for low/mid/high temper levels."""

    temp = int(temper_temperature_c)
    if temp <= 250:
        base = generate_martensite_structure(
            size=size,
            seed=seed,
            needle_count=3200,
            needle_length_px=(8, 70),
            packet_spread_deg=16.0,
        )
        image = base["image"].astype(np.float32)
        image = image * 0.92 + 12.0
        style = "low_temper_troostite"
    elif temp <= 450:
        sorbite = generate_sorbite_structure(
            size=size,
            seed=seed + 77,
            mode="temper",
        )["image"].astype(np.float32)
        mart = generate_martensite_structure(
            size=size,
            seed=seed,
            needle_count=1400,
            needle_length_px=(6, 38),
            packet_spread_deg=23.0,
        )["image"].astype(np.float32)
        mart_share = float(np.clip((450.0 - float(temp)) / 200.0, 0.08, 0.24))
        image = sorbite * (1.0 - mart_share) + mart * mart_share
        style = "medium_temper_sorbite"
    else:
        coarse = generate_sorbite_structure(
            size=size,
            seed=seed + 15,
            mode="temper",
        )["image"].astype(np.float32)
        grains = generate_grain_structure(
            size=size,
            seed=seed + 8,
            mean_grain_size_px=108,
            grain_size_jitter=0.18,
            equiaxed=1.05,
            boundary_width_px=2,
            boundary_contrast=0.26,
        )["image"].astype(np.float32)
        image = coarse * 0.76 + grains * 0.24
        style = "high_temper_coarse_sorbite"

    if ndimage is not None:
        image = ndimage.gaussian_filter(image, sigma=0.7)
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    return {"image": image_u8, "metadata": {"temper_temperature_c": temp, "style": style}}


def generate_troostite_structure(
    size: tuple[int, int],
    seed: int,
    mode: str = "quench",
) -> dict[str, Any]:
    """Generate educational troostite-like morphology."""
    mode_l = str(mode).strip().lower()
    mart = generate_martensite_structure(
        size=size,
        seed=seed + 13,
        needle_count=(2600 if mode_l != "temper" else 1900),
        needle_length_px=((6, 46) if mode_l != "temper" else (5, 34)),
        packet_spread_deg=(21.0 if mode_l != "temper" else 24.0),
    )["image"].astype(np.float32)
    fine = generate_pearlite_structure(
        size=size,
        seed=seed + 17,
        pearlite_fraction=(0.62 if mode_l != "temper" else 0.66),
        lamella_period_px=(3.4 if mode_l != "temper" else 2.9),
        colony_size_px=(60.0 if mode_l != "temper" else 54.0),
        ferrite_brightness=(172 if mode_l != "temper" else 170),
        cementite_brightness=(68 if mode_l != "temper" else 74),
    )["image"].astype(np.float32)
    mix = fine * (0.78 if mode_l == "temper" else 0.62) + mart * (0.22 if mode_l == "temper" else 0.38)
    if ndimage is not None:
        mix = ndimage.gaussian_filter(mix, sigma=(0.64 if mode_l == "temper" else 0.7))
    return {"image": np.clip(mix, 0, 255).astype(np.uint8), "metadata": {"mode": str(mode), "style": "troostite"}}


def generate_sorbite_structure(
    size: tuple[int, int],
    seed: int,
    mode: str = "temper",
    include_particles: bool = True,
) -> dict[str, Any]:
    """Generate sorbite-like morphology: dispersed carbides in ferritic matrix."""
    mode_l = str(mode).strip().lower()
    rng = np.random.default_rng(int(seed) + 9011)
    height, width = size

    matrix = generate_grain_structure(
        size=size,
        seed=seed + 29,
        mean_grain_size_px=(58.0 if mode_l == "temper" else 56.0),
        grain_size_jitter=0.18,
        boundary_width_px=1,
        boundary_contrast=(0.16 if mode_l == "temper" else 0.16),
    )
    matrix_img = matrix["image"].astype(np.float32)
    boundaries = matrix.get("boundaries")

    base_level = 72.0 if mode_l == "temper" else 72.0
    mix = matrix_img * (0.42 if mode_l == "temper" else 0.42) + base_level
    low_field = _smooth_random_field(rng, size=size, sigma=12.0)
    mid_field = _smooth_random_field(rng, size=size, sigma=4.0)
    mix += low_field * 10.0
    mix += mid_field * 6.0
    if isinstance(boundaries, np.ndarray):
        mix[boundaries] -= 8.0

    fine_field = _smooth_random_field(rng, size=size, sigma=0.95)
    fine_fraction = 0.08
    fine_thr = float(np.quantile(fine_field, 1.0 - fine_fraction))
    fine_mask = fine_field >= fine_thr
    if ndimage is not None:
        fine_mask = ndimage.binary_opening(fine_mask, iterations=1)
    bright_tone = 174.0
    mix[fine_mask] = mix[fine_mask] * 0.45 + bright_tone * 0.55

    dark_field = _smooth_random_field(rng, size=size, sigma=2.4)
    dark_thr = float(np.quantile(dark_field, 0.02))
    dark_mask = dark_field <= dark_thr
    mix[dark_mask] -= 16.0

    image_u8 = np.clip(mix, 0.0, 255.0).astype(np.uint8)
    particle_count = 0
    if bool(include_particles):
        particle_count = max(8, (height * width) // (42000 if mode_l == "temper" else 42000))
        image_u8 = _draw_irregular_particles(
            image=image_u8,
            rng=rng,
            count=particle_count,
            radius_range=((1.8, 6.0) if mode_l == "temper" else (1.8, 6.0)),
            tone_range=((156, 216) if mode_l == "temper" else (156, 216)),
        )

    if mode_l == "quench":
        laths = generate_martensite_structure(
            size=size,
            seed=seed + 204,
            needle_count=1200,
            needle_length_px=(6, 34),
            packet_spread_deg=24.0,
        )["image"].astype(np.float32)
        out = image_u8.astype(np.float32) * 0.84 + laths * 0.16
    else:
        out = image_u8.astype(np.float32)

    if ndimage is not None:
        out = ndimage.gaussian_filter(out, sigma=(0.6 if mode_l == "temper" else 0.6))
        detail = out - ndimage.gaussian_filter(out, sigma=1.35)
        out += detail * (0.2 if mode_l == "temper" else 0.2)

    return {
        "image": np.clip(out, 0, 255).astype(np.uint8),
        "metadata": {
            "mode": str(mode),
            "style": ("sorbite" if mode_l == "temper" else "sorbite_quench_transitional"),
            "particle_count": int(particle_count),
            "fine_fraction": float(fine_fraction),
        },
    }
