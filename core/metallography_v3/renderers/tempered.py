"""Отпуск мартенсита (§2.11–2.13 справочника).

Обслуживает:
  * tempered_low (§2.11, 150–250°C, ε-карбиды внутри реек)
  * tempered_medium / troostite_temper (§2.12, 350–500°C)
  * tempered_high / sorbite_temper (§2.13, 500–650°C, Q+T "улучшение")
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.generator_grains import generate_grain_structure
from core.metallography_v3.realism_utils import (
    boundary_mask_from_labels,
    multiscale_noise,
    normalize01,
    rescale_to_u8,
)
from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.renderers import martensite as _r_martensite
from core.metallography_v3.system_generators.base import (
    SystemGenerationContext,
    soft_unsharp,
)

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


HANDLES_STAGES: frozenset[str] = frozenset(
    {
        "tempered_low",
        "tempered_medium",
        "tempered_high",
        "troostite_temper",
        "sorbite_temper",
    }
)


# --- tones (§2.11, §2.12, §2.13) ———————————————————————————————
_TONE_LOW_MATRIX_MULT = 0.72  # §2.11: общее затемнение 0.70-0.75x
_TONE_LOW_CARBIDE = 28.0
_TONE_MEDIUM_MATRIX = 70.0
_TONE_MEDIUM_INTERIOR = 95.0
_TONE_MEDIUM_CARBIDE = 30.0
_TONE_HIGH_MATRIX = 158.0
_TONE_HIGH_BOUNDARY = 78.0
_TONE_HIGH_CARBIDE = 30.0
_TONE_HIGH_HALO = 175.0


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    # Aliases → primary:
    if stage == "troostite_temper":
        return _render_medium(context=context, seed_split=seed_split)
    if stage == "sorbite_temper":
        return _render_high(context=context, seed_split=seed_split)

    if stage == "tempered_low":
        return _render_low(context=context, seed_split=seed_split)
    if stage == "tempered_medium":
        return _render_medium(context=context, seed_split=seed_split)
    if stage == "tempered_high":
        return _render_high(context=context, seed_split=seed_split)
    raise ValueError(f"tempered renderer has no branch for stage {stage!r}")


# --- tempered_low (§2.11) ———————————————————————————————————————


def _render_low(
    *, context: SystemGenerationContext, seed_split: dict[str, int]
) -> RendererOutput:
    # Реечная база от Phase 4 martensite renderer.
    base = _r_martensite._render_lath(
        context=context, seed_split=seed_split, base_grain_size_px=55.0
    )
    img = base.image_gray.astype(np.float32)

    # Общее затемнение + тёплый сдвиг (§2.11). Применяем в float,
    # итоговый rescale вернёт в [0,255].
    img *= _TONE_LOW_MATRIX_MULT
    img += 1.5  # лёгкий warm сдвиг; RGB не дифференцируется в
               # grayscale, сохраняем скалярный.

    # Poisson ε-карбиды (tone ≤ _TONE_LOW_CARBIDE).
    h, w = img.shape
    rng = np.random.default_rng(
        int(seed_split.get("seed_particles", context.seed)) + 101
    )
    n_carbides = max(40, int(0.005 * h * w))
    for _ in range(n_carbides):
        cy = int(rng.integers(0, h))
        cx = int(rng.integers(0, w))
        # Короткий штрих 2-3 px под псевдо-60°.
        theta = float(rng.uniform(0.0, math.pi))
        length = int(rng.integers(2, 4))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        for t in range(-length, length + 1):
            y = int(cy + t * sin_t)
            x = int(cx + t * cos_t)
            if 0 <= y < h and 0 <= x < w:
                img[y, x] = min(img[y, x], _TONE_LOW_CARBIDE)

    image_gray = rescale_to_u8(np.clip(img, 0.0, 255.0), lo=10.0, hi=235.0)
    image_gray = soft_unsharp(image_gray, amount=0.22)

    martensite_mask = np.ones(img.shape, dtype=np.uint8)
    cementite_mask = (image_gray <= 55).astype(np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "MARTENSITE": martensite_mask,
            "CEMENTITE": cementite_mask,
        },
        morphology_trace={
            "family": "tempered_low",
            "stage": "tempered_low",
            "brightness_multiplier": _TONE_LOW_MATRIX_MULT,
            "epsilon_carbide_count": n_carbides,
            "lath_geometry_preserved": True,
        },
        rendered_layers=["MARTENSITE", "CEMENTITE"],
        fragment_area=base.fragment_area,
    )


# --- tempered_medium / troostite_temper (§2.12) ——————————————————


def _render_medium(
    *, context: SystemGenerationContext, seed_split: dict[str, int]
) -> RendererOutput:
    # Реечная база → размытие (имитация рекристаллизации).
    base = _r_martensite._render_lath(
        context=context, seed_split=seed_split, base_grain_size_px=60.0
    )
    img = base.image_gray.astype(np.float32)
    if ndimage is not None:
        img = ndimage.gaussian_filter(img, sigma=1.0)

    # Overlay — базовый тон ~70 (§2.12).
    img = 0.55 * img + 0.45 * _TONE_MEDIUM_MATRIX

    # Высокочастотный Perlin (velvet).
    size = img.shape
    seed = int(seed_split.get("seed_particles", context.seed))
    hf_noise = (
        multiscale_noise(size=size, seed=seed + 37, scales=((3.0, 0.5), (1.2, 0.5)))
        - 0.5
    )
    img += hf_noise * 15.0

    # Poisson-карбиды 50-200 нм (1-2 px) с биасом на бывшие границы реек.
    h, w = size
    rng = np.random.default_rng(seed + 59)
    labels = generate_grain_structure(
        size=size,
        seed=seed + 61,
        mean_grain_size_px=55.0,
        grain_size_jitter=0.22,
        boundary_width_px=1,
        boundary_contrast=0.0,
        elongation=1.0,
    )["labels"]
    boundaries = boundary_mask_from_labels(labels, width=2)
    ys_b, xs_b = np.where(boundaries)
    n_carbides = max(200, int(0.018 * h * w))
    cementite_mask = np.zeros(size, dtype=bool)
    for _ in range(n_carbides):
        # 60% на границах, 40% внутри.
        if rng.random() < 0.60 and len(xs_b) > 0:
            idx = int(rng.integers(0, len(xs_b)))
            cy = int(ys_b[idx])
            cx = int(xs_b[idx])
        else:
            cy = int(rng.integers(0, h))
            cx = int(rng.integers(0, w))
        r = int(rng.integers(1, 3))
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        cementite_mask[y0:y1, x0:x1] |= disk
    img[cementite_mask] = _TONE_MEDIUM_CARBIDE

    image_gray = rescale_to_u8(np.clip(img, 0.0, 255.0), lo=15.0, hi=220.0)
    image_gray = soft_unsharp(image_gray, amount=0.18)

    troostite_mask = np.ones(size, dtype=np.uint8)
    troostite_mask[cementite_mask] = 0
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "TROOSTITE": troostite_mask,
            "CEMENTITE": cementite_mask.astype(np.uint8),
        },
        morphology_trace={
            "family": "tempered_medium",
            "stage": "tempered_medium",
            "blur_sigma_px": 1.0,
            "high_freq_perlin_amplitude_rgb": 15.0,
            "carbide_on_former_lath_boundary_probability": 0.60,
            "n_carbides": n_carbides,
        },
        rendered_layers=["TROOSTITE", "CEMENTITE"],
        fragment_area=int(h * w // 200),
    )


# --- tempered_high / sorbite_temper (§2.13, Q+T) —————————————————


def _render_high(
    *, context: SystemGenerationContext, seed_split: dict[str, int]
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = int(seed_split.get("seed_topology", context.seed))

    # Полигональный феррит 3-6 px (2-10 μm по §2.13).
    labels = generate_grain_structure(
        size=size,
        seed=seed,
        mean_grain_size_px=4.5,
        grain_size_jitter=0.25,
        boundary_width_px=1,
        boundary_contrast=0.0,
        elongation=1.0,
    )["labels"]
    boundaries = boundary_mask_from_labels(labels, width=1)

    # Матрица: светлый полигональный феррит.
    rng = np.random.default_rng(seed + 67)
    n_grains = int(labels.max()) + 1
    per_grain_jitter = rng.normal(0.0, 5.0, size=n_grains).astype(np.float32)
    img = (_TONE_HIGH_MATRIX + per_grain_jitter[labels]).astype(np.float32)
    img += (
        multiscale_noise(size=size, seed=seed + 71, scales=((8.0, 0.6), (2.5, 0.4)))
        - 0.5
    ) * 6.0

    # Границы зёрен.
    img = np.where(boundaries > 0, _TONE_HIGH_BOUNDARY, img)

    # Poisson-карбиды Fe₃C: 70% на границах, 30% внутри. Радиус
    # lognormal(~0.4 μm) ≈ lognormal(1.5 px).
    ys_b, xs_b = np.where(boundaries)
    n_carbides = max(150, int(0.012 * h * w))
    cementite_mask = np.zeros(size, dtype=bool)
    halo_mask = np.zeros(size, dtype=bool)
    for _ in range(n_carbides):
        if rng.random() < 0.70 and len(xs_b) > 0:
            idx = int(rng.integers(0, len(xs_b)))
            cy = int(ys_b[idx])
            cx = int(xs_b[idx])
        else:
            cy = int(rng.integers(0, h))
            cx = int(rng.integers(0, w))
        r = float(rng.lognormal(mean=math.log(1.3), sigma=0.35))
        r = max(0.9, min(3.0, r))
        r_int = int(math.ceil(r))
        y0 = max(0, cy - r_int)
        y1 = min(h, cy + r_int + 1)
        x0 = max(0, cx - r_int)
        x1 = min(w, cx + r_int + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        disk = dist2 <= r ** 2
        halo = (dist2 > r ** 2) & (dist2 <= (r + 0.8) ** 2)
        cementite_mask[y0:y1, x0:x1] |= disk
        halo_mask[y0:y1, x0:x1] |= halo

    img[halo_mask & ~cementite_mask] = _TONE_HIGH_HALO
    img[cementite_mask] = _TONE_HIGH_CARBIDE

    image_gray = rescale_to_u8(np.clip(img, 0.0, 255.0), lo=20.0, hi=230.0)
    image_gray = soft_unsharp(image_gray, amount=0.22)

    ferrite_mask = np.ones(size, dtype=np.uint8)
    ferrite_mask[cementite_mask] = 0
    ferrite_mask[boundaries > 0] = 0
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "FERRITE": ferrite_mask,
            "CEMENTITE": cementite_mask.astype(np.uint8),
            "SORBITE": ferrite_mask,
        },
        morphology_trace={
            "family": "tempered_high",
            "stage": "tempered_high",
            "polygonal_ferrite_grain_count": n_grains,
            "boundary_bias": 0.70,
            "no_lath_geometry": True,
            "no_retained_austenite": True,
            "n_carbides": n_carbides,
        },
        rendered_layers=["SORBITE", "FERRITE", "CEMENTITE"],
        fragment_area=int(h * w // max(1, n_grains)),
    )
