"""Видманштеттов феррит (§2.10 справочника).

Обслуживает: widmanstatten_ferrite — иглы феррита 50-500 × 2-20 μm в
направлениях {60°, 120°} из границ PAG, на фоне перлита.

Реализация:
- Voronoi PAG ~85 px.
- Аллотриоморфный феррит: дилация boundary mask на 2 px.
- Иглы: per-PAG выбираем 2 направления из {0°, 60°, 120°}, сэмплируем
  Poisson-точки на границах зерна и рисуем полосы.
- Фон между иглами — перлит с ламельной модуляцией.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.generator_grains import generate_grain_structure
from core.metallography_v3.realism_utils import (
    boundary_mask_from_labels,
    multiscale_noise,
    rescale_to_u8,
)
from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import (
    SystemGenerationContext,
    soft_unsharp,
)

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


HANDLES_STAGES: frozenset[str] = frozenset({"widmanstatten_ferrite"})


_TONE_FERRITE_NEEDLE = 220.0
_TONE_PEARLITE_MATRIX = 100.0
_TONE_PAG_BOUNDARY = 35.0


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = int(seed_split.get("seed_topology", context.seed))

    labels = generate_grain_structure(
        size=size,
        seed=seed,
        mean_grain_size_px=85.0,
        grain_size_jitter=0.25,
        boundary_width_px=1,
        boundary_contrast=0.0,
        elongation=1.0,
    )["labels"]
    boundaries = boundary_mask_from_labels(labels, width=2)
    n_grains = int(labels.max()) + 1

    # Фон — перлит.
    img = np.full(size, _TONE_PEARLITE_MATRIX, dtype=np.float32)
    pearlite_lam = np.sin(
        2.0 * math.pi * (np.arange(w)[None, :] + 0.5 * np.arange(h)[:, None]) / 3.5
    ).astype(np.float32)
    img += pearlite_lam * 6.0
    img += (
        multiscale_noise(size=size, seed=seed + 17, scales=((12.0, 0.6), (4.0, 0.4)))
        - 0.5
    ) * 8.0

    # Аллотриоморфный феррит на PAG-границах (дилация).
    if ndimage is not None:
        allotriomorph = ndimage.binary_dilation(boundaries, iterations=1)
    else:
        allotriomorph = boundaries.copy()
    img[allotriomorph] = _TONE_FERRITE_NEEDLE - 10.0

    rng = np.random.default_rng(int(seed) + 41)
    ferrite_mask = allotriomorph.copy()

    direction_options = np.array([0.0, 60.0, 120.0], dtype=np.float32)
    n_needles_per_pag = 8

    ys_b, xs_b = np.where(boundaries)
    if len(xs_b) > 0:
        for grain_id in range(n_grains):
            grain_mask = labels == grain_id
            if not grain_mask.any():
                continue
            dirs = rng.choice(direction_options, size=2, replace=False)
            grain_boundary = boundaries & grain_mask
            bys, bxs = np.where(grain_boundary)
            if len(bxs) < 4:
                continue
            for _ in range(n_needles_per_pag):
                idx = int(rng.integers(0, len(bxs)))
                cy = int(bys[idx])
                cx = int(bxs[idx])
                theta_deg = float(dirs[rng.integers(0, 2)])
                theta = math.radians(theta_deg + float(rng.normal(0.0, 2.5)))
                length = int(rng.uniform(45, 120))
                width = float(rng.uniform(2.5, 5.0))
                _draw_needle(
                    img=img,
                    ferrite_mask=ferrite_mask,
                    cy=cy, cx=cx, length=length, width=width, theta=theta,
                    grain_mask=grain_mask,
                )

    # PAG-границы обратно — тонкая тёмная линия.
    img = np.where(boundaries > 0, _TONE_PAG_BOUNDARY, img)

    image_gray = rescale_to_u8(np.clip(img, 0.0, 255.0), lo=20.0, hi=240.0)
    image_gray = soft_unsharp(image_gray, amount=0.22)

    ferrite_u8 = ferrite_mask.astype(np.uint8)
    pearlite_u8 = ((~ferrite_mask) & (boundaries == 0)).astype(np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "FERRITE": ferrite_u8,
            "PEARLITE": pearlite_u8,
        },
        morphology_trace={
            "family": "widmanstatten_ferrite",
            "stage": "widmanstatten_ferrite",
            "prior_austenite_grain_count": n_grains,
            "needle_directions_deg": [0.0, 60.0, 120.0],
            "ferrite_coverage": float(ferrite_mask.mean()),
        },
        rendered_layers=["FERRITE", "PEARLITE"],
        fragment_area=int(h * w // max(1, n_grains * 4)),
    )


def _draw_needle(
    *,
    img: np.ndarray,
    ferrite_mask: np.ndarray,
    cy: int, cx: int,
    length: int, width: float, theta: float,
    grain_mask: np.ndarray,
) -> None:
    """Узкая полоса (игла) ферритного тона в направлении theta из точки
    (cy, cx). Игла clip'ается grain_mask, so she не пересекает соседние зёрна."""
    h, w = img.shape
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    half_w = width / 2.0

    bbox_r = int(math.ceil(length + width))
    y0 = max(0, cy - bbox_r)
    y1 = min(h, cy + bbox_r + 1)
    x0 = max(0, cx - bbox_r)
    x1 = min(w, cx + bbox_r + 1)
    if y1 <= y0 or x1 <= x0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dy = yy - cy
    dx = xx - cx
    along = dx * cos_t + dy * sin_t
    across = -dx * sin_t + dy * cos_t

    inside = (along >= 0) & (along <= length) & (np.abs(across) <= half_w)
    inside &= grain_mask[y0:y1, x0:x1]
    if not inside.any():
        return

    img[y0:y1, x0:x1] = np.where(inside, _TONE_FERRITE_NEEDLE, img[y0:y1, x0:x1])
    ferrite_mask[y0:y1, x0:x1] |= inside
