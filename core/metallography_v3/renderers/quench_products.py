"""Закалочные продукты мелкопластинчатого перлита (§2.8–2.9 справочника).

Отделены от tempered — это прямые продукты закалки через перегиб
С-кривой ТТТ, не отпуск.

Обслуживает:
  * troostite_quench (§2.8, S₀≈0.1 μm — не разрешается, "чёрные кляксы")
  * sorbite_quench (§2.9, S₀ 0.2–0.3 μm — различимая штриховка)
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
        "troostite_quench",
        "sorbite_quench",
    }
)


# --- tones (§2.8, §2.9) ———————————————————————————————————————
_TONE_TROOSTITE_BLOB = 42.0
_TONE_TROOSTITE_BG = 85.0
_TONE_SORBITE_FERRITE = 180.0
_TONE_SORBITE_CEMENTITE = 65.0


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    if stage == "troostite_quench":
        return _render_troostite(
            context=context,
            seed_split=seed_split,
            phase_fractions=phase_fractions,
        )
    if stage == "sorbite_quench":
        return _render_sorbite(
            context=context,
            seed_split=seed_split,
            phase_fractions=phase_fractions,
        )
    raise ValueError(
        f"quench_products renderer has no branch for stage {stage!r}"
    )


def _make_retained_austenite_mask(
    *,
    size: tuple[int, int],
    seed: int,
    target_fraction: float,
) -> np.ndarray:
    """Строит блочную маску γост на основе multiscale_noise threshold.

    Используется в обоих quench renderer'ах при
    ``phase_fractions["AUSTENITE"] > 0`` (RA-инжекция из
    ``render_fe_c_unified``).
    """
    if target_fraction <= 0.0:
        return np.zeros(size, dtype=bool)
    frac = float(max(0.005, min(0.45, target_fraction)))
    field = multiscale_noise(
        size=size, seed=int(seed), scales=((22.0, 0.65), (8.0, 0.35))
    )
    threshold = float(np.quantile(field, 1.0 - frac))
    mask = field >= threshold
    if ndimage is not None:
        mask = ndimage.binary_opening(mask, iterations=1)
    return mask


# --- troostite quench (§2.8) ————————————————————————————————————


def _render_troostite(
    *,
    context: SystemGenerationContext,
    seed_split: dict[str, int],
    phase_fractions: dict[str, float] | None = None,
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = int(seed_split.get("seed_topology", context.seed))
    phase_fractions = phase_fractions or {}
    ra_target = float(phase_fractions.get("AUSTENITE", 0.0))

    # Фон — средне-тёмный, изотропный.
    img = np.full(size, _TONE_TROOSTITE_BG, dtype=np.float32)
    img += (
        multiscale_noise(size=size, seed=seed + 11, scales=((12.0, 0.6), (4.0, 0.4)))
        - 0.5
    ) * 8.0

    # PAG-сетка для bias ядер к границам.
    labels = generate_grain_structure(
        size=size,
        seed=seed + 19,
        mean_grain_size_px=55.0,
        grain_size_jitter=0.2,
        boundary_width_px=1,
        boundary_contrast=0.0,
        elongation=1.0,
    )["labels"]
    boundaries = boundary_mask_from_labels(labels, width=3)

    rng = np.random.default_rng(int(seed) + 29)
    area = h * w
    # ~0.05 per μm² на границах, 0.02 per μm² в объёме (§2.8).
    # Примем 1 μm ≈ 1 px для тестовых целей.
    n_boundary_blobs = max(20, int(0.05 * area / 25))  # density по boundary mask
    n_interior_blobs = max(15, int(0.02 * area / 50))

    ys_b, xs_b = np.where(boundaries)
    # Boundary-blobs.
    for _ in range(n_boundary_blobs):
        if len(xs_b) == 0:
            break
        idx = int(rng.integers(0, len(xs_b)))
        cy = int(ys_b[idx])
        cx = int(xs_b[idx])
        r = float(rng.uniform(2.5, 5.0))
        size_um = float(rng.lognormal(mean=math.log(4.0), sigma=0.35))
        _paint_blob(img, cy=cy, cx=cx, sigma=r, scale=size_um)

    # Interior-blobs.
    for _ in range(n_interior_blobs):
        cy = int(rng.integers(0, h))
        cx = int(rng.integers(0, w))
        r = float(rng.uniform(2.0, 4.5))
        size_um = float(rng.lognormal(mean=math.log(3.5), sigma=0.35))
        _paint_blob(img, cy=cy, cx=cx, sigma=r, scale=size_um)

    # Лёгкий Perlin-шум поверх, §2.8 (amp=6).
    img += (
        multiscale_noise(size=size, seed=seed + 41, scales=((1.5, 1.0),))
        - 0.5
    ) * 6.0

    # Если phase_orchestrator инжектирует γост (RA fraction > 0),
    # карвим блочную маску и заливаем её светлым тоном (§2.4). Mask
    # возвращается из renderer'а, чтобы phase_visibility_report
    # отчитывал реальное покрытие.
    ra_mask = _make_retained_austenite_mask(
        size=size, seed=seed + 97, target_fraction=ra_target
    )
    if ra_mask.any():
        img[ra_mask] = 225.0  # RA bright tone по §2.4

    image_gray = rescale_to_u8(np.clip(img, 0.0, 255.0), lo=15.0, hi=230.0)
    image_gray = soft_unsharp(image_gray, amount=0.18)

    troostite_mask = ((image_gray <= 100) & ~ra_mask).astype(np.uint8)
    cementite_mask = ((image_gray <= 55) & ~ra_mask).astype(np.uint8)
    ra_mask_u8 = ra_mask.astype(np.uint8)
    phase_masks: dict[str, np.ndarray] = {
        "TROOSTITE": troostite_mask,
        "CEMENTITE": cementite_mask,
    }
    rendered_layers = ["TROOSTITE", "CEMENTITE"]
    if ra_target > 0.0:
        phase_masks["AUSTENITE"] = ra_mask_u8
        rendered_layers.append("AUSTENITE")
    return RendererOutput(
        image_gray=image_gray,
        phase_masks=phase_masks,
        morphology_trace={
            "family": "troostite_quench",
            "stage": "troostite_quench",
            "n_boundary_blobs": n_boundary_blobs,
            "n_interior_blobs": n_interior_blobs,
            "anisotropy_present": False,
            "retained_austenite_fraction_target": ra_target,
            "retained_austenite_fraction_actual": float(ra_mask.mean()),
        },
        rendered_layers=rendered_layers,
        fragment_area=25,
    )


def _paint_blob(
    img: np.ndarray, *, cy: int, cx: int, sigma: float, scale: float
) -> None:
    """Гауссов blob с тёмным ядром — имитация троостит-колонии."""
    h, w = img.shape
    radius = int(math.ceil(max(scale, sigma) * 1.6))
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    # Гауссовский falloff умноженный на step-like clip до `scale`.
    gauss = np.exp(-dist2 / (2.0 * sigma ** 2))
    inside_scale = dist2 <= scale ** 2
    contribution = np.where(inside_scale, gauss, gauss * 0.4)
    # Темнение: blend текущий пиксель к _TONE_TROOSTITE_BLOB.
    patch = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = patch * (1.0 - contribution) + _TONE_TROOSTITE_BLOB * contribution


# --- sorbite quench (§2.9) ——————————————————————————————————————


def _render_sorbite(
    *,
    context: SystemGenerationContext,
    seed_split: dict[str, int],
    phase_fractions: dict[str, float] | None = None,
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = int(seed_split.get("seed_topology", context.seed))
    phase_fractions = phase_fractions or {}
    ra_target = float(phase_fractions.get("AUSTENITE", 0.0))

    # Voronoi колоний 5-20 μm ≈ 12 px среднее.
    labels = generate_grain_structure(
        size=size,
        seed=seed + 23,
        mean_grain_size_px=14.0,
        grain_size_jitter=0.25,
        boundary_width_px=1,
        boundary_contrast=0.0,
        elongation=1.0,
    )["labels"]
    n_colonies = int(labels.max()) + 1

    rng = np.random.default_rng(int(seed) + 31)
    theta = rng.uniform(0.0, math.pi, size=n_colonies).astype(np.float32)
    period = rng.normal(0.25, 0.04, size=n_colonies).astype(np.float32)
    # period — μm, конвертируем в px (предположим 5 px/μm для этого масштаба).
    period_px = np.clip(period * 14.0, 2.5, 5.0)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=n_colonies).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cos_t = np.cos(theta[labels])
    sin_t = np.sin(theta[labels])
    proj = xx * cos_t + yy * sin_t

    # Per-colony period for each pixel.
    local_period = period_px[labels]

    # Фазовая модуляция phi (Perlin-like).
    phi_noise = (
        multiscale_noise(size=size, seed=seed + 43, scales=((7.0, 0.65), (2.5, 0.35)))
        - 0.5
    ).astype(np.float32)
    phi = phase[labels] + phi_noise * 0.6

    wave = 0.5 + 0.5 * np.sin(2.0 * math.pi / local_period * proj + phi)
    # Ламельная штриховка: tone α (светлая) vs Fe₃C (тёмная).
    img = _TONE_SORBITE_CEMENTITE + wave * (_TONE_SORBITE_FERRITE - _TONE_SORBITE_CEMENTITE)

    # Лёгкий Perlin для realism.
    img += (
        multiscale_noise(size=size, seed=seed + 57, scales=((10.0, 0.6), (3.5, 0.4)))
        - 0.5
    ) * 10.0

    # γост инжекция (как в troostite). Блочные светлые пятна поверх
    # штриховки, маска возвращается отдельно для phase_visibility_report.
    ra_mask = _make_retained_austenite_mask(
        size=size, seed=seed + 103, target_fraction=ra_target
    )
    if ra_mask.any():
        img[ra_mask] = 225.0

    image_gray = rescale_to_u8(np.clip(img, 0.0, 255.0), lo=20.0, hi=230.0)
    image_gray = soft_unsharp(image_gray, amount=0.22)

    sorbite_mask = (~ra_mask).astype(np.uint8)
    cementite_mask = ((image_gray <= 85) & ~ra_mask).astype(np.uint8)
    ra_mask_u8 = ra_mask.astype(np.uint8)
    phase_masks: dict[str, np.ndarray] = {
        "SORBITE": sorbite_mask,
        "CEMENTITE": cementite_mask,
    }
    rendered_layers = ["SORBITE", "CEMENTITE"]
    if ra_target > 0.0:
        phase_masks["AUSTENITE"] = ra_mask_u8
        rendered_layers.append("AUSTENITE")
    return RendererOutput(
        image_gray=image_gray,
        phase_masks=phase_masks,
        morphology_trace={
            "family": "sorbite_quench",
            "stage": "sorbite_quench",
            "colony_count": n_colonies,
            "period_px_mean": float(local_period.mean()),
            "per_colony_orientation": True,
            "retained_austenite_fraction_target": ra_target,
            "retained_austenite_fraction_actual": float(ra_mask.mean()),
        },
        rendered_layers=rendered_layers,
        fragment_area=int(h * w // max(1, n_colonies)),
    )
