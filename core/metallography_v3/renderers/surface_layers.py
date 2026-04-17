"""Поверхностные композиционные слои (§3.2-3.3 справочника).

Обслуживает:
  * decarburized_layer (§3.2, FFD→MAD→core, вертикальный градиент,
    верх светлее)
  * carburized_layer (§3.3, surface мартенсит → core α+P, верх темнее)

Принципиально построчная композиция: C(y) = erfc-профиль. Для каждой
y-строки выбираем локальный тон, интерполирующий между поверхностью и
сердцевиной.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.metallography_v3.realism_utils import (
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
        "decarburized_layer",
        "carburized_layer",
    }
)

REQUIRES_SURFACE_COMPOSITION: bool = True


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    if stage == "decarburized_layer":
        return _render_decarburized(context=context, seed_split=seed_split)
    if stage == "carburized_layer":
        return _render_carburized(context=context, seed_split=seed_split)
    raise ValueError(f"surface_layers renderer has no branch for {stage!r}")


def _erfc(y_norm: np.ndarray) -> np.ndarray:
    """Аппроксимация erfc через tanh (без scipy.special)."""
    return 1.0 - np.tanh(2.5 * y_norm)


# --- decarburized (§3.2) —————————————————————————————————————


def _render_decarburized(
    *, context: SystemGenerationContext, seed_split: dict[str, int]
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = int(seed_split.get("seed_topology", context.seed))

    # Вертикальная координата нормированная [0, 1].
    y_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    # C-profile: 0 на поверхности (FFD) → C_core на сердцевине.
    # `c_frac` = 1 - erfc(y_norm * 2.5) → 0..1.
    c_frac = 1.0 - _erfc(y_norm * 2.2)
    c_frac = np.clip(c_frac, 0.0, 1.0)
    # Tone: 228 у поверхности (чистый феррит) → 140 у сердцевины (α+P).
    tone_surface = 228.0
    tone_core = 140.0
    base_col = tone_surface + (tone_core - tone_surface) * c_frac
    img = np.broadcast_to(base_col, size).astype(np.float32).copy()

    # Лёгкая текстура — per-row шум.
    img += (
        multiscale_noise(size=size, seed=seed + 13, scales=((10.0, 0.6), (3.0, 0.4)))
        - 0.5
    ) * 10.0

    # Перлит в MAD-зоне (средняя часть): случайные тёмные blobs
    # пропорционально c_frac.
    rng = np.random.default_rng(seed + 37)
    n_pearlite_colonies = 300
    for _ in range(n_pearlite_colonies):
        cy = int(rng.integers(0, h))
        # Вероятность появления перлита = c_frac[cy].
        if rng.random() > float(c_frac[cy, 0]) * 0.8:
            continue
        cx = int(rng.integers(0, w))
        r = float(rng.uniform(2.0, 6.0))
        y0 = max(0, int(cy - r))
        y1 = min(h, int(cy + r + 1))
        x0 = max(0, int(cx - r))
        x1 = min(w, int(cx + r + 1))
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        img[y0:y1, x0:x1][disk] = 100.0

    # Сглаживание границ зон (σ=15 px).
    if ndimage is not None:
        img = ndimage.gaussian_filter(img, sigma=(8.0, 1.0))  # только по y

    image_gray = rescale_to_u8(np.clip(img, 0.0, 255.0), lo=30.0, hi=245.0)
    image_gray = soft_unsharp(image_gray, amount=0.18)

    # Фаза-маски: ferrite-frac = 1 - c_frac, pearlite-frac = c_frac * 0.3.
    c_frac_2d = np.broadcast_to(c_frac, size)
    ferrite_mask = (c_frac_2d < 0.3).astype(np.uint8)
    pearlite_mask = (image_gray <= 120).astype(np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "FERRITE": ferrite_mask,
            "PEARLITE": pearlite_mask,
        },
        morphology_trace={
            "family": "decarburized_layer",
            "stage": "decarburized_layer",
            "profile": "erfc",
            "surface_tone": tone_surface,
            "core_tone": tone_core,
            "vertical_gradient": "bright_top_dark_bottom",
        },
        rendered_layers=["FERRITE", "PEARLITE"],
        fragment_area=int(h * w // 100),
    )


# --- carburized (§3.3) ——————————————————————————————————————


def _render_carburized(
    *, context: SystemGenerationContext, seed_split: dict[str, int]
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = int(seed_split.get("seed_topology", context.seed))

    y_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    # C-surface высокий (~1.0), core низкий (~0.2).
    c_frac = _erfc(y_norm * 2.5)
    c_frac = np.clip(c_frac, 0.0, 1.0)
    # После закалки: верх — тёмный мартенсит (60), сердцевина — светлая
    # α+P матрица (165). C_frac высокий → тёмный тон.
    tone_surface = 60.0
    tone_core = 170.0
    base_col = tone_surface + (tone_core - tone_surface) * (1.0 - c_frac)
    img = np.broadcast_to(base_col, size).astype(np.float32).copy()

    # Subsurface текстура: тёмные мартенситные штрихи в верхней зоне.
    rng = np.random.default_rng(seed + 41)
    n_needles = 200
    for _ in range(n_needles):
        cy = int(rng.integers(0, int(h * 0.6)))  # только в верхней части
        # Вероятность появления иглы высокая при high c_frac[cy].
        if rng.random() > float(c_frac[cy, 0]):
            continue
        cx = int(rng.integers(0, w))
        length = int(rng.uniform(6, 14))
        theta = float(rng.uniform(0.0, math.pi))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        for t in range(-length, length + 1):
            y = int(cy + t * sin_t)
            x = int(cx + t * cos_t)
            if 0 <= y < h and 0 <= x < w:
                img[y, x] = 35.0

    # Перлит в средней зоне.
    n_pearlite = 150
    for _ in range(n_pearlite):
        cy = int(rng.integers(int(h * 0.4), int(h * 0.85)))
        if rng.random() > 0.5:
            continue
        cx = int(rng.integers(0, w))
        r = float(rng.uniform(2.0, 5.0))
        y0 = max(0, int(cy - r))
        y1 = min(h, int(cy + r + 1))
        x0 = max(0, int(cx - r))
        x1 = min(w, int(cx + r + 1))
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        img[y0:y1, x0:x1][disk] = 95.0

    img += (
        multiscale_noise(size=size, seed=seed + 53, scales=((10.0, 0.6), (3.0, 0.4)))
        - 0.5
    ) * 8.0

    if ndimage is not None:
        img = ndimage.gaussian_filter(img, sigma=(6.0, 0.8))

    image_gray = rescale_to_u8(np.clip(img, 0.0, 255.0), lo=20.0, hi=220.0)
    image_gray = soft_unsharp(image_gray, amount=0.18)

    # Фазовые маски — по зонам.
    c_frac_2d = np.broadcast_to(c_frac, size)
    martensite_mask = (c_frac_2d >= 0.55).astype(np.uint8)
    ferrite_mask = (c_frac_2d <= 0.25).astype(np.uint8)
    pearlite_mask = (image_gray <= 110).astype(np.uint8)
    cementite_mask = (image_gray <= 50).astype(np.uint8)
    austenite_mask = (image_gray >= 200).astype(np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "MARTENSITE": martensite_mask,
            "PEARLITE": pearlite_mask,
            "FERRITE": ferrite_mask,
            "CEMENTITE": cementite_mask,
            "AUSTENITE": austenite_mask,
        },
        morphology_trace={
            "family": "carburized_layer",
            "stage": "carburized_layer",
            "profile": "erfc",
            "surface_tone": tone_surface,
            "core_tone": tone_core,
            "vertical_gradient": "dark_top_bright_bottom",
        },
        rendered_layers=["MARTENSITE", "PEARLITE", "FERRITE", "CEMENTITE", "AUSTENITE"],
        fragment_area=int(h * w // 150),
    )
