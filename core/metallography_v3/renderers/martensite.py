"""Мартенситное семейство (§2.1–2.4 справочника).

Обслуживает: lath (§2.1 — реечный, C<0.6%),
plate/lenticular с midrib (§2.2 — пластинчатый, C>1.0%),
mixed (§2.3 — смешанный, 0.6-1.0%C). Retained austenite (§2.4)
инжектится post-process'ом в ``render_fe_c_unified`` (см. ~1770+),
этот модуль её не трогает.

Реализация — прагматичная (не полная 5-уровневая иерархия PAG →
packet → block → subblock → lath). Морфологические признаки:
  * Lath: анизотропный multiscale_noise (10:1) + стриповая
    модуляция вдоль per-PAG ориентации.
  * Plate: «crosscut needles» — набор линзовидных игл из случайной
    базовой ориентации + {±60°, ±90°, ±120°} с убывающей длиной,
    у каждой — центральная тёмная midrib-линия.
  * Mixed: линейная интерполяция plate_fraction по углероду,
    сначала пластины, остаток заливается lath-текстурой.
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
        "martensite",
        "martensite_tetragonal",
        "martensite_cubic",
    }
)


# --- tones (§2.1, §2.2, §2.3) ———————————————————————————————————
_TONE_LATH_BODY = 67.0
_TONE_LATH_DARK = 45.0
_TONE_PLATE_BODY = 70.0
_TONE_PLATE_HALO = 95.0
_TONE_MIDRIB = 20.0
_TONE_BOUNDARY = 27.0
_TONE_RETAINED_AUSTENITE = 225.0


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    if stage == "martensite_cubic":
        return _render_lath(context=context, seed_split=seed_split)
    if stage == "martensite_tetragonal":
        return _render_plate(context=context, seed_split=seed_split)
    if stage == "martensite":
        return _render_mixed(context=context, seed_split=seed_split)
    raise ValueError(f"martensite renderer has no branch for stage {stage!r}")


# --- helpers —————————————————————————————————————————————————————


def _grain_labels(
    size: tuple[int, int], seed: int, mean_size_px: float
) -> np.ndarray:
    out = generate_grain_structure(
        size=size,
        seed=int(seed),
        mean_grain_size_px=max(32.0, float(mean_size_px)),
        grain_size_jitter=0.20,
        boundary_width_px=1,
        boundary_contrast=0.0,
        elongation=1.0,
    )
    return out["labels"]


def _anisotropic_noise(
    size: tuple[int, int],
    seed: int,
    theta_per_grain: np.ndarray,
    labels: np.ndarray,
    *,
    scale_long: float = 80.0,
    scale_trans: float = 8.0,
) -> np.ndarray:
    """Анизотропный «бархатный» шум с per-PAG ориентацией.

    Для каждого пикселя проекция координаты на «длинную» ось зерна
    используется как sin-основа, перпендикулярная ось — как мелкий шум.
    """
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = theta_per_grain[labels]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    along = xx * cos_t + yy * sin_t
    across = -xx * sin_t + yy * cos_t

    rng = np.random.default_rng(int(seed))
    phase = rng.uniform(0.0, 2.0 * math.pi, size=labels.max() + 1).astype(np.float32)
    wave_long = np.sin(2.0 * math.pi * along / max(3.0, float(scale_long)) + phase[labels])
    wave_trans = np.sin(2.0 * math.pi * across / max(2.0, float(scale_trans)))
    mixed = 0.55 * wave_long + 0.45 * wave_trans
    return mixed.astype(np.float32)


def _per_grain_angles(n_grains: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.uniform(0.0, math.pi, size=n_grains).astype(np.float32)


def _finalize(image_f32: np.ndarray) -> np.ndarray:
    img = rescale_to_u8(np.clip(image_f32, 0.0, 255.0), lo=10.0, hi=245.0)
    return soft_unsharp(img, amount=0.28)


# --- lath / cubic (§2.1) ——————————————————————————————————————————


def _render_lath(
    *,
    context: SystemGenerationContext,
    seed_split: dict[str, int],
    base_grain_size_px: float = 50.0,
) -> RendererOutput:
    size = context.size
    seed = int(seed_split.get("seed_topology", context.seed))
    labels = _grain_labels(size, seed, mean_size_px=base_grain_size_px)
    boundaries = boundary_mask_from_labels(labels, width=2)
    n_grains = int(labels.max()) + 1

    # Per-PAG ориентация (packet/block упрощены — один доминирующий угол
    # на PAG в Phase 4, полная 5-уровневая иерархия в будущей доработке).
    theta = _per_grain_angles(n_grains, seed + 17)

    # Анизотропный шум ~10:1 вдоль орientation.
    anis = _anisotropic_noise(
        size,
        seed + 23,
        theta_per_grain=theta,
        labels=labels,
        scale_long=80.0,
        scale_trans=8.0,
    )
    # Дополнительная мелкозернистая bath-текстура (рейки).
    fine = multiscale_noise(
        size=size, seed=seed + 41, scales=((4.0, 0.6), (1.8, 0.4))
    )

    # Собираем картинку: среднее = TONE_LATH_BODY, модуляция ±амплитуда.
    amplitude = 14.0
    img = (
        _TONE_LATH_BODY
        + anis * amplitude
        + (normalize01(fine) - 0.5) * 8.0
    ).astype(np.float32)

    # Небольшое дополнительное затемнение в «тёмных» минимумах волны
    # (имитация блочных границ).
    img = np.where(anis < -0.75, _TONE_LATH_DARK, img)

    # PAG-границы — тёмные.
    img = np.where(boundaries > 0, _TONE_BOUNDARY, img)

    image_gray = _finalize(img)

    martensite_mask = np.ones(size, dtype=np.uint8)
    martensite_mask[boundaries > 0] = 0
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={"MARTENSITE": martensite_mask},
        morphology_trace={
            "family": "martensite_lath",
            "stage": "martensite_cubic",
            "prior_austenite_grain_count": n_grains,
            "anisotropy_ratio": 10.0,
            "scale_long_px": 80.0,
            "scale_trans_px": 8.0,
        },
        rendered_layers=["MARTENSITE"],
        fragment_area=int(size[0] * size[1] // max(1, n_grains)),
    )


# --- plate / tetragonal (§2.2) ————————————————————————————————————


def _render_plate(
    *,
    context: SystemGenerationContext,
    seed_split: dict[str, int],
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = int(seed_split.get("seed_topology", context.seed))
    labels = _grain_labels(size, seed, mean_size_px=80.0)
    boundaries = boundary_mask_from_labels(labels, width=2)
    n_grains = int(labels.max()) + 1

    # Фон — dislocation-halo (tone ~95), заметно темнее RA. Mean
    # plate-картинки ≈ body-tone по §2.2 (MARTENSITE_TETRAGONAL 82% +
    # RA 13% + cementite 5% по phase_composition).
    img = np.full(size, _TONE_PLATE_HALO, dtype=np.float32)
    img += (
        multiscale_noise(size=size, seed=seed + 31, scales=((20.0, 0.6), (5.0, 0.4)))
        - 0.5
    ) * 8.0

    # Крупные блоки RA (γост) — 13% по §2.4.
    ra_field = multiscale_noise(
        size=size, seed=seed + 61, scales=((28.0, 0.65), (9.0, 0.35))
    )
    ra_threshold = float(np.quantile(ra_field, 0.87))  # ~13% RA
    ra_mask = ra_field >= ra_threshold
    if ndimage is not None:
        ra_mask = ndimage.binary_opening(ra_mask, iterations=1)
    img[ra_mask] = _TONE_RETAINED_AUSTENITE

    rng = np.random.default_rng(int(seed) + 97)
    # Для каждой PAG — набор игл.
    for grain_id in range(n_grains):
        grain_mask = labels == grain_id
        if not np.any(grain_mask):
            continue
        ys, xs = np.where(grain_mask)
        if len(xs) < 16:
            continue
        cy_pag = float(ys.mean())
        cx_pag = float(xs.mean())
        # Размер PAG-зерна как диапазон длины иглы.
        grain_extent = float(max(ys.max() - ys.min(), xs.max() - xs.min()))
        max_needle_length = 0.9 * grain_extent

        # Базовая ориентация.
        theta0 = float(rng.uniform(0.0, math.pi))
        direction_offsets = np.array(
            [0.0, math.pi / 3, -math.pi / 3, math.pi / 2, 2 * math.pi / 3, -2 * math.pi / 3],
            dtype=np.float32,
        )
        weights = np.array([0.35, 0.2, 0.2, 0.1, 0.1, 0.05], dtype=np.float32)

        n_needles = int(rng.poisson(18)) + 8
        for i in range(n_needles):
            offset = float(rng.choice(direction_offsets, p=weights))
            theta = theta0 + offset + rng.normal(0.0, math.radians(3.0))
            # Уменьшающаяся длина, §2.2.
            length = max(6.0, max_needle_length * 0.55 ** (i / max(1, n_needles)))
            aspect = float(rng.uniform(12.0, 25.0))
            width = max(1.2, length / aspect)

            # Центр иглы слегка смещён от центра PAG.
            jitter_x = rng.normal(0.0, 0.25 * grain_extent)
            jitter_y = rng.normal(0.0, 0.25 * grain_extent)
            cy = cy_pag + jitter_y
            cx = cx_pag + jitter_x

            _draw_plate_with_midrib(
                img=img,
                cy=cy,
                cx=cx,
                length=length,
                width=width,
                theta=theta,
                grain_mask=grain_mask,
            )

    # PAG-границы.
    img = np.where(boundaries > 0, _TONE_BOUNDARY, img)

    image_gray = _finalize(img)

    # Маска мартенсита = всё, кроме явно светлых RA-областей.
    plate_mask = (image_gray < 190).astype(np.uint8)
    plate_mask[boundaries > 0] = 0
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "MARTENSITE": plate_mask,
            "AUSTENITE": (image_gray >= 190).astype(np.uint8),
        },
        morphology_trace={
            "family": "martensite_plate",
            "stage": "martensite_tetragonal",
            "prior_austenite_grain_count": n_grains,
            "midrib_present": True,
        },
        rendered_layers=["MARTENSITE", "AUSTENITE"],
        fragment_area=int(size[0] * size[1] // max(1, n_grains * 4)),
    )


def _draw_plate_with_midrib(
    *,
    img: np.ndarray,
    cy: float,
    cx: float,
    length: float,
    width: float,
    theta: float,
    grain_mask: np.ndarray,
) -> None:
    """Рисует двояковыпуклую линзу мартенситной пластины с midrib.

    Через bounding-box + расстояние до оси. Игла ограничена PAG-зерном
    (grain_mask).
    """
    h, w = img.shape
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    half_L = length / 2.0
    half_W = width / 2.0

    # Bounding box иглы: наибольший проекционный радиус.
    bbox_r = int(math.ceil(max(half_L, half_W) * 1.1))
    y0 = max(0, int(cy - bbox_r))
    y1 = min(h, int(cy + bbox_r + 1))
    x0 = max(0, int(cx - bbox_r))
    x1 = min(w, int(cx + bbox_r + 1))
    if y1 <= y0 or x1 <= x0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dy = yy - cy
    dx = xx - cx
    along = dx * cos_t + dy * sin_t
    across = -dx * sin_t + dy * cos_t

    # Линза: |along|≤L/2 AND |across| ≤ W/2 × sqrt(1-(2along/L)^2).
    along_norm = along / max(1e-6, half_L)
    width_at = half_W * np.sqrt(np.clip(1.0 - along_norm ** 2, 0.0, 1.0))
    inside = (np.abs(along) <= half_L) & (np.abs(across) <= width_at)

    # Clip к PAG-зерну (чтобы не рисовать за границами).
    inside &= grain_mask[y0:y1, x0:x1]
    if not np.any(inside):
        return

    # Тело пластины: тон body + небольшой halo на краях.
    edge_factor = np.abs(across) / (width_at + 1e-6)  # [0,1], 0 = ось
    edge_factor = np.clip(edge_factor, 0.0, 1.0)
    plate_tone = _TONE_PLATE_BODY + edge_factor * (_TONE_PLATE_HALO - _TONE_PLATE_BODY)
    patch = img[y0:y1, x0:x1]
    patch[inside] = plate_tone[inside]

    # Midrib — центральная линия (|across| ≤ 0.5 px).
    midrib = inside & (np.abs(across) <= 0.8)
    patch[midrib] = _TONE_MIDRIB

    img[y0:y1, x0:x1] = patch


# --- mixed (§2.3) ——————————————————————————————————————————————


def _render_mixed(
    *,
    context: SystemGenerationContext,
    seed_split: dict[str, int],
) -> RendererOutput:
    c_wt = float((context.composition_wt or {}).get("C", 0.8))
    # plate_fraction = clip((C - 0.4) / 0.6, 0, 1), §2.3.
    plate_frac = max(0.0, min(1.0, (c_wt - 0.4) / 0.6))

    # Сначала рендерим lath-фон.
    lath = _render_lath(context=context, seed_split=seed_split, base_grain_size_px=55.0)
    if plate_frac <= 0.001:
        return RendererOutput(
            image_gray=lath.image_gray,
            phase_masks=lath.phase_masks,
            morphology_trace={
                **lath.morphology_trace,
                "family": "martensite_mixed",
                "plate_fraction": plate_frac,
            },
            rendered_layers=lath.rendered_layers,
            fragment_area=lath.fragment_area,
        )

    # Поверх — plate-иглы, но только на части PAG-зёрен (по plate_frac).
    plate = _render_plate(context=context, seed_split=seed_split)
    # Смешиваем: plate-пиксели (tone ≤160, содержат mid-body tone) берём
    # с весом plate_frac; оставшееся — lath-фон.
    plate_img = plate.image_gray.astype(np.float32)
    lath_img = lath.image_gray.astype(np.float32)
    plate_fg_mask = plate_img < 150  # где нарисованы пластины
    img = lath_img.copy()
    # Доля пластин ограничивается plate_frac (вероятностный mask).
    rng = np.random.default_rng(
        int(seed_split.get("seed_topology", context.seed)) + 57
    )
    select = rng.random(plate_fg_mask.shape) < plate_frac
    blend_mask = plate_fg_mask & select
    img[blend_mask] = plate_img[blend_mask]

    image_gray = _finalize(img)
    martensite_mask = (image_gray < 200).astype(np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={"MARTENSITE": martensite_mask},
        morphology_trace={
            "family": "martensite_mixed",
            "stage": "martensite",
            "plate_fraction": plate_frac,
            "c_wt": c_wt,
        },
        rendered_layers=["MARTENSITE"],
        fragment_area=lath.fragment_area,
    )
