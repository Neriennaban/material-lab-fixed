"""Бейнитное семейство (§2.5–2.7 справочника).

Обслуживает:
  * bainite_upper (§2.5, feathery packets, плёнки Fe₃C параллельно
    осям реек)
  * bainite_lower (§2.6, acicular laths + 60° hatch, ОДНО направление
    внутри каждой иглы — отличие от отпущенного мартенсита)
  * carbide_free_bainite (§2.7, Si≥1.5%, бархат αb + блоки γR,
    НЕТ точечных карбидов) — новая стадия в SYSTEM_STAGE_ORDER
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
        "bainite_upper",
        "bainite_lower",
        "carbide_free_bainite",
    }
)


# --- tones (§2.5, §2.6, §2.7) ———————————————————————————————————
# Upper bainite
_TONE_UB_MATRIX = 140.0
_TONE_UB_CEMENTITE_FILM = 60.0
_TONE_UB_INTER_SHEAF = 205.0
# Lower bainite (тона занижены от сырых §2.6 значений, т.к. после
# rescale_to_u8(lo=15, hi=245) диапазон растягивается — нужно, чтобы
# needle body после финализации попадал в «тёмную» зону ≤65).
_TONE_LB_BACKGROUND = 125.0
_TONE_LB_NEEDLE_BODY = 58.0
_TONE_LB_HATCH = 22.0
# Carbide-free bainite
_TONE_CFB_MATRIX = 92.0
_TONE_CFB_RA_BLOCK = 225.0
_TONE_CFB_MA_CORE = 60.0


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    if stage == "bainite_upper":
        return _render_upper(context=context, seed_split=seed_split)
    if stage == "bainite_lower":
        return _render_lower(context=context, seed_split=seed_split)
    if stage == "carbide_free_bainite":
        return _render_cfb(context=context, seed_split=seed_split)
    raise ValueError(f"bainite renderer has no branch for stage {stage!r}")


# --- helpers —————————————————————————————————————————————————————


def _grain_labels(
    size: tuple[int, int], seed: int, mean_size_px: float
) -> np.ndarray:
    out = generate_grain_structure(
        size=size,
        seed=int(seed),
        mean_grain_size_px=max(32.0, float(mean_size_px)),
        grain_size_jitter=0.22,
        boundary_width_px=1,
        boundary_contrast=0.0,
        elongation=1.0,
    )
    return out["labels"]


def _finalize(image_f32: np.ndarray) -> np.ndarray:
    img = rescale_to_u8(np.clip(image_f32, 0.0, 255.0), lo=15.0, hi=245.0)
    return soft_unsharp(img, amount=0.25)


def _per_grain_theta(n_grains: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.uniform(0.0, math.pi, size=n_grains).astype(np.float32)


# --- upper bainite (§2.5) ————————————————————————————————————————


def _render_upper(
    *, context: SystemGenerationContext, seed_split: dict[str, int]
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = int(seed_split.get("seed_topology", context.seed))
    labels = _grain_labels(size, seed, mean_size_px=60.0)
    boundaries = boundary_mask_from_labels(labels, width=2)
    n_grains = int(labels.max()) + 1

    theta = _per_grain_theta(n_grains, seed + 19)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cos_t = np.cos(theta[labels])
    sin_t = np.sin(theta[labels])
    along = xx * cos_t + yy * sin_t
    across = -xx * sin_t + yy * cos_t

    # Перистая анизотропия: медленная волна вдоль оси (packet), быстрая
    # поперёк (lath stripes).
    rng = np.random.default_rng(int(seed) + 29)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=n_grains).astype(np.float32)
    long_wave = np.sin(2.0 * math.pi * along / 65.0 + phase[labels])
    trans_wave = np.sin(2.0 * math.pi * across / 3.5)

    # Базовая текстура (§2.5 matrix).
    img = (
        _TONE_UB_MATRIX
        + long_wave * 14.0
        + trans_wave * 8.0
    ).astype(np.float32)

    # Плёнки Fe₃C ∥ оси: там, где trans_wave близок к 0, и long_wave
    # умеренно положителен — рисуем тёмные точечные плёнки.
    cementite_film = (np.abs(trans_wave) < 0.22) & (long_wave > -0.3)
    img[cementite_film] = _TONE_UB_CEMENTITE_FILM

    # Inter-sheaf зоны (§2.5: ~5% светлых клиньев между пучками).
    inter_mask = long_wave > 0.88
    img[inter_mask] = _TONE_UB_INTER_SHEAF

    # Границы PAG.
    img = np.where(boundaries > 0, 38.0, img)

    # Micro-noise для реалистичности.
    img += (
        multiscale_noise(size=size, seed=seed + 41, scales=((12.0, 0.65), (4.0, 0.35)))
        - 0.5
    ) * 6.0

    image_gray = _finalize(img)

    cementite_mask = cementite_film.astype(np.uint8)
    bainite_mask = np.ones(size, dtype=np.uint8)
    bainite_mask[cementite_film] = 0
    bainite_mask[boundaries > 0] = 0
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "BAINITE": bainite_mask,
            "CEMENTITE": cementite_mask,
        },
        morphology_trace={
            "family": "bainite_upper_feathery",
            "stage": "bainite_upper",
            "prior_austenite_grain_count": n_grains,
            "anisotropy_ratio": 10.0,
            "cementite_film_fraction": float(cementite_mask.mean()),
        },
        rendered_layers=["BAINITE", "CEMENTITE"],
        fragment_area=int(size[0] * size[1] // max(1, n_grains * 2)),
    )


# --- lower bainite (§2.6) ————————————————————————————————————————


def _render_lower(
    *, context: SystemGenerationContext, seed_split: dict[str, int]
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = int(seed_split.get("seed_topology", context.seed))

    img = np.full(size, _TONE_LB_BACKGROUND, dtype=np.float32)
    img += (
        multiscale_noise(size=size, seed=seed + 53, scales=((16.0, 0.65), (4.0, 0.35)))
        - 0.5
    ) * 8.0

    # Иглы-щепа: Poisson sampling центров + случайных ориентаций.
    rng = np.random.default_rng(int(seed) + 61)
    area = h * w
    # density ~0.0018/px² для 192×192 дает ~65 игл.
    n_needles = max(18, int(0.0018 * area))
    needle_mask_total = np.zeros(size, dtype=bool)
    hatch_mask_total = np.zeros(size, dtype=bool)

    for _ in range(n_needles):
        cy = float(rng.integers(0, h))
        cx = float(rng.integers(0, w))
        theta = float(rng.uniform(0.0, math.pi))
        length = float(rng.uniform(25.0, 55.0))
        width = float(rng.uniform(1.2, 2.8))
        _draw_lower_needle_with_hatch(
            img=img,
            needle_mask=needle_mask_total,
            hatch_mask=hatch_mask_total,
            cy=cy, cx=cx,
            length=length, width=width, theta=theta,
        )

    image_gray = _finalize(img)

    bainite_mask = needle_mask_total.astype(np.uint8)
    cementite_mask = hatch_mask_total.astype(np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "BAINITE": bainite_mask,
            "CEMENTITE": cementite_mask,
        },
        morphology_trace={
            "family": "bainite_lower_acicular",
            "stage": "bainite_lower",
            "needle_count": n_needles,
            "hatch_fraction": float(cementite_mask.mean()),
            "single_hatch_direction_per_needle": True,
        },
        rendered_layers=["BAINITE", "CEMENTITE"],
        fragment_area=int(area // max(1, n_needles)),
    )


def _draw_lower_needle_with_hatch(
    *,
    img: np.ndarray,
    needle_mask: np.ndarray,
    hatch_mask: np.ndarray,
    cy: float,
    cx: float,
    length: float,
    width: float,
    theta: float,
) -> None:
    """Рисует иглу + 60° hatch-штрихи в ОДНОМ направлении внутри неё."""
    h, w = img.shape
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    bbox_r = int(math.ceil(length / 2.0 * 1.1))
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

    inside = (np.abs(along) <= length / 2.0) & (np.abs(across) <= width / 2.0)
    # Тело иглы.
    patch = img[y0:y1, x0:x1]
    patch[inside] = _TONE_LB_NEEDLE_BODY
    # Штрихи под 60° к оси иглы. ОДНО направление на иглу.
    # Координаты вдоль штриха — комбинация along/across под +60°.
    hatch_cos = math.cos(math.radians(60.0))
    hatch_sin = math.sin(math.radians(60.0))
    hatch_proj = along * hatch_cos + across * hatch_sin
    # Периодическая модуляция даёт узкие hatch-линии.
    hatch_wave = np.sin(2.0 * math.pi * hatch_proj / 3.5)
    hatch = inside & (np.abs(hatch_wave) > 0.88)
    patch[hatch] = _TONE_LB_HATCH

    img[y0:y1, x0:x1] = patch
    needle_mask[y0:y1, x0:x1] |= inside
    hatch_mask[y0:y1, x0:x1] |= hatch


# --- carbide-free bainite (§2.7) ————————————————————————————————


def _render_cfb(
    *, context: SystemGenerationContext, seed_split: dict[str, int]
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = int(seed_split.get("seed_topology", context.seed))
    labels = _grain_labels(size, seed, mean_size_px=55.0)
    n_grains = int(labels.max()) + 1
    theta = _per_grain_theta(n_grains, seed + 79)

    # Очень мелкий анизотропный шум 10:1 (scale_long=20, scale_trans=2).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cos_t = np.cos(theta[labels])
    sin_t = np.sin(theta[labels])
    along = xx * cos_t + yy * sin_t
    across = -xx * sin_t + yy * cos_t
    rng = np.random.default_rng(int(seed) + 83)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=n_grains).astype(np.float32)
    velvet_long = np.sin(2.0 * math.pi * along / 20.0 + phase[labels])
    velvet_trans = np.sin(2.0 * math.pi * across / 2.0)
    velvet = 0.5 * velvet_long + 0.5 * velvet_trans

    img = (_TONE_CFB_MATRIX + velvet * 8.0).astype(np.float32)
    img += (
        multiscale_noise(size=size, seed=seed + 91, scales=((8.0, 0.6), (2.2, 0.4)))
        - 0.5
    ) * 5.0

    # Блоки γR: Poisson disk с ≈10% coverage.
    ra_field = multiscale_noise(
        size=size, seed=seed + 107, scales=((18.0, 0.65), (6.0, 0.35))
    )
    ra_threshold = float(np.quantile(ra_field, 0.80))
    ra_mask = ra_field >= ra_threshold
    if ndimage is not None:
        ra_mask = ndimage.binary_opening(ra_mask, iterations=1)
    img[ra_mask] = _TONE_CFB_RA_BLOCK

    # 30% RA-блоков имеют тёмное мартенситное ядро (M/A), §2.7.
    ma_mask = np.zeros(size, dtype=bool)
    if ndimage is not None and ra_mask.any():
        labeled, n_blocks = ndimage.label(ra_mask)
        ma_blocks = rng.random(n_blocks + 1) < 0.30
        ma_blocks[0] = False  # background label
        # Ядро = эрозия блока на 2 пикселя.
        core_candidates = ndimage.binary_erosion(ra_mask, iterations=2)
        for block_id, use in enumerate(ma_blocks):
            if not use:
                continue
            block_mask = (labeled == block_id) & core_candidates
            ma_mask |= block_mask
        img[ma_mask] = _TONE_CFB_MA_CORE

    image_gray = _finalize(img)

    ra_u8 = ra_mask.astype(np.uint8)
    martensite_u8 = ma_mask.astype(np.uint8)
    bainite_u8 = (~ra_mask).astype(np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "BAINITE": bainite_u8,
            "AUSTENITE": ra_u8,
            "MARTENSITE": martensite_u8,
        },
        morphology_trace={
            "family": "bainite_cfb",
            "stage": "carbide_free_bainite",
            "prior_austenite_grain_count": n_grains,
            "ra_fraction": float(ra_mask.mean()),
            "ma_fraction_in_ra": float(ma_mask.sum() / max(1, ra_mask.sum())),
            "no_pointwise_carbides": True,
        },
        rendered_layers=["BAINITE", "AUSTENITE", "MARTENSITE"],
        fragment_area=int(h * w // max(1, n_grains * 3)),
    )
