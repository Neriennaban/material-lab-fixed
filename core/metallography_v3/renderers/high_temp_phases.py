"""Высокотемпературные и жидкие фазы (§1.4, §1.5, §3.1).

Обслуживает:
  * austenite (§1.4, γ-Fe, равноосные + annealing twins)
  * delta_ferrite (§1.5, вермикулярные островки)
  * alpha_gamma (переход α ↔ γ)
  * gamma_cementite (γ + Fe₃C на высоких T)
  * liquid (жидкая фаза)
  * liquid_gamma (жидкость + γ-дендриты, §3.1 cast dendrites)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.generator_grains import generate_grain_structure
from core.metallography_v3.realism_utils import (
    boundary_mask_from_labels,
    build_twins_from_labels,
    low_frequency_field,
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
        "austenite",
        "delta_ferrite",
        "alpha_gamma",
        "gamma_cementite",
        "liquid",
        "liquid_gamma",
    }
)


# --- tones (§1.4, §1.5, §3.1) —————————————————————————————————
_TONE_AUSTENITE_INTERIOR = 227.0
_TONE_AUSTENITE_BOUNDARY = 88.0
_TONE_AUSTENITE_TWIN = 200.0
_TONE_FERRITE = 220.0
_TONE_DELTA_ISLAND = 143.0
_TONE_LIQUID_MIN = 208.0
_TONE_LIQUID_MAX = 245.0
_TONE_CEMENTITE_PARTICLE = 35.0
_TONE_DENDRITE_AXIS = 215.0
_TONE_INTERDENDRITIC = 80.0


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    if stage == "austenite":
        return _render_austenite(context=context, seed_split=seed_split)
    if stage == "delta_ferrite":
        return _render_delta_ferrite(
            context=context, seed_split=seed_split, phase_fractions=phase_fractions
        )
    if stage == "alpha_gamma":
        return _render_alpha_gamma(
            context=context, seed_split=seed_split, phase_fractions=phase_fractions
        )
    if stage == "gamma_cementite":
        return _render_gamma_cementite(
            context=context, seed_split=seed_split, phase_fractions=phase_fractions
        )
    if stage == "liquid":
        return _render_liquid(context=context, seed_split=seed_split)
    if stage == "liquid_gamma":
        return _render_liquid_gamma(
            context=context, seed_split=seed_split, phase_fractions=phase_fractions
        )
    raise NotImplementedError(
        f"high_temp_phases renderer has no branch for stage {stage!r}"
    )


# --- helpers —————————————————————————————————————————————————————————


def _grain_labels(size: tuple[int, int], seed: int, mean_size_px: float) -> np.ndarray:
    """Равноосные Voronoi-зёрна."""
    out = generate_grain_structure(
        size=size,
        seed=int(seed),
        mean_grain_size_px=max(18.0, float(mean_size_px)),
        grain_size_jitter=0.22,
        boundary_width_px=1,
        boundary_contrast=0.0,
        elongation=1.0,
    )
    return out["labels"]


def _per_grain_interior_tone(
    labels: np.ndarray, base_tone: float, seed: int, jitter: float = 3.5
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    count = int(labels.max()) + 1
    per_grain = base_tone + rng.normal(0.0, jitter, size=count).astype(np.float32)
    return per_grain[labels].astype(np.float32)


def _finalize(image_f32: np.ndarray) -> np.ndarray:
    img = rescale_to_u8(np.clip(image_f32, 0.0, 255.0), lo=20.0, hi=250.0)
    return soft_unsharp(img, amount=0.25)


# --- austenite (§1.4) ——————————————————————————————————————


def _render_austenite(
    *, context: SystemGenerationContext, seed_split: dict[str, int]
) -> RendererOutput:
    size = context.size
    seed = seed_split.get("seed_topology", context.seed)
    labels = _grain_labels(size, seed, mean_size_px=48.0)
    boundaries = boundary_mask_from_labels(labels, width=2)

    # Per-grain mean brightness с небольшим джиттером для «orientation response».
    img = _per_grain_interior_tone(
        labels, _TONE_AUSTENITE_INTERIOR, seed + 11, jitter=3.5
    )

    # Двойники отжига (parallel straight bands внутри зёрен).
    twin_mask = build_twins_from_labels(
        labels=labels,
        seed=seed + 23,
        fraction_of_grains=0.45,
        spacing_px=14.0,
        width_px=1.8,
    )
    rng_twins = np.random.default_rng(int(seed) + 31)
    twin_tone = _TONE_AUSTENITE_TWIN + rng_twins.normal(0.0, 2.0, img.shape).astype(
        np.float32
    )
    img = np.where(twin_mask, twin_tone, img)

    # Границы зёрен — тёмные тонкие линии.
    img = np.where(boundaries > 0, _TONE_AUSTENITE_BOUNDARY, img)

    # Micro-noise для «этченного» вида.
    img = img + (
        multiscale_noise(size=size, seed=seed + 41, scales=((18.0, 0.6), (5.0, 0.4)))
        - 0.5
    ) * 6.0

    image_gray = _finalize(img)

    austenite_mask = np.ones(size, dtype=np.uint8)
    austenite_mask[boundaries > 0] = 0
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={"AUSTENITE": austenite_mask},
        morphology_trace={
            "family": "high_temp_phase",
            "stage": "austenite",
            "grain_count": int(labels.max()) + 1,
            "twin_pixels": int(twin_mask.sum()),
        },
        rendered_layers=["AUSTENITE"],
        fragment_area=int(size[0] * size[1] // max(1, int(labels.max()) + 1)),
    )


# --- δ-ferrite (§1.5) ————————————————————————————————————————


def _render_delta_ferrite(
    *,
    context: SystemGenerationContext,
    seed_split: dict[str, int],
    phase_fractions: dict[str, float],
) -> RendererOutput:
    size = context.size
    seed = seed_split.get("seed_topology", context.seed)

    base_result = _render_austenite(context=context, seed_split=seed_split)
    img = base_result.image_gray.astype(np.float32)

    # Анизотропный шум с вытяжкой по X.
    h, w = size
    stretch = 3.0
    src_w = max(8, int(round(w / stretch)))
    anis_noise_raw = multiscale_noise(
        size=(h, src_w),
        seed=seed + 53,
        scales=((26.0, 0.7), (9.0, 0.3)),
    )
    if ndimage is not None:
        anis = ndimage.zoom(anis_noise_raw, zoom=(1.0, w / src_w), order=1)[:h, :w]
    else:
        anis = np.repeat(anis_noise_raw, int(round(stretch)), axis=1)[:h, :w]
    anis = normalize01(anis.astype(np.float32))

    # Целевая доля δ из §1.5: 2-15%.
    target_frac = float(phase_fractions.get("DELTA_FERRITE", 0.15))
    target_frac = max(0.02, min(0.15, target_frac))
    threshold = float(np.quantile(anis, 1.0 - target_frac))
    island_mask = anis >= threshold

    if ndimage is not None:
        island_mask = ndimage.binary_opening(island_mask, iterations=1)

    # Островки — тёмнее аустенитной матрицы, с микровариацией по «зёрнам».
    delta_tone_field = _per_grain_interior_tone(
        _grain_labels(size, seed + 67, mean_size_px=32.0),
        _TONE_DELTA_ISLAND,
        seed + 71,
        jitter=4.5,
    )
    img = np.where(island_mask, delta_tone_field, img)

    image_gray = _finalize(img)

    austenite_mask = (~island_mask).astype(np.uint8)
    delta_mask = island_mask.astype(np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={"DELTA_FERRITE": delta_mask, "AUSTENITE": austenite_mask},
        morphology_trace={
            "family": "high_temp_phase",
            "stage": "delta_ferrite",
            "delta_fraction_actual": float(island_mask.mean()),
            "anisotropy_ratio": stretch,
        },
        rendered_layers=["AUSTENITE", "DELTA_FERRITE"],
        fragment_area=max(1, int(island_mask.sum() // 8)),
    )


# --- liquid (§3.1) —————————————————————————————————————————————


def _render_liquid(
    *, context: SystemGenerationContext, seed_split: dict[str, int]
) -> RendererOutput:
    size = context.size
    seed = seed_split.get("seed_noise", context.seed)

    low = low_frequency_field(size, seed=seed + 3, sigma=28.0)
    noise = multiscale_noise(size=size, seed=seed + 13, scales=((40.0, 0.6), (12.0, 0.4)))
    field = normalize01(0.6 * low + 0.4 * noise)
    img = _TONE_LIQUID_MIN + field * (_TONE_LIQUID_MAX - _TONE_LIQUID_MIN)

    image_gray = _finalize(img)
    liquid_mask = np.ones(size, dtype=np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={"LIQUID": liquid_mask},
        morphology_trace={
            "family": "high_temp_phase",
            "stage": "liquid",
            "tone_range": [_TONE_LIQUID_MIN, _TONE_LIQUID_MAX],
        },
        rendered_layers=["LIQUID"],
        fragment_area=size[0] * size[1],
    )


# --- liquid_gamma (§3.1 dendrites) ——————————————————————————————


def _render_liquid_gamma(
    *,
    context: SystemGenerationContext,
    seed_split: dict[str, int],
    phase_fractions: dict[str, float],
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = seed_split.get("seed_topology", context.seed)

    liq = _render_liquid(context=context, seed_split=seed_split).image_gray.astype(
        np.float32
    )

    rng = np.random.default_rng(int(seed) + 97)
    dendrite_mask = np.zeros(size, dtype=bool)
    n_trunks = int(rng.integers(3, 8))
    trunk_len = int(0.75 * min(h, w))
    arm_spacing_px = float(rng.integers(8, 18))
    for _ in range(n_trunks):
        cx = int(rng.integers(int(0.15 * w), int(0.85 * w)))
        cy = int(rng.integers(int(0.15 * h), int(0.85 * h)))
        theta = float(rng.uniform(0.0, math.pi))
        _draw_dendrite(
            mask=dendrite_mask,
            start=(cy, cx),
            trunk_len=trunk_len,
            trunk_angle=theta,
            arm_spacing=arm_spacing_px,
            arm_max_len=int(0.35 * min(h, w)),
            rng=rng,
        )
    if ndimage is not None:
        dendrite_mask = ndimage.binary_dilation(dendrite_mask, iterations=1)

    img = liq.copy()
    img = np.where(dendrite_mask, _TONE_DENDRITE_AXIS, img)

    if ndimage is not None:
        halo = ndimage.binary_dilation(dendrite_mask, iterations=3) & ~dendrite_mask
        img = np.where(
            halo,
            _TONE_INTERDENDRITIC + (img - _TONE_INTERDENDRITIC) * 0.4,
            img,
        )
        img = ndimage.gaussian_filter(img, sigma=1.8)

    image_gray = _finalize(img)

    gamma_mask = dendrite_mask.astype(np.uint8)
    liquid_mask = (~dendrite_mask).astype(np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={"AUSTENITE": gamma_mask, "LIQUID": liquid_mask},
        morphology_trace={
            "family": "high_temp_phase",
            "stage": "liquid_gamma",
            "n_trunks": n_trunks,
            "sdas_proxy_px": arm_spacing_px,
            "dendrite_fraction": float(dendrite_mask.mean()),
        },
        rendered_layers=["LIQUID", "AUSTENITE"],
        fragment_area=int(dendrite_mask.sum() // max(1, n_trunks)),
    )


def _draw_dendrite(
    *,
    mask: np.ndarray,
    start: tuple[int, int],
    trunk_len: int,
    trunk_angle: float,
    arm_spacing: float,
    arm_max_len: int,
    rng: np.random.Generator,
) -> None:
    """L-система: ствол + перпендикулярные ветви уменьшающейся длины."""
    h, w = mask.shape
    cy, cx = start
    cos_t = math.cos(trunk_angle)
    sin_t = math.sin(trunk_angle)
    cos_p = -sin_t
    sin_p = cos_t

    half = trunk_len // 2
    for s in range(-half, half + 1):
        y = int(cy + s * sin_t)
        x = int(cx + s * cos_t)
        if 0 <= y < h and 0 <= x < w:
            mask[y, x] = True

    step = int(max(4.0, arm_spacing))
    for s in range(-half, half + 1, step):
        y0 = int(cy + s * sin_t)
        x0 = int(cx + s * cos_t)
        if not (0 <= y0 < h and 0 <= x0 < w):
            continue
        frac = 1.0 - abs(s) / max(1.0, half)
        arm_len = int(arm_max_len * (0.4 + 0.6 * frac))
        for sign in (-1, +1):
            for t in range(0, arm_len):
                y = int(y0 + sign * t * sin_p)
                x = int(x0 + sign * t * cos_p)
                if 0 <= y < h and 0 <= x < w:
                    mask[y, x] = True
                else:
                    break
                if arm_len > 20 and t % max(4, int(arm_spacing * 0.6)) == 0 and t > 0:
                    tert_len = int(0.35 * arm_len * (1.0 - t / arm_len))
                    for sign2 in (-1, +1):
                        for u in range(0, tert_len):
                            y2 = int(y + sign2 * u * cos_t)
                            x2 = int(x + sign2 * u * sin_t)
                            if 0 <= y2 < h and 0 <= x2 < w:
                                mask[y2, x2] = True


# --- alpha_gamma ——————————————————————————————————————————————


def _render_alpha_gamma(
    *,
    context: SystemGenerationContext,
    seed_split: dict[str, int],
    phase_fractions: dict[str, float],
) -> RendererOutput:
    size = context.size
    seed = seed_split.get("seed_topology", context.seed)
    labels = _grain_labels(size, seed, mean_size_px=40.0)
    boundaries = boundary_mask_from_labels(labels, width=2)

    rng = np.random.default_rng(int(seed) + 17)
    count = int(labels.max()) + 1
    target_ferrite_frac = float(phase_fractions.get("FERRITE", 0.55))
    target_ferrite_frac = max(0.05, min(0.95, target_ferrite_frac))
    is_ferrite = rng.random(count) < target_ferrite_frac

    jitter = rng.normal(0.0, 3.0, size=labels.shape).astype(np.float32)
    interior_tone = np.where(
        is_ferrite[labels],
        _TONE_FERRITE + jitter,
        _TONE_AUSTENITE_INTERIOR + jitter,
    )
    img = interior_tone.astype(np.float32)
    img = np.where(boundaries > 0, _TONE_AUSTENITE_BOUNDARY, img)
    img = img + (
        multiscale_noise(size=size, seed=seed + 27, scales=((16.0, 0.6), (5.0, 0.4)))
        - 0.5
    ) * 5.0

    image_gray = _finalize(img)
    ferrite_pix = (is_ferrite[labels]) & (boundaries == 0)
    austenite_pix = (~is_ferrite[labels]) & (boundaries == 0)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "FERRITE": ferrite_pix.astype(np.uint8),
            "AUSTENITE": austenite_pix.astype(np.uint8),
        },
        morphology_trace={
            "family": "high_temp_phase",
            "stage": "alpha_gamma",
            "ferrite_grain_count": int(is_ferrite.sum()),
        },
        rendered_layers=["FERRITE", "AUSTENITE"],
        fragment_area=int(size[0] * size[1] // max(1, count)),
    )


# --- gamma_cementite ——————————————————————————————————————————


def _render_gamma_cementite(
    *,
    context: SystemGenerationContext,
    seed_split: dict[str, int],
    phase_fractions: dict[str, float],
) -> RendererOutput:
    size = context.size
    h, w = size
    seed = seed_split.get("seed_topology", context.seed)

    aus = _render_austenite(context=context, seed_split=seed_split)
    img = aus.image_gray.astype(np.float32)
    labels = _grain_labels(size, seed, mean_size_px=48.0)
    boundaries = boundary_mask_from_labels(labels, width=2)

    rng = np.random.default_rng(int(seed) + 83)
    target_frac = float(phase_fractions.get("CEMENTITE", 0.28))
    area = h * w
    mean_radius_px = 3.5
    particle_area = math.pi * mean_radius_px ** 2
    n_particles = int(target_frac * area / particle_area)
    n_particles = max(50, min(n_particles, 4000))

    boundary_ys, boundary_xs = np.where(boundaries > 0)
    has_boundaries = len(boundary_xs) > 0

    cementite_mask = np.zeros(size, dtype=bool)
    for _ in range(n_particles):
        if has_boundaries and rng.random() < 0.75:
            idx = int(rng.integers(0, len(boundary_xs)))
            cy = int(boundary_ys[idx])
            cx = int(boundary_xs[idx])
        else:
            cy = int(rng.integers(0, h))
            cx = int(rng.integers(0, w))
        r = float(rng.lognormal(mean=math.log(mean_radius_px), sigma=0.3))
        r = max(1.5, min(8.0, r))
        y0 = max(0, int(cy - r * 1.2))
        y1 = min(h, int(cy + r * 1.2 + 1))
        x0 = max(0, int(cx - r * 1.2))
        x1 = min(w, int(cx + r * 1.2 + 1))
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        cementite_mask[y0:y1, x0:x1] |= disk

    img = np.where(cementite_mask, _TONE_CEMENTITE_PARTICLE, img)
    if ndimage is not None:
        img = ndimage.gaussian_filter(img, sigma=0.6)

    image_gray = _finalize(img)
    austenite_mask = (~cementite_mask & (boundaries == 0)).astype(np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "CEMENTITE": cementite_mask.astype(np.uint8),
            "AUSTENITE": austenite_mask,
        },
        morphology_trace={
            "family": "high_temp_phase",
            "stage": "gamma_cementite",
            "cementite_fraction_actual": float(cementite_mask.mean()),
            "n_particles": n_particles,
        },
        rendered_layers=["AUSTENITE", "CEMENTITE"],
        fragment_area=int(particle_area),
    )
