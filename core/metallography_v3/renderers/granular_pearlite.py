"""Зернистый (сфероидизированный) перлит (§1.9 справочника).

Обслуживает: granular_pearlite — глобули Fe₃C 0.1-2 μm в ферритной
матрице 5-20 μm. Типичная структура после сфероидизации (780°C, 3ч +
медл. охлаждение 15°C/ч) — сталь ШХ15 (шарикоподшипниковая),
У10-У12 в состоянии поставки.

Реализация:
- `generate_pure_ferrite_micrograph` для ферритной матрицы с мелкими
  зёрнами (mean_eq_d_px=12).
- Poisson-disk глобули Fe₃C с lognormal радиусом, биас к границам
  ферритных зёрен.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.metallography_v3.pure_ferrite_generator import (
    generate_pure_ferrite_micrograph,
)
from core.metallography_v3.realism_utils import (
    boundary_mask_from_labels,
    rescale_to_u8,
)
from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import (
    SystemGenerationContext,
    soft_unsharp,
)


HANDLES_STAGES: frozenset[str] = frozenset({"granular_pearlite"})


_TONE_FERRITE_MATRIX = 210.0
_TONE_CEMENTITE_GLOBULE = 240.0  # nital bright (Fe₃C не травится)
_TONE_CEMENTITE_CORE = 160.0  # для контраста в центре globule
_TONE_BOUNDARY = 105.0


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

    # Ферритная матрица через стандартный генератор.
    ferrite_out = generate_pure_ferrite_micrograph(
        size=size,
        seed=seed,
        mean_eq_d_px=12.0,  # §1.9: 5-20 μm
        size_sigma=0.25,
        relax_iter=1,
        boundary_width_px=1.2,
        boundary_depth=0.10,
        blur_sigma_px=0.4,
    )
    img = ferrite_out["image_gray"].astype(np.float32)
    labels = np.asarray(ferrite_out["labels"])
    boundaries = boundary_mask_from_labels(labels, width=1)

    # Poisson-disk глобули Fe₃C.
    rng = np.random.default_rng(seed + 67)
    # Плотность ≈0.035/px² → на 192×192 ~1290 глобул.
    n_globules = max(200, int(0.035 * h * w))
    ys_b, xs_b = np.where(boundaries)
    has_b = len(xs_b) > 0

    globule_mask = np.zeros(size, dtype=bool)
    for _ in range(n_globules):
        # 55% на границах, 45% внутри зерна.
        if has_b and rng.random() < 0.55:
            idx = int(rng.integers(0, len(xs_b)))
            cy = int(ys_b[idx])
            cx = int(xs_b[idx])
        else:
            cy = int(rng.integers(0, h))
            cx = int(rng.integers(0, w))
        r = float(rng.lognormal(mean=math.log(1.2), sigma=0.3))
        r = max(0.7, min(3.0, r))
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
        core = dist2 <= (r * 0.5) ** 2
        # Яркая окантовка + чуть темнее центр (слабый 3D-эффект).
        img_patch = img[y0:y1, x0:x1]
        img_patch[disk] = _TONE_CEMENTITE_GLOBULE
        img_patch[core] = _TONE_CEMENTITE_CORE
        img[y0:y1, x0:x1] = img_patch
        globule_mask[y0:y1, x0:x1] |= disk

    image_gray = rescale_to_u8(np.clip(img, 0.0, 255.0), lo=30.0, hi=250.0)
    image_gray = soft_unsharp(image_gray, amount=0.20)

    ferrite_mask = (~globule_mask).astype(np.uint8)
    return RendererOutput(
        image_gray=image_gray,
        phase_masks={
            "FERRITE": ferrite_mask,
            "CEMENTITE": globule_mask.astype(np.uint8),
        },
        morphology_trace={
            "family": "granular_pearlite",
            "stage": "granular_pearlite",
            "ferrite_grain_count": int(labels.max()) + 1,
            "globule_count": n_globules,
            "globule_coverage": float(globule_mask.mean()),
            "boundary_bias": 0.55,
        },
        rendered_layers=["FERRITE", "CEMENTITE"],
        fragment_area=int(h * w // max(1, n_globules)),
    )
