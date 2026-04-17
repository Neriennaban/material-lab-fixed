"""Белые чугуны + ледебурит (§1.6, §1.10 справочника).

Обслуживает:
  * ledeburite (Ld′ — леопардова шкура при 20°C)
  * white_cast_iron_eutectic (100% Ld′)
  * white_cast_iron_hypoeutectic (Ld′ + первичные γ-дендриты → перлит)
  * white_cast_iron_hypereutectic (Ld′ + первичный Fe₃C_I «ножи»)

Переиспользует готовые utilities проекта:
  - ``fe_c_textures.texture_ledeburite_leopard`` — leopard-матрица
    §1.6 (bright cementite ~218 + dark pearlite blobs ~60, двухмасштабный
    smooth noise threshold).
  - ``fe_c_dendrites.render_fe_c_austenite_dendrites`` — первичные
    γ-дендриты для гипоэвтектического чугуна.
  - ``fe_c_primary_cementite.render_primary_cementite_needles`` —
    длинные белые пластины Fe₃C_I для заэвтектического чугуна.

Эти хелперы остаются на своих местах (не дублируются); этот модуль —
адаптер под контракт RendererOutput + диспетчер стадий.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import SystemGenerationContext


HANDLES_STAGES: frozenset[str] = frozenset(
    {
        "ledeburite",
        "white_cast_iron_eutectic",
        "white_cast_iron_hypoeutectic",
        "white_cast_iron_hypereutectic",
    }
)


# --- tones из §1.6 ———————————————————————————————————
_LEOPARD_PEARLITE_TONE_MAX = 110  # пиксели ≤110 считаются pearlite в маске


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    # Ленивые импорты, чтобы избежать циклов при инициализации
    # renderers/__init__ ← fe_c_unified ← renderers.
    from core.metallography_v3.system_generators.fe_c_dendrites import (
        render_fe_c_austenite_dendrites,
    )
    from core.metallography_v3.system_generators.fe_c_primary_cementite import (
        render_primary_cementite_needles,
    )
    from core.metallography_v3.system_generators.fe_c_textures import (
        texture_ledeburite_leopard,
    )

    size = context.size
    seed = int(seed_split.get("seed_topology", context.seed))
    c_wt = float((context.composition_wt or {}).get("C", 4.3))
    cooling_rate = float(
        (context.thermal_summary or {}).get(
            "max_effective_cooling_rate_c_per_s", 5.0
        )
    ) or 5.0

    # Leopard-матрица — общая для всех четырёх стадий (§1.6).
    base = texture_ledeburite_leopard(size=size, seed=seed)

    if stage == "white_cast_iron_eutectic":
        return _finalize_eutectic(base, size=size, stage=stage)
    if stage == "ledeburite":
        return _finalize_eutectic(
            base,
            size=size,
            stage=stage,
            family="ledeburite_transformed",
        )
    if stage == "white_cast_iron_hypoeutectic":
        out = render_fe_c_austenite_dendrites(
            size=size,
            seed=seed + 401,
            c_wt=c_wt,
            base_image=base,
            cooling_rate_c_per_s=cooling_rate,
        )
        image = out["image"]
        dendrite_mask = np.asarray(out["dendrite_mask"], dtype=bool)
        ledeburite_mask = (~dendrite_mask).astype(np.uint8)
        pearlite_mask = dendrite_mask.astype(np.uint8)
        trace: dict[str, Any] = {
            "family": "white_cast_iron_hypoeutectic",
            "stage": stage,
            "leopard_seed": seed,
            "cooling_rate_c_per_s": cooling_rate,
        }
        trace.update(out.get("metadata", {}))
        return RendererOutput(
            image_gray=np.asarray(image, dtype=np.uint8),
            phase_masks={
                "LEDEBURITE": ledeburite_mask,
                "PEARLITE": pearlite_mask,
            },
            morphology_trace=trace,
            rendered_layers=["LEDEBURITE", "PEARLITE"],
            fragment_area=max(
                48, int(dendrite_mask.sum() // max(1, int(dendrite_mask.sum() ** 0.5 / 8)))
            ),
        )

    if stage == "white_cast_iron_hypereutectic":
        out = render_primary_cementite_needles(
            size=size,
            seed=seed + 501,
            c_wt=c_wt,
            base_image=base,
            cooling_rate_c_per_s=cooling_rate,
        )
        image = out["image"]
        needle_mask = np.asarray(out["needle_mask"], dtype=bool)
        ledeburite_mask = (~needle_mask).astype(np.uint8)
        primary_mask = needle_mask.astype(np.uint8)
        trace = {
            "family": "white_cast_iron_hypereutectic",
            "stage": stage,
            "leopard_seed": seed,
            "cooling_rate_c_per_s": cooling_rate,
        }
        trace.update(out.get("metadata", {}))
        return RendererOutput(
            image_gray=np.asarray(image, dtype=np.uint8),
            phase_masks={
                "LEDEBURITE": ledeburite_mask,
                "CEMENTITE_PRIMARY": primary_mask,
            },
            morphology_trace=trace,
            rendered_layers=["LEDEBURITE", "CEMENTITE_PRIMARY"],
            fragment_area=max(80, int(needle_mask.sum() // 4)),
        )

    raise ValueError(
        f"white_cast_iron renderer has no branch for stage {stage!r}"
    )


def _finalize_eutectic(
    base: np.ndarray,
    *,
    size: tuple[int, int],
    stage: str,
    family: str = "white_cast_iron_eutectic",
) -> RendererOutput:
    """Eutectic / ledeburite — 100% Ld′ без primary фаз.

    Разбиение на PEARLITE + CEMENTITE маски идёт по порогу яркости:
    тёмные blob'ы (≤110) → pearlite islands, остальное → эвтект. цементит.
    """
    image = np.asarray(base, dtype=np.uint8)
    pearlite_mask = (image <= _LEOPARD_PEARLITE_TONE_MAX).astype(np.uint8)
    cementite_mask = (image > _LEOPARD_PEARLITE_TONE_MAX).astype(np.uint8)
    return RendererOutput(
        image_gray=image,
        phase_masks={
            "LEDEBURITE": np.ones(size, dtype=np.uint8),
            "PEARLITE": pearlite_mask,
            "CEMENTITE": cementite_mask,
        },
        morphology_trace={
            "family": family,
            "stage": stage,
            "pearlite_island_fraction": float(pearlite_mask.mean()),
            "cementite_matrix_fraction": float(cementite_mask.mean()),
        },
        rendered_layers=["LEDEBURITE", "PEARLITE", "CEMENTITE"],
        fragment_area=max(48, int(size[0] * size[1] * 0.04)),
    )
