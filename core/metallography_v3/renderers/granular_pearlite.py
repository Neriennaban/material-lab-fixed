"""Зернистый (сфероидизированный) перлит (§1.9 справочника).

Обслуживает: granular_pearlite — глобули Fe₃C 0.1–2 μm в ферритной
матрице 5–20 μm. Типичная структура после сфероидизации (780°C, 3ч + медл.
охл. 15°C/ч) — сталь ШХ15 для подшипников, У10–У12 в состоянии
поставки.

Планируется переиспользовать ``generate_pure_ferrite_micrograph`` с
меньшим ``mean_eq_d_px`` + Poisson-disk глобули поверх.

Phase 1 stub: регистрируется в диспетчере, но не подключается в runtime.
Реализация — отдельный sub-plan (Phase 8).
"""
from __future__ import annotations

from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import SystemGenerationContext

HANDLES_STAGES: frozenset[str] = frozenset({"granular_pearlite"})


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    raise NotImplementedError(
        f"granular_pearlite renderer for stage {stage!r} not implemented yet — "
        "see whimsical-wandering-dawn.md Phase 8"
    )
