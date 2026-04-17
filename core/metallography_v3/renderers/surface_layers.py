"""Поверхностные композиционные слои (§3.2–3.3 справочника).

Обслуживает:
  * decarburized_layer (§3.2, градиент обезуглероживания FFD→MAD→core)
  * carburized_layer (§3.3, гра_C (surface)→core + мартенсит после закалки)

Принципиально иной рендер-контракт — построчная композиция по нормали к
поверхности. Поэтому модуль помечен флагом
``REQUIRES_SURFACE_COMPOSITION = True`` — в Phase 8 sub-plan'е диспетчер
получит отдельную ветку для таких модулей.

Phase 1 stub: регистрируется в диспетчере, но не подключается в runtime.
Реализация — отдельный sub-plan (Phase 8).
"""
from __future__ import annotations

from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import SystemGenerationContext

HANDLES_STAGES: frozenset[str] = frozenset(
    {
        "decarburized_layer",
        "carburized_layer",
    }
)

# Флаг для будущего отдельного dispatch-branch в fe_c_unified.
# На Phase 1 не используется, но проверяется тестом test_dispatch_table.
REQUIRES_SURFACE_COMPOSITION: bool = True


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    raise NotImplementedError(
        f"surface_layers renderer for stage {stage!r} not implemented yet — "
        "see whimsical-wandering-dawn.md Phase 8"
    )
