"""Закалочные продукты мелкопластинчатого перлита (§2.8–2.9 справочника).

Отделены от tempered — это прямые продукты закалки через перегиб
С-кривой ТТТ, не отпуск.

Обслуживает:
  * troostite_quench (§2.8, S₀ ≈ 0.1 μm — не разрешается, "чёрные кляксы")
  * sorbite_quench (§2.9, S₀ 0.2–0.3 μm — различимая штриховка)

Phase 1 stub: регистрируется в диспетчере, но не подключается в runtime.
Реализация — отдельный sub-plan (Phase 6).
"""
from __future__ import annotations

from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import SystemGenerationContext

HANDLES_STAGES: frozenset[str] = frozenset(
    {
        "troostite_quench",
        "sorbite_quench",
    }
)


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    raise NotImplementedError(
        f"quench_products renderer for stage {stage!r} not implemented yet — "
        "see whimsical-wandering-dawn.md Phase 6"
    )
