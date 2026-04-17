"""Бейнитное семейство (§2.5–2.7 справочника).

Обслуживает: upper bainite (§2.5, feathery packets), lower bainite
(§2.6, acicular laths + 60° carbide hash), carbide-free bainite (§2.7,
Si≥1.5%, nanobainite).

Phase 1 stub: регистрируется в диспетчере, но не подключается в runtime.
Реализация — отдельный sub-plan (Phase 5).
"""
from __future__ import annotations

from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import SystemGenerationContext

HANDLES_STAGES: frozenset[str] = frozenset(
    {
        "bainite_upper",
        "bainite_lower",
        "carbide_free_bainite",
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
        f"bainite renderer for stage {stage!r} not implemented yet — "
        "see whimsical-wandering-dawn.md Phase 5"
    )
