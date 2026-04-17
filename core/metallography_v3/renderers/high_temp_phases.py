"""Высокотемпературные и жидкие фазы (§1.4, §1.5, §3.1).

Обслуживает:
  * austenite (§1.4, γ-Fe, равноосные + annealing twins)
  * delta_ferrite (§1.5, вермикулярные островки)
  * alpha_gamma (переход α ↔ γ)
  * gamma_cementite (γ + Fe₃C на высоких T)
  * liquid (жидкая фаза)
  * liquid_gamma (жидкость + γ-дендриты, §3.1 cast dendrites)

Phase 1 stub: регистрируется в диспетчере, но не подключается в runtime.
Реализация — отдельный sub-plan (Phase 2, рекомендован первым).
"""
from __future__ import annotations

from core.metallography_v3.renderers._common import RendererOutput
from core.metallography_v3.system_generators.base import SystemGenerationContext

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


def render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> RendererOutput:
    raise NotImplementedError(
        f"high_temp_phases renderer for stage {stage!r} not implemented yet — "
        "see whimsical-wandering-dawn.md Phase 2"
    )
