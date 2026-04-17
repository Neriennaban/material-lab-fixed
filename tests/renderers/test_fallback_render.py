"""Phase 9 regression: _fallback_render обслуживает unknown stages.

Мотивация — CodeRabbit review на PR #12 (P1):
`resolve_fe_c_stage` пропускает custom/typo stage names verbatim в
`render_fe_c_unified`. До Phase 9 эту роль играл `_generic_render`;
после cleanup функция удалена — unknown stages падают в
`_fallback_render`. Если этот путь бросает exception, `generate_fe_c`
catches silently и делегирует в legacy `run_phase_map_system`, что
меняет pipeline/metadata неожиданно. Этот тест гарантирует, что
fallback остаётся работоспособным.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import (
    _fallback_render,
    render_fe_c_unified,
)


def _make_ctx(stage: str, seed: int = 2026, size=(128, 128)):
    return SystemGenerationContext(
        size=size,
        seed=seed,
        inferred_system="fe-c",
        stage=stage,
        phase_fractions={"FERRITE": 0.6, "PEARLITE": 0.4},
        composition_wt={"Fe": 99.6, "C": 0.4},
        processing=ProcessingState(
            temperature_c=20.0,
            cooling_mode="equilibrium",
        ),
        thermal_summary={"max_effective_cooling_rate_c_per_s": 5.0},
    )


class FallbackRenderTests(unittest.TestCase):
    def test_fallback_direct_call_returns_valid_output(self) -> None:
        """Прямой вызов _fallback_render должен вернуть валидный uint8
        image и phase_masks без exception."""
        ctx = _make_ctx("custom_unknown_stage_xyz")
        seed_split = {
            "seed_topology": 1,
            "seed_boundary": 2,
            "seed_particles": 3,
            "seed_lamella": 4,
            "seed_noise": 5,
        }
        image_gray, phase_masks, rendered_layers, fragment_area, trace = (
            _fallback_render(
                context=ctx,
                stage="custom_unknown_stage_xyz",
                normalized_fractions={"FERRITE": 0.6, "PEARLITE": 0.4},
                seed_split=seed_split,
            )
        )
        self.assertEqual(image_gray.shape, (128, 128))
        self.assertEqual(image_gray.dtype, np.uint8)
        self.assertIn("FERRITE", phase_masks)
        self.assertIn("PEARLITE", phase_masks)
        self.assertGreater(int(phase_masks["FERRITE"].sum()), 0)
        self.assertGreater(int(phase_masks["PEARLITE"].sum()), 0)
        self.assertEqual(trace.get("family"), "fallback_generic")
        self.assertEqual(trace.get("stage"), "custom_unknown_stage_xyz")
        self.assertGreater(fragment_area, 0)
        self.assertIn("FERRITE", rendered_layers)

    def test_unknown_stage_through_dispatcher_uses_fallback(self) -> None:
        """Полный path: render_fe_c_unified с unknown stage → fallback.

        НЕ должен бросать ValueError/TypeError — должен вернуть
        корректный SystemGenerationResult с family='fallback_generic'.
        """
        ctx = _make_ctx("some_typo_stage_name")
        out = render_fe_c_unified(ctx)
        self.assertEqual(out.image_gray.shape, (128, 128))
        self.assertEqual(out.image_gray.dtype, np.uint8)
        trace = out.metadata.get("fe_c_phase_render", {}).get(
            "morphology_trace", {}
        )
        self.assertEqual(trace.get("family"), "fallback_generic")


if __name__ == "__main__":
    unittest.main()
