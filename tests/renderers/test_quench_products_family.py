"""Phase 6 — тесты quench_products renderer'а.

Семейство: troostite_quench (§2.8, не разрешается, isotropic blobs),
sorbite_quench (§2.9, резрешается, per-colony ориентация штриховки).
"""
from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


_STAGE_DEFAULTS = {
    "troostite_quench": (
        {"TROOSTITE": 0.88, "CEMENTITE": 0.12},
        {"Fe": 99.4, "C": 0.6},
        550.0,
    ),
    "sorbite_quench": (
        {"SORBITE": 0.84, "CEMENTITE": 0.16},
        {"Fe": 99.45, "C": 0.55},
        620.0,
    ),
}


def _make_ctx(stage: str, seed: int = 2026, size=(192, 192)):
    fractions, composition, temperature_c = _STAGE_DEFAULTS[stage]
    return SystemGenerationContext(
        size=size,
        seed=seed,
        inferred_system="fe-c",
        stage=stage,
        phase_fractions=fractions,
        composition_wt=composition,
        processing=ProcessingState(
            temperature_c=temperature_c,
            cooling_mode="quench_oil",
        ),
        thermal_summary={"max_effective_cooling_rate_c_per_s": 100.0},
    )


class QuenchProductsFamilyTests(unittest.TestCase):
    def test_all_stages_render(self) -> None:
        for stage in _STAGE_DEFAULTS:
            with self.subTest(stage=stage):
                out = render_fe_c_unified(_make_ctx(stage))
                self.assertEqual(out.image_gray.shape, (192, 192))
                self.assertEqual(out.image_gray.dtype, np.uint8)
                self.assertTrue(out.phase_masks)

    def test_troostite_is_dark_and_isotropic(self) -> None:
        """§2.8 — тёмные кляксы без направленной анизотропии."""
        out = render_fe_c_unified(_make_ctx("troostite_quench"))
        img = out.image_gray.astype(np.float32)
        mean = float(img.mean())
        self.assertLess(
            mean, 130.0,
            f"troostite quench: too bright ({mean:.1f}), should be dark",
        )
        # Изотропность: разница между градиентами по X и Y невелика.
        grad_x = float(np.abs(np.diff(img, axis=1)).mean())
        grad_y = float(np.abs(np.diff(img, axis=0)).mean())
        diff = abs(grad_x - grad_y)
        self.assertLess(
            diff, 3.5,
            f"troostite quench: anisotropy detected "
            f"(grad_x={grad_x:.2f}, grad_y={grad_y:.2f}, diff={diff:.2f})",
        )

    def test_sorbite_has_colony_striping(self) -> None:
        """§2.9 — различимая штриховка в колониях, среднее вокруг ~115."""
        out = render_fe_c_unified(_make_ctx("sorbite_quench"))
        img = out.image_gray
        mean = float(img.mean())
        std = float(img.std())
        self.assertGreater(
            mean, 80.0,
            f"sorbite quench: too dark (mean={mean:.1f})",
        )
        self.assertLess(
            mean, 180.0,
            f"sorbite quench: too bright (mean={mean:.1f})",
        )
        # Вариация отражает ламельную структуру колоний.
        self.assertGreater(
            std, 15.0,
            f"sorbite quench: lamella contrast too weak (std={std:.1f})",
        )

    def test_family_trace_strings(self) -> None:
        expected = {
            "troostite_quench": "troostite_quench",
            "sorbite_quench": "sorbite_quench",
        }
        for stage, fam in expected.items():
            with self.subTest(stage=stage):
                out = render_fe_c_unified(_make_ctx(stage))
                trace = out.metadata.get("fe_c_phase_render", {}).get(
                    "morphology_trace", {}
                )
                self.assertEqual(
                    trace.get("family"), fam,
                    f"{stage}: family={trace.get('family')!r}, expected {fam!r}",
                )

    def test_determinism(self) -> None:
        for stage in _STAGE_DEFAULTS:
            with self.subTest(stage=stage):
                a = render_fe_c_unified(_make_ctx(stage, seed=777)).image_gray
                b = render_fe_c_unified(_make_ctx(stage, seed=777)).image_gray
                self.assertTrue(np.array_equal(a, b), f"{stage} not deterministic")


if __name__ == "__main__":
    unittest.main()
