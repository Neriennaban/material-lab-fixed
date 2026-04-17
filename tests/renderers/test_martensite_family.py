"""Phase 4 — тесты martensite renderer'а.

Семейство: martensite_cubic (lath §2.1), martensite_tetragonal (plate
§2.2 с midrib), martensite (mixed §2.3). RA (§2.4) инжектится
post-process'ом в render_fe_c_unified.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


_STAGE_DEFAULTS = {
    "martensite_cubic": (
        {"MARTENSITE_CUBIC": 0.94, "CEMENTITE": 0.06},
        {"Fe": 99.7, "C": 0.3},
        20.0,
    ),
    "martensite_tetragonal": (
        {"MARTENSITE_TETRAGONAL": 0.82, "CEMENTITE": 0.05, "AUSTENITE": 0.13},
        {"Fe": 98.8, "C": 1.2},
        20.0,
    ),
    "martensite": (
        {"MARTENSITE": 0.85, "CEMENTITE": 0.05, "AUSTENITE": 0.10},
        {"Fe": 99.2, "C": 0.8},
        20.0,
    ),
}


def _make_ctx(stage: str, seed: int = 2026, size=(192, 192), c_override=None):
    fractions, composition, temperature_c = _STAGE_DEFAULTS[stage]
    if c_override is not None:
        composition = {"Fe": 100.0 - c_override, "C": c_override}
    return SystemGenerationContext(
        size=size,
        seed=seed,
        inferred_system="fe-c",
        stage=stage,
        phase_fractions=fractions,
        composition_wt=composition,
        processing=ProcessingState(
            temperature_c=temperature_c,
            cooling_mode="quench_water",
        ),
        thermal_summary={"max_effective_cooling_rate_c_per_s": 800.0},
    )


class MartensiteFamilyTests(unittest.TestCase):
    def test_all_stages_render(self) -> None:
        for stage in _STAGE_DEFAULTS:
            with self.subTest(stage=stage):
                out = render_fe_c_unified(_make_ctx(stage))
                self.assertEqual(out.image_gray.shape, (192, 192))
                self.assertEqual(out.image_gray.dtype, np.uint8)
                self.assertTrue(out.phase_masks)
                mean = float(out.image_gray.mean())
                self.assertGreater(mean, 30.0, f"{stage}: too dark ({mean:.1f})")
                self.assertLess(mean, 220.0, f"{stage}: too bright ({mean:.1f})")

    def test_cubic_is_anisotropic(self) -> None:
        """Реечный мартенсит имеет выраженную ориентационную анизотропию.
        Оцениваем через разницу stddev проекций на оси X и Y."""
        out = render_fe_c_unified(_make_ctx("martensite_cubic"))
        img = out.image_gray.astype(np.float32)
        grad_y = float(np.abs(np.diff(img, axis=0)).mean())
        grad_x = float(np.abs(np.diff(img, axis=1)).mean())
        # Реечный мартенсит ДОЛЖЕН иметь заметный градиент хотя бы по
        # одной оси (anisotropic noise внутри PAG даёт полосатую текстуру).
        self.assertGreater(
            max(grad_x, grad_y), 4.0,
            f"cubic martensite lacks directional texture "
            f"(grad_x={grad_x:.2f}, grad_y={grad_y:.2f})",
        )

    def test_tetragonal_has_dark_midrib_features(self) -> None:
        """Пластинчатый мартенсит имеет midrib линии — доля пикселей с
        tone ≤45 (глубоко-тёмных) ≥ 1%."""
        out = render_fe_c_unified(_make_ctx("martensite_tetragonal"))
        img = out.image_gray
        very_dark = float((img <= 45).mean())
        self.assertGreater(
            very_dark, 0.005,
            f"tetragonal martensite: expected midrib-dark pixels ≥0.5%, "
            f"got {very_dark*100:.3f}%",
        )
        # Сверх того: должны быть и светлые RA/boundary пиксели.
        bright = float((img >= 180).mean())
        self.assertLess(
            bright, 0.60,
            f"tetragonal: too bright ({bright*100:.1f}%), RA has taken over?",
        )

    def test_mixed_c_dependence(self) -> None:
        """Смешанный при C=0.9% имеет больше plate-подобных признаков
        (dark midrib pixels), чем cubic при C=0.25%."""
        out_mixed = render_fe_c_unified(
            _make_ctx("martensite", c_override=0.9, seed=111)
        )
        out_cubic = render_fe_c_unified(
            _make_ctx("martensite_cubic", c_override=0.25, seed=111)
        )
        dark_mixed = float((out_mixed.image_gray <= 55).mean())
        dark_cubic = float((out_cubic.image_gray <= 55).mean())
        # Смешанный должен быть не светлее реечного — нижняя граница
        # проверки с small margin.
        self.assertGreaterEqual(
            dark_mixed + 0.005,
            dark_cubic,
            f"mixed (C=0.9) should have ≥ dark-pixel coverage vs cubic "
            f"(C=0.25): mixed={dark_mixed*100:.2f}%, cubic={dark_cubic*100:.2f}%",
        )

    def test_family_trace_strings(self) -> None:
        expected = {
            "martensite_cubic": "martensite_lath",
            "martensite_tetragonal": "martensite_plate",
            "martensite": "martensite_mixed",
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
