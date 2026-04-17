"""Phase 2 — тесты high_temp_phases renderer'а."""
from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


_STAGE_DEFAULTS = {
    "austenite": ({"AUSTENITE": 1.0}, {"Fe": 99.2, "C": 0.8}, 900.0),
    "delta_ferrite": (
        {"DELTA_FERRITE": 0.15, "AUSTENITE": 0.85},
        {"Fe": 99.95, "C": 0.05},
        1450.0,
    ),
    "liquid": ({"LIQUID": 1.0}, {"Fe": 99.5, "C": 0.5}, 1600.0),
    "liquid_gamma": (
        {"LIQUID": 0.62, "AUSTENITE": 0.38},
        {"Fe": 99.7, "C": 0.3},
        1480.0,
    ),
    "alpha_gamma": (
        {"FERRITE": 0.55, "AUSTENITE": 0.45},
        {"Fe": 99.7, "C": 0.3},
        800.0,
    ),
    "gamma_cementite": (
        {"AUSTENITE": 0.72, "CEMENTITE": 0.28},
        {"Fe": 98.8, "C": 1.2},
        900.0,
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
            cooling_mode="equilibrium",
        ),
    )


class HighTempPhasesFamilyTests(unittest.TestCase):
    """Smoke-тесты: каждая стадия рендерится, маски не пустые, тона в
    разумном диапазоне."""

    def test_austenite_renders_with_boundaries_and_twins(self) -> None:
        out = render_fe_c_unified(_make_ctx("austenite"))
        self.assertEqual(out.image_gray.shape, (192, 192))
        self.assertEqual(out.image_gray.dtype, np.uint8)
        self.assertIn("AUSTENITE", out.phase_masks)
        self.assertGreater(int(out.phase_masks["AUSTENITE"].sum()), 1000)
        mean = float(out.image_gray.mean())
        self.assertGreater(mean, 170.0, f"austenite too dark: mean={mean:.1f}")
        self.assertLess(mean, 240.0, f"austenite too bright: mean={mean:.1f}")
        std = float(out.image_gray.std())
        self.assertGreater(std, 6.0, "no textural variation — no twins?")

    def test_delta_ferrite_has_dark_islands(self) -> None:
        out = render_fe_c_unified(_make_ctx("delta_ferrite"))
        img = out.image_gray
        dark_frac = float((img < 180).mean())
        self.assertGreater(
            dark_frac, 0.02,
            f"δ-ferrite should have dark islands (expected >=2%), got {dark_frac*100:.1f}%",
        )
        self.assertLess(
            dark_frac, 0.25,
            f"δ-ferrite islands too extensive: {dark_frac*100:.1f}%",
        )

    def test_liquid_has_no_grain_boundaries(self) -> None:
        out = render_fe_c_unified(_make_ctx("liquid"))
        img = out.image_gray.astype(np.int16)
        self.assertEqual(out.image_gray.dtype, np.uint8)
        grad = np.abs(np.diff(img, axis=0)).mean() + np.abs(np.diff(img, axis=1)).mean()
        self.assertLess(
            grad, 20.0,
            f"liquid has too sharp transitions (mean |grad|={grad:.1f}), "
            "expected smooth gradient",
        )

    def test_liquid_gamma_is_heterogeneous(self) -> None:
        out = render_fe_c_unified(_make_ctx("liquid_gamma"))
        self.assertIn("LIQUID", out.phase_masks)
        self.assertIn("AUSTENITE", out.phase_masks)
        std = float(out.image_gray.std())
        self.assertGreater(std, 15.0, "liquid_gamma should have dendrite contrast")

    def test_alpha_gamma_has_ferrite_and_austenite(self) -> None:
        out = render_fe_c_unified(_make_ctx("alpha_gamma"))
        self.assertIn("FERRITE", out.phase_masks)
        self.assertIn("AUSTENITE", out.phase_masks)
        f = int(out.phase_masks["FERRITE"].sum())
        a = int(out.phase_masks["AUSTENITE"].sum())
        self.assertGreater(f, 500)
        self.assertGreater(a, 500)

    def test_gamma_cementite_has_dark_carbides(self) -> None:
        out = render_fe_c_unified(_make_ctx("gamma_cementite"))
        img = out.image_gray
        # Ожидаем минимум небольшое количество очень тёмных пикселей
        # (карбиды).
        very_dark = float((img < 80).mean())
        self.assertGreater(
            very_dark, 0.005,
            f"gamma_cementite: expected dark carbide pixels, got {very_dark*100:.3f}%",
        )

    def test_all_stages_deterministic(self) -> None:
        for stage in _STAGE_DEFAULTS:
            with self.subTest(stage=stage):
                a = render_fe_c_unified(_make_ctx(stage, seed=777)).image_gray
                b = render_fe_c_unified(_make_ctx(stage, seed=777)).image_gray
                self.assertTrue(
                    np.array_equal(a, b),
                    f"{stage} is not deterministic with same seed",
                )


if __name__ == "__main__":
    unittest.main()
