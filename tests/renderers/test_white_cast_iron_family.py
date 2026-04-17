"""Phase 3 — тесты white_cast_iron renderer'а.

Семейство: ledeburite (Ld′ при 20°C), white_cast_iron_eutectic,
white_cast_iron_hypoeutectic, white_cast_iron_hypereutectic.
Справочник §1.6, §1.10.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


_STAGE_DEFAULTS = {
    "ledeburite": (
        {"PEARLITE": 0.49, "CEMENTITE": 0.51},
        {"Fe": 95.7, "C": 4.3},
        500.0,
    ),
    "white_cast_iron_eutectic": (
        {"LEDEBURITE": 1.0},
        {"Fe": 95.7, "C": 4.3},
        20.0,
    ),
    "white_cast_iron_hypoeutectic": (
        {"LEDEBURITE": 0.65, "PEARLITE": 0.35},
        {"Fe": 97.0, "C": 3.0},
        20.0,
    ),
    "white_cast_iron_hypereutectic": (
        {"LEDEBURITE": 0.70, "CEMENTITE_PRIMARY": 0.30},
        {"Fe": 94.5, "C": 5.5},
        20.0,
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
        processing=ProcessingState(temperature_c=temperature_c, cooling_mode="equilibrium"),
        thermal_summary={"max_effective_cooling_rate_c_per_s": 5.0},
    )


class WhiteCastIronFamilyTests(unittest.TestCase):
    def test_all_stages_render(self) -> None:
        for stage in _STAGE_DEFAULTS:
            with self.subTest(stage=stage):
                out = render_fe_c_unified(_make_ctx(stage))
                self.assertEqual(out.image_gray.shape, (192, 192))
                self.assertEqual(out.image_gray.dtype, np.uint8)
                self.assertTrue(out.phase_masks, f"{stage}: empty phase masks")

    def test_leopard_has_strong_contrast(self) -> None:
        """§1.6: цементитная матрица (~240) vs перлитные островки (~100),
        ΔR ≈ 140. Проверяем присутствие и светлых (≥200) и тёмных (≤110)
        пикселей одновременно."""
        for stage in ("ledeburite", "white_cast_iron_eutectic"):
            with self.subTest(stage=stage):
                out = render_fe_c_unified(_make_ctx(stage))
                img = out.image_gray
                bright_frac = float((img >= 200).mean())
                dark_frac = float((img <= 110).mean())
                self.assertGreater(
                    bright_frac, 0.10,
                    f"{stage}: expected bright cementite matrix ≥10% "
                    f"(got {bright_frac*100:.1f}%)",
                )
                self.assertGreater(
                    dark_frac, 0.05,
                    f"{stage}: expected dark pearlite islands ≥5% "
                    f"(got {dark_frac*100:.1f}%)",
                )
                # Глобальная амплитуда — показатель двухфазности.
                self.assertGreater(
                    int(img.max()) - int(img.min()),
                    120,
                    f"{stage}: contrast Δ too low",
                )

    def test_hypoeutectic_has_primary_dendrites(self) -> None:
        """Hypoeutectic содержит первичные γ-дендриты, превращающиеся в
        перлит при 20°C."""
        out = render_fe_c_unified(_make_ctx("white_cast_iron_hypoeutectic"))
        self.assertIn("PEARLITE", out.phase_masks)
        pearlite_frac = float(out.phase_masks["PEARLITE"].mean())
        self.assertGreater(
            pearlite_frac, 0.01,
            f"hypoeutectic: primary pearlite (dendrites) mask too small "
            f"({pearlite_frac*100:.2f}%)",
        )

    def test_hypereutectic_has_bright_plates(self) -> None:
        """§1.10в: первичный Fe₃C_I — длинные яркие (≥240) пластины."""
        out = render_fe_c_unified(_make_ctx("white_cast_iron_hypereutectic"))
        img = out.image_gray
        very_bright = float((img >= 240).mean())
        self.assertGreater(
            very_bright, 0.005,
            f"hypereutectic: expected bright primary cementite plates "
            f"(≥240), got {very_bright*100:.3f}%",
        )

    def test_family_trace_strings(self) -> None:
        """`morphology_trace.family` должен соответствовать ожиданиям
        PresetIntegrationTest."""
        expected_family = {
            "white_cast_iron_eutectic": "white_cast_iron_eutectic",
            "white_cast_iron_hypoeutectic": "white_cast_iron_hypoeutectic",
            "white_cast_iron_hypereutectic": "white_cast_iron_hypereutectic",
        }
        for stage, fam in expected_family.items():
            with self.subTest(stage=stage):
                out = render_fe_c_unified(_make_ctx(stage))
                trace = out.metadata.get("fe_c_phase_render", {}).get(
                    "morphology_trace", {}
                )
                self.assertEqual(
                    trace.get("family"),
                    fam,
                    f"{stage}: family trace mismatch (got {trace.get('family')!r})",
                )

    def test_determinism(self) -> None:
        for stage in _STAGE_DEFAULTS:
            with self.subTest(stage=stage):
                a = render_fe_c_unified(_make_ctx(stage, seed=777)).image_gray
                b = render_fe_c_unified(_make_ctx(stage, seed=777)).image_gray
                self.assertTrue(np.array_equal(a, b), f"{stage} not deterministic")


if __name__ == "__main__":
    unittest.main()
