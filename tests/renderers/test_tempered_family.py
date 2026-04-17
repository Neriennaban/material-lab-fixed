"""Phase 7 — тесты tempered renderer'а.

Семейство: tempered_low (§2.11), tempered_medium/troostite_temper
(§2.12), tempered_high/sorbite_temper (§2.13, Q+T).
"""
from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


_STAGE_DEFAULTS = {
    "tempered_low": (
        {"MARTENSITE": 0.92, "CEMENTITE": 0.08},
        {"Fe": 99.55, "C": 0.45},
        220.0,
    ),
    "tempered_medium": (
        {"TROOSTITE": 0.70, "CEMENTITE": 0.20, "FERRITE": 0.10},
        {"Fe": 99.55, "C": 0.45},
        420.0,
    ),
    "troostite_temper": (
        {"TROOSTITE": 0.70, "CEMENTITE": 0.20, "FERRITE": 0.10},
        {"Fe": 99.55, "C": 0.45},
        420.0,
    ),
    "tempered_high": (
        {"SORBITE": 0.42, "FERRITE": 0.40, "CEMENTITE": 0.18},
        {"Fe": 99.6, "C": 0.4},
        580.0,
    ),
    "sorbite_temper": (
        {"SORBITE": 0.42, "FERRITE": 0.40, "CEMENTITE": 0.18},
        {"Fe": 99.6, "C": 0.4},
        580.0,
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
            cooling_mode="tempered",
        ),
        thermal_summary={"max_effective_cooling_rate_c_per_s": 50.0},
    )


class TemperedFamilyTests(unittest.TestCase):
    def test_all_stages_render(self) -> None:
        for stage in _STAGE_DEFAULTS:
            with self.subTest(stage=stage):
                out = render_fe_c_unified(_make_ctx(stage))
                self.assertEqual(out.image_gray.shape, (192, 192))
                self.assertEqual(out.image_gray.dtype, np.uint8)
                self.assertTrue(out.phase_masks)

    def test_low_is_darker_than_raw_martensite(self) -> None:
        """§2.11: отпуск при 150-250°C даёт общее затемнение ~0.72× vs
        реечного закалочного мартенсита."""
        low = render_fe_c_unified(_make_ctx("tempered_low")).image_gray.mean()
        # Реечный при той же C (0.45) для сравнения.
        raw_ctx = SystemGenerationContext(
            size=(192, 192),
            seed=2026,
            inferred_system="fe-c",
            stage="martensite_cubic",
            phase_fractions={"MARTENSITE_CUBIC": 0.94, "CEMENTITE": 0.06},
            composition_wt={"Fe": 99.55, "C": 0.45},
            processing=ProcessingState(
                temperature_c=20.0,
                cooling_mode="quench_water",
            ),
            thermal_summary={"max_effective_cooling_rate_c_per_s": 800.0},
        )
        raw = render_fe_c_unified(raw_ctx).image_gray.mean()
        self.assertLess(
            low, raw,
            f"tempered_low (mean={low:.1f}) not darker than raw "
            f"martensite_cubic (mean={raw:.1f})",
        )

    def test_medium_has_velvet_texture(self) -> None:
        """§2.12 — «бархатная» матрица с высокочастотным шумом + точки
        карбидов: хорошая текстурная вариация."""
        out = render_fe_c_unified(_make_ctx("tempered_medium"))
        std = float(out.image_gray.std())
        self.assertGreater(
            std, 12.0,
            f"tempered_medium: velvet texture too flat (std={std:.1f})",
        )

    def test_high_is_polygonal_and_bright(self) -> None:
        """§2.13 — Q+T: полигональная ферритная матрица, светлее чем
        low/medium + тёмные точечные карбиды."""
        out = render_fe_c_unified(_make_ctx("tempered_high"))
        mean = float(out.image_gray.mean())
        self.assertGreater(
            mean, 115.0,
            f"tempered_high: not bright enough for polygonal ferrite "
            f"(mean={mean:.1f})",
        )
        # Тёмные точечные карбиды.
        dark_spots = float((out.image_gray <= 50).mean())
        self.assertGreater(
            dark_spots, 0.015,
            f"tempered_high: dark carbide spots insufficient "
            f"({dark_spots*100:.2f}%)",
        )

    def test_aliases_render_identically(self) -> None:
        """troostite_temper ≡ tempered_medium, sorbite_temper ≡ tempered_high."""
        pairs = [
            ("troostite_temper", "tempered_medium"),
            ("sorbite_temper", "tempered_high"),
        ]
        for alias, primary in pairs:
            with self.subTest(pair=f"{alias} vs {primary}"):
                a = render_fe_c_unified(_make_ctx(alias, seed=4242)).image_gray
                b = render_fe_c_unified(_make_ctx(primary, seed=4242)).image_gray
                # Алиасы должны выдать идентичную картинку при одинаковом seed.
                diff = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
                self.assertLess(
                    diff, 3,
                    f"{alias} and {primary} diverge (max pixel diff {diff})",
                )

    def test_family_trace_strings(self) -> None:
        expected = {
            "tempered_low": "tempered_low",
            "tempered_medium": "tempered_medium",
            "troostite_temper": "tempered_medium",
            "tempered_high": "tempered_high",
            "sorbite_temper": "tempered_high",
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
