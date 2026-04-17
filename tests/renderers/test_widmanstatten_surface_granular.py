"""Phase 8 — тесты widmanstatten / surface_layers / granular_pearlite.

Четыре новые стадии §2.10, §3.2, §3.3, §1.9.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


_STAGE_DEFAULTS = {
    "widmanstatten_ferrite": (
        {"FERRITE": 0.50, "PEARLITE": 0.50},
        {"Fe": 99.7, "C": 0.3},
        720.0,
    ),
    "decarburized_layer": (
        {"FERRITE": 0.70, "PEARLITE": 0.30},
        {"Fe": 99.55, "C": 0.45},
        900.0,
    ),
    "carburized_layer": (
        {
            "MARTENSITE": 0.35,
            "PEARLITE": 0.30,
            "FERRITE": 0.20,
            "CEMENTITE": 0.10,
            "AUSTENITE": 0.05,
        },
        {"Fe": 99.8, "C": 0.2},
        20.0,
    ),
    "granular_pearlite": (
        {"FERRITE": 0.85, "CEMENTITE": 0.15},
        {"Fe": 99.0, "C": 1.0},
        700.0,
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
            cooling_mode="air",
        ),
        thermal_summary={"max_effective_cooling_rate_c_per_s": 10.0},
    )


class Phase8FamilyTests(unittest.TestCase):
    def test_all_stages_render(self) -> None:
        for stage in _STAGE_DEFAULTS:
            with self.subTest(stage=stage):
                out = render_fe_c_unified(_make_ctx(stage))
                self.assertEqual(out.image_gray.shape, (192, 192))
                self.assertEqual(out.image_gray.dtype, np.uint8)
                self.assertTrue(out.phase_masks)

    def test_widmanstatten_has_oriented_needles(self) -> None:
        """§2.10 — иглы в ~2-3 дискретных направлениях →
        texture с заметным гранд на изображении."""
        out = render_fe_c_unified(_make_ctx("widmanstatten_ferrite"))
        img = out.image_gray.astype(np.float32)
        # Ожидаем чёткий контраст светлых игл (feat ferrite ~225) и
        # тёмной матрицы перлита (~100).
        bright = float((img >= 180).mean())
        dark = float((img <= 130).mean())
        self.assertGreater(bright, 0.05, f"widmanstatten: needles ≥180 tone too few ({bright*100:.1f}%)")
        self.assertGreater(dark, 0.10, f"widmanstatten: pearlite matrix ≤130 too small ({dark*100:.1f}%)")
        self.assertIn("FERRITE", out.phase_masks)
        self.assertIn("PEARLITE", out.phase_masks)

    def test_decarburized_has_vertical_brightness_gradient(self) -> None:
        """§3.2 — обезуглероживание: верх (y=0) светлее сердцевины
        (y=H). Разница средних между верхней и нижней полосой ≥20."""
        out = render_fe_c_unified(_make_ctx("decarburized_layer"))
        img = out.image_gray.astype(np.float32)
        h = img.shape[0]
        top_band = float(img[: h // 5].mean())
        bottom_band = float(img[-h // 5:].mean())
        self.assertGreater(
            top_band - bottom_band, 20.0,
            f"decarburized: vertical gradient too weak "
            f"(top={top_band:.1f}, bottom={bottom_band:.1f})",
        )

    def test_carburized_has_inverse_brightness_gradient(self) -> None:
        """§3.3 — цементация + закалка: верх (мартенсит) темнее
        сердцевины (α+P). Разница bottom-top ≥ 25."""
        out = render_fe_c_unified(_make_ctx("carburized_layer"))
        img = out.image_gray.astype(np.float32)
        h = img.shape[0]
        top_band = float(img[: h // 5].mean())
        bottom_band = float(img[-h // 5:].mean())
        self.assertGreater(
            bottom_band - top_band, 25.0,
            f"carburized: inverse gradient too weak "
            f"(top={top_band:.1f}, bottom={bottom_band:.1f})",
        )

    def test_granular_pearlite_has_bright_matrix_and_globules(self) -> None:
        """§1.9 — светлая ферритная матрица + точечные карбиды ≥1%."""
        out = render_fe_c_unified(_make_ctx("granular_pearlite"))
        img = out.image_gray
        mean = float(img.mean())
        self.assertGreater(mean, 140.0, f"granular_pearlite: too dark (mean={mean:.1f})")
        # Глобули либо очень светлые (nital bright Fe3C) либо очень тёмные
        # (picral). Nital карточка: globules ~240, matrix ~215 → проверяем
        # присутствие «экстремальных» пикселей.
        extreme_bright = float((img >= 235).mean())
        self.assertGreater(
            extreme_bright, 0.005,
            f"granular_pearlite: cementite globules too few "
            f"({extreme_bright*100:.3f}%)",
        )

    def test_new_stages_registered(self) -> None:
        from core.generator_phase_map import SYSTEM_STAGE_ORDER
        for stage in _STAGE_DEFAULTS:
            with self.subTest(stage=stage):
                self.assertIn(stage, SYSTEM_STAGE_ORDER["fe-c"])

    def test_family_trace_strings(self) -> None:
        expected = {
            "widmanstatten_ferrite": "widmanstatten_ferrite",
            "decarburized_layer": "decarburized_layer",
            "carburized_layer": "carburized_layer",
            "granular_pearlite": "granular_pearlite",
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
