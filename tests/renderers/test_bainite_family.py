"""Phase 5 — тесты bainite renderer'а.

Семейство: bainite_upper (§2.5 feathery), bainite_lower (§2.6 acicular
+ 60° carbide hash, ОДНО направление), carbide_free_bainite (§2.7,
новая стадия для Si≥1.5%).
"""
from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


_STAGE_DEFAULTS = {
    "bainite_upper": (
        {"BAINITE": 0.78, "CEMENTITE": 0.22},
        {"Fe": 99.55, "C": 0.45},
        480.0,
    ),
    "bainite_lower": (
        {"BAINITE": 0.85, "CEMENTITE": 0.15},
        {"Fe": 99.3, "C": 0.7},
        320.0,
    ),
    "carbide_free_bainite": (
        {"BAINITE": 0.70, "AUSTENITE": 0.25, "MARTENSITE": 0.05},
        {"Fe": 97.6, "C": 0.4, "Si": 1.8},
        300.0,
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
            cooling_mode="isothermal",
        ),
        thermal_summary={"max_effective_cooling_rate_c_per_s": 50.0},
    )


class BainiteFamilyTests(unittest.TestCase):
    def test_all_stages_render(self) -> None:
        for stage in _STAGE_DEFAULTS:
            with self.subTest(stage=stage):
                out = render_fe_c_unified(_make_ctx(stage))
                self.assertEqual(out.image_gray.shape, (192, 192))
                self.assertEqual(out.image_gray.dtype, np.uint8)
                self.assertTrue(out.phase_masks)
                mean = float(out.image_gray.mean())
                self.assertGreater(mean, 40.0, f"{stage}: too dark ({mean:.1f})")
                self.assertLess(mean, 200.0, f"{stage}: too bright ({mean:.1f})")

    def test_upper_feathery_anisotropy(self) -> None:
        """§2.5 — упорядоченная перистая текстура: заметный градиент
        хотя бы по одной оси."""
        out = render_fe_c_unified(_make_ctx("bainite_upper"))
        img = out.image_gray.astype(np.float32)
        grad_x = float(np.abs(np.diff(img, axis=1)).mean())
        grad_y = float(np.abs(np.diff(img, axis=0)).mean())
        self.assertGreater(
            max(grad_x, grad_y), 3.5,
            f"upper bainite lacks feathery texture (grads: x={grad_x:.2f}, y={grad_y:.2f})",
        )

    def test_lower_has_dark_needles(self) -> None:
        """§2.6 — нижний бейнит: чёткие тёмные иглы-щепа."""
        out = render_fe_c_unified(_make_ctx("bainite_lower"))
        dark_frac = float((out.image_gray <= 65).mean())
        self.assertGreater(
            dark_frac, 0.08,
            f"lower bainite: dark needle fraction too low ({dark_frac*100:.1f}%)",
        )

    def test_cfb_has_bright_blocks_and_no_dense_carbides(self) -> None:
        """§2.7 — CFB: блоки γR видны как светлые области; плотных
        тёмных точечных карбидов быть не должно."""
        out = render_fe_c_unified(_make_ctx("carbide_free_bainite"))
        img = out.image_gray
        bright_blocks = float((img >= 200).mean())
        self.assertGreater(
            bright_blocks, 0.02,
            f"CFB: expected bright γR blocks ≥2%, got {bright_blocks*100:.1f}%",
        )
        # Проверка «нет плотных карбидов»: доля очень-тёмных пикселей
        # (≤35) должна быть мала (<5%) — в CFB карбидов нет.
        very_dark = float((img <= 35).mean())
        self.assertLess(
            very_dark, 0.05,
            f"CFB: too many very-dark pixels ({very_dark*100:.1f}%), "
            "should have no sharp carbide dots",
        )

    def test_cfb_stage_registered(self) -> None:
        """Новая стадия carbide_free_bainite зарегистрирована в
        SYSTEM_STAGE_ORDER + UI labels."""
        from core.generator_phase_map import SYSTEM_STAGE_ORDER
        self.assertIn("carbide_free_bainite", SYSTEM_STAGE_ORDER["fe-c"])

    def test_family_trace_strings(self) -> None:
        expected = {
            "bainite_upper": "bainite_upper_feathery",
            "bainite_lower": "bainite_lower_acicular",
            "carbide_free_bainite": "bainite_cfb",
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
