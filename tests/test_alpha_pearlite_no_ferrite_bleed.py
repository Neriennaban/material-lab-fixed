"""Guard: в alpha_pearlite перлитные пиксели не должны иметь яркости,
попадающей в диапазон чистого феррита (≥150). См. план, §A."""
from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


class AlphaPearliteNoFerriteBleedTests(unittest.TestCase):
    def test_pearlite_pixels_are_uniformly_dark(self) -> None:
        out = render_fe_c_unified(
            SystemGenerationContext(
                size=(256, 256),
                seed=2026,
                inferred_system="fe-c",
                stage="alpha_pearlite",
                phase_fractions={"FERRITE": 0.40, "PEARLITE": 0.60},
                composition_wt={"Fe": 99.55, "C": 0.45},
                processing=ProcessingState(
                    temperature_c=20.0,
                    cooling_mode="equilibrium",
                ),
            )
        )
        pearlite_mask = out.phase_masks["PEARLITE"] > 0
        self.assertGreater(int(pearlite_mask.sum()), 1000)

        image = out.image_gray
        pearlite_pixels = image[pearlite_mask]
        # Справочник §1.3: перлит при ×500 — средний тон ~90, должен быть
        # заметно темнее чистого феррита (150-240). Допуск: среднее ≤ 140.
        pearlite_mean = float(pearlite_pixels.mean())
        self.assertLess(
            pearlite_mean,
            140.0,
            f"pearlite слишком светлый (mean={pearlite_mean:.1f}) — "
            "вероятно, внутри перлита остались ферритные ламели",
        )
        # Доля пикселей перлита, попадающих в ферритный диапазон (≥170),
        # должна быть мала — иначе перлит визуально «течёт» в феррит.
        bleed = float((pearlite_pixels >= 170).mean())
        self.assertLess(
            bleed,
            0.08,
            f"слишком много светлых пикселей внутри перлита (bleed={bleed:.3f}); "
            "ферритный слой не удалён",
        )

    def test_existing_pearlite_stage_unchanged(self) -> None:
        """pure pearlite НЕ должен пострадать от фикса alpha_pearlite."""
        out = render_fe_c_unified(
            SystemGenerationContext(
                size=(160, 160),
                seed=2026,
                inferred_system="fe-c",
                stage="pearlite",
                phase_fractions={"PEARLITE": 1.0},
                composition_wt={"Fe": 99.23, "C": 0.77},
                processing=ProcessingState(
                    temperature_c=20.0,
                    cooling_mode="equilibrium",
                ),
            )
        )
        # В pure pearlite ферритные ламели остаются → ожидаем светлые пиксели.
        pearlite_mask = out.phase_masks["PEARLITE"] > 0
        bright_frac = float((out.image_gray[pearlite_mask] >= 170).mean())
        self.assertGreater(
            bright_frac,
            0.08,
            "pure pearlite потерял светлые ферритные ламели — "
            "фикс затронул не только alpha_pearlite",
        )


if __name__ == "__main__":
    unittest.main()
