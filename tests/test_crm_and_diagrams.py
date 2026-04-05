from __future__ import annotations

import unittest

from core.crm_fe_c_generator import generate_crm_fe_c_rgb, phase_fractions_fe_c
from core.phase_diagrams import render_detailed_diagram


class CrmAndDiagramTests(unittest.TestCase):
    def test_crm_generator_output_shape_and_fractions(self) -> None:
        image, fractions = generate_crm_fe_c_rgb(
            width=160,
            height=128,
            carbon_pct=0.8,
            grains_count=90,
            seed=42,
            iron_type="auto",
            distortion_level=0.6,
        )
        self.assertEqual(image.shape, (128, 160, 3))
        self.assertAlmostEqual(sum(float(v) for v in fractions.values()), 1.0, places=6)

    def test_phase_fraction_behavior(self) -> None:
        low_c = phase_fractions_fe_c(0.1, "auto")
        high_c = phase_fractions_fe_c(1.2, "auto")
        self.assertGreater(low_c.get("ferrite", 0.0), 0.0)
        self.assertGreater(high_c.get("cementite", 0.0), low_c.get("cementite", 0.0))

    def test_diagram_renderer(self) -> None:
        image = render_detailed_diagram(
            system="al-si",
            composition={"Al": 88.0, "Si": 12.0},
            temperature_c=560.0,
            size=(700, 380),
        )
        self.assertEqual(image.size, (700, 380))


if __name__ == "__main__":
    unittest.main()

