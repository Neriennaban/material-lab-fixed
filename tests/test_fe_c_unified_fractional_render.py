from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


class FeCUnifiedFractionalRenderTests(unittest.TestCase):
    def test_alpha_gamma_has_two_phase_masks(self) -> None:
        ctx = SystemGenerationContext(
            size=(128, 128),
            seed=501,
            inferred_system="fe-c",
            stage="alpha_gamma",
            phase_fractions={"FERRITE": 0.65, "AUSTENITE": 0.35},
            composition_wt={"Fe": 99.6, "C": 0.4},
            processing=ProcessingState(temperature_c=760.0, cooling_mode="equilibrium"),
        )
        out = render_fe_c_unified(ctx)
        masks = out.phase_masks
        self.assertIn("FERRITE", masks)
        self.assertIn("AUSTENITE", masks)
        ferr = float((masks["FERRITE"] > 0).mean())
        aust = float((masks["AUSTENITE"] > 0).mean())
        self.assertGreater(ferr, 0.25)
        self.assertGreater(aust, 0.15)

    def test_liquid_gamma_has_liquid_and_solid_masks(self) -> None:
        ctx = SystemGenerationContext(
            size=(128, 128),
            seed=777,
            inferred_system="fe-c",
            stage="liquid_gamma",
            phase_fractions={"LIQUID": 0.55, "AUSTENITE": 0.45},
            composition_wt={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=1450.0, cooling_mode="equilibrium"),
        )
        out = render_fe_c_unified(ctx)
        masks = out.phase_masks
        self.assertIn("LIQUID", masks)
        self.assertIn("AUSTENITE", masks)
        liquid = float((masks["LIQUID"] > 0).mean())
        solid = float((masks["AUSTENITE"] > 0).mean())
        self.assertGreater(liquid, 0.2)
        self.assertGreater(solid, 0.2)


if __name__ == "__main__":
    unittest.main()
