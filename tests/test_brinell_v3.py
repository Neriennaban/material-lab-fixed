from __future__ import annotations

import unittest

from core.metallography_v3.hardness_brinell import hbw_estimate_from_microstructure, hbw_from_indent


class BrinellV3Tests(unittest.TestCase):
    def test_direct_hbw(self) -> None:
        out = hbw_from_indent(load_kgf=187.5, ball_d_mm=2.5, indent_d_mm=0.9)
        self.assertIn("HBW", out)
        self.assertGreater(out["HBW"], 0.0)

    def test_estimated_hbw(self) -> None:
        out = hbw_estimate_from_microstructure(
            system="fe-c",
            stage="martensite_tetragonal",
            phase_fractions={"MARTENSITE_T": 0.85, "CEMENTITE": 0.15},
            effect_vector={"dislocation_proxy": 0.7},
        )
        self.assertIn("HBW", out)
        self.assertGreater(out["HBW"], 200.0)


if __name__ == "__main__":
    unittest.main()
