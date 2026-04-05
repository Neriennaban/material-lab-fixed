from __future__ import annotations

import unittest

from core.calphad.kinetics import run_jmak_lsw


class CalphadKineticsTests(unittest.TestCase):
    def test_jmak_lsw_ranges(self) -> None:
        result = run_jmak_lsw(
            system="al-cu-mg",
            temperature_c=180.0,
            aging_hours=10.0,
            base_phase_fractions={"ALPHA": 0.9, "THETA": 0.1},
        )
        self.assertTrue(result.get("enabled"))
        self.assertGreaterEqual(float(result.get("jmak_fraction", 0.0)), 0.0)
        self.assertLessEqual(float(result.get("jmak_fraction", 0.0)), 1.0)
        self.assertGreaterEqual(float(result.get("precipitate_fraction", 0.0)), 0.0)
        self.assertLessEqual(float(result.get("precipitate_fraction", 0.0)), 0.45)
        self.assertGreater(float(result.get("mean_radius_nm", 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
