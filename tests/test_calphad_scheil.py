from __future__ import annotations

import unittest
from pathlib import Path

from core.calphad.db_manager import resolve_database_reference
from core.calphad.scheil import run_scheil
from core.contracts_v2 import ThermoBackendConfig


class CalphadScheilTests(unittest.TestCase):
    def test_scheil_solid_fraction_is_monotonic(self) -> None:
        profile = Path("profiles") / "calphad_profile_v2.json"
        thermo = ThermoBackendConfig(db_profile_path=str(profile))
        db_ref = resolve_database_reference(system="fe-c", thermo=thermo, profile_path=profile)
        series = run_scheil(
            db_ref=db_ref,
            system="fe-c",
            composition={"Fe": 99.2, "C": 0.8},
            t_start_c=1500.0,
            t_end_c=20.0,
            d_t_c=25.0,
        )
        self.assertTrue(series.get("enabled"))
        traj = series.get("trajectory", [])
        self.assertGreater(len(traj), 2)
        prev = -1.0
        for item in traj:
            cur = float(item.get("f_solid", 0.0))
            self.assertGreaterEqual(cur + 1e-9, prev)
            prev = cur
            liq = float(item.get("f_liquid", 0.0))
            self.assertAlmostEqual(cur + liq, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
