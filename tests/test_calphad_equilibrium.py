from __future__ import annotations

import unittest
from pathlib import Path

from core.calphad.db_manager import resolve_database_reference
from core.calphad.engine_pycalphad import run_equilibrium
from core.contracts_v2 import ThermoBackendConfig


class CalphadEquilibriumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = Path("profiles") / "calphad_profile_v2.json"
        self.thermo = ThermoBackendConfig(db_profile_path=str(self.profile))

    def test_equilibrium_for_supported_systems(self) -> None:
        cases = [
            ("fe-c", {"Fe": 99.2, "C": 0.8}, 780.0),
            ("fe-si", {"Fe": 98.6, "Si": 1.4}, 900.0),
            ("al-si", {"Al": 88.0, "Si": 12.0}, 580.0),
            ("cu-zn", {"Cu": 68.0, "Zn": 32.0}, 700.0),
            ("al-cu-mg", {"Al": 93.0, "Cu": 4.4, "Mg": 1.5}, 520.0),
        ]
        for system, composition, temp in cases:
            with self.subTest(system=system):
                db_ref = resolve_database_reference(system=system, thermo=self.thermo, profile_path=self.profile)
                result = run_equilibrium(
                    db_ref=db_ref,
                    system=system,
                    composition=composition,
                    temperature_c=temp,
                    pressure_pa=101325.0,
                )
                self.assertTrue(result.stable_phases)
                self.assertLessEqual(abs(sum(result.stable_phases.values()) - 1.0), 1e-6)
                self.assertTrue(0.0 <= result.liquid_fraction <= 1.0)
                self.assertTrue(0.0 <= result.solid_fraction <= 1.0)


if __name__ == "__main__":
    unittest.main()
