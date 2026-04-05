from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.metallography_v3.phase_orchestrator import estimate_auto_phase_fractions


class FeCTemperingTablesInterpolationTests(unittest.TestCase):
    def test_quench_water20_interpolation_points(self) -> None:
        proc = ProcessingState(temperature_c=30.0, cooling_mode="quenched")
        qsum = {"medium_code_resolved": "water_20", "effect_applied": True}
        thermal = {"operation_inference": {"has_quench": True, "has_temper": False}}

        cases = [
            (0.05, {"MARTENSITE": 1.0, "AUSTENITE": 0.0, "CEMENTITE": 0.0}),
            (0.40, {"MARTENSITE": 0.95, "AUSTENITE": 0.05, "CEMENTITE": 0.0}),
            (0.75, {"MARTENSITE": 0.862, "AUSTENITE": 0.138, "CEMENTITE": 0.0}),
            (0.80, {"MARTENSITE": 0.84, "AUSTENITE": 0.15, "CEMENTITE": 0.01}),
        ]
        for c_wt, expected in cases:
            with self.subTest(c_wt=c_wt):
                phases = estimate_auto_phase_fractions(
                    "fe-c",
                    "martensite",
                    {"Fe": 100.0 - c_wt, "C": c_wt},
                    proc,
                    thermal_summary=thermal,
                    quench_summary=qsum,
                )
                self.assertAlmostEqual(float(phases.get("MARTENSITE", 0.0)), float(expected["MARTENSITE"]), delta=0.02)
                self.assertAlmostEqual(float(phases.get("AUSTENITE", 0.0)), float(expected["AUSTENITE"]), delta=0.02)
                self.assertAlmostEqual(float(phases.get("CEMENTITE", 0.0)), float(expected["CEMENTITE"]), delta=0.01)


if __name__ == "__main__":
    unittest.main()
