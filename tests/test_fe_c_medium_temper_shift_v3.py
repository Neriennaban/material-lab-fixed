from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.contracts_v3 import PhaseModelConfigV3
from core.metallography_v3.phase_orchestrator import build_phase_bundle


class FeCMediumTemperShiftV3Tests(unittest.TestCase):
    def test_medium_shift_changes_temper_stage_near_boundary(self) -> None:
        thermal_summary = {
            "temperature_max_c": 840.0,
            "observed_temperature_c": 35.0,
            "max_effective_cooling_rate_c_per_s": 20.0,
            "operation_inference": {
                "has_quench": True,
                "has_temper": True,
                "temper_peak_temperature_c": 500.0,
                "temper_total_hold_s": 1800.0,
            },
        }

        water_bundle = build_phase_bundle(
            composition={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=35.0, cooling_mode="quenched"),
            system_hint="fe-c",
            phase_model=PhaseModelConfigV3(),
            thermal_summary=thermal_summary,
            quench_summary={"medium_code": "water_20", "severity_effective": 0.88, "temper_shift_c": {"low": 0.0, "medium": 0.0, "high": 0.0}},
        )

        oil_bundle = build_phase_bundle(
            composition={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=35.0, cooling_mode="quenched"),
            system_hint="fe-c",
            phase_model=PhaseModelConfigV3(),
            thermal_summary=thermal_summary,
            quench_summary={"medium_code": "oil_20_80", "severity_effective": 0.45, "temper_shift_c": {"low": 70.0, "medium": 70.0, "high": 70.0}},
        )

        self.assertEqual(water_bundle.stage, "sorbite_temper")
        self.assertEqual(oil_bundle.stage, "troostite_temper")


if __name__ == "__main__":
    unittest.main()
