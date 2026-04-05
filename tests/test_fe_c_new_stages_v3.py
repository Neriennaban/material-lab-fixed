from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.contracts_v3 import PhaseModelConfigV3
from core.metallography_v3.phase_orchestrator import build_phase_bundle


class FeCNewStagesV3Tests(unittest.TestCase):
    def test_martensite_tetragonal_stage(self) -> None:
        bundle = build_phase_bundle(
            composition={"Fe": 99.0, "C": 1.0},
            processing=ProcessingState(temperature_c=120.0, cooling_mode="quenched"),
            system_hint="fe-c",
            phase_model=PhaseModelConfigV3(),
            thermal_summary={"temperature_max_c": 860.0, "hold_time_s": 200.0, "max_effective_cooling_rate_c_per_s": 35.0, "observed_temperature_c": 120.0},
            quench_summary={"medium_code": "water_20", "severity_effective": 0.9},
        )
        self.assertEqual(bundle.stage, "martensite_tetragonal")

    def test_sorbite_quench_stage(self) -> None:
        bundle = build_phase_bundle(
            composition={"Fe": 99.3, "C": 0.7},
            processing=ProcessingState(temperature_c=220.0, cooling_mode="quenched"),
            system_hint="fe-c",
            phase_model=PhaseModelConfigV3(),
            thermal_summary={"temperature_max_c": 820.0, "hold_time_s": 180.0, "max_effective_cooling_rate_c_per_s": 8.0, "observed_temperature_c": 220.0},
            quench_summary={"medium_code": "oil_20_80", "severity_effective": 0.35},
        )
        self.assertEqual(bundle.stage, "sorbite_quench")


if __name__ == "__main__":
    unittest.main()
