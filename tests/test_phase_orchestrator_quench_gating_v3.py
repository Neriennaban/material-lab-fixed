from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.contracts_v3 import PhaseModelConfigV3
from core.metallography_v3.phase_orchestrator import build_phase_bundle


class PhaseOrchestratorQuenchGatingTests(unittest.TestCase):
    def test_medium_does_not_force_quench_stage_without_curve_quench(self) -> None:
        processing = ProcessingState(temperature_c=35.0, cooling_mode="slow_cool")
        thermal_summary = {
            "temperature_max_c": 840.0,
            "observed_temperature_c": 35.0,
            "max_effective_cooling_rate_c_per_s": 18.0,
            "operation_inference": {
                "has_quench": False,
                "has_temper": False,
                "quench_detected_by_curve": False,
            },
        }

        quench_w = {
            "medium_code_resolved": "water_20",
            "severity_effective": 1.0,
            "effect_applied": False,
            "as_quenched_prediction": {
                "retained_austenite_fraction_est": 0.12,
            },
        }
        quench_o = {
            "medium_code_resolved": "oil_20_80",
            "severity_effective": 0.45,
            "effect_applied": False,
            "as_quenched_prediction": {
                "retained_austenite_fraction_est": 0.28,
            },
        }

        bundle_w = build_phase_bundle(
            composition={"Fe": 99.2, "C": 0.8},
            processing=processing,
            system_hint="fe-c",
            phase_model=PhaseModelConfigV3(),
            thermal_summary=thermal_summary,
            quench_summary=quench_w,
        )
        bundle_o = build_phase_bundle(
            composition={"Fe": 99.2, "C": 0.8},
            processing=processing,
            system_hint="fe-c",
            phase_model=PhaseModelConfigV3(),
            thermal_summary=thermal_summary,
            quench_summary=quench_o,
        )

        self.assertEqual(str(bundle_w.stage), str(bundle_o.stage))
        self.assertNotIn(
            str(bundle_w.stage),
            {
                "martensite",
                "martensite_tetragonal",
                "martensite_cubic",
                "troostite_quench",
                "sorbite_quench",
            },
        )


if __name__ == "__main__":
    unittest.main()

