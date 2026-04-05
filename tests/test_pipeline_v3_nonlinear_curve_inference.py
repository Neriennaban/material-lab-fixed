from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, ThermalPointV3, ThermalTransitionV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3NonlinearCurveInferenceTests(unittest.TestCase):
    def test_transition_mode_changes_inference_summary(self) -> None:
        pipeline = MetallographyPipelineV3()
        base_points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=260.0, temperature_c=840.0),
            ThermalPointV3(time_s=420.0, temperature_c=840.0),
            ThermalPointV3(time_s=520.0, temperature_c=40.0),
        ]

        req_linear = MetallographyRequestV3(
            sample_id="linear_curve",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            resolution=(96, 96),
            seed=301,
        )
        req_linear.thermal_program.points = list(base_points)
        req_linear.thermal_program.points[2] = ThermalPointV3(
            time_s=420.0,
            temperature_c=840.0,
            transition_to_next=ThermalTransitionV3(model="linear", curvature=1.0, segment_medium_code="water_20"),
        )
        req_linear.thermal_program.quench.medium_code = "water_20"

        req_non = MetallographyRequestV3(
            sample_id="nonlinear_curve",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            resolution=(96, 96),
            seed=301,
        )
        req_non.thermal_program.points = list(base_points)
        req_non.thermal_program.points[2] = ThermalPointV3(
            time_s=420.0,
            temperature_c=840.0,
            transition_to_next=ThermalTransitionV3(model="sigmoid", curvature=3.2, segment_medium_code="water_20"),
        )
        req_non.thermal_program.quench.medium_code = "water_20"

        out_linear = pipeline.generate(req_linear)
        out_non = pipeline.generate(req_non)

        thermal_lin = dict(out_linear.metadata.get("thermal_program_summary", {}))
        thermal_non = dict(out_non.metadata.get("thermal_program_summary", {}))
        self.assertNotEqual(
            float(thermal_lin.get("max_effective_cooling_rate_c_per_s", 0.0)),
            float(thermal_non.get("max_effective_cooling_rate_c_per_s", 0.0)),
        )
        self.assertIn("segment_transition_report", out_non.metadata)
        self.assertTrue(len(list(out_non.metadata.get("segment_transition_report", []))) > 0)


if __name__ == "__main__":
    unittest.main()

