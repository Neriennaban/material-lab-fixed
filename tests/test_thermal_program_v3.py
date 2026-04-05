from __future__ import annotations

import unittest

from core.contracts_v3 import ThermalPointV3, ThermalProgramV3, ThermalTransitionV3
from core.metallography_v3.thermal_program_v3 import (
    build_segments,
    effective_processing_from_thermal,
    infer_operations_from_thermal_program,
    sample_thermal_program,
    validate_thermal_program,
)


class ThermalProgramV3Tests(unittest.TestCase):
    def test_validate_and_segments(self) -> None:
        program = ThermalProgramV3(
            points=[
                ThermalPointV3(time_s=0.0, temperature_c=20.0),
                ThermalPointV3(time_s=120.0, temperature_c=850.0),
                ThermalPointV3(time_s=300.0, temperature_c=850.0),
                ThermalPointV3(time_s=420.0, temperature_c=30.0),
            ],
            sampling_mode="per_degree",
            degree_step_c=10.0,
            max_frames=200,
        )
        check = validate_thermal_program(program)
        self.assertTrue(check["is_valid"])
        segs = build_segments(program.points)
        self.assertGreaterEqual(len(segs), 3)
        kinds = {s.kind for s in segs}
        self.assertIn("heat", kinds)
        self.assertIn("hold", kinds)
        self.assertIn("cool", kinds)

    def test_sampling(self) -> None:
        program = ThermalProgramV3(
            points=[
                ThermalPointV3(time_s=0.0, temperature_c=20.0),
                ThermalPointV3(time_s=180.0, temperature_c=720.0),
            ],
            sampling_mode="per_degree",
            degree_step_c=20.0,
            max_frames=80,
        )
        rows = sample_thermal_program(program)
        self.assertTrue(rows)
        self.assertLessEqual(len(rows), 80)
        self.assertIn("model", rows[0])
        self.assertIn("segment_medium_code", rows[0])

    def test_old_points_payload_without_transition_is_supported(self) -> None:
        program = ThermalProgramV3.from_dict(
            {
                "points": [
                    {"time_s": 0.0, "temperature_c": 20.0, "label": "A"},
                    {"time_s": 200.0, "temperature_c": 800.0, "label": "B"},
                ],
                "sampling_mode": "points",
            }
        )
        self.assertEqual(len(program.points), 2)
        self.assertEqual(program.points[0].transition_to_next.model, "linear")
        rows = sample_thermal_program(program)
        self.assertTrue(rows)

    def test_nonlinear_transition_changes_midpoint_temperature(self) -> None:
        linear_program = ThermalProgramV3(
            points=[
                ThermalPointV3(
                    time_s=0.0,
                    temperature_c=20.0,
                    transition_to_next=ThermalTransitionV3(model="linear", curvature=1.0, segment_medium_code="air"),
                ),
                ThermalPointV3(time_s=100.0, temperature_c=820.0),
            ],
            sampling_mode="per_degree",
            degree_step_c=80.0,
            max_frames=40,
        )
        nonlinear_program = ThermalProgramV3(
            points=[
                ThermalPointV3(
                    time_s=0.0,
                    temperature_c=20.0,
                    transition_to_next=ThermalTransitionV3(model="sigmoid", curvature=3.0, segment_medium_code="air"),
                ),
                ThermalPointV3(time_s=100.0, temperature_c=820.0),
            ],
            sampling_mode="per_degree",
            degree_step_c=80.0,
            max_frames=40,
        )
        rows_lin = sample_thermal_program(linear_program)
        rows_non = sample_thermal_program(nonlinear_program)
        self.assertEqual(len(rows_lin), len(rows_non))
        mid_idx = max(1, len(rows_lin) // 4)
        self.assertNotAlmostEqual(
            float(rows_lin[mid_idx]["temperature_c"]),
            float(rows_non[mid_idx]["temperature_c"]),
            places=3,
        )

    def test_quench_requires_curve_segment_even_with_quench_medium(self) -> None:
        program = ThermalProgramV3(
            points=[
                ThermalPointV3(time_s=0.0, temperature_c=20.0),
                ThermalPointV3(time_s=220.0, temperature_c=840.0),
                ThermalPointV3(time_s=460.0, temperature_c=840.0),
                ThermalPointV3(time_s=760.0, temperature_c=700.0),
            ],
            sampling_mode="per_degree",
            degree_step_c=6.0,
            max_frames=220,
        )
        program.quench.medium_code = "water_20"
        program.quench.bath_temperature_c = 20.0
        program.quench.sample_temperature_c = 840.0
        op_payload = infer_operations_from_thermal_program(program)
        op_summary = dict(op_payload.get("summary", {}))
        self.assertFalse(bool(op_summary.get("has_quench", False)))
        self.assertFalse(bool(op_summary.get("quench_detected_by_curve", False)))
        self.assertEqual(str(op_summary.get("quench_presence_rule", "")), "curve_only")
        self.assertEqual(str(op_summary.get("stage_inference_profile", "")), "fe_c_temper_curve_v2")

        processing, runtime_summary, quench_summary = effective_processing_from_thermal(program)
        self.assertNotEqual(str(processing.cooling_mode), "quenched")
        self.assertFalse(bool(runtime_summary.get("quench_effect_applied", True)))
        self.assertEqual(str(runtime_summary.get("quench_effect_reason", "")), "no_quench_segment")
        self.assertFalse(bool(quench_summary.get("effect_applied", True)))
        self.assertEqual(str(quench_summary.get("effect_reason", "")), "no_quench_segment")
        warns = list(quench_summary.get("warnings", []))
        self.assertTrue(any("влияние среды не применяется" in str(w).lower() for w in warns))


if __name__ == "__main__":
    unittest.main()
