from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.contracts_v2 import GenerationRequestV2, ProcessRoute, ProcessingOperation, ProcessingState
from core.pipeline_v2 import GenerationPipelineV2


class PipelineV2RouteBatchTests(unittest.TestCase):
    def test_route_batch_exports_step_series_and_metadata(self) -> None:
        pipeline = GenerationPipelineV2()
        request = GenerationRequestV2(
            mode="batch",
            composition={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            generator="phase_map",
            generator_params={"system": "fe-c", "stage": "auto"},
            seed=909,
            resolution=(160, 160),
            strict_validation=True,
            route_policy="route_driven",
            process_route=ProcessRoute(
                route_name="qt",
                step_preview_enabled=True,
                operations=[
                    ProcessingOperation(method="quench_water", temperature_c=860.0, duration_min=30.0, cooling_mode="quenched"),
                    ProcessingOperation(method="temper_medium", temperature_c=400.0, duration_min=90.0, cooling_mode="tempered"),
                ],
            ),
        )

        with tempfile.TemporaryDirectory() as td:
            result = pipeline.generate_batch(requests=[request], output_dir=td, file_prefix="route_ut")
            self.assertEqual(len(result.rows), 1)
            row = result.rows[0]
            self.assertTrue(row["validation_passed"])
            self.assertEqual(row["route_name"], "qt")
            self.assertEqual(int(row["step_count"]), 2)
            self.assertNotEqual(row["final_stage"], "")
            self.assertGreater(float(row["hv_estimate"]), 0.0)
            self.assertGreater(float(row["uts_estimate_mpa"]), 0.0)

            metadata_path = Path(row["metadata_path"])
            self.assertTrue(metadata_path.exists())
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIn("process_route", payload)
            self.assertIn("route_timeline", payload)
            self.assertIn("resolved_stage_by_step", payload)
            self.assertIn("final_effect_vector", payload)
            self.assertIn("property_indicators", payload)
            self.assertIn("route_validation", payload)

            step_paths = payload.get("step_series_images", [])
            self.assertEqual(len(step_paths), 2)
            for step in step_paths:
                self.assertTrue(Path(step).exists())


if __name__ == "__main__":
    unittest.main()

