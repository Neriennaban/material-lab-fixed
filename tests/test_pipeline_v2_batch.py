from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.contracts_v2 import GenerationRequestV2, ProcessingState
from core.pipeline_v2 import GenerationPipelineV2


class PipelineV2BatchTests(unittest.TestCase):
    def test_batch_exports_index_and_handles_partial_errors(self) -> None:
        pipeline = GenerationPipelineV2()
        requests = [
            GenerationRequestV2(
                mode="batch",
                composition={"Fe": 99.2, "C": 0.8},
                processing=ProcessingState(temperature_c=780.0, cooling_mode="equilibrium"),
                generator="dendritic_cast",
                generator_params={},
                seed=100,
                resolution=(256, 256),
                strict_validation=True,
            ),
            GenerationRequestV2(
                mode="batch",
                composition={"Xx": 100.0},
                processing=ProcessingState(),
                generator="dendritic_cast",
                generator_params={},
                seed=101,
                resolution=(256, 256),
                strict_validation=True,
            ),
        ]

        with tempfile.TemporaryDirectory() as td:
            result = pipeline.generate_batch(requests=requests, output_dir=td, file_prefix="ut")
            self.assertEqual(len(result.rows), 2)
            self.assertTrue(result.csv_index_path.exists())

            first = result.rows[0]
            second = result.rows[1]
            self.assertTrue(first["validation_passed"])
            self.assertTrue(Path(first["image_path"]).exists())
            self.assertTrue(Path(first["metadata_path"]).exists())
            self.assertFalse(second["validation_passed"])
            self.assertNotEqual(second["error"], "")

    def test_metadata_contains_v2_fields(self) -> None:
        pipeline = GenerationPipelineV2()
        output = pipeline.generate(
            GenerationRequestV2(
                mode="direct",
                composition={"Al": 88.0, "Si": 12.0},
                processing=ProcessingState(temperature_c=580.0, cooling_mode="equilibrium"),
                generator="auto",
                generator_params={},
                seed=33,
                resolution=(192, 192),
                strict_validation=True,
            )
        )
        payload = output.metadata_json_safe()
        self.assertIn("generator_version", payload)
        self.assertIn("diagram_snapshot_params", payload)
        self.assertIn("phase_fraction_estimate", payload)
        self.assertIn("validation_report", payload)
        self.assertIn("auto_generator", payload)


if __name__ == "__main__":
    unittest.main()
