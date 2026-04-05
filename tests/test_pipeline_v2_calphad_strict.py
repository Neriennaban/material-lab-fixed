from __future__ import annotations

import unittest
from pathlib import Path

from core.contracts_v2 import GenerationRequestV2, ProcessingState, ThermoBackendConfig
from core.pipeline_v2 import GenerationPipelineV2


class PipelineV2CalphadStrictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = GenerationPipelineV2()

    def test_custom_system_blocked(self) -> None:
        req = GenerationRequestV2(
            mode="direct",
            composition={"Ni": 70.0, "Cr": 20.0, "Mo": 10.0},
            processing=ProcessingState(temperature_c=950.0, cooling_mode="equilibrium"),
            generator="auto",
            resolution=(64, 64),
            strict_validation=True,
        )
        with self.assertRaises(ValueError) as ctx:
            self.pipeline.generate(req)
        self.assertIn("SYSTEM_UNSUPPORTED", str(ctx.exception))

    def test_missing_db_is_hard_stop(self) -> None:
        thermo = ThermoBackendConfig(
            strict_mode=True,
            db_overrides={"fe-c": str(Path("missing") / "fe_c.tdb")},
        )
        req = GenerationRequestV2(
            mode="direct",
            composition={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=780.0, cooling_mode="equilibrium"),
            generator="auto",
            thermo=thermo,
            resolution=(64, 64),
            strict_validation=True,
        )
        with self.assertRaises(ValueError) as ctx:
            self.pipeline.generate(req)
        self.assertIn("DB_MISSING", str(ctx.exception))

    def test_supported_system_returns_calphad_metadata(self) -> None:
        req = GenerationRequestV2(
            mode="direct",
            composition={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=780.0, cooling_mode="equilibrium"),
            generator="auto",
            resolution=(64, 64),
            strict_validation=True,
        )
        out = self.pipeline.generate(req)
        self.assertEqual(out.metadata.get("auto_generator", {}).get("selected_generator"), "calphad_phase")
        calphad = out.metadata.get("calphad", {})
        self.assertIsInstance(calphad, dict)
        self.assertIn("equilibrium_result", calphad)


if __name__ == "__main__":
    unittest.main()
