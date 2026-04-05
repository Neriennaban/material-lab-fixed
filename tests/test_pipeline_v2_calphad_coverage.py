from __future__ import annotations

import unittest

from core.contracts_v2 import GenerationRequestV2, ProcessingState
from core.pipeline_v2 import GenerationPipelineV2


class PipelineV2CalphadCoverageTests(unittest.TestCase):
    def test_auto_uses_calphad_for_supported_systems(self) -> None:
        pipeline = GenerationPipelineV2()
        cases = [
            ("fe-c", {"Fe": 99.2, "C": 0.8}, 780.0, "equilibrium"),
            ("fe-si", {"Fe": 98.6, "Si": 1.4}, 900.0, "equilibrium"),
            ("al-si", {"Al": 88.0, "Si": 12.0}, 580.0, "equilibrium"),
            ("cu-zn", {"Cu": 68.0, "Zn": 32.0}, 700.0, "equilibrium"),
            ("al-cu-mg", {"Al": 93.0, "Cu": 4.4, "Mg": 1.5}, 520.0, "solutionized"),
        ]
        for system, composition, temp, mode in cases:
            with self.subTest(system=system):
                out = pipeline.generate(
                    GenerationRequestV2(
                        mode="direct",
                        composition=composition,
                        processing=ProcessingState(temperature_c=temp, cooling_mode=mode),
                        generator="auto",
                        resolution=(64, 64),
                        strict_validation=True,
                    )
                )
                self.assertEqual(out.metadata.get("auto_generator", {}).get("selected_generator"), "calphad_phase")
                self.assertIn("calphad", out.metadata)
                eq = out.metadata.get("calphad", {}).get("equilibrium_result", {})
                self.assertIsInstance(eq, dict)
                self.assertTrue(eq.get("stable_phases"))


if __name__ == "__main__":
    unittest.main()
