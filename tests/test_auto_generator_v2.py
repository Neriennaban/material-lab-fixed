from __future__ import annotations

import unittest

from core.contracts_v2 import GenerationRequestV2, ProcessRoute, ProcessingOperation, ProcessingState
from core.pipeline_v2 import GenerationPipelineV2


class AutoGeneratorV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = GenerationPipelineV2()

    def test_si_monocrystal_uses_calphad_phase(self) -> None:
        out = self.pipeline.generate(
            GenerationRequestV2(
                mode="direct",
                composition={"Si": 99.999},
                processing=ProcessingState(temperature_c=25.0, cooling_mode="equilibrium"),
                generator="auto",
                generator_params={},
                seed=42,
                resolution=(128, 128),
                strict_validation=True,
            )
        )
        auto = out.metadata.get("auto_generator", {})
        self.assertTrue(auto.get("enabled"))
        self.assertEqual(auto.get("selected_generator"), "calphad_phase")
        self.assertIn("calphad", out.metadata)

    def test_route_driven_fe_c_prefers_calphad_phase(self) -> None:
        out = self.pipeline.generate(
            GenerationRequestV2(
                mode="direct",
                composition={"Fe": 99.2, "C": 0.8},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
                generator="auto",
                generator_params={},
                seed=77,
                resolution=(160, 160),
                strict_validation=True,
                route_policy="route_driven",
                process_route=ProcessRoute(
                    route_name="qt",
                    operations=[
                        ProcessingOperation(method="quench_water", temperature_c=860.0, duration_min=30.0, cooling_mode="quenched"),
                        ProcessingOperation(method="temper_medium", temperature_c=400.0, duration_min=90.0, cooling_mode="tempered"),
                    ],
                ),
            )
        )
        auto = out.metadata.get("auto_generator", {})
        self.assertEqual(auto.get("selected_generator"), "calphad_phase")
        self.assertIn("calphad", out.metadata)
        self.assertIn("equilibrium_result", out.metadata.get("calphad", {}))

    def test_custom_system_is_blocked_in_strict_calphad_mode(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.pipeline.generate(
                GenerationRequestV2(
                    mode="direct",
                    composition={"Ni": 70.0, "Cr": 20.0, "Mo": 10.0},
                    processing=ProcessingState(temperature_c=980.0, cooling_mode="equilibrium"),
                    generator="auto",
                    generator_params={},
                    seed=108,
                    resolution=(128, 128),
                    strict_validation=True,
                )
            )
        self.assertIn("SYSTEM_UNSUPPORTED", str(ctx.exception))

    def test_universal_auto_alias_is_supported(self) -> None:
        req = GenerationRequestV2.from_dict(
            {
                "composition": {"Fe": 99.2, "C": 0.8},
                "processing": {"temperature_c": 780.0, "cooling_mode": "equilibrium"},
                "generator": "universal_auto",
                "resolution": [128, 128],
                "seed": 11,
            }
        )
        self.assertEqual(req.generator, "auto")


if __name__ == "__main__":
    unittest.main()
