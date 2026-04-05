from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3LegacyWaterMappingTests(unittest.TestCase):
    def test_legacy_water_is_mapped_by_bath_temperature(self) -> None:
        pipeline = MetallographyPipelineV3()

        request_hot = MetallographyRequestV3(
            sample_id="legacy_water_hot",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            resolution=(96, 96),
            seed=333,
        )
        request_hot.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=300.0, temperature_c=840.0),
            ThermalPointV3(time_s=480.0, temperature_c=840.0),
            ThermalPointV3(time_s=560.0, temperature_c=100.0),
        ]
        request_hot.thermal_program.quench.medium_code = "water"
        request_hot.thermal_program.quench.bath_temperature_c = 100.0

        out_hot = pipeline.generate(request_hot)
        resolved_hot = str(dict(out_hot.metadata.get("quench_medium_profile", {})).get("medium_code_resolved", ""))
        self.assertEqual(resolved_hot, "water_100")

        request_cold = MetallographyRequestV3(
            sample_id="legacy_water_cold",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            resolution=(96, 96),
            seed=334,
        )
        request_cold.thermal_program.points = request_hot.thermal_program.points
        request_cold.thermal_program.quench.medium_code = "water"
        request_cold.thermal_program.quench.bath_temperature_c = 20.0

        out_cold = pipeline.generate(request_cold)
        resolved_cold = str(dict(out_cold.metadata.get("quench_medium_profile", {})).get("medium_code_resolved", ""))
        self.assertEqual(resolved_cold, "water_20")


if __name__ == "__main__":
    unittest.main()
