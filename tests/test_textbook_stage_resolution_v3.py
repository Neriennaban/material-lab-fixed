from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class TextbookStageResolutionV3Tests(unittest.TestCase):
    def test_cast_profile_resolves_to_solid_stage(self) -> None:
        pipeline = MetallographyPipelineV3()
        req = MetallographyRequestV3(
            sample_id="alsi_cast_stage",
            composition_wt={"Al": 88.0, "Si": 12.0},
            system_hint="al-si",
            resolution=(96, 96),
            seed=901,
            strict_validation=True,
        )
        req.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=120.0, temperature_c=730.0),
            ThermalPointV3(time_s=240.0, temperature_c=730.0),
            ThermalPointV3(time_s=900.0, temperature_c=30.0),
        ]
        out = pipeline.generate(req)
        stage = str(out.metadata.get("final_stage", "")).lower()
        self.assertNotIn("liquid", stage)
        self.assertIn(stage, {"alpha_eutectic", "eutectic", "primary_si_eutectic"})

    def test_normalize_profile_resolves_to_room_temperature_observation(self) -> None:
        pipeline = MetallographyPipelineV3()
        req = MetallographyRequestV3(
            sample_id="steel_norm_stage",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            resolution=(96, 96),
            seed=902,
            strict_validation=True,
        )
        req.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=240.0, temperature_c=780.0),
            ThermalPointV3(time_s=420.0, temperature_c=780.0),
            ThermalPointV3(time_s=1300.0, temperature_c=25.0),
        ]
        out = pipeline.generate(req)
        stage = str(out.metadata.get("final_stage", "")).lower()
        self.assertNotIn("liquid", stage)
        self.assertIn(stage, {"pearlite", "alpha_pearlite", "pearlite_cementite"})


if __name__ == "__main__":
    unittest.main()

