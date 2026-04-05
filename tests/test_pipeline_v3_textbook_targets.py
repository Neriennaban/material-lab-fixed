from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3TextbookTargetsTests(unittest.TestCase):
    def test_new_fe_c_constituents_are_present_in_textbook_profile(self) -> None:
        pipeline = MetallographyPipelineV3()
        request = MetallographyRequestV3(
            sample_id="target_constituents_troostite",
            composition_wt={"Fe": 99.3, "C": 0.7},
            system_hint="fe-c",
            seed=19,
            resolution=(96, 96),
        )
        request.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0, label="start"),
            ThermalPointV3(time_s=200.0, temperature_c=820.0, label="austenitize"),
            ThermalPointV3(time_s=360.0, temperature_c=820.0, label="hold"),
            ThermalPointV3(time_s=420.0, temperature_c=120.0, label="quench"),
        ]
        request.thermal_program.quench.medium_code = "brine_20_30"
        request.thermal_program.quench.quench_time_s = 40.0
        request.thermal_program.quench.bath_temperature_c = 20.0
        request.thermal_program.quench.sample_temperature_c = 820.0
        request.thermal_program.quench.custom_severity_factor = 0.35
        request.synthesis_profile.profile_id = "textbook_steel_bw"

        output = pipeline.generate(request)
        self.assertEqual(output.metadata.get("final_stage"), "troostite_quench")

        textbook_profile = output.metadata.get("textbook_profile", {})
        self.assertIsInstance(textbook_profile, dict)
        targets = textbook_profile.get("target_microconstituents", [])
        self.assertIsInstance(targets, list)
        self.assertIn("TROOSTITE", targets)
        self.assertIn("CEMENTITE", targets)

    def test_phase_fraction_names_are_merged_into_targets(self) -> None:
        pipeline = MetallographyPipelineV3()
        request = MetallographyRequestV3(
            sample_id="target_constituents_manual",
            composition_wt={"Fe": 99.0, "C": 1.0},
            system_hint="fe-c",
            seed=23,
            resolution=(96, 96),
        )
        request.phase_model.phase_control_mode = "manual_only"
        request.phase_model.manual_phase_fractions = {
            "MARTENSITE_T": 0.82,
            "CEMENTITE": 0.18,
        }
        request.synthesis_profile.profile_id = "textbook_steel_bw"

        output = pipeline.generate(request)
        textbook_profile = output.metadata.get("textbook_profile", {})
        targets = textbook_profile.get("target_microconstituents", [])
        self.assertIn("MARTENSITE_TETRAGONAL", targets)
        self.assertIn("CEMENTITE", targets)


if __name__ == "__main__":
    unittest.main()
