from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, SynthesisProfileV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3ProModeTests(unittest.TestCase):
    def test_pipeline_loads_pro_realistic_preset(self) -> None:
        pipeline = MetallographyPipelineV3()
        payload = pipeline.load_preset("fe_c_eutectoid_pro_realistic")
        request = pipeline.request_from_preset(payload)
        self.assertEqual(str(request.synthesis_profile.generation_mode), "pro_realistic")
        self.assertEqual(str(request.synthesis_profile.system_generator_mode), "system_fe_c")

    def test_pipeline_emits_pro_realistic_metadata_blocks(self) -> None:
        pipeline = MetallographyPipelineV3()
        request = MetallographyRequestV3(
            sample_id="fe_c_pro_mode",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic", system_generator_mode="system_fe_c"),
            resolution=(128, 128),
            seed=501,
        )
        request.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=240.0, temperature_c=840.0),
            ThermalPointV3(time_s=420.0, temperature_c=840.0),
            ThermalPointV3(time_s=800.0, temperature_c=20.0),
        ]

        out = pipeline.generate(request)

        self.assertEqual(str(out.metadata.get("system_generator", {}).get("resolved_mode", "")), "pro_fe_c")
        self.assertIn("continuous_transformation_state", out.metadata)
        self.assertIn("surface_state_summary", out.metadata)
        self.assertIn("reflected_light_model", out.metadata)
        self.assertIn("validation_pro", out.metadata)
        self.assertIn("grain_size_astm_number_proxy", out.metadata.get("validation_pro", {}))
        self.assertIn("two_point_corr_curve", out.metadata.get("validation_pro", {}))
        self.assertIn("bainite_activation_progress", out.metadata.get("kinetics_model", {}))
        self.assertIn("ferrite_effective_exposure_s", out.metadata.get("kinetics_model", {}))
        self.assertIn("pearlite_progress", out.metadata.get("morphology_state", {}))
        cts = out.metadata["continuous_transformation_state"]
        kinetics = out.metadata["kinetics_model"]
        morph = out.metadata["morphology_state"]
        self.assertNotIn("competition_index", cts)
        self.assertNotIn("competition_index", kinetics)
        self.assertNotIn("competition_index", morph)
        for key in (
            "transformation_family",
            "ferrite_morphology_family",
            "bainite_morphology_family",
            "martensite_morphology_family",
            "pearlite_morphology_family",
        ):
            self.assertEqual(cts[key], morph[key], key)
        for key in (
            "ferrite_pearlite_competition_index",
            "ferrite_progress",
            "pearlite_progress",
            "bainite_activation_progress",
            "martensite_conversion_progress",
            "ferrite_effective_exposure_s",
            "pearlite_effective_exposure_s",
            "bainite_effective_exposure_s",
            "martensite_effective_exposure_s",
            "diffusional_equivalent_time_s",
            "hardenability_factor",
            "continuous_cooling_shift_factor",
            "ferrite_nucleation_drive",
            "pearlite_nucleation_drive",
            "bainite_nucleation_drive",
        ):
            self.assertEqual(cts[key], kinetics[key], key)
            self.assertEqual(cts[key], morph[key], key)
        self.assertIn("thermodynamics", cts["provenance"])
        self.assertIn("diffusional_transformations", cts["provenance"])
        self.assertEqual(str(out.metadata.get("final_stage", "")), "pearlite")
        self.assertEqual(out.image_gray.shape, (128, 128))

    def test_pipeline_emits_tempered_family_metadata(self) -> None:
        pipeline = MetallographyPipelineV3()
        request = MetallographyRequestV3(
            sample_id="fe_c_tempered_pro_mode",
            composition_wt={"Fe": 99.15, "C": 0.85},
            system_hint="fe-c",
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic", system_generator_mode="system_fe_c"),
            resolution=(128, 128),
            seed=777,
        )
        request.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=180.0, temperature_c=860.0),
            ThermalPointV3(time_s=360.0, temperature_c=860.0),
            ThermalPointV3(time_s=390.0, temperature_c=320.0),
            ThermalPointV3(time_s=990.0, temperature_c=320.0),
            ThermalPointV3(time_s=1080.0, temperature_c=320.0),
        ]
        out = pipeline.generate(request)
        morph = out.metadata.get("morphology_state", {})
        val = out.metadata.get("validation_pro", {})
        self.assertEqual(str(out.metadata.get("system_generator", {}).get("resolved_mode", "")), "pro_fe_c")
        self.assertEqual(str(morph.get("transformation_family", "")), "tempered_martensitic_family")
        self.assertIn(str(morph.get("martensite_morphology_family", "")), {"mixed_lath_plate", "plate_dominant"})
        self.assertIn("martensite_lath_density", val)
        self.assertIn("artifact_risk_scores", val)
        self.assertIn("scratch_trace_revelation_risk", out.metadata.get("surface_state_summary", {}))
        self.assertIn("false_porosity_pullout_risk", out.metadata.get("surface_state_summary", {}))
        self.assertIn("relief_dominance_risk", val)
        self.assertIn("stain_deposit_contrast_dominance_risk", val)


if __name__ == "__main__":
    unittest.main()
