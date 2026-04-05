from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3TextbookPresetsTests(unittest.TestCase):
    def test_textbook_presets_have_readability_pass(self) -> None:
        pipeline = MetallographyPipelineV3()
        stems = [
            "fe_c_hypoeutectoid_textbook",
            "fe_c_eutectoid_textbook",
            "fe_c_hypereutectoid_textbook",
            "steel_quenched_textbook",
            "steel_tempered_400_textbook",
            "LR1_ASTM5",
            "steel45_normalized_textbook",
            "steel45_improved_textbook",
            "steel_u8_tool_textbook",
            "cast_iron_grey_textbook",
            "alsi_eutectic_textbook",
            "brass_alpha_beta_textbook",
        ]
        for stem in stems:
            payload = pipeline.load_preset(stem)
            req = MetallographyRequestV3.from_dict(payload)
            req.resolution = (128, 128)
            out = pipeline.generate(req)
            textbook_profile = out.metadata.get("textbook_profile", {})
            self.assertIsInstance(textbook_profile, dict, msg=f"{stem}: textbook_profile missing")
            self.assertIn("pass", textbook_profile, msg=f"{stem}: textbook_profile.pass missing")
            if not bool(textbook_profile.get("pass")):
                self.assertIsInstance(
                    textbook_profile.get("achieved_readability", {}),
                    dict,
                    msg=f"{stem}: achieved_readability missing when pass=false",
                )
            sysgen = out.metadata.get("system_generator", {})
            self.assertIsInstance(sysgen, dict, msg=f"{stem}: system_generator missing")
            self.assertTrue(str(sysgen.get("resolved_mode", "")).startswith("system_"), msg=f"{stem}: invalid system_generator")
            electron_guidance = out.metadata.get("electron_microscopy_guidance", {})
            self.assertIsInstance(electron_guidance, dict, msg=f"{stem}: electron_microscopy_guidance missing")
            self.assertIn("primary_recommendation", electron_guidance, msg=f"{stem}: electron primary recommendation missing")
            self.assertIn("sem_guidance", electron_guidance, msg=f"{stem}: sem guidance missing")
            self.assertIn("tem_guidance", electron_guidance, msg=f"{stem}: tem guidance missing")
            if str(out.metadata.get("inferred_system", "")) == "fe-c":
                fe_c_unified = dict(sysgen.get("fe_c_unified", {}))
                self.assertTrue(bool(fe_c_unified.get("enabled", False)), msg=f"{stem}: fe_c_unified not enabled")
                props = dict(out.metadata.get("property_indicators", {}))
                self.assertEqual(props.get("property_model_source"), "hybrid_textbook_calculator_v1", msg=f"{stem}: hybrid source")
                self.assertEqual(props.get("reference_dataset"), "textbook_material_properties", msg=f"{stem}: reference dataset")
                if payload.get("expected_properties"):
                    expected_validation = out.metadata.get("expected_properties_validation", {})
                    self.assertIsInstance(expected_validation, dict, msg=f"{stem}: expected_properties_validation missing")
                    self.assertIn("pass", expected_validation, msg=f"{stem}: expected_properties_validation.pass missing")


if __name__ == "__main__":
    unittest.main()
