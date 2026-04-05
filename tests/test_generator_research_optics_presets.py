from __future__ import annotations

import json
import unittest
from pathlib import Path

from ui_qt.sample_factory_window_v3 import LAB_RESEARCH_TEMPLATE_PRESETS


class GeneratorResearchOpticsPresetsTests(unittest.TestCase):
    def test_research_template_list_contains_expected_presets(self) -> None:
        preset_keys = {key for _, key in LAB_RESEARCH_TEMPLATE_PRESETS}
        self.assertIn("fe_c_eutectoid_research_optics_bessel", preset_keys)
        self.assertIn("fe_c_eutectoid_research_optics_stir", preset_keys)
        self.assertIn("fe_c_eutectoid_research_optics_hybrid", preset_keys)

    def test_research_preset_files_define_psf_profiles(self) -> None:
        expected = {
            "fe_c_eutectoid_research_optics_bessel": "bessel_extended_dof",
            "fe_c_eutectoid_research_optics_stir": "stir_sectioning",
            "fe_c_eutectoid_research_optics_hybrid": "lens_axicon_hybrid",
        }
        for name, profile in expected.items():
            path = Path("presets_v3") / f"{name}.json"
            self.assertTrue(path.exists(), msg=str(path))
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            self.assertEqual(str(payload.get("microscope_profile", {}).get("psf_profile", "")), profile)
            self.assertEqual(str(payload.get("synthesis_profile", {}).get("generation_mode", "")), "pro_realistic")


if __name__ == "__main__":
    unittest.main()
