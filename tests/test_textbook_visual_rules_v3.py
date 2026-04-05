from __future__ import annotations

import unittest

import numpy as np

from core.generator_calphad_phase import generate_calphad_phase_structure


class TextbookVisualRulesV3Tests(unittest.TestCase):
    def test_fe_c_tone_hierarchy_textbook_profile(self) -> None:
        out = generate_calphad_phase_structure(
            size=(192, 192),
            seed=4401,
            system="fe-c",
            phase_fractions={"FERRITE": 0.52, "PEARLITE": 0.34, "CEMENTITE": 0.14},
            generation_mode="edu_engineering",
            phase_emphasis_style="contrast_texture",
            phase_fraction_tolerance_pct=20.0,
            visual_profile_id="textbook_steel_bw",
        )
        image = out["image"].astype(np.float32)
        masks = out.get("phase_masks", {})
        ferrite = image[masks["FERRITE"] > 0].mean()
        pearlite = image[masks["PEARLITE"] > 0].mean()
        cementite = image[masks["CEMENTITE"] > 0].mean()
        self.assertGreater(ferrite, pearlite)
        self.assertGreater(pearlite, cementite)

        vis = out.get("metadata", {}).get("phase_visibility_report", {})
        self.assertGreater(float(vis.get("separability_score", 0.0)), 0.18)


if __name__ == "__main__":
    unittest.main()
