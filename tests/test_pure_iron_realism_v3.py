from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v3 import (
    EtchProfileV3,
    MetallographyRequestV3,
    PrepOperationV3,
    SamplePrepRouteV3,
    SynthesisProfileV3,
)
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PureIronRealismV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = MetallographyPipelineV3()

    def test_fe_pure_preset_stays_bright_and_clean(self) -> None:
        payload = self.pipeline.load_preset("fe_pure_brightfield_v3")
        req = MetallographyRequestV3.from_dict(payload)
        req.resolution = (256, 256)
        out = self.pipeline.generate(req)
        img = out.image_gray.astype(np.float32)
        prep = dict(out.metadata.get("prep_summary", {}))
        etch = dict(out.metadata.get("etch_summary", {}))
        pure = dict(out.metadata.get("pure_iron_baseline", {}))

        self.assertTrue(bool(pure.get("applied", False)))
        self.assertGreater(float(np.quantile(img, 0.05)), 95.0)
        self.assertGreater(float(img.mean()), 135.0)
        self.assertLess(float(prep.get("scratch_mean", 1.0)), 0.12)
        self.assertLess(float(etch.get("stain_level_mean", 1.0)), 0.18)
        self.assertGreater(float(pure.get("boundary_visibility_score", 0.0)), 0.45)

    def test_auto_zero_carbon_ferrite_stays_nearly_white(self) -> None:
        req = MetallographyRequestV3(
            sample_id="pure_auto",
            composition_wt={"Fe": 100.0, "C": 0.0},
            resolution=(256, 256),
            seed=123,
        )
        out = self.pipeline.generate(req)
        img = out.image_gray.astype(np.float32)

        self.assertEqual(str(out.metadata.get("inferred_system", "")), "fe-c")
        self.assertEqual(str(out.metadata.get("final_stage", "")), "ferrite")
        self.assertTrue(
            bool(out.metadata.get("pure_iron_baseline", {}).get("applied", False))
        )
        self.assertGreater(float(img.mean()), 192.0)
        self.assertGreater(float(np.quantile(img, 0.01)), 96.0)
        self.assertGreater(float(np.quantile(img, 0.05)), 125.0)
        self.assertLess(
            float(out.metadata.get("etch_summary", {}).get("stain_level_mean", 1.0)),
            0.12,
        )

    def test_textbook_steel_brightfield_avoids_black_spots(self) -> None:
        req = MetallographyRequestV3(
            sample_id="pure_edu_textbook",
            composition_wt={"Fe": 100.0, "C": 0.0},
            synthesis_profile=SynthesisProfileV3(
                profile_id="textbook_steel_bw",
                generation_mode="edu_engineering",
                composition_sensitivity_mode="educational",
                contrast_target=1.2,
                boundary_sharpness=1.2,
                artifact_level=0.2,
                phase_emphasis_style="contrast_texture",
            ),
            resolution=(512, 512),
            seed=42,
        )
        out = self.pipeline.generate(req)
        img = out.image_gray.astype(np.float32)
        qc = dict(out.metadata.get("quality_metrics", {}))

        self.assertTrue(
            bool(out.metadata.get("pure_iron_baseline", {}).get("applied", False))
        )
        self.assertGreater(float(np.quantile(img, 0.01)), 72.0)
        self.assertGreaterEqual(float(np.quantile(img, 0.05)), 124.0)
        self.assertLess(float((img < 40.0).mean()), 0.0005)
        self.assertLess(
            float(qc.get("unexpected_dark_spot_largest_component_px", 0.0)), 28.0
        )

    def test_pro_realistic_ferrite_empty_prep_uses_bright_negative_control(
        self,
    ) -> None:
        req = MetallographyRequestV3(
            sample_id="pure_pro_ferrite",
            composition_wt={"Fe": 100.0, "C": 0.0},
            synthesis_profile=SynthesisProfileV3(
                profile_id="textbook_steel_bw",
                generation_mode="pro_realistic",
                composition_sensitivity_mode="realistic",
                contrast_target=1.2,
                boundary_sharpness=1.2,
                artifact_level=0.2,
                phase_emphasis_style="contrast_texture",
            ),
            resolution=(512, 512),
            seed=42,
        )
        out = self.pipeline.generate(req)
        img = out.image_gray.astype(np.float32)
        pure = dict(out.metadata.get("pure_iron_baseline", {}))
        prep = dict(out.metadata.get("prep_summary", {}))
        qc = dict(out.metadata.get("quality_metrics", {}))

        self.assertEqual(
            str(out.metadata.get("engineering_trace", {}).get("backend", "")),
            "pro_realistic_fe_c_v1",
        )
        self.assertTrue(bool(pure.get("applied", False)))
        self.assertTrue(bool(prep.get("implicit_baseline_route_applied", False)))
        self.assertGreater(float(img.mean()), 176.0)
        self.assertGreater(float(np.quantile(img, 0.01)), 70.0)
        self.assertGreater(float(np.quantile(img, 0.05)), 100.0)
        self.assertEqual(
            str(
                out.metadata.get("spatial_morphology_state", {})
                .get("pure_ferrite_generator", {})
                .get("generator", "")
            ),
            "pure_ferrite_power_voronoi_v1",
        )
        self.assertTrue(bool(out.metadata.get("single_phase_negative_control", False)))
        self.assertFalse(
            bool(out.metadata.get("multiphase_separability_applicable", True))
        )
        self.assertTrue(
            bool(out.metadata.get("textbook_profile", {}).get("pass", False))
        )
        self.assertTrue(bool(qc.get("passed", False)))

    def test_aggressive_pure_iron_route_raises_artifacts_vs_baseline(self) -> None:
        baseline_payload = self.pipeline.load_preset("fe_pure_brightfield_v3")
        baseline = MetallographyRequestV3.from_dict(baseline_payload)
        baseline.resolution = (192, 192)
        baseline_out = self.pipeline.generate(baseline)

        aggressive = MetallographyRequestV3.from_dict(baseline_payload)
        aggressive.sample_id = "Fe_pure_aggressive"
        aggressive.resolution = (192, 192)
        aggressive.prep_route = SamplePrepRouteV3(
            steps=[
                PrepOperationV3(
                    method="grinding_600",
                    duration_s=180.0,
                    abrasive_um=28.0,
                    load_n=32.0,
                    rpm=280.0,
                    coolant="none",
                ),
                PrepOperationV3(
                    method="polishing_1um",
                    duration_s=210.0,
                    abrasive_um=1.0,
                    load_n=20.0,
                    rpm=160.0,
                    cloth_type="long_nap",
                    slurry_type="diamond",
                ),
            ],
            roughness_target_um=0.07,
            contamination_level=0.04,
        )
        aggressive.etch_profile = EtchProfileV3(
            reagent="nital_2",
            time_s=12.0,
            temperature_c=22.0,
            agitation="active",
            overetch_factor=1.2,
        )
        aggressive_out = self.pipeline.generate(aggressive)

        base_img = baseline_out.image_gray.astype(np.float32)
        aggr_img = aggressive_out.image_gray.astype(np.float32)
        base_prep = dict(baseline_out.metadata.get("prep_summary", {}))
        aggr_prep = dict(aggressive_out.metadata.get("prep_summary", {}))
        base_etch = dict(baseline_out.metadata.get("etch_summary", {}))
        aggr_etch = dict(aggressive_out.metadata.get("etch_summary", {}))

        self.assertLessEqual(
            float(np.quantile(aggr_img, 0.05)), float(np.quantile(base_img, 0.05))
        )
        self.assertGreater(
            float(aggr_prep.get("scratch_mean", 0.0)),
            float(base_prep.get("scratch_mean", 0.0)),
        )
        self.assertGreater(
            float(aggr_prep.get("tempering_by_grinding_risk", 0.0)),
            float(base_prep.get("tempering_by_grinding_risk", 0.0)),
        )
        self.assertGreater(
            float(aggr_etch.get("stain_level_mean", 0.0)),
            float(base_etch.get("stain_level_mean", 0.0)),
        )
        self.assertLess(
            float(
                aggressive_out.metadata.get("pure_iron_baseline", {}).get(
                    "cleanliness_score", 1.0
                )
            ),
            float(
                baseline_out.metadata.get("pure_iron_baseline", {}).get(
                    "cleanliness_score", 0.0
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
