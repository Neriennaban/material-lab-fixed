from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v3 import MetallographyRequestV3
from core.virtual_lab_backend import MicroscopeState, VirtualLabBackend


class VirtualLabBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = VirtualLabBackend(generator_version="v3.0.0+tests")
        payload = cls.backend.pipeline.load_preset("steel_tempered_400_textbook")
        payload["resolution"] = [384, 384]
        payload["sample_id"] = "test_realtime_backend"
        cls.slide = cls.backend.generate_slide(payload)
        pure_payload = cls.backend.pipeline.load_preset("fe_pure_brightfield_v3")
        pure_payload["resolution"] = [256, 256]
        pure_payload["sample_id"] = "test_pure_iron_backend"
        cls.pure_slide = cls.backend.generate_slide(pure_payload)

    def test_generated_slide_contains_process_metadata(self) -> None:
        meta = self.slide.metadata
        self.assertIn("etch_summary", meta)
        self.assertIn("prep_summary", meta)
        self.assertIn("phase_model", meta)
        self.assertIn("microscope_ready", meta)
        self.assertTrue(self.slide.inferred_system)
        self.assertTrue(self.slide.final_stage)

    def test_render_frame_returns_expected_shape_and_focus(self) -> None:
        frame, meta = self.backend.render_microscope_frame(
            self.slide,
            MicroscopeState(
                objective=400, stage_x=0.45, stage_y=0.55, output_size=(384, 384)
            ),
        )
        self.assertEqual(frame.shape, (384, 384))
        self.assertGreaterEqual(float(meta["focus_quality"]), 0.95)
        self.assertEqual(int(meta["objective"]), 400)
        self.assertEqual(meta["sample_id"], "test_realtime_backend")
        self.assertIn("pyramid_level", meta)

    def test_render_frame_propagates_psf_profile_metadata(self) -> None:
        _frame, meta = self.backend.render_microscope_frame(
            self.slide,
            MicroscopeState(
                objective=400,
                stage_x=0.45,
                stage_y=0.55,
                output_size=(384, 384),
                psf_profile="stir_sectioning",
                psf_strength=0.8,
                sectioning_shear_deg=40.0,
            ),
        )
        self.assertEqual(str(meta["psf_profile"]), "stir_sectioning")
        self.assertTrue(bool(meta["sectioning_active"]))
        self.assertGreater(float(meta["sectioning_suppression_score"]), 0.0)

    def test_render_frame_propagates_pure_iron_baseline_metadata(self) -> None:
        _frame, meta = self.backend.render_microscope_frame(
            self.pure_slide,
            MicroscopeState(
                objective=200, stage_x=0.5, stage_y=0.5, output_size=(256, 256)
            ),
        )
        self.assertTrue(bool(meta.get("pure_iron_baseline_applied", False)))
        self.assertGreater(float(meta.get("pure_iron_cleanliness_score", 0.0)), 0.5)
        self.assertGreater(
            float(meta.get("pure_iron_dark_defect_suppression", 0.0)), 0.4
        )
        self.assertFalse(bool(meta.get("pure_iron_dark_defect_warning", False)))
        self.assertEqual(
            str(meta.get("pure_iron_electropolish_profile", "")),
            "pure_iron_electropolish",
        )
        self.assertGreater(
            float(meta.get("pure_iron_polarized_extinction_score", 0.0)), 0.8
        )
        self.assertTrue(bool(meta.get("single_phase_negative_control", False)))
        self.assertEqual(
            str(meta.get("optical_recommendation", {}).get("default_mode", "")),
            "brightfield",
        )
        self.assertIsInstance(meta.get("electron_microscopy_guidance", {}), dict)

    def test_render_frame_propagates_optical_mode_and_changes_image(self) -> None:
        bright, bright_meta = self.backend.render_microscope_frame(
            self.pure_slide,
            MicroscopeState(
                objective=200,
                stage_x=0.5,
                stage_y=0.5,
                output_size=(256, 256),
                optical_mode="brightfield",
            ),
        )
        dark, dark_meta = self.backend.render_microscope_frame(
            self.pure_slide,
            MicroscopeState(
                objective=200,
                stage_x=0.5,
                stage_y=0.5,
                output_size=(256, 256),
                optical_mode="darkfield",
                optical_mode_parameters={"scatter_sensitivity": 1.2},
            ),
        )
        self.assertEqual(str(bright_meta.get("optical_mode", "")), "brightfield")
        self.assertEqual(str(dark_meta.get("optical_mode", "")), "darkfield")
        self.assertGreater(
            float(abs(bright.astype("float32").mean() - dark.astype("float32").mean())),
            5.0,
        )
        self.assertIn("scatter_pass_fraction", dark_meta)

    def test_render_frame_supports_magnetic_etching_for_pure_iron(self) -> None:
        frame, meta = self.backend.render_microscope_frame(
            self.pure_slide,
            MicroscopeState(
                objective=400,
                stage_x=0.5,
                stage_y=0.5,
                output_size=(256, 256),
                optical_mode="magnetic_etching",
            ),
        )
        self.assertEqual(str(meta.get("optical_mode", "")), "magnetic_etching")
        self.assertTrue(bool(meta.get("magnetic_field_active", False)))
        self.assertGreater(float(meta.get("ferromagnetic_fraction", 0.0)), 0.9)
        self.assertGreater(float(meta.get("magnetic_signal_fraction", 0.0)), 0.02)
        self.assertEqual(frame.shape, (256, 256))
        self.assertIn("electron_microscopy_guidance", meta)

    def test_render_frame_keeps_auto_zero_carbon_ferrite_bright_in_brightfield(
        self,
    ) -> None:
        slide = self.backend.generate_slide(
            MetallographyRequestV3(
                sample_id="pure_auto_backend",
                composition_wt={"Fe": 100.0, "C": 0.0},
                resolution=(256, 256),
                seed=123,
            )
        )
        frame, meta = self.backend.render_microscope_frame(
            slide,
            MicroscopeState(
                objective=200, stage_x=0.5, stage_y=0.5, output_size=(256, 256)
            ),
        )
        arr = frame.astype("float32")
        self.assertTrue(bool(meta.get("pure_iron_baseline_applied", False)))
        self.assertEqual(str(meta.get("optical_mode", "")), "brightfield")
        self.assertGreater(float(arr.mean()), 182.0)
        self.assertGreater(float(__import__("numpy").quantile(arr, 0.01)), 108.0)
        self.assertGreater(float(__import__("numpy").quantile(arr, 0.05)), 128.0)

    def test_render_frame_keeps_pro_realistic_ferrite_bright_with_empty_prep(
        self,
    ) -> None:
        slide = self.backend.generate_slide(
            {
                "sample_id": "pure_pro_backend",
                "composition_wt": {"Fe": 100.0, "C": 0.0},
                "resolution": [256, 256],
                "seed": 42,
                "synthesis_profile": {
                    "profile_id": "textbook_steel_bw",
                    "generation_mode": "pro_realistic",
                    "composition_sensitivity_mode": "realistic",
                    "contrast_target": 1.2,
                    "boundary_sharpness": 1.2,
                    "artifact_level": 0.2,
                    "phase_emphasis_style": "contrast_texture",
                },
            }
        )
        frame, meta = self.backend.render_microscope_frame(
            slide,
            MicroscopeState(
                objective=200, stage_x=0.5, stage_y=0.5, output_size=(256, 256)
            ),
        )
        arr = frame.astype(np.float32)
        self.assertTrue(bool(meta.get("pure_iron_baseline_applied", False)))
        self.assertGreater(float(arr.mean()), 170.0)
        self.assertGreater(float(np.quantile(arr, 0.05)), 110.0)

    def test_higher_objective_gives_smaller_crop(self) -> None:
        _frame_a, meta_a = self.backend.render_microscope_frame(
            self.slide,
            MicroscopeState(
                objective=200, stage_x=0.45, stage_y=0.55, output_size=(384, 384)
            ),
        )
        _frame_b, meta_b = self.backend.render_microscope_frame(
            self.slide,
            MicroscopeState(
                objective=600, stage_x=0.45, stage_y=0.55, output_size=(384, 384)
            ),
        )
        crop_a = meta_a["crop_size_px"]
        crop_b = meta_b["crop_size_px"]
        self.assertGreater(int(crop_a[0]), int(crop_b[0]))
        self.assertGreater(int(crop_a[1]), int(crop_b[1]))

    def test_focus_quality_decreases_with_same_error_on_higher_magnification(
        self,
    ) -> None:
        target_200 = self.backend.focus_target_mm(200, 0.5, 0.5)
        target_600 = self.backend.focus_target_mm(600, 0.5, 0.5)
        quality_200 = self.backend.focus_quality(200, target_200 + 0.25, target_200)
        quality_600 = self.backend.focus_quality(600, target_600 + 0.25, target_600)
        self.assertGreater(quality_200, quality_600)


if __name__ == "__main__":
    unittest.main()
