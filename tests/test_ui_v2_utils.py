import unittest

from core.ui_v2_utils import (
    build_capture_metadata,
    choose_scale_bar,
    estimate_um_per_px,
    normalize_capture_metadata,
    normalize_compare_mode,
)


class UiV2UtilsTests(unittest.TestCase):
    def test_compare_mode_normalization(self) -> None:
        self.assertEqual(normalize_compare_mode("Один кадр"), "single")
        self.assertEqual(normalize_compare_mode("До/После"), "before_after")
        self.assertEqual(normalize_compare_mode("step_by_step"), "step_by_step")
        self.assertEqual(normalize_compare_mode("Фазовый переход (кривая)"), "phase_transition_curve")
        self.assertEqual(normalize_compare_mode("unknown"), "single")

    def test_scale_bar_calculation(self) -> None:
        um_per_px = estimate_um_per_px(
            um_per_px_100x=1.2,
            crop_size_px=(512, 512),
            output_size_px=(1024, 1024),
        )
        self.assertAlmostEqual(um_per_px, 0.6)
        scale = choose_scale_bar(um_per_px)
        self.assertTrue(scale["enabled"])
        self.assertGreater(scale["bar_px"], 0)
        self.assertGreater(scale["bar_nm"], 0)

    def test_capture_metadata_backward_compat(self) -> None:
        payload = build_capture_metadata(
            source_image="a.png",
            source_metadata="a.json",
            microscope_params={"magnification": 200},
            view_meta={"crop_size_px": [100, 100]},
            route_summary={"final_stage": "pearlite"},
            session_id="sid",
            capture_index=3,
            reticle_enabled=True,
            scale_bar={"enabled": True, "um_per_px": 1.0, "bar_um": 100.0, "bar_px": 100.0},
            controls_state={"objective": 200, "focus_coarse": 0.9, "focus_fine": 0.0, "stage_x": 0.5, "stage_y": 0.5},
        )
        normalized = normalize_capture_metadata(payload)
        self.assertIn("microscope_params", normalized)
        self.assertIn("route_summary", normalized)
        self.assertIn("controls_state", normalized)
        self.assertIn("source_generator_version", normalized)
        self.assertIn("prep_signature", normalized)
        self.assertIn("etch_signature", normalized)
        self.assertIn("quality_metrics", normalized)
        self.assertEqual(normalized["capture_index"], 3)


if __name__ == "__main__":
    unittest.main()
