import unittest

import numpy as np

from core.imaging import simulate_microscope_view


def _checkerboard(size: int = 256, period: int = 8) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    board = ((yy // period + xx // period) % 2) * 255
    return board.astype(np.uint8)


def _sharpness_metric(image: np.ndarray) -> float:
    arr = image.astype(np.float32)
    gx = np.abs(np.diff(arr, axis=1)).mean()
    gy = np.abs(np.diff(arr, axis=0)).mean()
    return float(gx + gy)


class ImagingFocusPhysicsTests(unittest.TestCase):
    def test_exact_focus_is_sharper_than_defocused_view(self) -> None:
        sample = _checkerboard()

        focused, _ = simulate_microscope_view(
            sample=sample,
            magnification=300,
            output_size=(256, 256),
            focus_distance_mm=20.0,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            seed=1,
        )
        defocused, _ = simulate_microscope_view(
            sample=sample,
            magnification=300,
            output_size=(256, 256),
            focus_distance_mm=21.0,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            seed=1,
        )

        self.assertGreater(_sharpness_metric(focused), _sharpness_metric(defocused))
        focused_std, meta_std = simulate_microscope_view(
            sample=sample,
            magnification=300,
            output_size=(256, 256),
            focus_distance_mm=20.0,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            psf_profile="standard",
            psf_strength=0.0,
            seed=1,
        )
        self.assertEqual(str(meta_std["psf_profile"]), "standard")
        self.assertFalse(bool(meta_std["psf_engineering_applied"]))
        self.assertEqual(focused_std.shape, focused.shape)

    def test_physical_defocus_radius_grows_with_error_and_magnification(self) -> None:
        sample = _checkerboard()

        _, meta_small = simulate_microscope_view(
            sample=sample,
            magnification=200,
            output_size=(256, 256),
            focus_distance_mm=20.2,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            seed=1,
        )
        _, meta_large = simulate_microscope_view(
            sample=sample,
            magnification=200,
            output_size=(256, 256),
            focus_distance_mm=21.0,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            seed=1,
        )
        _, meta_high_mag = simulate_microscope_view(
            sample=sample,
            magnification=600,
            output_size=(256, 256),
            focus_distance_mm=21.0,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            seed=1,
        )

        small_radius = float(meta_small["focus_optics"]["defocus_radius_px"])
        large_radius = float(meta_large["focus_optics"]["defocus_radius_px"])
        high_mag_radius = float(meta_high_mag["focus_optics"]["defocus_radius_px"])

        self.assertLess(small_radius, large_radius)
        self.assertLess(large_radius, high_mag_radius)
        self.assertIn("focus_profile_mode", meta_large)
        self.assertIn("effective_dof_factor", meta_large)

    def test_optical_modes_change_image_and_emit_mode_metadata(self) -> None:
        sample = _checkerboard(period=12)

        bright, bright_meta = simulate_microscope_view(
            sample=sample,
            magnification=300,
            output_size=(256, 256),
            focus_distance_mm=20.0,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            optical_mode="brightfield",
            seed=3,
        )
        dark, dark_meta = simulate_microscope_view(
            sample=sample,
            magnification=300,
            output_size=(256, 256),
            focus_distance_mm=20.0,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            optical_mode="darkfield",
            optical_mode_parameters={"scatter_sensitivity": 1.2},
            seed=3,
        )
        polarized, polarized_meta = simulate_microscope_view(
            sample=np.full((256, 256), 170, dtype=np.uint8),
            magnification=300,
            output_size=(256, 256),
            focus_distance_mm=20.0,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            optical_mode="polarized",
            optical_mode_parameters={"crossed_polars": True},
            optical_context={"pure_iron_baseline_applied": True},
            seed=3,
        )
        phase_pos, phase_pos_meta = simulate_microscope_view(
            sample=sample,
            magnification=300,
            output_size=(256, 256),
            focus_distance_mm=20.0,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            optical_mode="phase_contrast",
            optical_mode_parameters={"phase_plate_type": "positive"},
            seed=3,
        )
        phase_neg, phase_neg_meta = simulate_microscope_view(
            sample=sample,
            magnification=300,
            output_size=(256, 256),
            focus_distance_mm=20.0,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            optical_mode="phase_contrast",
            optical_mode_parameters={"phase_plate_type": "negative"},
            seed=3,
        )

        self.assertEqual(str(bright_meta["optical_mode"]), "brightfield")
        self.assertEqual(str(dark_meta["optical_mode"]), "darkfield")
        self.assertGreater(float(np.mean(np.abs(bright.astype(np.float32) - dark.astype(np.float32)))), 2.0)
        self.assertIn("scatter_pass_fraction", dark_meta)
        self.assertEqual(str(polarized_meta["optical_mode"]), "polarized")
        self.assertLess(float(polarized.mean()), 40.0)
        self.assertTrue(bool(polarized_meta.get("crossed_polars", False)))
        self.assertEqual(str(phase_pos_meta["phase_plate_type"]), "positive")
        self.assertEqual(str(phase_neg_meta["phase_plate_type"]), "negative")
        self.assertGreater(float(np.mean(np.abs(phase_pos.astype(np.float32) - phase_neg.astype(np.float32)))), 1.0)
        limit_summary = dict(bright_meta.get("optical_limit_summary", {}))
        self.assertGreater(float(limit_summary.get("objective_numerical_aperture", 0.0)), 0.0)
        self.assertGreater(float(limit_summary.get("optical_resolution_limit_um", 0.0)), 0.0)
        self.assertGreater(float(limit_summary.get("approx_depth_of_field_um", 0.0)), 0.0)
        self.assertIn("empty_magnification_risk", limit_summary)

    def test_magnetic_etching_emits_ferromagnetic_metadata_for_pure_iron(self) -> None:
        sample = np.full((256, 256), 168, dtype=np.uint8)
        magnetic, meta = simulate_microscope_view(
            sample=sample,
            magnification=400,
            output_size=(256, 256),
            focus_distance_mm=20.0,
            focus_target_mm=20.0,
            um_per_px_100x=1.0,
            brightness=1.0,
            contrast=1.0,
            vignette_strength=0.0,
            uneven_strength=0.0,
            noise_sigma=0.0,
            optical_mode="magnetic_etching",
            optical_context={"pure_iron_baseline_applied": True},
            seed=4,
        )
        self.assertEqual(str(meta.get("optical_mode", "")), "magnetic_etching")
        self.assertTrue(bool(meta.get("magnetic_field_active", False)))
        self.assertGreater(float(meta.get("ferromagnetic_fraction", 0.0)), 0.95)
        self.assertGreater(float(meta.get("magnetic_signal_fraction", 0.0)), 0.05)
        self.assertLess(float(magnetic.mean()), 168.0)


if __name__ == "__main__":
    unittest.main()
