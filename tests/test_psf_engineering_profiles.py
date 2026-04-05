from __future__ import annotations

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


class PsfEngineeringProfilesTests(unittest.TestCase):
    def test_bessel_extended_dof_preserves_more_detail_than_standard_under_defocus(self) -> None:
        sample = _checkerboard()
        standard, meta_standard = simulate_microscope_view(
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
            psf_profile="standard",
            psf_strength=0.0,
            seed=7,
        )
        bessel, meta_bessel = simulate_microscope_view(
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
            psf_profile="bessel_extended_dof",
            psf_strength=1.0,
            seed=7,
        )
        self.assertGreater(_sharpness_metric(bessel), _sharpness_metric(standard))
        self.assertEqual(str(meta_standard["psf_profile"]), "standard")
        self.assertEqual(str(meta_bessel["psf_profile"]), "bessel_extended_dof")
        self.assertGreater(float(meta_bessel["effective_dof_factor"]), 1.0)

    def test_hybrid_profile_interpolates_between_standard_and_bessel(self) -> None:
        sample = _checkerboard()
        standard, _ = simulate_microscope_view(
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
            psf_profile="standard",
            psf_strength=0.0,
            seed=8,
        )
        bessel, _ = simulate_microscope_view(
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
            psf_profile="bessel_extended_dof",
            psf_strength=1.0,
            seed=8,
        )
        hybrid, meta_hybrid = simulate_microscope_view(
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
            psf_profile="lens_axicon_hybrid",
            psf_strength=1.0,
            hybrid_balance=0.5,
            seed=8,
        )
        sharp_standard = _sharpness_metric(standard)
        sharp_bessel = _sharpness_metric(bessel)
        sharp_hybrid = _sharpness_metric(hybrid)
        self.assertGreater(sharp_hybrid, sharp_standard)
        self.assertLess(sharp_hybrid, sharp_bessel)
        self.assertEqual(str(meta_hybrid["psf_profile"]), "lens_axicon_hybrid")
        self.assertGreater(float(meta_hybrid["effective_dof_factor"]), 1.0)

    def test_stir_sectioning_emits_sectioning_metadata(self) -> None:
        sample = _checkerboard()
        _image, meta = simulate_microscope_view(
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
            psf_profile="stir_sectioning",
            psf_strength=0.9,
            sectioning_shear_deg=40.0,
            seed=9,
        )
        self.assertEqual(str(meta["psf_profile"]), "stir_sectioning")
        self.assertTrue(bool(meta["sectioning_active"]))
        self.assertGreater(float(meta["sectioning_suppression_score"]), 0.0)


if __name__ == "__main__":
    unittest.main()
