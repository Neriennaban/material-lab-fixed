"""Tests for the Phase C1 quality-metrics module."""

from __future__ import annotations

import math
import unittest

import numpy as np

from core.metallography_v3.quality_metrics_ferro import (
    fft_lamellae_period_px,
    grain_size_astm,
    histogram_intersection,
    hough_orientation_histogram,
    phase_fraction_error,
    ssim_vs_reference,
)


class PhaseFractionErrorTest(unittest.TestCase):
    def test_zero_error_for_perfect_masks(self) -> None:
        # Use a 100×100 mask so ``int(0.7 * 100) = 70`` rows is
        # exactly 70 % of the area — no rounding artefacts.
        h, w = 100, 100
        ferrite_mask = np.zeros((h, w), dtype=np.uint8)
        ferrite_mask[:70, :] = 1  # exactly 70 % coverage
        pearlite_mask = 1 - ferrite_mask
        result = phase_fraction_error(
            phase_masks={"FERRITE": ferrite_mask, "PEARLITE": pearlite_mask},
            expected_fractions={"FERRITE": 0.7, "PEARLITE": 0.3},
        )
        self.assertAlmostEqual(result["actual"]["FERRITE"], 0.7, places=5)
        self.assertAlmostEqual(result["actual"]["PEARLITE"], 0.3, places=5)
        self.assertLess(result["max_relative_error_pct"], 1.0)

    def test_missing_mask_treated_as_zero(self) -> None:
        result = phase_fraction_error(
            phase_masks={"FERRITE": np.ones((10, 10), dtype=np.uint8)},
            expected_fractions={"FERRITE": 1.0, "PEARLITE": 0.5},
        )
        self.assertEqual(result["actual"]["PEARLITE"], 0.0)
        self.assertGreater(result["relative_error_pct"]["PEARLITE"], 99.0)


class HistogramIntersectionTest(unittest.TestCase):
    def test_identical_grayscale_returns_one(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        self.assertAlmostEqual(histogram_intersection(a, a), 1.0, places=4)

    def test_identical_rgb_returns_one(self) -> None:
        rng = np.random.default_rng(43)
        a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        self.assertAlmostEqual(histogram_intersection(a, a), 1.0, places=4)

    def test_unrelated_distributions_below_one(self) -> None:
        rng = np.random.default_rng(44)
        a = rng.integers(0, 256, (96, 96), dtype=np.uint8)
        b = rng.integers(0, 256, (96, 96), dtype=np.uint8)
        score = histogram_intersection(a, b)
        self.assertGreater(score, 0.5)
        self.assertLess(score, 1.0)

    def test_shape_mismatch_raises(self) -> None:
        a = np.zeros((10, 10), dtype=np.uint8)
        b = np.zeros((11, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            histogram_intersection(a, b)


class SSIMTest(unittest.TestCase):
    def test_identical_returns_one(self) -> None:
        rng = np.random.default_rng(45)
        a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        self.assertAlmostEqual(ssim_vs_reference(a, a), 1.0, places=4)

    def test_negation_returns_minus_one_or_close(self) -> None:
        a = np.linspace(0, 255, 64 * 64).reshape(64, 64).astype(np.uint8)
        b = (255 - a).astype(np.uint8)
        score = ssim_vs_reference(a, b)
        # Negative correlation; both skimage SSIM and the Pearson
        # fallback should be ≤ 0 here.
        self.assertLessEqual(score, 0.0)


class FFTLamellaePeriodTest(unittest.TestCase):
    def test_recovers_known_period(self) -> None:
        for period in (6.0, 8.0, 12.0):
            with self.subTest(period=period):
                yy, xx = np.mgrid[0:128, 0:128]
                img = (np.sin(2.0 * math.pi * xx / period) * 80 + 128).astype(
                    np.uint8
                )
                detected = fft_lamellae_period_px(img)
                self.assertAlmostEqual(detected, period, delta=1.5)

    def test_uniform_image_returns_zero(self) -> None:
        img = np.full((96, 96), 128, dtype=np.uint8)
        self.assertEqual(fft_lamellae_period_px(img), 0.0)


class HoughOrientationTest(unittest.TestCase):
    def test_uniform_image_yields_uniform_histogram(self) -> None:
        img = np.full((96, 96), 200, dtype=np.uint8)
        hist = hough_orientation_histogram(img, bins=8)
        self.assertEqual(hist.shape, (8,))
        # Pixel-flat image → histogram falls back to uniform 1/N.
        self.assertAlmostEqual(float(hist.max()), 1.0 / 8.0, places=4)

    def test_directional_pattern_has_dominant_bin(self) -> None:
        # Vertical stripes → dominant horizontal gradient (angle ≈ 0).
        yy, xx = np.mgrid[0:128, 0:128]
        img = ((np.sin(2.0 * math.pi * xx / 8.0) > 0).astype(np.uint8)) * 200
        hist = hough_orientation_histogram(img, bins=8)
        # The peak bin should hold significantly more than the uniform
        # baseline of 0.125.
        self.assertGreater(float(hist.max()), 0.30)


class GrainSizeASTMTest(unittest.TestCase):
    def test_known_grid_label_map(self) -> None:
        # 4×4 grid of 16 grains in a 64×64 image with 1 µm/px ≈
        # 16 µm grains. ASTM E112 for ~16 µm grains lands around 7-8.
        labels = np.zeros((64, 64), dtype=np.int32)
        for r in range(4):
            for c in range(4):
                labels[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16] = r * 4 + c
        astm = grain_size_astm(labels, um_per_px=1.0)
        self.assertGreater(astm, 5.0)
        self.assertLess(astm, 11.0)

    def test_finer_grid_yields_larger_astm_number(self) -> None:
        # Smaller grains → larger ASTM number (E112 monotonicity).
        coarse = np.zeros((64, 64), dtype=np.int32)
        for r in range(4):
            for c in range(4):
                coarse[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16] = r * 4 + c
        fine = np.zeros((64, 64), dtype=np.int32)
        for r in range(8):
            for c in range(8):
                fine[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8] = r * 8 + c
        self.assertGreater(
            grain_size_astm(fine, um_per_px=1.0),
            grain_size_astm(coarse, um_per_px=1.0),
        )

    def test_no_intersections_returns_nan(self) -> None:
        labels = np.zeros((10, 10), dtype=np.int32)
        result = grain_size_astm(labels, um_per_px=1.0)
        self.assertTrue(math.isnan(result))


if __name__ == "__main__":
    unittest.main()
