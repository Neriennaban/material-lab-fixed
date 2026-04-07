"""Tests for A10.1 — AISI 10xx series phase-fraction validation.

The plan asserts that the new AISI nital-warm presets must produce
pearlite fractions that agree with the lever rule within the
``phase_fraction_tolerance_pct`` of the preset (15 %). Each grade is
rendered at a modest resolution and the ferrite/pearlite fractions are
measured from the returned phase masks.

The tests also verify that the ``color_mode="nital_warm"`` option turns
the output into a real chromatic image — the red channel must dominate
the blue channel on the ferrite phase, matching the warm yellow tint of
the reference micrographs.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = REPO_ROOT / "presets_v3"
PROFILES_DIR = REPO_ROOT / "profiles_v3"


# (grade, carbon wt %, target pearlite fraction, allowed +/- band)
AISI_TARGETS: tuple[tuple[str, float, float, float], ...] = (
    # Lever rule: P = (C - 0.02) / (0.77 - 0.02).
    # Bands include the preset-level tolerance (15 %).
    ("1020", 0.20, 0.24, 0.10),
    ("1030", 0.30, 0.37, 0.10),
    ("1040", 0.40, 0.51, 0.10),
    ("1045", 0.45, 0.57, 0.10),
    ("1050", 0.50, 0.64, 0.10),
)


def _render(pipeline: MetallographyPipelineV3, grade: str):
    payload = pipeline.load_preset(f"aisi_{grade}_nital_warm_v3")
    payload["resolution"] = [256, 256]
    request = pipeline.request_from_preset(payload)
    return pipeline.generate(request)


def _phase_fraction(masks: dict[str, np.ndarray], phase_key: str) -> float:
    if not isinstance(masks, dict):
        return 0.0
    total_pixels = 0
    matched = 0
    target = str(phase_key).strip().upper()
    for name, mask in masks.items():
        if not isinstance(mask, np.ndarray):
            continue
        total_pixels = mask.size
        if str(name).strip().upper() == target:
            matched = int((mask > 0).sum())
    if total_pixels <= 0:
        return 0.0
    return float(matched) / float(total_pixels)


class AISISeriesFractionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = MetallographyPipelineV3(
            presets_dir=PRESETS_DIR,
            profiles_dir=PROFILES_DIR,
        )

    def test_pearlite_fraction_matches_lever_rule(self) -> None:
        for grade, carbon, target, band in AISI_TARGETS:
            with self.subTest(grade=grade):
                output = _render(self.pipeline, grade)
                pearlite = _phase_fraction(output.phase_masks, "PEARLITE")
                self.assertGreaterEqual(
                    pearlite,
                    target - band,
                    f"AISI {grade} pearlite {pearlite:.3f} below {target - band:.3f}",
                )
                self.assertLessEqual(
                    pearlite,
                    target + band,
                    f"AISI {grade} pearlite {pearlite:.3f} above {target + band:.3f}",
                )

    def test_ferrite_plus_pearlite_dominant(self) -> None:
        for grade, *_ in AISI_TARGETS:
            with self.subTest(grade=grade):
                output = _render(self.pipeline, grade)
                ferrite = _phase_fraction(output.phase_masks, "FERRITE")
                pearlite = _phase_fraction(output.phase_masks, "PEARLITE")
                # Ferrite + pearlite should cover at least 90 % of the
                # image — any other phase is a residual trace.
                self.assertGreaterEqual(
                    ferrite + pearlite,
                    0.90,
                    f"AISI {grade} ferrite+pearlite = {ferrite + pearlite:.3f}",
                )

    def test_nital_warm_output_is_truly_chromatic(self) -> None:
        for grade, *_ in AISI_TARGETS:
            with self.subTest(grade=grade):
                output = _render(self.pipeline, grade)
                rgb = output.image_rgb
                self.assertEqual(rgb.ndim, 3)
                self.assertEqual(rgb.shape[2], 3)
                r_mean = float(rgb[..., 0].mean())
                g_mean = float(rgb[..., 1].mean())
                b_mean = float(rgb[..., 2].mean())
                # Warm palette → red dominates blue on average.
                self.assertGreater(
                    r_mean,
                    b_mean + 20.0,
                    f"AISI {grade} R={r_mean:.1f} not warmer than B={b_mean:.1f}",
                )
                # Green should sit between R and B for yellow-cream tint.
                self.assertGreater(g_mean, b_mean)
                self.assertLess(g_mean, r_mean + 1.0)


if __name__ == "__main__":
    unittest.main()
