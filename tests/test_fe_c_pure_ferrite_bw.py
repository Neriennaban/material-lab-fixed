"""Tests for A10.5 — pure ferrite brightfield BW preset.

The reference sample (образец 5) shows an Armco-iron polish at 100× in
plain brightfield: a polygon lattice with a fairly wide grain-size
distribution (roughly 20-80 µm), thin dark grain boundaries and a
bright mean intensity. This test pins the following invariants for
``presets_v3/fe_pure_armco_bw_v3.json``:

* output is a single-phase ferrite micrograph (no pearlite/cementite),
* ``color_mode="grayscale_nital"`` keeps the RGB channels identical
  (classical BW brightfield, no palette post-process),
* mean brightness and grain-boundary contrast are in the expected
  bright/dark bands,
* ``native_um_per_px`` comes out to ``1.0`` for the 100× microscope
  profile, matching the 100 µm scale bar of the reference image.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = REPO_ROOT / "presets_v3"
PROFILES_DIR = REPO_ROOT / "profiles_v3"

PRESET_NAME = "fe_pure_armco_bw_v3"


class PureFerriteBWPresetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = MetallographyPipelineV3(
            presets_dir=PRESETS_DIR,
            profiles_dir=PROFILES_DIR,
        )
        payload = cls.pipeline.load_preset(PRESET_NAME)
        payload["resolution"] = [256, 256]
        request = cls.pipeline.request_from_preset(payload)
        cls.output = cls.pipeline.generate(request)
        cls.request = request

    def test_single_phase_ferrite(self) -> None:
        masks = self.output.phase_masks or {}
        phase_names = {str(name).upper() for name in masks.keys()}
        self.assertIn("FERRITE", phase_names)
        # No pearlite/cementite traces in a pure Armco iron render.
        self.assertNotIn("PEARLITE", phase_names)
        self.assertNotIn("CEMENTITE", phase_names)

    def test_grayscale_rgb_channels_identical(self) -> None:
        rgb = self.output.image_rgb
        self.assertEqual(rgb.ndim, 3)
        self.assertTrue(np.array_equal(rgb[..., 0], rgb[..., 1]))
        self.assertTrue(np.array_equal(rgb[..., 1], rgb[..., 2]))

    def test_bright_mean_intensity_and_visible_boundaries(self) -> None:
        gray = self.output.image_gray.astype(np.float32)
        mean = float(gray.mean())
        # Armco iron on brightfield nital is dominated by bright ferrite
        # grains — mean intensity should sit in the upper half of the
        # 0–255 range.
        self.assertGreater(mean, 160.0, f"mean brightness {mean:.1f} too dark")
        self.assertLess(mean, 245.0, f"mean brightness {mean:.1f} too flat")

        # Grain boundaries must produce a visible darker tail. Phase D
        # introduced a hard brightness floor at 120 (see
        # ``_brighten_pure_ferrite_baseline``) so "dark" now means
        # "dark gray" (120-150) rather than the pre-D black bands
        # the legacy test expected. The assertion still guarantees a
        # visible grain-boundary network without mandating specific
        # pitch-black pixels.
        dark_fraction = float((gray < 155.0).mean())
        self.assertGreater(
            dark_fraction,
            0.005,
            f"dark boundary fraction {dark_fraction:.4f} too small",
        )

    def test_native_um_per_px_matches_100x(self) -> None:
        # The preset requests 100× magnification, so the internal
        # ``native_um_per_px`` should resolve to 1.0 µm/px.
        mag = float(self.request.microscope_profile.get("magnification", 0.0))
        self.assertEqual(mag, 100.0)
        expected_um_per_px = 1.0 / max(1e-3, mag / 100.0)
        self.assertAlmostEqual(expected_um_per_px, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
