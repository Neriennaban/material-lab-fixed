"""Tests for A10.3 — DIC polarised reflected-light grain colouring.

The reference image (образец 1) shows Armco iron under DIC/Nomarski
reflected polarised light: every grain receives a slightly different
hue from the polariser, while grain boundaries glow with rim lighting
from the relief effect. The new ``dic_polarized`` ``color_mode``
re-creates this look by sampling random HSV colours per grain label.

These tests pin the following invariants for the new
``presets_v3/fe_armco_dic_polarized_v3.json`` preset:

* the pipeline produces a true RGB output with channel-by-channel
  variation (not a grayscale stack);
* hue diversity across the image is non-trivial — at least dozens of
  unique R-channel values, indicating per-grain colouring;
* the grayscale half of the result still tracks the underlying
  ferrite render so the structural information is preserved.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from core.metallography_v3.fe_c_color_palette import apply_color_palette
from core.metallography_v3.fe_c_palettes import DIC_POLARIZED_MODE
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = REPO_ROOT / "presets_v3"
PROFILES_DIR = REPO_ROOT / "profiles_v3"


PRESET_NAME = "fe_armco_dic_polarized_v3"


class DicPolarizedPresetTest(unittest.TestCase):
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

    def test_color_mode_propagated_to_synthesis_profile(self) -> None:
        self.assertEqual(
            self.pipeline.load_preset(PRESET_NAME)["synthesis_profile"]["color_mode"],
            DIC_POLARIZED_MODE,
        )

    def test_rgb_output_is_truly_chromatic(self) -> None:
        rgb = self.output.image_rgb
        self.assertEqual(rgb.ndim, 3)
        self.assertEqual(rgb.shape[2], 3)
        self.assertFalse(
            np.array_equal(rgb[..., 0], rgb[..., 1]),
            "DIC polarised output channels must differ",
        )
        self.assertFalse(
            np.array_equal(rgb[..., 1], rgb[..., 2]),
            "DIC polarised output channels must differ",
        )

    def test_per_grain_hue_diversity(self) -> None:
        # Per-grain HSV colouring should leave many distinct values in
        # the red channel — at least 60 across a 256×256 frame.
        rgb = self.output.image_rgb
        unique_r = int(np.unique(rgb[..., 0]).size)
        self.assertGreater(
            unique_r,
            60,
            f"Expected ≥60 unique R values for DIC, got {unique_r}",
        )

    def test_grayscale_information_preserved(self) -> None:
        gray = self.output.image_gray
        # Grayscale Armco mean is bright (~200+); the colouring
        # post-process must not damage the underlying structure.
        self.assertGreater(float(gray.mean()), 160.0)

    def test_dic_palette_is_no_op_without_labels(self) -> None:
        """Pure-function safety: when called with ``labels=None`` the
        DIC palette degrades to grayscale-stack instead of crashing."""
        gray = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape(64, 64)
        rgb = apply_color_palette(
            image_gray=gray,
            phase_masks=None,
            color_mode=DIC_POLARIZED_MODE,
            seed=42,
            labels=None,
        )
        self.assertEqual(rgb.shape, (64, 64, 3))
        self.assertTrue(np.array_equal(rgb[..., 0], rgb[..., 1]))


if __name__ == "__main__":
    unittest.main()
