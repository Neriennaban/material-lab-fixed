"""Tests for A10.4 — Klemm I tint-etched upper bainite preset.

The reference image (образец 3) shows a directional blue/yellow upper
bainite obtained with Klemm I tint etching. The new
``presets_v3/steel_45_upper_bainite_klemm_v3.json`` preset re-creates
the look by combining the existing pearlitic morphology with the
``tint_etch_blue_yellow`` palette in the post-process colourer.

These tests verify the high-level invariants:

* the preset propagates ``color_mode="tint_etch_blue_yellow"`` into the
  pipeline;
* the resulting image is a real RGB frame (channels differ);
* the blue channel dominates the red channel on average — the
  characteristic Klemm I "blue bainite" look;
* the texture renderers ``texture_bainite_upper`` and
  ``texture_bainite_lower`` produce non-trivial outputs and are
  registered in ``fe_c_texture_map``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from core.metallography_v3.fe_c_palettes import TINT_ETCH_BLUE_YELLOW_MODE
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3
from core.metallography_v3.system_generators.fe_c_textures import (
    fe_c_texture_map,
    texture_bainite_lower,
    texture_bainite_upper,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = REPO_ROOT / "presets_v3"
PROFILES_DIR = REPO_ROOT / "profiles_v3"

PRESET_NAME = "steel_45_upper_bainite_klemm_v3"


class BainiteTextureRegistryTest(unittest.TestCase):
    def test_upper_lower_textures_registered(self) -> None:
        registry = fe_c_texture_map()
        self.assertIn("BAINITE_UPPER", registry)
        self.assertIn("BAINITE_LOWER", registry)
        self.assertIn("BAINITE", registry)  # legacy still present

    def test_upper_bainite_texture_has_visible_lamellae(self) -> None:
        img = texture_bainite_upper((128, 128), seed=42)
        self.assertEqual(img.shape, (128, 128))
        self.assertEqual(img.dtype, np.uint8)
        # Coarse feathery texture should have a meaningful contrast.
        self.assertGreater(float(img.std()), 5.0)

    def test_lower_bainite_texture_has_needle_contrast(self) -> None:
        img = texture_bainite_lower((128, 128), seed=42)
        self.assertEqual(img.shape, (128, 128))
        self.assertEqual(img.dtype, np.uint8)
        self.assertGreater(float(img.std()), 5.0)


class TintEtchPresetIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = MetallographyPipelineV3(
            presets_dir=PRESETS_DIR,
            profiles_dir=PROFILES_DIR,
        )
        payload = cls.pipeline.load_preset(PRESET_NAME)
        payload["resolution"] = [256, 256]
        cls.output = cls.pipeline.generate(cls.pipeline.request_from_preset(payload))

    def test_color_mode_in_preset(self) -> None:
        synth = self.pipeline.load_preset(PRESET_NAME)["synthesis_profile"]
        self.assertEqual(synth["color_mode"], TINT_ETCH_BLUE_YELLOW_MODE)

    def test_rgb_output_is_chromatic(self) -> None:
        rgb = self.output.image_rgb
        self.assertEqual(rgb.ndim, 3)
        self.assertEqual(rgb.shape[2], 3)
        differing = int((rgb[..., 0] != rgb[..., 2]).sum())
        # The vast majority of pixels must have a different red and
        # blue channel — the palette is genuinely RGB.
        self.assertGreater(differing, rgb.shape[0] * rgb.shape[1] * 0.80)

    def test_blue_dominates_on_average(self) -> None:
        rgb = self.output.image_rgb
        b_mean = float(rgb[..., 2].mean())
        r_mean = float(rgb[..., 0].mean())
        # Klemm I tint should make the matrix look noticeably blue.
        self.assertGreater(
            b_mean,
            r_mean + 25.0,
            f"blue ({b_mean:.1f}) should dominate red ({r_mean:.1f})",
        )


if __name__ == "__main__":
    unittest.main()
