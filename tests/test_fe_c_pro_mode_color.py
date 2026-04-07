"""Tests for C1.1 — extending pro-realistic generation to honour
``synthesis_profile.color_mode``.

The plan calls for the colour palette infrastructure introduced in
A10.0 to also be reachable from the ``generation_mode="pro_realistic"``
path. ``generate_pro_realistic_fe_c`` now applies
``apply_color_palette`` itself when the synthesis profile picks a non
default mode and surfaces the resulting RGB array under
``"image_rgb"`` so ``MetallographyPipelineV3`` can plumb it through.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = REPO_ROOT / "presets_v3"
PROFILES_DIR = REPO_ROOT / "profiles_v3"

_PRO_PRESET = "fe_c_hypoeutectoid_pro_realistic"


def _render_with_color_mode(color_mode: str):
    pipeline = MetallographyPipelineV3(
        presets_dir=PRESETS_DIR,
        profiles_dir=PROFILES_DIR,
    )
    payload = copy.deepcopy(pipeline.load_preset(_PRO_PRESET))
    payload["resolution"] = [192, 192]
    synth = dict(payload.get("synthesis_profile") or {})
    synth["color_mode"] = color_mode
    payload["synthesis_profile"] = synth
    return pipeline.generate(pipeline.request_from_preset(payload))


class ProModeColorPaletteTest(unittest.TestCase):
    def test_default_grayscale_mode_unchanged(self) -> None:
        out = _render_with_color_mode("grayscale_nital")
        rgb = out.image_rgb
        self.assertEqual(rgb.ndim, 3)
        self.assertEqual(rgb.shape[2], 3)
        # Default mode produces a stacked-channel RGB.
        self.assertTrue(np.array_equal(rgb[..., 0], rgb[..., 1]))
        self.assertTrue(np.array_equal(rgb[..., 1], rgb[..., 2]))

    def test_nital_warm_mode_produces_chromatic_rgb(self) -> None:
        out = _render_with_color_mode("nital_warm")
        rgb = out.image_rgb
        self.assertEqual(rgb.ndim, 3)
        self.assertEqual(rgb.shape[2], 3)
        self.assertFalse(
            np.array_equal(rgb[..., 0], rgb[..., 2]),
            "pro-mode nital_warm output should be chromatic",
        )
        # Warm palette: red dominates blue on average.
        self.assertGreater(
            float(rgb[..., 0].mean()), float(rgb[..., 2].mean()) + 10.0
        )

    def test_grayscale_information_preserved(self) -> None:
        gray_a = _render_with_color_mode("grayscale_nital").image_gray
        gray_b = _render_with_color_mode("nital_warm").image_gray
        # The grayscale half is computed before the colour palette,
        # so switching to ``nital_warm`` must not perturb it.
        self.assertTrue(np.array_equal(gray_a, gray_b))

    def test_tint_etch_mode_works_under_pro(self) -> None:
        out = _render_with_color_mode("tint_etch_blue_yellow")
        rgb = out.image_rgb
        self.assertEqual(rgb.shape[2], 3)
        # Hypoeutectoid steel ≈ 50/50 ferrite (yellow) + pearlite
        # (blue) — so we cannot pin which channel dominates, only
        # require that the result is genuinely chromatic.
        self.assertFalse(np.array_equal(rgb[..., 0], rgb[..., 2]))
        # And both yellow and blue regions must exist somewhere.
        warm_pixels = int(((rgb[..., 0] > rgb[..., 2] + 30)).sum())
        cool_pixels = int(((rgb[..., 2] > rgb[..., 0] + 10)).sum())
        self.assertGreater(warm_pixels, 100)
        self.assertGreater(cool_pixels, 100)


if __name__ == "__main__":
    unittest.main()
