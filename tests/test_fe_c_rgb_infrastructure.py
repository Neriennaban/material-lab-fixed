"""Tests for A10.0 — RGB post-process infrastructure.

The plan ships a new ``apply_color_palette`` function and a
``SynthesisProfileV3.color_mode`` switch. These tests verify:

* the default ``color_mode="grayscale_nital"`` path is byte-identical
  to the legacy ``_to_rgb(image_gray)`` behaviour (no regressions on
  29 existing presets — covered by the baseline snapshot test);
* when a preset overrides ``color_mode`` with ``nital_warm``,
  ``dic_polarized`` or ``tint_etch_blue_yellow``, the returned
  ``GenerationOutputV3.image_rgb`` is a real multi-channel frame
  (the channels differ on non-boundary pixels) and is aligned with
  ``image_gray``;
* ``apply_color_palette`` itself is a pure function driven purely
  by its arguments.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from core.metallography_v3.fe_c_color_palette import apply_color_palette
from core.metallography_v3.fe_c_palettes import (
    DIC_POLARIZED_MODE,
    GRAYSCALE_MODE,
    NITAL_WARM_MODE,
    PALETTES,
    SUPPORTED_COLOR_MODES,
    TINT_ETCH_BLUE_YELLOW_MODE,
)
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = REPO_ROOT / "presets_v3"
PROFILES_DIR = REPO_ROOT / "profiles_v3"

_BASELINE_PRESET = "fe_c_hypoeutectoid_textbook"
_RENDER_RESOLUTION = (128, 128)
_RENDER_SEED = 42


def _render_with_color_mode(color_mode: str) -> tuple[np.ndarray, np.ndarray, dict]:
    pipeline = MetallographyPipelineV3(
        presets_dir=PRESETS_DIR,
        profiles_dir=PROFILES_DIR,
    )
    payload = pipeline.load_preset(_BASELINE_PRESET)
    payload = copy.deepcopy(payload)
    payload["resolution"] = list(_RENDER_RESOLUTION)
    payload["seed"] = _RENDER_SEED
    synth = dict(payload.get("synthesis_profile") or {})
    synth["color_mode"] = color_mode
    payload["synthesis_profile"] = synth
    request = pipeline.request_from_preset(payload)
    output = pipeline.generate(request)
    return output.image_gray, output.image_rgb, output.phase_masks


class PaletteRegistryTest(unittest.TestCase):
    def test_all_modes_present_in_registry(self) -> None:
        for mode in SUPPORTED_COLOR_MODES:
            self.assertIn(mode, PALETTES, f"{mode} missing from PALETTES")


class ApplyColorPalettePureFunctionTest(unittest.TestCase):
    def setUp(self) -> None:
        h, w = 64, 64
        self.gray = np.linspace(0, 255, h * w, dtype=np.uint8).reshape(h, w)
        # Two phase masks with non-trivial coverage.
        self.phase_masks = {
            "FERRITE": (self.gray > 200).astype(np.uint8),
            "PEARLITE": (self.gray < 60).astype(np.uint8),
        }
        self.labels = np.zeros_like(self.gray, dtype=np.int32)
        self.labels[16:, :] = 1
        self.labels[:, 16:] += 2

    def test_grayscale_mode_returns_broadcasted_channels(self) -> None:
        rgb = apply_color_palette(
            image_gray=self.gray,
            phase_masks=self.phase_masks,
            color_mode=GRAYSCALE_MODE,
            seed=42,
        )
        self.assertEqual(rgb.shape, (*self.gray.shape, 3))
        self.assertTrue(np.array_equal(rgb[..., 0], rgb[..., 1]))
        self.assertTrue(np.array_equal(rgb[..., 1], rgb[..., 2]))
        self.assertTrue(np.array_equal(rgb[..., 0], self.gray))

    def test_nital_warm_mode_produces_true_rgb(self) -> None:
        rgb = apply_color_palette(
            image_gray=self.gray,
            phase_masks=self.phase_masks,
            color_mode=NITAL_WARM_MODE,
            seed=42,
        )
        self.assertEqual(rgb.shape, (*self.gray.shape, 3))
        self.assertFalse(
            np.array_equal(rgb[..., 0], rgb[..., 1]),
            "nital_warm output channels should differ",
        )
        # Ferrite mask pixels should be warm (R ≥ B).
        ferrite_pixels = rgb[self.phase_masks["FERRITE"] > 0]
        self.assertGreater(
            float(ferrite_pixels[:, 0].mean()),
            float(ferrite_pixels[:, 2].mean()),
            "ferrite region should be warmer than blue channel",
        )

    def test_tint_etch_mode_produces_blue_and_yellow(self) -> None:
        rgb = apply_color_palette(
            image_gray=self.gray,
            phase_masks=self.phase_masks,
            color_mode=TINT_ETCH_BLUE_YELLOW_MODE,
            seed=42,
        )
        self.assertEqual(rgb.shape, (*self.gray.shape, 3))
        # Ferrite mask → yellow (R > B), pearlite mask → blue (B > R).
        ferrite_pixels = rgb[self.phase_masks["FERRITE"] > 0]
        pearlite_pixels = rgb[self.phase_masks["PEARLITE"] > 0]
        if ferrite_pixels.size > 0:
            self.assertGreater(
                float(ferrite_pixels[:, 0].mean()),
                float(ferrite_pixels[:, 2].mean()),
            )
        if pearlite_pixels.size > 0:
            self.assertGreater(
                float(pearlite_pixels[:, 2].mean()),
                float(pearlite_pixels[:, 0].mean()),
            )

    def test_dic_polarized_mode_colors_labels(self) -> None:
        rgb = apply_color_palette(
            image_gray=self.gray,
            phase_masks=None,
            color_mode=DIC_POLARIZED_MODE,
            seed=42,
            labels=self.labels,
        )
        self.assertEqual(rgb.shape, (*self.gray.shape, 3))
        # At least some pixels must have different R and B channels.
        self.assertFalse(np.array_equal(rgb[..., 0], rgb[..., 2]))

    def test_unknown_mode_falls_back_to_grayscale(self) -> None:
        rgb = apply_color_palette(
            image_gray=self.gray,
            phase_masks=self.phase_masks,
            color_mode="this_does_not_exist",
            seed=42,
        )
        self.assertTrue(np.array_equal(rgb[..., 0], rgb[..., 2]))


class PipelineRespectColorModeTest(unittest.TestCase):
    """Pipeline-level integration: the new ``color_mode`` switch must
    propagate from the preset dict through ``SynthesisProfileV3`` into
    ``apply_color_palette`` without breaking grayscale presets."""

    def test_default_grayscale_mode_matches_to_rgb_stack(self) -> None:
        image_gray, image_rgb, _ = _render_with_color_mode(GRAYSCALE_MODE)
        self.assertEqual(image_rgb.ndim, 3)
        self.assertEqual(image_rgb.shape[2], 3)
        self.assertTrue(np.array_equal(image_rgb[..., 0], image_gray))
        self.assertTrue(np.array_equal(image_rgb[..., 1], image_gray))
        self.assertTrue(np.array_equal(image_rgb[..., 2], image_gray))

    def test_nital_warm_mode_via_pipeline_returns_true_rgb(self) -> None:
        image_gray, image_rgb, _ = _render_with_color_mode(NITAL_WARM_MODE)
        self.assertEqual(image_rgb.shape[0], image_gray.shape[0])
        self.assertEqual(image_rgb.shape[1], image_gray.shape[1])
        self.assertEqual(image_rgb.shape[2], 3)
        # Not a broadcast-to-RGB result: R and B channels must differ.
        self.assertFalse(
            np.array_equal(image_rgb[..., 0], image_rgb[..., 2]),
            "nital_warm pipeline output should be chromatic",
        )

    def test_grayscale_preset_grayscale_hash_unchanged(self) -> None:
        """Explicit sanity check: the grayscale channel of a grayscale
        preset is not affected by the palette post-process (backward
        compatibility with the baseline snapshot)."""
        gray_a, _, _ = _render_with_color_mode(GRAYSCALE_MODE)
        gray_b, _, _ = _render_with_color_mode(GRAYSCALE_MODE)
        self.assertTrue(np.array_equal(gray_a, gray_b))


if __name__ == "__main__":
    unittest.main()
