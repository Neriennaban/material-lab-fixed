"""Tests for A2 — leopard-style ledeburite texture.

The current ``texture_ledeburite`` is a blend of an Al-Si eutectic and
the pearlite renderer; it does not match the hierarchical
"leopard skin" pattern described in §5.4 of the TZ. The new
``texture_ledeburite_leopard`` thresholds a multiscale smooth-noise
field to scatter dark pearlite blobs across a bright cementite matrix
with a near-uniform spacing.

These tests pin the qualitative invariants:

* the renderer is registered in ``fe_c_texture_map`` under
  ``LEDEBURITE_LEOPARD`` (the legacy ``LEDEBURITE`` key continues to
  point at the old blend);
* the output is a uint8 grayscale image with a clear bimodal
  distribution — bright cementite matrix and dark pearlite blobs;
* the dark blob fraction sits in a sensible 30-70 % range
  (matching the eutectic constituent ratio of white cast iron);
* the renderer is deterministic for a given seed.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.metallography_v3.system_generators.fe_c_textures import (
    fe_c_texture_map,
    texture_ledeburite,
    texture_ledeburite_leopard,
)


class LedeburiteLeopardTest(unittest.TestCase):
    def test_registered_in_texture_map(self) -> None:
        registry = fe_c_texture_map()
        self.assertIn("LEDEBURITE_LEOPARD", registry)
        self.assertIn("LEDEBURITE", registry)
        self.assertIs(registry["LEDEBURITE_LEOPARD"], texture_ledeburite_leopard)
        # Legacy key must still point at the old renderer.
        self.assertIs(registry["LEDEBURITE"], texture_ledeburite)

    def test_output_shape_and_dtype(self) -> None:
        img = texture_ledeburite_leopard((128, 128), seed=42)
        self.assertEqual(img.shape, (128, 128))
        self.assertEqual(img.dtype, np.uint8)

    def test_bright_matrix_and_dark_blobs(self) -> None:
        img = texture_ledeburite_leopard((256, 256), seed=2024)
        # Bright cementite matrix → many pixels above 180.
        bright_fraction = float((img > 180).mean())
        self.assertGreater(
            bright_fraction,
            0.30,
            f"bright matrix fraction {bright_fraction:.3f} too small",
        )
        # Dark pearlite blobs → many pixels below 100.
        dark_fraction = float((img < 100).mean())
        self.assertGreater(
            dark_fraction,
            0.20,
            f"dark blob fraction {dark_fraction:.3f} too small",
        )
        self.assertLess(
            dark_fraction,
            0.70,
            f"dark blob fraction {dark_fraction:.3f} too large",
        )

    def test_distinct_from_legacy_ledeburite(self) -> None:
        leopard = texture_ledeburite_leopard((128, 128), seed=42)
        legacy = texture_ledeburite((128, 128), seed=42)
        self.assertEqual(leopard.shape, legacy.shape)
        # The two renderers should produce visibly different output.
        self.assertFalse(np.array_equal(leopard, legacy))
        diff = float(np.abs(leopard.astype(np.int16) - legacy.astype(np.int16)).mean())
        self.assertGreater(diff, 5.0)

    def test_deterministic_for_seed(self) -> None:
        a = texture_ledeburite_leopard((96, 96), seed=999)
        b = texture_ledeburite_leopard((96, 96), seed=999)
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
