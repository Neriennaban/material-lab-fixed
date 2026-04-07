"""Tests for A1 — primary cementite needle renderer.

Reference: §5.3.В of the TZ. The renderer paints bright Fe₃C needles
on top of a leopard ledeburite base. The tests pin the qualitative
invariants:

* output is a uint8 grayscale image of the requested shape;
* needle pixels are *brighter* than the base, occupy a sensible
  fraction of the field, and cluster around 2-3 dominant orientations;
* higher %C → more needles than lower %C;
* faster cooling → fewer needles than slow cooling at the same %C;
* the renderer is deterministic for a fixed seed.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.metallography_v3.system_generators.fe_c_primary_cementite import (
    render_primary_cementite_needles,
)
from core.metallography_v3.system_generators.fe_c_textures import (
    texture_ledeburite_leopard,
)


class PrimaryCementiteNeedlesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.size = (192, 192)
        self.base = texture_ledeburite_leopard(self.size, seed=4242)

    def test_output_shape_and_dtype(self) -> None:
        result = render_primary_cementite_needles(
            size=self.size,
            seed=42,
            c_wt=5.5,
            base_image=self.base,
        )
        img = result["image"]
        self.assertEqual(img.shape, self.size)
        self.assertEqual(img.dtype, np.uint8)
        self.assertIn("needle_count", result["metadata"])
        self.assertIn("primary_directions_rad", result["metadata"])

    def test_needles_brighter_than_base(self) -> None:
        result = render_primary_cementite_needles(
            size=self.size,
            seed=42,
            c_wt=5.5,
            base_image=self.base,
        )
        mask = result["needle_mask"].astype(bool)
        self.assertGreater(int(mask.sum()), 200, "no needle pixels detected")
        needle_pixels = result["image"][mask]
        base_pixels = self.base[mask] if hasattr(self.base, "shape") else None
        self.assertGreater(float(needle_pixels.mean()), 230.0)
        if base_pixels is not None:
            self.assertGreater(float(needle_pixels.mean()), float(base_pixels.mean()))

    def test_higher_carbon_yields_more_needles(self) -> None:
        low_c = render_primary_cementite_needles(
            size=self.size,
            seed=42,
            c_wt=4.5,
            base_image=self.base,
            cooling_rate_c_per_s=5.0,
        )["metadata"]["needle_count"]
        high_c = render_primary_cementite_needles(
            size=self.size,
            seed=42,
            c_wt=6.0,
            base_image=self.base,
            cooling_rate_c_per_s=5.0,
        )["metadata"]["needle_count"]
        self.assertGreater(high_c, low_c)

    def test_faster_cooling_yields_fewer_needles(self) -> None:
        slow = render_primary_cementite_needles(
            size=self.size,
            seed=42,
            c_wt=5.5,
            base_image=self.base,
            cooling_rate_c_per_s=2.0,
        )["metadata"]["needle_count"]
        fast = render_primary_cementite_needles(
            size=self.size,
            seed=42,
            c_wt=5.5,
            base_image=self.base,
            cooling_rate_c_per_s=200.0,
        )["metadata"]["needle_count"]
        self.assertGreater(slow, fast)

    def test_deterministic_for_seed(self) -> None:
        a = render_primary_cementite_needles(
            size=self.size, seed=999, c_wt=5.0, base_image=self.base
        )["image"]
        b = render_primary_cementite_needles(
            size=self.size, seed=999, c_wt=5.0, base_image=self.base
        )["image"]
        self.assertTrue(np.array_equal(a, b))

    def test_three_dominant_orientations(self) -> None:
        meta = render_primary_cementite_needles(
            size=self.size,
            seed=42,
            c_wt=5.5,
            base_image=self.base,
        )["metadata"]
        self.assertEqual(len(meta["primary_directions_rad"]), 3)


if __name__ == "__main__":
    unittest.main()
