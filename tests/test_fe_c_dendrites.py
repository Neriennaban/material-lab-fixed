"""Tests for A3 — Fe-C austenite dendrite renderer for hypoeutectic
white cast iron.

The TZ §5.5 says the carbon excess below the eutectic point drives the
amount of austenite that crystallises as primary dendrites: the closer
to 2.14 % C, the more austenite (and thus more dendritic coverage);
the closer to 4.3 % C, the less. Cooling rate has the opposite effect:
fast cooling fragments the dendrites into many small trunks, slow
cooling produces fewer but bigger ones.

The renderer also needs to be deterministic for a fixed seed and to
expose a usable per-pixel mask of the dendritic regions.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.metallography_v3.system_generators.fe_c_dendrites import (
    render_fe_c_austenite_dendrites,
)
from core.metallography_v3.system_generators.fe_c_textures import (
    texture_ledeburite_leopard,
)


class FeCDendritesTest(unittest.TestCase):
    SIZE = (192, 192)

    @classmethod
    def setUpClass(cls) -> None:
        cls.base = texture_ledeburite_leopard(cls.SIZE, seed=4242)

    def _render(self, **kwargs):
        return render_fe_c_austenite_dendrites(
            size=self.SIZE,
            seed=42,
            base_image=self.base,
            **kwargs,
        )

    def test_output_shape_and_keys(self) -> None:
        out = self._render(c_wt=3.0)
        self.assertEqual(out["image"].shape, self.SIZE)
        self.assertEqual(out["image"].dtype, np.uint8)
        self.assertEqual(out["dendrite_mask"].shape, self.SIZE)
        self.assertIn("trunk_count", out["metadata"])
        self.assertIn("trunk_length_px", out["metadata"])
        self.assertIn("max_branch_order", out["metadata"])

    def test_more_dendrites_at_lower_carbon(self) -> None:
        # 2.2 % C is closer to the steel-cast iron limit, so the melt
        # contains more primary austenite → more dendrites than at 4.0 %.
        low_c = self._render(c_wt=2.2)
        high_c = self._render(c_wt=4.0)
        self.assertGreater(
            int(low_c["dendrite_mask"].sum()),
            int(high_c["dendrite_mask"].sum()),
            "lower carbon should yield more dendritic coverage",
        )

    def test_faster_cooling_yields_more_trunks(self) -> None:
        slow = self._render(c_wt=3.0, cooling_rate_c_per_s=0.5)
        fast = self._render(c_wt=3.0, cooling_rate_c_per_s=20.0)
        self.assertGreater(
            int(fast["metadata"]["trunk_count"]),
            int(slow["metadata"]["trunk_count"]),
        )

    def test_dendrites_darker_than_base(self) -> None:
        out = self._render(c_wt=3.0)
        mask = out["dendrite_mask"].astype(bool)
        if mask.any():
            dendrite_pixels = out["image"][mask].astype(np.float32)
            base_pixels = self.base[mask].astype(np.float32)
            self.assertLess(
                float(dendrite_pixels.mean()),
                float(base_pixels.mean()),
                "dendrite pixels should be darker than the base",
            )

    def test_branch_order_increases_with_carbon_excess(self) -> None:
        # Greater carbon excess (low C, near 2.14) should produce more
        # branching orders than near the eutectic (high C, near 4.3).
        excess_high = self._render(c_wt=2.3)["metadata"]["max_branch_order"]
        excess_low = self._render(c_wt=4.1)["metadata"]["max_branch_order"]
        self.assertGreaterEqual(excess_high, excess_low)

    def test_deterministic_for_seed(self) -> None:
        a = render_fe_c_austenite_dendrites(
            size=(96, 96), seed=999, c_wt=3.0
        )["image"]
        b = render_fe_c_austenite_dendrites(
            size=(96, 96), seed=999, c_wt=3.0
        )["image"]
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
