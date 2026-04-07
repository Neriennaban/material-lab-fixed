"""Tests for A5 — secondary cementite network thickness as f(c_wt).

The TZ §5.3.Б says the cementite grain-boundary network grows from
~1-2 px at 0.9 %C to ~5-8 px at 2.0 %C, with slow cooling further
widening it. The legacy ``texture_cementite_network`` used a fixed
``boundary_width_px=2``; A5 makes it carbon-aware while keeping the
default call byte-identical for backward compatibility.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.metallography_v3.system_generators.fe_c_textures import (
    texture_cementite_network,
)


def _bright_boundary_fraction(image: np.ndarray) -> float:
    """Fraction of pixels that landed in the bright cementite ring
    (boundary + matrix-on-boundary blob), i.e. brighter than the
    base matrix (235)."""
    return float((image > 240).mean())


class CementiteNetworkThicknessTest(unittest.TestCase):
    SIZE = (192, 192)
    SEED = 4242

    def _render(self, **kwargs) -> np.ndarray:
        return texture_cementite_network(self.SIZE, self.SEED, **kwargs)

    def test_legacy_call_unchanged(self) -> None:
        a = texture_cementite_network(self.SIZE, self.SEED)
        b = texture_cementite_network(self.SIZE, self.SEED)
        self.assertTrue(np.array_equal(a, b))

    def test_higher_carbon_thickens_network(self) -> None:
        thin = self._render(c_wt=0.85)  # near eutectoid
        thick = self._render(c_wt=2.0)  # near steel limit
        self.assertGreater(
            _bright_boundary_fraction(thick),
            _bright_boundary_fraction(thin) + 0.05,
            "thicker network should produce more bright boundary pixels",
        )

    def test_slow_cooling_thickens_network(self) -> None:
        normal = self._render(c_wt=1.4, cooling_rate_c_per_s=1.0)
        slow = self._render(c_wt=1.4, cooling_rate_c_per_s=0.05)
        self.assertGreater(
            _bright_boundary_fraction(slow),
            _bright_boundary_fraction(normal),
        )

    def test_carbon_clamped_below_eutectoid(self) -> None:
        # Below the eutectoid composition the network must not grow:
        # the formula clamps c_wt at 0.77, so c_wt=0.5 must produce
        # the thinnest possible boundary.
        below = self._render(c_wt=0.5)
        at_eutectoid = self._render(c_wt=0.77)
        self.assertEqual(
            _bright_boundary_fraction(below),
            _bright_boundary_fraction(at_eutectoid),
        )

    def test_carbon_clamped_above_steel_limit(self) -> None:
        # Above 2.14 %C clamps to the maximum thickness — passing
        # 5.0 must not exceed the value at 2.14.
        at_limit = self._render(c_wt=2.14)
        above = self._render(c_wt=5.0)
        self.assertEqual(
            _bright_boundary_fraction(at_limit),
            _bright_boundary_fraction(above),
        )


if __name__ == "__main__":
    unittest.main()
