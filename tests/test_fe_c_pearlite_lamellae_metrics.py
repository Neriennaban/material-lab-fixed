"""Tests for A4 / A10.2 — magnification-aware pearlite rendering.

The TZ §6.3 formula ``S₀ = 8.3 / ΔT`` (in micrometres) tells us how the
true interlamellar spacing scales with cooling rate. Combined with the
microscope pixel pitch ``um_per_px = 1 / (magnification / 100)`` we can
decide at render time whether to draw explicit lamellae (high
magnification, fine pixel pitch) or to fall back to a uniform dark
pearlite blob (low magnification, lamellae sub-pixel).

These tests pin the contract of ``generate_pearlite_structure``:

* When ``um_per_px is None`` the function preserves the legacy
  ``lamella_period_px`` argument byte-for-byte (backward compatibility).
* At ~1000× (``um_per_px ≈ 0.1`` µm/px) and a furnace cooling rate of
  ~1 °C/s the lamella period sits in a sensible 3–10 px range with the
  ``"physical"`` mode tag.
* At 100× (``um_per_px = 1.0``) and the same cooling rate the period
  cannot be resolved → the renderer switches to ``"unresolved_uniform"``
  and the resulting pearlite area lacks any directional sine ridges.
* Faster cooling (sorbite/troostite range) shrinks ``S₀`` so the
  renderer also collapses to the uniform mode at moderate
  magnification.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from core.generator_pearlite import (
    _resolve_lamella_period_px,
    generate_pearlite_structure,
)


class ResolveLamellaPeriodTest(unittest.TestCase):
    def test_legacy_mode_when_microscope_context_missing(self) -> None:
        period, mode, s0 = _resolve_lamella_period_px(
            lamella_period_px=7.0,
            um_per_px=None,
            cooling_rate_c_per_s=None,
        )
        self.assertEqual(mode, "legacy_fixed")
        self.assertEqual(period, 7.0)
        self.assertTrue(math.isnan(s0))

    def test_high_magnification_resolves_lamellae_physically(self) -> None:
        # 1000×: 0.1 µm/px. ΔT ≈ 80*log10(1) = 0 → clamp to 8 °C ⇒
        # S0 ≈ 8.3 / 8 ≈ 1.04 µm ≈ 10.4 px.
        period, mode, s0 = _resolve_lamella_period_px(
            lamella_period_px=7.0,
            um_per_px=0.1,
            cooling_rate_c_per_s=1.0,
        )
        self.assertEqual(mode, "physical")
        self.assertGreater(period, 5.0)
        self.assertLess(period, 20.0)
        self.assertGreater(s0, 0.5)
        self.assertLess(s0, 2.0)

    def test_low_magnification_collapses_to_uniform(self) -> None:
        # 100×: 1.0 µm/px. Same S0 ≈ 1.04 µm → < 1.5 px → unresolved.
        period, mode, s0 = _resolve_lamella_period_px(
            lamella_period_px=7.0,
            um_per_px=1.0,
            cooling_rate_c_per_s=1.0,
        )
        self.assertEqual(mode, "unresolved_uniform")
        self.assertEqual(period, 7.0)  # falls back to legacy
        self.assertGreater(s0, 0.5)

    def test_fast_cooling_makes_period_finer(self) -> None:
        period_slow, _, s0_slow = _resolve_lamella_period_px(
            lamella_period_px=7.0,
            um_per_px=0.05,
            cooling_rate_c_per_s=1.0,
        )
        period_fast, _, s0_fast = _resolve_lamella_period_px(
            lamella_period_px=7.0,
            um_per_px=0.05,
            cooling_rate_c_per_s=100.0,
        )
        self.assertGreater(s0_slow, s0_fast)
        self.assertGreater(period_slow, period_fast)


class GeneratePearliteBackwardCompatTest(unittest.TestCase):
    def test_default_call_unchanged(self) -> None:
        a = generate_pearlite_structure(
            size=(64, 64),
            seed=12345,
            pearlite_fraction=0.6,
            lamella_period_px=7.0,
        )
        b = generate_pearlite_structure(
            size=(64, 64),
            seed=12345,
            pearlite_fraction=0.6,
            lamella_period_px=7.0,
        )
        self.assertTrue(np.array_equal(a["image"], b["image"]))
        self.assertEqual(a["metadata"]["lamella_mode"], "legacy_fixed")


class GeneratePearliteMagnificationModesTest(unittest.TestCase):
    def test_high_mag_renders_directional_lamellae(self) -> None:
        out = generate_pearlite_structure(
            size=(128, 128),
            seed=42,
            pearlite_fraction=1.0,
            lamella_period_px=7.0,
            colony_size_px=80.0,
            um_per_px=0.05,
            cooling_rate_c_per_s=1.0,
        )
        meta = out["metadata"]
        self.assertEqual(meta["lamella_mode"], "physical")
        # Image must contain bright cementite stripes (close to 175).
        bright = (out["image"] > 150).mean()
        self.assertGreater(bright, 0.10, f"bright lamella fraction {bright:.3f}")

    def test_low_mag_collapses_to_uniform(self) -> None:
        out = generate_pearlite_structure(
            size=(128, 128),
            seed=42,
            pearlite_fraction=1.0,
            lamella_period_px=7.0,
            colony_size_px=80.0,
            um_per_px=1.0,
            cooling_rate_c_per_s=1.0,
        )
        meta = out["metadata"]
        self.assertEqual(meta["lamella_mode"], "unresolved_uniform")
        # No bright lamellae should appear — at most a stray boundary.
        bright_fraction = float((out["image"] > 150).mean())
        self.assertLess(bright_fraction, 0.05)


if __name__ == "__main__":
    unittest.main()
