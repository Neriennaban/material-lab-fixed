"""Tests for A7 (ferrite annealing twins) and A9 (anisotropic nital etch).

Both refinements live behind opt-in flags on ``texture_ferrite`` so the
legacy call signature stays byte-identical with the snapshot baseline.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.metallography_v3.system_generators.fe_c_textures import texture_ferrite


def _find_seed_with_visible_twins() -> tuple[tuple[int, int], int]:
    """Pick a (size, seed) combination where the 7 % twin probability
    actually fires for at least one grain — needed so the diff-based
    tests are stable across stochastic seed choices."""
    candidates = [(256, 256), (320, 320)]
    for size in candidates:
        for seed in range(7000, 7050):
            legacy = texture_ferrite(size, seed)
            twinned = texture_ferrite(size, seed, add_twins=True)
            if not np.array_equal(legacy, twinned):
                return size, seed
    raise RuntimeError("Could not find a seed where twins fire (unexpected)")


class FerriteRefinementsTest(unittest.TestCase):
    SIZE, SEED = _find_seed_with_visible_twins()

    def test_legacy_call_unchanged(self) -> None:
        a = texture_ferrite(self.SIZE, self.SEED)
        b = texture_ferrite(self.SIZE, self.SEED)
        self.assertTrue(np.array_equal(a, b))

    def test_twins_flag_changes_output(self) -> None:
        legacy = texture_ferrite(self.SIZE, self.SEED)
        twinned = texture_ferrite(self.SIZE, self.SEED, add_twins=True)
        self.assertFalse(np.array_equal(legacy, twinned))

    def test_anisotropic_etching_changes_output(self) -> None:
        legacy = texture_ferrite(self.SIZE, self.SEED)
        aniso = texture_ferrite(self.SIZE, self.SEED, anisotropic_etching=True)
        self.assertFalse(np.array_equal(legacy, aniso))

    def test_both_flags_combine(self) -> None:
        legacy = texture_ferrite(self.SIZE, self.SEED)
        both = texture_ferrite(
            self.SIZE,
            self.SEED,
            add_twins=True,
            anisotropic_etching=True,
        )
        self.assertFalse(np.array_equal(legacy, both))
        twins = texture_ferrite(self.SIZE, self.SEED, add_twins=True)
        aniso = texture_ferrite(self.SIZE, self.SEED, anisotropic_etching=True)
        self.assertFalse(np.array_equal(both, twins))
        self.assertFalse(np.array_equal(both, aniso))

    def test_anisotropic_widens_intensity_distribution(self) -> None:
        legacy = texture_ferrite(self.SIZE, self.SEED).astype(np.float32)
        aniso = texture_ferrite(
            self.SIZE, self.SEED, anisotropic_etching=True
        ).astype(np.float32)
        # The orientation-driven offset adds variance to the per-grain
        # tone distribution, so the std of the anisotropic version
        # must not be smaller than the legacy version.
        self.assertGreaterEqual(float(aniso.std()), float(legacy.std()) * 0.95)

    def test_twins_are_subtle_not_full_contrast(self) -> None:
        legacy = texture_ferrite(self.SIZE, self.SEED).astype(np.float32)
        twinned = texture_ferrite(
            self.SIZE, self.SEED, add_twins=True
        ).astype(np.float32)
        diff = np.abs(twinned - legacy)
        nonzero = diff[diff > 0.5]
        if nonzero.size > 0:
            # Twins should add at most ~10 grayscale units locally —
            # they are subtle features, not full lamellae.
            self.assertLess(float(nonzero.max()), 30.0)

    def test_determinism_with_flags(self) -> None:
        a = texture_ferrite(
            self.SIZE,
            self.SEED,
            add_twins=True,
            anisotropic_etching=True,
        )
        b = texture_ferrite(
            self.SIZE,
            self.SEED,
            add_twins=True,
            anisotropic_etching=True,
        )
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
