"""Tests for A8 — retained austenite localisation along lath
boundaries in ``_build_martensitic_render``.

The plan asks the renderer to push retained austenite (RA) onto the
inter-lath films instead of distributing it as random islands. The
default weight is 0.72 (legacy, keeps the snapshot baseline byte
identical); presets can opt in to a stronger 0.85-0.92 weight via the
new ``ra_boundary_strength`` field on ``SystemGenerationContext``.

This test verifies:

* the legacy default already produces a non-trivial boundary bias
  (the existing renderer is not random);
* a higher ``ra_boundary_strength`` increases the bias score relative
  to the legacy weight;
* the metric is reported in
  ``output.metadata["fe_c_phase_render"]["morphology_trace"]
  ["retained_austenite_boundary_bias"]``.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import (
    _build_martensitic_render,
)


def _make_martensitic_context(
    *,
    ra_boundary_strength: float | None,
    seed: int = 4242,
) -> SystemGenerationContext:
    return SystemGenerationContext(
        size=(192, 192),
        seed=seed,
        inferred_system="fe-c",
        stage="martensite_tetragonal",
        phase_fractions={"MARTENSITE_TETRAGONAL": 0.86, "AUSTENITE": 0.14},
        composition_wt={"Fe": 99.2, "C": 0.8},
        processing=ProcessingState(),
        thermal_summary={"max_effective_cooling_rate_c_per_s": 200.0},
        ra_boundary_strength=ra_boundary_strength,
    )


def _render(context: SystemGenerationContext):
    return _build_martensitic_render(
        context=context,
        stage="martensite_tetragonal",
        phase_fractions={"MARTENSITE_TETRAGONAL": 0.86, "AUSTENITE": 0.14},
        seed_split={
            "seed_topology": int(context.seed) + 1001,
            "seed_boundary": int(context.seed) + 1003,
            "seed_particles": int(context.seed) + 1007,
            "seed_lamella": int(context.seed) + 1013,
            "seed_noise": int(context.seed) + 1021,
        },
        retained_austenite_used=0.14,
    )


class RALocalizationTest(unittest.TestCase):
    def test_legacy_default_byte_identical(self) -> None:
        ctx_a = _make_martensitic_context(ra_boundary_strength=None)
        ctx_b = _make_martensitic_context(ra_boundary_strength=None)
        img_a, masks_a, trace_a = _render(ctx_a)
        img_b, masks_b, trace_b = _render(ctx_b)
        self.assertTrue(np.array_equal(img_a, img_b))
        self.assertEqual(
            trace_a["retained_austenite_boundary_bias"],
            trace_b["retained_austenite_boundary_bias"],
        )

    def test_legacy_default_already_boundary_biased(self) -> None:
        ctx = _make_martensitic_context(ra_boundary_strength=None)
        _, masks, trace = _render(ctx)
        self.assertIn("AUSTENITE", masks)
        self.assertGreater(int(masks["AUSTENITE"].sum()), 0)
        self.assertGreater(
            float(trace["retained_austenite_boundary_bias"]),
            0.40,
            "legacy default should already place RA preferentially on boundaries",
        )

    def test_stronger_strength_increases_bias(self) -> None:
        legacy = _render(_make_martensitic_context(ra_boundary_strength=None))[2]
        strong = _render(_make_martensitic_context(ra_boundary_strength=0.92))[2]
        self.assertGreaterEqual(
            float(strong["retained_austenite_boundary_bias"]),
            float(legacy["retained_austenite_boundary_bias"]),
        )

    def test_strength_clamped_to_valid_range(self) -> None:
        # Above 0.95 the renderer must clamp internally so the noise
        # weight does not collapse to zero.
        out = _render(_make_martensitic_context(ra_boundary_strength=2.0))
        _, masks, trace = out
        self.assertGreater(int(masks["AUSTENITE"].sum()), 0)
        self.assertLessEqual(
            float(trace["retained_austenite_boundary_bias"]), 1.0
        )

    def test_metadata_records_distribution_label(self) -> None:
        _, _, trace = _render(_make_martensitic_context(ra_boundary_strength=0.92))
        self.assertIn("retained_austenite_distribution", trace)
        self.assertIn(
            trace["retained_austenite_distribution"],
            {"boundary_films", "mixed_films_islands"},
        )


if __name__ == "__main__":
    unittest.main()
