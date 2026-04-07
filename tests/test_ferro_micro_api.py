"""Tests for the ``ferro_micro_api`` public-API facade (Phase B2).

These tests focus on the public surface only — the underlying
``MetallographyPipelineV3`` is exercised by the rest of the suite.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.metallography_v3 import ferro_micro_api as fm


class GenerateTest(unittest.TestCase):
    def test_default_call_returns_rgb_with_phases(self) -> None:
        sample = fm.generate(
            carbon=0.45,
            width=128,
            height=128,
            seed=42,
        )
        self.assertEqual(sample.image.shape, (128, 128, 3))
        self.assertEqual(sample.image.dtype, np.uint8)
        self.assertEqual(sample.image_gray.shape, (128, 128))
        self.assertGreater(len(sample.phase_masks), 0)
        self.assertIsNone(sample.info)

    def test_color_mode_propagated(self) -> None:
        sample = fm.generate(
            carbon=0.45,
            width=128,
            height=128,
            color_mode="nital_warm",
            seed=42,
        )
        rgb = sample.image
        self.assertFalse(np.array_equal(rgb[..., 0], rgb[..., 2]))

    def test_return_info_populates_phase_fractions(self) -> None:
        sample = fm.generate(
            carbon=0.20,
            width=128,
            height=128,
            seed=42,
            return_info=True,
        )
        self.assertIsNotNone(sample.info)
        phases = sample.info["phases"]
        self.assertGreater(sum(phases.values()), 0.5)
        # Hypoeutectoid carbon → ferrite + pearlite expected.
        self.assertIn("FERRITE", phases)
        self.assertIn("PEARLITE", phases)

    def test_seed_determinism(self) -> None:
        a = fm.generate(carbon=0.45, width=96, height=96, seed=42)
        b = fm.generate(carbon=0.45, width=96, height=96, seed=42)
        self.assertTrue(np.array_equal(a.image, b.image))

    def test_thermal_program_override(self) -> None:
        # Custom thermal program — auto-build path is bypassed.
        program = [
            {"time_s": 0.0, "temperature_c": 20.0, "label": "start", "locked": True},
            {"time_s": 600.0, "temperature_c": 870.0, "label": "austenitize"},
            {"time_s": 1800.0, "temperature_c": 870.0, "label": "hold"},
            {"time_s": 3600.0, "temperature_c": 20.0, "label": "cool"},
        ]
        sample = fm.generate(
            carbon=0.45,
            width=96,
            height=96,
            thermal_program=program,
            seed=42,
        )
        self.assertEqual(sample.image.shape, (96, 96, 3))


class PresetsTest(unittest.TestCase):
    def test_alias_list_contains_known_grades(self) -> None:
        aliases = fm.presets.list_aliases()
        self.assertIn("armco", aliases)
        self.assertIn("steel_20", aliases)
        self.assertIn("cast_iron_white_hypereutectic", aliases)

    def test_armco_preset_renders_single_phase(self) -> None:
        sample = fm.presets.armco(width=96, height=96, return_info=True)
        self.assertEqual(sample.image.shape, (96, 96, 3))
        phase_names = {n.upper() for n in sample.phase_masks.keys()}
        self.assertEqual(phase_names, {"FERRITE"})

    def test_steel_20_preset_renders_with_pearlite(self) -> None:
        sample = fm.presets.steel_20(width=96, height=96, return_info=True)
        phase_names = {n.upper() for n in sample.phase_masks.keys()}
        self.assertIn("FERRITE", phase_names)
        self.assertIn("PEARLITE", phase_names)
        # Should be predominantly ferrite at 0.20 % C.
        ferrite_fraction = sample.info["phases"]["FERRITE"]
        self.assertGreater(ferrite_fraction, 0.5)

    def test_white_cast_iron_preset_has_multiple_phases(self) -> None:
        sample = fm.presets.cast_iron_white_hypereutectic(width=96, height=96)
        phases = {n.upper() for n in sample.phase_masks.keys()}
        self.assertGreaterEqual(len(phases), 3)

    def test_unknown_alias_raises(self) -> None:
        with self.assertRaises(AttributeError):
            fm.presets.steel_NONEXISTENT  # type: ignore[attr-defined]

    def test_color_mode_override_in_preset(self) -> None:
        sample = fm.presets.armco(
            width=96, height=96, color_mode="dic_polarized"
        )
        rgb = sample.image
        # DIC palette must produce a chromatic frame.
        self.assertFalse(np.array_equal(rgb[..., 0], rgb[..., 2]))


if __name__ == "__main__":
    unittest.main()
