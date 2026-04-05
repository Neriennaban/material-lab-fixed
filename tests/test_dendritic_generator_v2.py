from __future__ import annotations

import unittest

import numpy as np

from core.generator_dendritic import generate_dendritic_cast


class DendriticGeneratorV2Tests(unittest.TestCase):
    def test_seed_determinism(self) -> None:
        out_a = generate_dendritic_cast(size=(180, 220), seed=777)
        out_b = generate_dendritic_cast(size=(180, 220), seed=777)
        self.assertTrue(np.array_equal(out_a["image_gray"], out_b["image_gray"]))
        self.assertTrue(np.array_equal(out_a["image_rgb"], out_b["image_rgb"]))

    def test_output_shapes_and_masks(self) -> None:
        out = generate_dendritic_cast(size=(160, 192), seed=21)
        self.assertEqual(out["image_gray"].shape, (160, 192))
        self.assertEqual(out["image_rgb"].shape, (160, 192, 3))
        masks = out["phase_masks"]
        self.assertIn("dendrite_core", masks)
        self.assertIn("interdendritic", masks)
        self.assertIn("porosity", masks)
        self.assertEqual(masks["porosity"].shape, (160, 192))

    def test_parameter_boundaries(self) -> None:
        out = generate_dendritic_cast(
            size=(128, 128),
            seed=1,
            cooling_rate=3000.0,
            thermal_gradient=-2.0,
            undercooling=999.0,
            primary_arm_spacing=2.0,
            secondary_arm_factor=3.0,
            interdendritic_fraction=1.2,
            porosity_fraction=0.9,
        )
        gray = out["image_gray"]
        self.assertEqual(gray.dtype, np.uint8)
        self.assertGreater(float(out["phase_masks"]["interdendritic"].mean()), 0.01)


if __name__ == "__main__":
    unittest.main()
