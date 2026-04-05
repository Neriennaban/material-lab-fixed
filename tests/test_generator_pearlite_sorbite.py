from __future__ import annotations

import unittest

import numpy as np

from core.generator_pearlite import generate_sorbite_structure


class SorbiteGeneratorTests(unittest.TestCase):
    def test_sorbite_is_deterministic_for_same_seed(self) -> None:
        a = generate_sorbite_structure(size=(256, 256), seed=123, mode="temper")["image"]
        b = generate_sorbite_structure(size=(256, 256), seed=123, mode="temper")["image"]
        self.assertTrue(np.array_equal(a, b))

    def test_temper_and_quench_modes_are_visually_distinct(self) -> None:
        temper = generate_sorbite_structure(size=(256, 256), seed=123, mode="temper")["image"].astype(np.float32)
        quench = generate_sorbite_structure(size=(256, 256), seed=123, mode="quench")["image"].astype(np.float32)
        mae = float(np.mean(np.abs(temper - quench)))
        self.assertGreater(mae, 8.0)


if __name__ == "__main__":
    unittest.main()
