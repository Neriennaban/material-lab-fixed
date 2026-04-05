from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


class FeCUnifiedDeterminismTests(unittest.TestCase):
    def _ctx(self, seed: int) -> SystemGenerationContext:
        return SystemGenerationContext(
            size=(160, 160),
            seed=seed,
            inferred_system="fe-c",
            stage="sorbite_temper",
            phase_fractions={"SORBITE": 0.62, "CEMENTITE": 0.22, "FERRITE": 0.16},
            composition_wt={"Fe": 99.25, "C": 0.75},
            processing=ProcessingState(temperature_c=420.0, cooling_mode="tempered"),
        )

    def test_same_seed_is_identical(self) -> None:
        a = render_fe_c_unified(self._ctx(901)).image_gray
        b = render_fe_c_unified(self._ctx(901)).image_gray
        self.assertTrue(np.array_equal(a, b))

    def test_different_seed_changes_image(self) -> None:
        a = render_fe_c_unified(self._ctx(901)).image_gray.astype(np.float32)
        b = render_fe_c_unified(self._ctx(902)).image_gray.astype(np.float32)
        mae = float(np.mean(np.abs(a - b)))
        self.assertGreater(mae, 1.0)


if __name__ == "__main__":
    unittest.main()
