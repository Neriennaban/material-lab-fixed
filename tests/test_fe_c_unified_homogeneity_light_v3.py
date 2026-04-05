from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


class FeCUnifiedHomogeneityLightTests(unittest.TestCase):
    def _ctx(self, seed: int) -> SystemGenerationContext:
        return SystemGenerationContext(
            size=(192, 192),
            seed=int(seed),
            inferred_system="fe-c",
            stage="troostite_temper",
            phase_fractions={"TROOSTITE": 0.62, "CEMENTITE": 0.2, "FERRITE": 0.18},
            composition_wt={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=35.0, cooling_mode="tempered"),
            thermal_summary={"operation_inference": {"has_quench": False, "has_temper": True}},
            quench_summary={"effect_applied": False, "medium_code_resolved": "water_20"},
        )

    def test_deterministic_render(self) -> None:
        a = render_fe_c_unified(self._ctx(551))
        b = render_fe_c_unified(self._ctx(551))
        self.assertTrue(np.array_equal(a.image_gray, b.image_gray))
        self.assertTrue(bool(dict(a.metadata.get("fe_c_phase_render", {})).get("homogeneity_mode", "") == "light"))

    def test_light_homogeneity_keeps_phase_content(self) -> None:
        out = render_fe_c_unified(self._ctx(552))
        img = out.image_gray.astype(np.float32)
        h, w = img.shape
        left = float(np.mean(img[:, : w // 2]))
        right = float(np.mean(img[:, w // 2 :]))
        self.assertLess(abs(left - right), 18.0)

        masks = dict(out.phase_masks)
        non_empty = [name for name, mask in masks.items() if isinstance(mask, np.ndarray) and np.any(mask > 0)]
        self.assertGreaterEqual(len(non_empty), 2)


if __name__ == "__main__":
    unittest.main()

