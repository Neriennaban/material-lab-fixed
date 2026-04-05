from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v3 import MetallographyRequestV3, SynthesisProfileV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3PreviewDeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = MetallographyPipelineV3()

    def _request(self, seed: int) -> MetallographyRequestV3:
        req = MetallographyRequestV3(
            sample_id="preview_det",
            system_hint="fe-c",
            composition_wt={"Fe": 99.2, "C": 0.8},
            resolution=(96, 96),
            seed=seed,
            synthesis_profile=SynthesisProfileV3(generation_mode="realistic_visual", system_generator_mode="system_auto"),
            microscope_profile={
                "simulate_preview": True,
                "magnification": 400,
                "focus": 0.92,
                "brightness": 1.0,
                "contrast": 1.0,
                "vignette_strength": 0.12,
                "uneven_strength": 0.08,
                "noise_sigma": 1.8,
                "add_dust": True,
                "add_scratches": True,
                "etch_uneven": 0.3,
            },
        )
        req.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=220.0, temperature_c=780.0),
            ThermalPointV3(time_s=420.0, temperature_c=780.0),
            ThermalPointV3(time_s=740.0, temperature_c=30.0),
        ]
        return req

    def test_same_seed_keeps_preview_reproducible(self) -> None:
        a = self.pipeline.generate(self._request(seed=777))
        b = self.pipeline.generate(self._request(seed=777))
        self.assertTrue(np.array_equal(a.image_gray, b.image_gray))
        self.assertEqual(str(a.metadata.get("preview_optics", {}).get("optical_mode", "")), "brightfield")
        self.assertEqual(str(a.metadata.get("preview_optics", {}).get("psf_profile", "")), "standard")

    def test_different_seed_changes_preview(self) -> None:
        a = self.pipeline.generate(self._request(seed=777)).image_gray.astype(np.float32)
        b = self.pipeline.generate(self._request(seed=778)).image_gray.astype(np.float32)
        mae = float(np.mean(np.abs(a - b)))
        self.assertGreater(mae, 1.0)


if __name__ == "__main__":
    unittest.main()
