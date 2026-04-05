from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import GenerationRequestV2, ProcessingState
from core.pipeline_v2 import GenerationPipelineV2


class PipelineV2CompositionEffectTests(unittest.TestCase):
    def test_same_phase_different_composition_changes_image(self) -> None:
        pipeline = GenerationPipelineV2()

        req_low_si = GenerationRequestV2(
            mode="direct",
            composition={"Fe": 99.5, "Si": 0.5},
            processing=ProcessingState(temperature_c=900.0, cooling_mode="equilibrium"),
            generator="auto",
            resolution=(96, 96),
            strict_validation=True,
        )
        req_high_si = GenerationRequestV2(
            mode="direct",
            composition={"Fe": 95.0, "Si": 5.0},
            processing=ProcessingState(temperature_c=900.0, cooling_mode="equilibrium"),
            generator="auto",
            resolution=(96, 96),
            strict_validation=True,
        )

        out_low = pipeline.generate(req_low_si)
        out_high = pipeline.generate(req_high_si)

        self.assertEqual(out_low.metadata.get("auto_generator", {}).get("selected_generator"), "calphad_phase")
        self.assertEqual(out_high.metadata.get("auto_generator", {}).get("selected_generator"), "calphad_phase")

        stable_low = out_low.metadata.get("calphad", {}).get("equilibrium_result", {}).get("stable_phases", {})
        stable_high = out_high.metadata.get("calphad", {}).get("equilibrium_result", {}).get("stable_phases", {})
        self.assertIsInstance(stable_low, dict)
        self.assertIsInstance(stable_high, dict)
        self.assertTrue(stable_low)
        self.assertTrue(stable_high)
        self.assertEqual(max(stable_low, key=stable_low.get), max(stable_high, key=stable_high.get))

        mae = float(np.abs(out_low.image_gray.astype(np.float32) - out_high.image_gray.astype(np.float32)).mean())
        self.assertGreater(mae, 1.0)

        comp_effect = out_low.metadata.get("composition_effect", {})
        self.assertIsInstance(comp_effect, dict)
        self.assertIn("mode", comp_effect)
        self.assertIn("solute_index", comp_effect)


if __name__ == "__main__":
    unittest.main()

