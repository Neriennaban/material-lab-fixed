from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v3 import MetallographyRequestV3, SynthesisProfileV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3CompositionEffectTests(unittest.TestCase):
    def test_single_phase_fe_si_responds_to_composition(self) -> None:
        pipeline = MetallographyPipelineV3()

        request_common = dict(
            sample_id="cmp_v3",
            system_hint="fe-si",
            resolution=(96, 96),
            seed=31415,
            strict_validation=True,
            synthesis_profile=SynthesisProfileV3(composition_sensitivity_mode="realistic"),
        )
        req_low = MetallographyRequestV3(composition_wt={"Fe": 99.5, "Si": 0.5}, **request_common)
        req_high = MetallographyRequestV3(composition_wt={"Fe": 95.0, "Si": 5.0}, **request_common)
        thermal_points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=250.0, temperature_c=900.0),
            ThermalPointV3(time_s=450.0, temperature_c=900.0),
            ThermalPointV3(time_s=1200.0, temperature_c=25.0),
        ]
        req_low.thermal_program.points = list(thermal_points)
        req_high.thermal_program.points = list(thermal_points)

        out_low = pipeline.generate(req_low)
        out_high = pipeline.generate(req_high)

        mae = float(np.abs(out_low.image_gray.astype(np.float32) - out_high.image_gray.astype(np.float32)).mean())
        self.assertGreater(mae, 1.0)

        comp_effect = out_low.metadata.get("composition_effect", {})
        self.assertIsInstance(comp_effect, dict)
        self.assertEqual(comp_effect.get("mode"), "realistic")
        self.assertIn("composition_hash", comp_effect)
        self.assertIn("seed_offset", comp_effect)

        vis = out_low.metadata.get("phase_visibility_report", {})
        self.assertIsInstance(vis, dict)
        self.assertIn("within_tolerance", vis)
        self.assertIn("separability_score", vis)

        trace = out_low.metadata.get("engineering_trace", {})
        self.assertIsInstance(trace, dict)
        self.assertIn("generation_mode", trace)
        self.assertIn("phase_emphasis_style", trace)
        self.assertIn("phase_fraction_tolerance_pct", trace)

        self.assertIn("phase_model_report", out_low.metadata)
        self.assertIn("system_resolution", out_low.metadata)
        self.assertNotIn("calphad", out_low.metadata)


if __name__ == "__main__":
    unittest.main()

