from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, PhaseModelConfigV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3CustomFallbackTests(unittest.TestCase):
    def test_custom_multicomponent_uses_fallback(self) -> None:
        pipeline = MetallographyPipelineV3()
        request = MetallographyRequestV3(
            sample_id="custom_case",
            composition_wt={"Ni": 62.0, "Cr": 24.0, "Mo": 9.0, "W": 5.0},
            system_hint=None,
            phase_model=PhaseModelConfigV3(
                phase_control_mode="auto_with_override",
                allow_custom_fallback=True,
            ),
            resolution=(96, 96),
            seed=77,
            strict_validation=True,
        )
        out = pipeline.generate(request)
        report = out.metadata.get("phase_model_report", {})
        self.assertIsInstance(report, dict)
        self.assertTrue(bool(report.get("fallback_used")))
        self.assertEqual(out.metadata.get("inferred_system"), "custom-multicomponent")


if __name__ == "__main__":
    unittest.main()

