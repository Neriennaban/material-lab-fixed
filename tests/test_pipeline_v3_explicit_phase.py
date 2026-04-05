from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, PhaseModelConfigV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3ExplicitPhaseTests(unittest.TestCase):
    def test_explicit_phase_metadata_blocks_present(self) -> None:
        pipeline = MetallographyPipelineV3()
        request = MetallographyRequestV3(
            sample_id="explicit_phase_case",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            phase_model=PhaseModelConfigV3(
                engine="explicit_rules_v3",
                phase_control_mode="auto_with_override",
                manual_phase_fractions={"FERRITE": 0.15, "PEARLITE": 0.85},
                manual_override_weight=0.35,
                allow_custom_fallback=True,
            ),
            resolution=(96, 96),
            seed=2026,
            strict_validation=True,
        )
        request.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=300.0, temperature_c=780.0),
            ThermalPointV3(time_s=540.0, temperature_c=780.0),
            ThermalPointV3(time_s=860.0, temperature_c=35.0),
        ]
        out = pipeline.generate(request)
        meta = out.metadata
        self.assertIn("phase_model_report", meta)
        self.assertIn("system_resolution", meta)
        self.assertIn("operations_from_curve", meta)
        self.assertIsInstance(meta["phase_model_report"], dict)
        self.assertIsInstance(meta["system_resolution"], dict)
        self.assertIsInstance(meta["operations_from_curve"], dict)
        self.assertIn("operation_inference", dict(meta.get("thermal_program_summary", {})))
        self.assertNotIn("calphad", meta)


if __name__ == "__main__":
    unittest.main()

