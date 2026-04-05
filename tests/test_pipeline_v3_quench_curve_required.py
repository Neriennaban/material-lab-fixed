from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3QuenchCurveRequiredTests(unittest.TestCase):
    def test_no_quench_segment_disables_quench_effect(self) -> None:
        pipeline = MetallographyPipelineV3()
        request = MetallographyRequestV3(
            sample_id="no_quench_curve",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            resolution=(96, 96),
            seed=1501,
        )
        request.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=260.0, temperature_c=840.0),
            ThermalPointV3(time_s=500.0, temperature_c=840.0),
            ThermalPointV3(time_s=900.0, temperature_c=700.0),
            ThermalPointV3(time_s=1300.0, temperature_c=600.0),
        ]
        request.thermal_program.quench.medium_code = "water_20"
        request.thermal_program.quench.bath_temperature_c = 20.0
        request.thermal_program.quench.sample_temperature_c = 840.0

        out = pipeline.generate(request)
        thermal = dict(out.metadata.get("thermal_program_summary", {}))
        quench = dict(out.metadata.get("quench_summary", {}))
        self.assertFalse(bool(thermal.get("quench_effect_applied", True)))
        self.assertEqual(str(thermal.get("quench_effect_reason", "")), "no_quench_segment")
        self.assertFalse(bool(quench.get("effect_applied", True)))
        self.assertEqual(str(quench.get("effect_reason", "")), "no_quench_segment")
        self.assertNotIn(
            str(out.metadata.get("final_stage", "")),
            {"martensite", "martensite_tetragonal", "martensite_cubic", "troostite_quench", "sorbite_quench"},
        )

    def test_without_quench_segment_medium_switch_does_not_change_stage_family(self) -> None:
        pipeline = MetallographyPipelineV3()
        base_points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=300.0, temperature_c=840.0),
            ThermalPointV3(time_s=540.0, temperature_c=840.0),
            ThermalPointV3(time_s=980.0, temperature_c=680.0),
            ThermalPointV3(time_s=1450.0, temperature_c=620.0),
        ]
        stages: list[str] = []
        for medium, bath_t in (("water_20", 20.0), ("oil_20_80", 60.0)):
            request = MetallographyRequestV3(
                sample_id=f"no_quench_{medium}",
                composition_wt={"Fe": 99.2, "C": 0.8},
                system_hint="fe-c",
                resolution=(96, 96),
                seed=1502,
            )
            request.thermal_program.points = list(base_points)
            request.thermal_program.quench.medium_code = medium
            request.thermal_program.quench.bath_temperature_c = bath_t
            request.thermal_program.quench.sample_temperature_c = 840.0
            out = pipeline.generate(request)
            thermal = dict(out.metadata.get("thermal_program_summary", {}))
            self.assertFalse(bool(thermal.get("quench_effect_applied", True)))
            stages.append(str(out.metadata.get("final_stage", "")))

        self.assertEqual(stages[0], stages[1])


if __name__ == "__main__":
    unittest.main()

