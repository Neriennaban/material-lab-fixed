from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, QuenchSettingsV3, SynthesisProfileV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3FeCUnifiedTests(unittest.TestCase):
    def test_pipeline_marks_fe_c_unified_metadata(self) -> None:
        pipeline = MetallographyPipelineV3()
        request = MetallographyRequestV3(
            sample_id="fe_c_unified_meta",
            composition_wt={"Fe": 99.3, "C": 0.7},
            system_hint="fe-c",
            synthesis_profile=SynthesisProfileV3(system_generator_mode="system_fe_c"),
            resolution=(128, 128),
            seed=321,
        )
        out = pipeline.generate(request)
        system_gen = dict(out.metadata.get("system_generator", {}))
        self.assertEqual(system_gen.get("resolved_mode"), "system_fe_c")
        fe_c_unified = dict(system_gen.get("fe_c_unified", {}))
        self.assertTrue(bool(fe_c_unified.get("enabled", False)))
        self.assertIn("resolved_stage", fe_c_unified)

        render_meta = dict(out.metadata.get("fe_c_phase_render", {}))
        self.assertIn("normalized_phase_fractions", render_meta)
        self.assertTrue(bool(render_meta.get("phase_masks_present", False)))
        self.assertIn(str(render_meta.get("fraction_source", "")), {"table_interpolated", "default_formula"})

    def test_pipeline_resolves_temper_stage_after_quench(self) -> None:
        pipeline = MetallographyPipelineV3()
        request = MetallographyRequestV3(
            sample_id="fe_c_temper_stage",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            synthesis_profile=SynthesisProfileV3(system_generator_mode="system_fe_c"),
            resolution=(128, 128),
            seed=322,
        )
        request.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0, label="start"),
            ThermalPointV3(time_s=300.0, temperature_c=840.0, label="austenitize"),
            ThermalPointV3(time_s=480.0, temperature_c=840.0, label="hold"),
            ThermalPointV3(time_s=560.0, temperature_c=90.0, label="quench"),
            ThermalPointV3(time_s=900.0, temperature_c=400.0, label="temper_heat"),
            ThermalPointV3(time_s=1320.0, temperature_c=400.0, label="temper_hold"),
            ThermalPointV3(time_s=1600.0, temperature_c=35.0, label="finish"),
        ]
        request.thermal_program.quench = QuenchSettingsV3(
            medium_code="water_20",
            quench_time_s=35.0,
            bath_temperature_c=20.0,
            sample_temperature_c=840.0,
        )

        out = pipeline.generate(request)
        self.assertEqual(str(out.metadata.get("final_stage", "")), "troostite_temper")
        inferred = dict(out.metadata.get("thermal_program_summary", {})).get("operation_inference", {})
        self.assertTrue(bool(dict(inferred).get("has_temper", False)))


if __name__ == "__main__":
    unittest.main()
