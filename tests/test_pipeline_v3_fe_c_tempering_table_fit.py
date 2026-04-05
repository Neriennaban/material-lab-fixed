from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3FeCTemperingTableFitTests(unittest.TestCase):
    def _build_request(self, *, c_wt: float, mode: str, seed: int) -> MetallographyRequestV3:
        req = MetallographyRequestV3(
            sample_id=f"table_fit_{mode}_{int(round(c_wt * 100))}",
            composition_wt={"Fe": 100.0 - c_wt, "C": c_wt},
            system_hint="fe-c",
            resolution=(96, 96),
            seed=seed,
        )
        if mode == "quench":
            req.thermal_program.points = [
                ThermalPointV3(time_s=0.0, temperature_c=20.0),
                ThermalPointV3(time_s=260.0, temperature_c=840.0),
                ThermalPointV3(time_s=420.0, temperature_c=840.0),
                ThermalPointV3(time_s=500.0, temperature_c=70.0),
            ]
        elif mode == "temper_low":
            req.thermal_program.points = [
                ThermalPointV3(time_s=0.0, temperature_c=20.0),
                ThermalPointV3(time_s=260.0, temperature_c=840.0),
                ThermalPointV3(time_s=420.0, temperature_c=840.0),
                ThermalPointV3(time_s=500.0, temperature_c=70.0),
                ThermalPointV3(time_s=760.0, temperature_c=220.0),
                ThermalPointV3(time_s=1120.0, temperature_c=220.0),
                ThermalPointV3(time_s=1400.0, temperature_c=30.0),
            ]
        elif mode == "temper_medium":
            req.thermal_program.points = [
                ThermalPointV3(time_s=0.0, temperature_c=20.0),
                ThermalPointV3(time_s=260.0, temperature_c=840.0),
                ThermalPointV3(time_s=420.0, temperature_c=840.0),
                ThermalPointV3(time_s=500.0, temperature_c=70.0),
                ThermalPointV3(time_s=760.0, temperature_c=400.0),
                ThermalPointV3(time_s=1120.0, temperature_c=400.0),
                ThermalPointV3(time_s=1400.0, temperature_c=30.0),
            ]
        else:
            req.thermal_program.points = [
                ThermalPointV3(time_s=0.0, temperature_c=20.0),
                ThermalPointV3(time_s=260.0, temperature_c=840.0),
                ThermalPointV3(time_s=420.0, temperature_c=840.0),
                ThermalPointV3(time_s=500.0, temperature_c=70.0),
                ThermalPointV3(time_s=760.0, temperature_c=600.0),
                ThermalPointV3(time_s=1120.0, temperature_c=600.0),
                ThermalPointV3(time_s=1400.0, temperature_c=30.0),
            ]
        req.thermal_program.quench.medium_code = "water_20"
        req.thermal_program.quench.bath_temperature_c = 20.0
        req.thermal_program.quench.sample_temperature_c = 840.0
        return req

    def test_table_fit_for_grades_and_modes(self) -> None:
        pipeline = MetallographyPipelineV3()
        expected_stage = {
            "quench": "martensite",
            "temper_low": "tempered_low",
            "temper_medium": "troostite_temper",
            "temper_high": "sorbite_temper",
        }
        seed = 700
        for c_wt in (0.2, 0.4, 0.6, 0.8):
            for mode in ("quench", "temper_low", "temper_medium", "temper_high"):
                with self.subTest(c_wt=c_wt, mode=mode):
                    out = pipeline.generate(self._build_request(c_wt=c_wt, mode=mode, seed=seed))
                    seed += 1
                    report = dict(out.metadata.get("phase_model_report", {}))
                    self.assertEqual(str(report.get("calibration_mode", "")), "table_interpolated")
                    self.assertEqual(str(report.get("calibration_profile", "")), "fe_c_tempering_tables_v1")
                    self.assertLessEqual(float(report.get("table_match_error_pct", 999.0)), 2.0)
                    final_stage = str(out.metadata.get("final_stage", ""))
                    if mode == "quench":
                        self.assertIn(final_stage, {"martensite", "martensite_tetragonal", "martensite_cubic"})
                    else:
                        self.assertEqual(final_stage, expected_stage[mode])


if __name__ == "__main__":
    unittest.main()
