from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3QuenchMediaProfilesTests(unittest.TestCase):
    def test_profiles_exported_in_metadata(self) -> None:
        pipeline = MetallographyPipelineV3()
        media_codes = ["water_20", "water_100", "brine_20_30", "oil_20_80"]
        for medium in media_codes:
            with self.subTest(medium=medium):
                request = MetallographyRequestV3(
                    sample_id=f"q_media_{medium}",
                    composition_wt={"Fe": 99.2, "C": 0.8},
                    system_hint="fe-c",
                    resolution=(96, 96),
                    seed=420,
                )
                request.thermal_program.points = [
                    ThermalPointV3(time_s=0.0, temperature_c=20.0),
                    ThermalPointV3(time_s=280.0, temperature_c=840.0),
                    ThermalPointV3(time_s=420.0, temperature_c=840.0),
                    ThermalPointV3(time_s=520.0, temperature_c=60.0),
                ]
                request.thermal_program.quench.medium_code = medium
                if medium == "water_100":
                    request.thermal_program.quench.bath_temperature_c = 100.0
                elif medium == "oil_20_80":
                    request.thermal_program.quench.bath_temperature_c = 60.0
                elif medium == "brine_20_30":
                    request.thermal_program.quench.bath_temperature_c = 25.0
                else:
                    request.thermal_program.quench.bath_temperature_c = 20.0

                out = pipeline.generate(request)
                profile = dict(out.metadata.get("quench_medium_profile", {}))
                temper_adj = dict(out.metadata.get("temper_adjustment", {}))
                as_quenched = dict(out.metadata.get("as_quenched_prediction", {}))
                thermal = dict(out.metadata.get("thermal_program_summary", {}))
                quench = dict(out.metadata.get("quench_summary", {}))
                self.assertEqual(str(profile.get("medium_code_resolved", "")), medium)
                self.assertIn("cooling_rate_band_800_400", profile)
                self.assertIn("hardness_hrc_as_quenched_range", profile)
                self.assertIn("stress_mpa_range", profile)
                self.assertIn("shift_c", temper_adj)
                self.assertIn("retained_austenite_fraction_est", as_quenched)
                self.assertFalse(bool(thermal.get("quench_effect_applied", True)))
                self.assertFalse(bool(quench.get("effect_applied", True)))
                self.assertEqual(str(thermal.get("quench_effect_reason", "")), "no_quench_segment")

    def test_high_temper_converges_to_sorbite_temper(self) -> None:
        pipeline = MetallographyPipelineV3()
        for medium, bath_t in (
            ("water_20", 20.0),
            ("water_100", 100.0),
            ("brine_20_30", 25.0),
            ("oil_20_80", 60.0),
        ):
            with self.subTest(medium=medium):
                request = MetallographyRequestV3(
                    sample_id=f"high_temper_{medium}",
                    composition_wt={"Fe": 99.2, "C": 0.8},
                    system_hint="fe-c",
                    resolution=(96, 96),
                    seed=421,
                )
                request.thermal_program.points = [
                    ThermalPointV3(time_s=0.0, temperature_c=20.0),
                    ThermalPointV3(time_s=300.0, temperature_c=840.0),
                    ThermalPointV3(time_s=480.0, temperature_c=840.0),
                    ThermalPointV3(time_s=560.0, temperature_c=90.0),
                    ThermalPointV3(time_s=900.0, temperature_c=600.0),
                    ThermalPointV3(time_s=1320.0, temperature_c=600.0),
                    ThermalPointV3(time_s=1650.0, temperature_c=35.0),
                ]
                request.thermal_program.quench.medium_code = medium
                request.thermal_program.quench.bath_temperature_c = bath_t
                request.thermal_program.quench.sample_temperature_c = 840.0
                out = pipeline.generate(request)
                self.assertEqual(str(out.metadata.get("final_stage", "")), "sorbite_temper")
                thermal = dict(out.metadata.get("thermal_program_summary", {}))
                quench = dict(out.metadata.get("quench_summary", {}))
                self.assertTrue(bool(thermal.get("quench_effect_applied", False)))
                self.assertTrue(bool(quench.get("effect_applied", False)))
                self.assertEqual(str(thermal.get("quench_effect_reason", "")), "curve_quench_detected")

    def test_medium_effect_requires_quench_segment(self) -> None:
        pipeline = MetallographyPipelineV3()
        request = MetallographyRequestV3(
            sample_id="medium_effect_requires_quench",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            resolution=(96, 96),
            seed=432,
        )
        # No quench segment
        request.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=260.0, temperature_c=840.0),
            ThermalPointV3(time_s=500.0, temperature_c=840.0),
            ThermalPointV3(time_s=900.0, temperature_c=700.0),
        ]
        request.thermal_program.quench.medium_code = "water_20"
        request.thermal_program.quench.bath_temperature_c = 20.0
        out_no = pipeline.generate(request)
        self.assertFalse(bool(dict(out_no.metadata.get("quench_summary", {})).get("effect_applied", True)))

        # Explicit quench segment
        request.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=260.0, temperature_c=840.0),
            ThermalPointV3(time_s=500.0, temperature_c=840.0),
            ThermalPointV3(time_s=580.0, temperature_c=60.0),
        ]
        out_yes = pipeline.generate(request)
        self.assertTrue(bool(dict(out_yes.metadata.get("quench_summary", {})).get("effect_applied", False)))


if __name__ == "__main__":
    unittest.main()
