from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.metallography_pro.transformation_fe_c import build_continuous_transformation_state


class ProSchedulerProgressTests(unittest.TestCase):
    def test_progress_metrics_are_present_and_bounded(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.2, "C": 0.8},
            stage="pearlite",
            phase_fractions={"PEARLITE": 0.8, "FERRITE": 0.2},
            processing=ProcessingState(temperature_c=680.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 840.0, "temperature_end_c": 680.0, "hold_time_s": 240.0},
        )
        self.assertGreaterEqual(state.ferrite_pearlite_competition_index, 0.0)
        self.assertLessEqual(state.ferrite_pearlite_competition_index, 1.0)
        self.assertGreaterEqual(state.bainite_activation_progress, 0.0)
        self.assertLessEqual(state.bainite_activation_progress, 1.0)
        self.assertGreaterEqual(state.martensite_conversion_progress, 0.0)
        self.assertLessEqual(state.martensite_conversion_progress, 1.0)
        self.assertGreaterEqual(state.ferrite_progress, 0.0)
        self.assertLessEqual(state.ferrite_progress, 1.0)
        self.assertGreaterEqual(state.pearlite_progress, 0.0)
        self.assertLessEqual(state.pearlite_progress, 1.0)
        self.assertGreaterEqual(state.ferrite_effective_exposure_s, 0.0)
        self.assertGreaterEqual(state.pearlite_effective_exposure_s, 0.0)
        self.assertGreaterEqual(state.bainite_effective_exposure_s, 0.0)
        self.assertGreaterEqual(state.martensite_effective_exposure_s, 0.0)
        meta = state.to_metadata()
        self.assertIn("ferrite_pearlite_competition_index", meta)
        self.assertEqual(meta["ferrite_pearlite_competition_index"], state.ferrite_pearlite_competition_index)
        self.assertNotIn("competition_index", meta)

    def test_family_effective_exposures_follow_temperature_window(self) -> None:
        ferrite_case = build_continuous_transformation_state(
            composition_wt={"Fe": 99.55, "C": 0.45},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.55, "PEARLITE": 0.45},
            processing=ProcessingState(temperature_c=770.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 840.0, "temperature_end_c": 770.0, "hold_time_s": 240.0},
        )
        pearlite_case = build_continuous_transformation_state(
            composition_wt={"Fe": 99.2, "C": 0.8},
            stage="pearlite",
            phase_fractions={"PEARLITE": 0.9, "FERRITE": 0.1},
            processing=ProcessingState(temperature_c=690.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 840.0, "temperature_end_c": 690.0, "hold_time_s": 240.0},
        )
        self.assertGreater(ferrite_case.ferrite_effective_exposure_s, ferrite_case.pearlite_effective_exposure_s)
        self.assertGreater(pearlite_case.pearlite_effective_exposure_s, pearlite_case.ferrite_effective_exposure_s)
        self.assertGreater(ferrite_case.ferrite_progress, pearlite_case.ferrite_progress)
        self.assertGreater(pearlite_case.pearlite_progress, ferrite_case.pearlite_progress)

    def test_bainite_progress_exceeds_pearlite_case_in_lower_c_window(self) -> None:
        pearlite_case = build_continuous_transformation_state(
            composition_wt={"Fe": 99.2, "C": 0.8},
            stage="pearlite",
            phase_fractions={"PEARLITE": 0.8, "FERRITE": 0.2},
            processing=ProcessingState(temperature_c=680.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 840.0, "temperature_end_c": 680.0, "hold_time_s": 240.0},
        )
        bainite_case = build_continuous_transformation_state(
            composition_wt={"Fe": 99.15, "C": 0.85},
            stage="bainite",
            phase_fractions={"BAINITE": 0.78, "CEMENTITE": 0.14, "AUSTENITE": 0.08},
            processing=ProcessingState(temperature_c=320.0, cooling_mode="oil_quench"),
            thermal_summary={"temperature_max_c": 860.0, "temperature_end_c": 320.0, "hold_time_s": 600.0},
            quench_summary={"effect_applied": True},
        )
        self.assertGreater(bainite_case.bainite_activation_progress, pearlite_case.bainite_activation_progress)
        self.assertGreater(bainite_case.time_in_lower_c_window_s, 0.0)
        self.assertGreater(bainite_case.bainite_effective_exposure_s, pearlite_case.bainite_effective_exposure_s)
        self.assertEqual(float(bainite_case.ferrite_effective_exposure_s), 0.0)
        self.assertEqual(float(bainite_case.pearlite_effective_exposure_s), 0.0)
        self.assertEqual(float(bainite_case.martensite_effective_exposure_s), 0.0)
        self.assertEqual(float(bainite_case.ferrite_progress), 0.0)
        self.assertEqual(float(bainite_case.pearlite_progress), 0.0)

    def test_martensite_conversion_progress_responds_to_below_ms_exposure(self) -> None:
        short = build_continuous_transformation_state(
            composition_wt={"Fe": 99.1, "C": 0.9},
            stage="martensite_tetragonal",
            phase_fractions={"MARTENSITE_TETRAGONAL": 0.88, "CEMENTITE": 0.12},
            processing=ProcessingState(temperature_c=120.0, cooling_mode="water_quench"),
            thermal_summary={"temperature_max_c": 860.0, "temperature_end_c": 120.0, "hold_time_s": 10.0},
            quench_summary={"effect_applied": True},
        )
        long = build_continuous_transformation_state(
            composition_wt={"Fe": 99.1, "C": 0.9},
            stage="martensite_tetragonal",
            phase_fractions={"MARTENSITE_TETRAGONAL": 0.88, "CEMENTITE": 0.12},
            processing=ProcessingState(temperature_c=120.0, cooling_mode="water_quench"),
            thermal_summary={"temperature_max_c": 860.0, "temperature_end_c": 120.0, "hold_time_s": 120.0},
            quench_summary={"effect_applied": True},
        )
        self.assertGreaterEqual(long.martensite_conversion_progress, short.martensite_conversion_progress)
        self.assertGreaterEqual(long.martensite_effective_exposure_s, short.martensite_effective_exposure_s)
        self.assertEqual(float(long.bainite_effective_exposure_s), 0.0)
        self.assertEqual(float(long.ferrite_effective_exposure_s), 0.0)
        self.assertEqual(float(long.pearlite_effective_exposure_s), 0.0)

    def test_segmented_thermal_history_populates_multiple_family_exposures(self) -> None:
        segmented = build_continuous_transformation_state(
            composition_wt={"Fe": 99.15, "C": 0.85},
            stage="bainite",
            phase_fractions={"BAINITE": 0.62, "PEARLITE": 0.18, "FERRITE": 0.12, "AUSTENITE": 0.08},
            processing=ProcessingState(temperature_c=320.0, cooling_mode="oil_quench"),
            thermal_summary={
                "temperature_max_c": 860.0,
                "temperature_end_c": 320.0,
                "segments": [
                    {"kind": "heat", "temp0_c": 20.0, "temp1_c": 860.0, "dt_s": 180.0},
                    {"kind": "hold", "temp0_c": 860.0, "temp1_c": 860.0, "dt_s": 180.0},
                    {"kind": "cool", "temp0_c": 860.0, "temp1_c": 760.0, "dt_s": 120.0},
                    {"kind": "hold", "temp0_c": 760.0, "temp1_c": 760.0, "dt_s": 120.0},
                    {"kind": "cool", "temp0_c": 760.0, "temp1_c": 680.0, "dt_s": 60.0},
                    {"kind": "hold", "temp0_c": 680.0, "temp1_c": 680.0, "dt_s": 180.0},
                    {"kind": "cool", "temp0_c": 680.0, "temp1_c": 320.0, "dt_s": 90.0},
                    {"kind": "hold", "temp0_c": 320.0, "temp1_c": 320.0, "dt_s": 420.0},
                ],
            },
            quench_summary={"effect_applied": True},
        )
        self.assertGreater(segmented.time_in_upper_c_window_s, 0.0)
        self.assertGreater(segmented.bainite_effective_exposure_s, 0.0)
        self.assertGreater(segmented.pearlite_effective_exposure_s, 0.0)
        self.assertGreater(segmented.bainite_activation_progress, 0.2)
        self.assertAlmostEqual(sum(segmented.family_weights.values()), 1.0, places=6)
        self.assertGreater(segmented.family_weights["bainitic_family"], segmented.family_weights["pearlitic_family"])


if __name__ == "__main__":
    unittest.main()
