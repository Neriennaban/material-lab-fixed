from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.metallography_pro.morphology_fe_c import build_spatial_morphology_state
from core.metallography_pro.transformation_fe_c import build_continuous_transformation_state


class ProTransformationAndMorphologyTests(unittest.TestCase):
    def test_faster_cooling_refines_spacing_in_continuous_state(self) -> None:
        slow = build_continuous_transformation_state(
            composition_wt={"Fe": 99.2, "C": 0.8},
            stage="pearlite",
            phase_fractions={"PEARLITE": 0.8, "FERRITE": 0.2},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 840.0, "hold_time_s": 180.0},
        )
        fast = build_continuous_transformation_state(
            composition_wt={"Fe": 99.2, "C": 0.8},
            stage="pearlite",
            phase_fractions={"PEARLITE": 0.8, "FERRITE": 0.2},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="normalized"),
            thermal_summary={"temperature_max_c": 840.0, "hold_time_s": 180.0},
        )
        self.assertGreater(slow.interlamellar_spacing_um_mean, fast.interlamellar_spacing_um_mean)

    def test_progress_metrics_track_path_not_only_final_temperature(self) -> None:
        short_hold = build_continuous_transformation_state(
            composition_wt={"Fe": 99.15, "C": 0.85},
            stage="bainite",
            phase_fractions={"BAINITE": 0.78, "CEMENTITE": 0.14, "AUSTENITE": 0.08},
            processing=ProcessingState(temperature_c=320.0, cooling_mode="oil_quench"),
            thermal_summary={"temperature_max_c": 860.0, "temperature_end_c": 320.0, "hold_time_s": 60.0},
            quench_summary={"effect_applied": True},
        )
        long_hold = build_continuous_transformation_state(
            composition_wt={"Fe": 99.15, "C": 0.85},
            stage="bainite",
            phase_fractions={"BAINITE": 0.78, "CEMENTITE": 0.14, "AUSTENITE": 0.08},
            processing=ProcessingState(temperature_c=320.0, cooling_mode="oil_quench"),
            thermal_summary={"temperature_max_c": 860.0, "temperature_end_c": 320.0, "hold_time_s": 600.0},
            quench_summary={"effect_applied": True},
        )
        self.assertGreater(long_hold.bainite_activation_progress, short_hold.bainite_activation_progress)
        self.assertGreaterEqual(long_hold.time_in_lower_c_window_s, short_hold.time_in_lower_c_window_s)

        mart_short = build_continuous_transformation_state(
            composition_wt={"Fe": 99.1, "C": 0.9},
            stage="martensite_tetragonal",
            phase_fractions={"MARTENSITE_TETRAGONAL": 0.88, "CEMENTITE": 0.12},
            processing=ProcessingState(temperature_c=120.0, cooling_mode="water_quench"),
            thermal_summary={"temperature_max_c": 860.0, "temperature_end_c": 120.0, "hold_time_s": 10.0},
            quench_summary={"effect_applied": True},
        )
        mart_long = build_continuous_transformation_state(
            composition_wt={"Fe": 99.1, "C": 0.9},
            stage="martensite_tetragonal",
            phase_fractions={"MARTENSITE_TETRAGONAL": 0.88, "CEMENTITE": 0.12},
            processing=ProcessingState(temperature_c=120.0, cooling_mode="water_quench"),
            thermal_summary={"temperature_max_c": 860.0, "temperature_end_c": 120.0, "hold_time_s": 120.0},
            quench_summary={"effect_applied": True},
        )
        self.assertGreater(mart_long.martensite_conversion_progress, mart_short.martensite_conversion_progress)

    def test_pearlitic_morphology_emits_lamellae_and_consistent_masks(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.45, "C": 0.55},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 830.0, "hold_time_s": 120.0},
        )
        morph = build_spatial_morphology_state(
            size=(128, 128),
            seed=101,
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        self.assertEqual(morph.phase_label_map.shape, (128, 128))
        self.assertIsNotNone(morph.lamella_field)
        self.assertIn("PEARLITE", morph.phase_masks)
        self.assertIn("FERRITE", morph.phase_masks)
        total = sum(float((mask > 0).mean()) for mask in morph.phase_masks.values())
        self.assertAlmostEqual(total, 1.0, places=3)
        self.assertIn("phase_boundaries", morph.feature_maps)
        self.assertGreater(int(morph.summary.get("prior_austenite_grain_count", 0)), 0)
        self.assertEqual(str(state.pearlite_morphology_family), "lamellar_colonies")

    def test_martensitic_morphology_emits_packet_field(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.1, "C": 0.9},
            stage="martensite_tetragonal",
            phase_fractions={"MARTENSITE_TETRAGONAL": 0.88, "CEMENTITE": 0.12},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="water_quench"),
            thermal_summary={"temperature_max_c": 860.0, "hold_time_s": 180.0},
            quench_summary={"effect_applied": True},
        )
        morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=102,
            stage="martensite_tetragonal",
            phase_fractions={"MARTENSITE_TETRAGONAL": 0.88, "CEMENTITE": 0.12},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        self.assertIsNotNone(morph.packet_field)
        self.assertIsNotNone(morph.packet_id_map)
        self.assertIn("martensite_laths_binary", morph.feature_maps)
        self.assertIn(str(state.martensite_morphology_family), {"mixed_lath_plate", "plate_dominant"})

    def test_ferrite_family_switches_with_cooling_context(self) -> None:
        slow = build_continuous_transformation_state(
            composition_wt={"Fe": 99.45, "C": 0.55},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 830.0, "temperature_end_c": 20.0, "hold_time_s": 120.0},
        )
        fast = build_continuous_transformation_state(
            composition_wt={"Fe": 99.45, "C": 0.55},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="normalized"),
            thermal_summary={"temperature_max_c": 830.0, "temperature_end_c": 20.0, "hold_time_s": 120.0},
        )
        slow_morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=120,
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            transformation_state=slow,
            native_um_per_px=0.5,
        )
        fast_morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=121,
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            transformation_state=fast,
            native_um_per_px=0.5,
        )
        self.assertEqual(str(slow.ferrite_morphology_family), "allotriomorphic")
        self.assertIn(str(fast.ferrite_morphology_family), {"allotriomorphic", "widmanstatten"})
        self.assertIn("allotriomorphic_ferrite_binary", slow_morph.feature_maps)
        self.assertNotIn("widmanstatten_sideplates_binary", slow_morph.feature_maps)
        if str(fast.ferrite_morphology_family) == "widmanstatten":
            self.assertIn("widmanstatten_sideplates_binary", fast_morph.feature_maps)
        else:
            self.assertIn("allotriomorphic_ferrite_binary", fast_morph.feature_maps)

    def test_upper_c_window_competition_changes_family_bias(self) -> None:
        ferrite_biased = build_continuous_transformation_state(
            composition_wt={"Fe": 99.75, "C": 0.25},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.72, "PEARLITE": 0.28},
            processing=ProcessingState(temperature_c=770.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 800.0, "temperature_end_c": 770.0, "hold_time_s": 240.0},
        )
        pearlite_biased = build_continuous_transformation_state(
            composition_wt={"Fe": 99.35, "C": 0.65},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.30, "PEARLITE": 0.70},
            processing=ProcessingState(temperature_c=680.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 800.0, "temperature_end_c": 680.0, "hold_time_s": 600.0},
        )
        self.assertLess(
            ferrite_biased.ferrite_pearlite_competition_index,
            pearlite_biased.ferrite_pearlite_competition_index,
        )
        self.assertLessEqual(ferrite_biased.ferrite_pearlite_competition_index, 0.42)
        self.assertGreaterEqual(pearlite_biased.ferrite_pearlite_competition_index, 0.58)
        self.assertEqual(str(ferrite_biased.transformation_family), "ferritic_family")
        self.assertEqual(str(pearlite_biased.transformation_family), "pearlitic_family")

    def test_bainite_family_emits_sheaf_features(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.15, "C": 0.85},
            stage="bainite",
            phase_fractions={"BAINITE": 0.78, "CEMENTITE": 0.14, "AUSTENITE": 0.08},
            processing=ProcessingState(temperature_c=320.0, cooling_mode="oil_quench"),
            thermal_summary={"temperature_max_c": 860.0, "temperature_end_c": 320.0, "hold_time_s": 600.0},
            quench_summary={"effect_applied": True},
        )
        morph = build_spatial_morphology_state(
            size=(128, 128),
            seed=103,
            stage="bainite",
            phase_fractions={"BAINITE": 0.78, "CEMENTITE": 0.14, "AUSTENITE": 0.08},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        self.assertEqual(str(state.transformation_family), "bainitic_family")
        self.assertIn(str(state.bainite_morphology_family), {"upper_bainite_sheaves", "lower_bainite_sheaves"})
        self.assertGreater(float(state.bainite_sheaf_length_um), 0.0)
        self.assertGreater(float(state.time_in_lower_c_window_s), 0.0)
        self.assertGreater(float(state.time_in_bainite_hold_s), 0.0)
        self.assertIn("bainite_sheaves_binary", morph.feature_maps)
        if str(state.bainite_morphology_family) == "upper_bainite_sheaves":
            self.assertIn("upper_bainite_sheaves_binary", morph.feature_maps)
        else:
            self.assertIn("lower_bainite_sheaves_binary", morph.feature_maps)
        self.assertGreater(float(morph.summary.get("bainite_spacing_px", 0.0)), 0.0)

    def test_metadata_provenance_reflects_bhadeshia_family_decisions(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.15, "C": 0.85},
            stage="troostite_quench",
            phase_fractions={"BAINITE": 0.64, "MARTENSITE": 0.18, "CEMENTITE": 0.10, "AUSTENITE": 0.08},
            processing=ProcessingState(temperature_c=340.0, cooling_mode="oil_quench"),
            thermal_summary={"temperature_max_c": 860.0, "temperature_end_c": 340.0, "hold_time_s": 420.0},
            quench_summary={"effect_applied": True},
        )
        self.assertIn("bainite_split", state.provenance)
        self.assertIn("ferrite_split", state.provenance)
        self.assertIn("derived_labels", state.provenance)

    def test_upper_and_lower_bainite_spacing_split_is_materialized(self) -> None:
        upper_state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.15, "C": 0.85},
            stage="bainite",
            phase_fractions={"BAINITE": 0.74, "CEMENTITE": 0.16, "AUSTENITE": 0.10},
            processing=ProcessingState(temperature_c=380.0, cooling_mode="oil_quench"),
            thermal_summary={"temperature_max_c": 860.0, "temperature_end_c": 380.0, "hold_time_s": 480.0},
            quench_summary={"effect_applied": True},
        )
        lower_state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.15, "C": 0.85},
            stage="bainite",
            phase_fractions={"BAINITE": 0.74, "CEMENTITE": 0.16, "AUSTENITE": 0.10},
            processing=ProcessingState(temperature_c=300.0, cooling_mode="oil_quench"),
            thermal_summary={"temperature_max_c": 860.0, "temperature_end_c": 300.0, "hold_time_s": 480.0},
            quench_summary={"effect_applied": True},
        )
        upper_morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=130,
            stage="bainite",
            phase_fractions={"BAINITE": 0.74, "CEMENTITE": 0.16, "AUSTENITE": 0.10},
            transformation_state=upper_state,
            native_um_per_px=0.5,
        )
        lower_morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=131,
            stage="bainite",
            phase_fractions={"BAINITE": 0.74, "CEMENTITE": 0.16, "AUSTENITE": 0.10},
            transformation_state=lower_state,
            native_um_per_px=0.5,
        )
        self.assertEqual(str(upper_state.bainite_morphology_family), "upper_bainite_sheaves")
        self.assertEqual(str(lower_state.bainite_morphology_family), "lower_bainite_sheaves")
        self.assertLess(float(lower_morph.summary["bainite_spacing_px"]), float(upper_morph.summary["bainite_spacing_px"]))


if __name__ == "__main__":
    unittest.main()
