from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.contracts_v3 import PhaseModelConfigV3
from core.metallography_v3.phase_orchestrator import (
    blend_phase_fractions,
    build_phase_bundle,
    estimate_auto_phase_fractions,
    infer_training_system,
    resolve_stage,
)


class PhaseOrchestratorV3Tests(unittest.TestCase):
    def test_infer_known_and_custom_system(self) -> None:
        known, conf, fallback = infer_training_system({"Fe": 99.2, "C": 0.8}, None)
        self.assertEqual(known, "fe-c")
        self.assertGreater(conf, 0.5)
        self.assertFalse(fallback)

        custom, conf_custom, fallback_custom = infer_training_system({"Ni": 60.0, "Cr": 40.0}, None)
        self.assertEqual(custom, "custom-multicomponent")
        self.assertTrue(fallback_custom)
        self.assertGreaterEqual(conf_custom, 0.2)

    def test_stage_resolution_and_auto_fraction_sum(self) -> None:
        proc = ProcessingState(temperature_c=650.0, cooling_mode="slow_cool")
        stage = resolve_stage("fe-c", {"Fe": 99.6, "C": 0.4}, proc)
        self.assertIn(stage, {"alpha_pearlite", "pearlite", "ferrite", "alpha_gamma"})

        phases = estimate_auto_phase_fractions("fe-c", stage, {"Fe": 99.6, "C": 0.4}, proc)
        self.assertTrue(phases)
        total = sum(float(v) for v in phases.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_eutectoid_pearlite_is_near_full(self) -> None:
        proc = ProcessingState(temperature_c=690.0, cooling_mode="slow_cool")
        phases = estimate_auto_phase_fractions("fe-c", "pearlite", {"Fe": 99.23, "C": 0.77}, proc)
        self.assertGreaterEqual(float(phases.get("PEARLITE", 0.0)), 0.98)
        self.assertLessEqual(float(phases.get("FERRITE", 0.0)), 0.02)
        self.assertAlmostEqual(float(phases.get("CEMENTITE", 0.0)), 0.0, delta=1e-6)

    def test_fe_c_table_alignment_for_05_and_77(self) -> None:
        proc = ProcessingState(temperature_c=680.0, cooling_mode="slow_cool")

        steel_05 = estimate_auto_phase_fractions("fe-c", "alpha_pearlite", {"Fe": 99.95, "C": 0.05}, proc)
        self.assertAlmostEqual(float(steel_05.get("FERRITE", 0.0)), 0.96, delta=0.01)
        self.assertAlmostEqual(float(steel_05.get("PEARLITE", 0.0)), 0.04, delta=0.01)
        self.assertAlmostEqual(float(steel_05.get("CEMENTITE", 0.0)), 0.0, delta=1e-6)

        steel_77 = estimate_auto_phase_fractions("fe-c", "pearlite", {"Fe": 99.23, "C": 0.77}, proc)
        self.assertAlmostEqual(float(steel_77.get("PEARLITE", 0.0)), 1.0, delta=1e-6)
        self.assertAlmostEqual(float(steel_77.get("CEMENTITE", 0.0)), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(steel_77.get("FERRITE", 0.0)), 0.0, delta=1e-6)

    def test_blend_modes(self) -> None:
        auto = {"FERRITE": 0.7, "PEARLITE": 0.3}
        manual = {"FERRITE": 0.2, "PEARLITE": 0.8}

        auto_only = blend_phase_fractions(auto, manual, "auto_only", 0.35)
        self.assertAlmostEqual(auto_only["FERRITE"], 0.7, places=4)
        self.assertAlmostEqual(auto_only["PEARLITE"], 0.3, places=4)

        manual_only = blend_phase_fractions(auto, manual, "manual_only", 0.35)
        self.assertAlmostEqual(manual_only["FERRITE"], 0.2, places=4)
        self.assertAlmostEqual(manual_only["PEARLITE"], 0.8, places=4)

        mixed = blend_phase_fractions(auto, manual, "auto_with_override", 0.5)
        self.assertAlmostEqual(mixed["FERRITE"], 0.45, places=4)
        self.assertAlmostEqual(mixed["PEARLITE"], 0.55, places=4)

    def test_build_phase_bundle_contains_report(self) -> None:
        model = PhaseModelConfigV3(
            phase_control_mode="auto_with_override",
            manual_phase_fractions={"FERRITE": 0.1, "PEARLITE": 0.9},
            manual_override_weight=0.4,
            allow_custom_fallback=True,
        )
        bundle = build_phase_bundle(
            composition={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=640.0, cooling_mode="slow_cool"),
            system_hint="fe-c",
            phase_model=model,
        )
        self.assertEqual(bundle.system, "fe-c")
        self.assertIsInstance(bundle.phase_fractions, dict)
        self.assertTrue(bundle.phase_fractions)
        report = bundle.phase_model_report
        self.assertIn("auto_phase_fractions", report)
        self.assertIn("manual_phase_fractions", report)
        self.assertIn("blended_phase_fractions", report)
        self.assertIn("fallback_used", report)
        self.assertIn("true_phase_fractions_after_blend", report)
        self.assertIn("pearlite_internal_true_phases", report)

    def test_steel_equilibrium_report_for_hypereutectoid_range(self) -> None:
        bundle = build_phase_bundle(
            composition={"Fe": 98.0, "C": 2.0},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            system_hint="fe-c",
            phase_model=PhaseModelConfigV3(),
        )
        self.assertEqual(bundle.stage, "pearlite_cementite")
        report = bundle.phase_model_report
        micro = dict(report.get("microconstituent_fractions_auto", {}))
        true_phases = dict(report.get("true_phase_fractions_auto", {}))
        self.assertGreater(float(micro.get("PEARLITE", 0.0)), 0.7)
        self.assertGreater(float(micro.get("CEMENTITE", 0.0)), 0.18)
        self.assertGreater(float(true_phases.get("CEMENTITE", 0.0)), 0.25)
        self.assertLess(float(true_phases.get("FERRITE", 0.0)), 0.75)

    def test_fe_c_temper_stage_uses_curve_inference(self) -> None:
        bundle = build_phase_bundle(
            composition={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=35.0, cooling_mode="quenched"),
            system_hint="fe-c",
            phase_model=PhaseModelConfigV3(),
            thermal_summary={
                "temperature_max_c": 840.0,
                "observed_temperature_c": 35.0,
                "max_effective_cooling_rate_c_per_s": 28.0,
                "operation_inference": {
                    "has_quench": True,
                    "has_temper": True,
                    "temper_peak_temperature_c": 410.0,
                    "temper_total_hold_s": 420.0,
                },
            },
            quench_summary={"medium_code": "water_20", "severity_effective": 0.82},
        )
        self.assertEqual(bundle.stage, "troostite_temper")

    def test_fe_c_grade_80_table_constituents_for_four_thermal_modes(self) -> None:
        proc = ProcessingState(temperature_c=35.0, cooling_mode="quenched")
        base_thermal = {
            "temperature_max_c": 840.0,
            "observed_temperature_c": 35.0,
            "max_effective_cooling_rate_c_per_s": 26.0,
        }
        qsum = {"medium_code_resolved": "water_20", "effect_applied": True}

        quench = estimate_auto_phase_fractions(
            "fe-c",
            "martensite",
            {"Fe": 99.2, "C": 0.8},
            proc,
            thermal_summary={**base_thermal, "operation_inference": {"has_quench": True, "has_temper": False}},
            quench_summary=qsum,
        )
        self.assertAlmostEqual(float(quench.get("MARTENSITE", 0.0)), 0.84, delta=0.02)
        self.assertAlmostEqual(float(quench.get("AUSTENITE", 0.0)), 0.15, delta=0.02)
        self.assertAlmostEqual(float(quench.get("CEMENTITE", 0.0)), 0.01, delta=0.01)

        low = estimate_auto_phase_fractions(
            "fe-c",
            "tempered_low",
            {"Fe": 99.2, "C": 0.8},
            proc,
            thermal_summary={**base_thermal, "operation_inference": {"has_quench": True, "has_temper": True}},
            quench_summary=qsum,
        )
        self.assertAlmostEqual(float(low.get("MARTENSITE", 0.0)), 0.84, delta=0.02)
        self.assertAlmostEqual(float(low.get("AUSTENITE", 0.0)), 0.15, delta=0.02)

        medium = estimate_auto_phase_fractions(
            "fe-c",
            "troostite_temper",
            {"Fe": 99.2, "C": 0.8},
            proc,
            thermal_summary={**base_thermal, "operation_inference": {"has_quench": True, "has_temper": True}},
            quench_summary=qsum,
        )
        self.assertAlmostEqual(float(medium.get("TROOSTITE", 0.0)), 0.945, delta=0.02)
        self.assertAlmostEqual(float(medium.get("AUSTENITE", 0.0)), 0.045, delta=0.02)
        self.assertAlmostEqual(float(medium.get("CEMENTITE", 0.0)), 0.01, delta=0.01)

        high = estimate_auto_phase_fractions(
            "fe-c",
            "sorbite_temper",
            {"Fe": 99.2, "C": 0.8},
            proc,
            thermal_summary={**base_thermal, "operation_inference": {"has_quench": True, "has_temper": True}},
            quench_summary=qsum,
        )
        self.assertAlmostEqual(float(high.get("SORBITE", 0.0)), 0.99, delta=0.02)
        self.assertAlmostEqual(float(high.get("CEMENTITE", 0.0)), 0.01, delta=0.01)


if __name__ == "__main__":
    unittest.main()
