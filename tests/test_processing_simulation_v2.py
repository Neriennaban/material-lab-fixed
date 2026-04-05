from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessRoute, ProcessingOperation, ProcessingState
from core.processing_simulation import simulate_process_route


class ProcessingSimulationV2Tests(unittest.TestCase):
    def _fe_c_route(self) -> ProcessRoute:
        return ProcessRoute(
            operations=[
                ProcessingOperation(method="quench_water", temperature_c=860.0, duration_min=30.0, cooling_mode="quenched"),
                ProcessingOperation(method="temper_medium", temperature_c=400.0, duration_min=90.0, cooling_mode="tempered"),
            ],
            route_name="fe_c_qt",
            step_preview_enabled=True,
        )

    def test_deterministic_for_same_seed_and_route(self) -> None:
        route = self._fe_c_route()
        kwargs = dict(
            composition={"Fe": 99.2, "C": 0.8},
            inferred_system="fe-c",
            route=route,
            initial_processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            generator="phase_map",
            base_seed=1234,
            step_preview_index=None,
        )
        a = simulate_process_route(**kwargs)
        b = simulate_process_route(**kwargs)
        self.assertEqual(a.final_stage, b.final_stage)
        self.assertEqual(a.final_effect_vector, b.final_effect_vector)
        self.assertEqual(a.route_timeline, b.route_timeline)

    def test_resolved_stage_and_properties_present(self) -> None:
        out = simulate_process_route(
            composition={"Fe": 99.2, "C": 0.8},
            inferred_system="fe-c",
            route=self._fe_c_route(),
            initial_processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            generator="phase_map",
            base_seed=55,
            step_preview_index=None,
        )
        self.assertEqual(out.final_stage, "tempered_medium")
        self.assertIn("hv_estimate", out.property_indicators)
        self.assertIn("uts_estimate_mpa", out.property_indicators)
        self.assertGreater(float(out.property_indicators["hv_estimate"]), 0.0)
        self.assertGreater(float(out.property_indicators["uts_estimate_mpa"]), 0.0)
        self.assertEqual(out.property_indicators["property_model_source"], "hybrid_textbook_calculator_v1")
        self.assertEqual(out.property_indicators["reference_dataset"], "textbook_material_properties")
        self.assertIn("compatibility_overlay_used", out.property_indicators)
        self.assertFalse(bool(out.property_indicators["fallback_used"]))

    def test_step_preview_uses_selected_step_state(self) -> None:
        route = self._fe_c_route()
        step0 = simulate_process_route(
            composition={"Fe": 99.2, "C": 0.8},
            inferred_system="fe-c",
            route=route,
            initial_processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            generator="phase_map",
            base_seed=777,
            step_preview_index=0,
        )
        final = simulate_process_route(
            composition={"Fe": 99.2, "C": 0.8},
            inferred_system="fe-c",
            route=route,
            initial_processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            generator="phase_map",
            base_seed=777,
            step_preview_index=None,
        )
        self.assertEqual(step0.final_stage, "martensite")
        self.assertEqual(final.final_stage, "tempered_medium")
        self.assertNotEqual(step0.final_effect_vector, final.final_effect_vector)

    def test_al_cu_mg_solution_age_route(self) -> None:
        route = ProcessRoute(
            operations=[
                ProcessingOperation(method="solution_treat", temperature_c=505.0, duration_min=60.0, cooling_mode="solutionized"),
                ProcessingOperation(method="quench_water", temperature_c=25.0, duration_min=5.0, cooling_mode="quenched"),
                ProcessingOperation(
                    method="age_artificial",
                    temperature_c=185.0,
                    duration_min=480.0,
                    cooling_mode="aged",
                    aging_hours=8.0,
                    aging_temperature_c=185.0,
                ),
            ],
            route_name="al_age",
        )
        out = simulate_process_route(
            composition={"Al": 93.1, "Cu": 4.4, "Mg": 1.5},
            inferred_system="al-cu-mg",
            route=route,
            initial_processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            generator="phase_map",
            base_seed=222,
            step_preview_index=None,
        )
        self.assertEqual(out.final_stage, "artificial_aged")
        self.assertTrue(out.route_validation.is_valid)
        self.assertEqual(out.property_indicators["property_model_source"], "legacy_properties_rules")
        self.assertEqual(out.property_indicators["reference_dataset"], "properties_rules.json")
        self.assertFalse(bool(out.property_indicators["fallback_used"]))


if __name__ == "__main__":
    unittest.main()
