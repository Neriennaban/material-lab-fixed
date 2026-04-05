from __future__ import annotations

import unittest

from core.generator_phase_map import resolve_phase_transition_state


class PhaseTransitionStateV2Tests(unittest.TestCase):
    def test_fe_c_crystallization_and_melting(self) -> None:
        base_processing = {"temperature_c": 1460.0, "cooling_mode": "equilibrium", "deformation_pct": 0.0}
        cool_state = resolve_phase_transition_state(
            system="fe-c",
            composition={"Fe": 99.2, "C": 0.8},
            processing=base_processing,
            thermal_slope=-12.0,
            requested_stage="auto",
        )
        self.assertEqual(cool_state["stage"], "liquid_gamma")
        self.assertEqual(cool_state["transition_kind"], "crystallization")
        self.assertGreater(float(cool_state["liquid_fraction"]), 0.0)
        self.assertLess(float(cool_state["liquid_fraction"]), 1.0)

        heat_state = resolve_phase_transition_state(
            system="fe-c",
            composition={"Fe": 99.2, "C": 0.8},
            processing=base_processing,
            thermal_slope=12.0,
            requested_stage="auto",
        )
        self.assertEqual(heat_state["transition_kind"], "melting")

    def test_supported_systems_fractional_ranges(self) -> None:
        cases = [
            ("al-si", {"Al": 88.0, "Si": 12.0}, {"temperature_c": 579.0, "cooling_mode": "equilibrium"}, "liquid_alpha"),
            ("cu-zn", {"Cu": 60.0, "Zn": 40.0}, {"temperature_c": 900.0, "cooling_mode": "equilibrium"}, "liquid_alpha"),
            ("fe-si", {"Fe": 98.6, "Si": 1.4}, {"temperature_c": 1490.0, "cooling_mode": "equilibrium"}, "liquid_ferrite"),
        ]
        for system, composition, processing, expected_stage in cases:
            with self.subTest(system=system):
                state = resolve_phase_transition_state(
                    system=system,
                    composition=composition,
                    processing=processing,
                    thermal_slope=-8.0,
                    requested_stage="auto",
                )
                self.assertEqual(state["stage"], expected_stage)
                self.assertGreater(float(state["liquid_fraction"]), 0.0)
                self.assertLess(float(state["liquid_fraction"]), 1.0)
                self.assertEqual(state["transition_kind"], "crystallization")


if __name__ == "__main__":
    unittest.main()
