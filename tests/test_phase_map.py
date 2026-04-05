from __future__ import annotations

import unittest

from core.generator_phase_map import (
    generate_phase_stage_structure,
    resolve_fe_c_stage,
    supported_stages,
)


class PhaseMapTests(unittest.TestCase):
    def test_fe_c_resolver_transitions(self) -> None:
        self.assertEqual(resolve_fe_c_stage(0.4, 1550.0, "equilibrium", "auto"), "liquid")
        self.assertEqual(resolve_fe_c_stage(0.4, 820.0, "equilibrium", "auto"), "austenite")
        self.assertEqual(resolve_fe_c_stage(0.4, 680.0, "equilibrium", "auto"), "alpha_pearlite")
        self.assertEqual(resolve_fe_c_stage(0.8, 120.0, "quenched", "auto"), "martensite")
        self.assertEqual(resolve_fe_c_stage(0.8, 400.0, "tempered", "auto"), "tempered_medium")

    def test_phase_map_generates_for_each_system(self) -> None:
        systems = {
            "fe-c": {"C": 0.6, "Fe": 99.4},
            "al-si": {"Al": 88.0, "Si": 12.0},
            "cu-zn": {"Cu": 68.0, "Zn": 32.0},
            "al-cu-mg": {"Al": 93.0, "Cu": 4.4, "Mg": 1.5},
            "fe-si": {"Fe": 98.6, "Si": 1.4},
        }
        for system, composition in systems.items():
            with self.subTest(system=system):
                result = generate_phase_stage_structure(
                    size=(192, 192),
                    seed=123,
                    system=system,
                    composition=composition,
                    stage="auto",
                    temperature_c=700.0,
                    cooling_mode="equilibrium",
                )
                self.assertEqual(result["image"].shape, (192, 192))
                stage = result["metadata"]["resolved_stage"]
                self.assertIn(stage, supported_stages(system))

    def test_fractional_transition_has_masks_and_fractions(self) -> None:
        result = generate_phase_stage_structure(
            size=(192, 192),
            seed=321,
            system="fe-c",
            composition={"Fe": 99.2, "C": 0.8},
            stage="auto",
            temperature_c=1460.0,
            cooling_mode="equilibrium",
            thermal_slope=-18.0,
        )
        self.assertEqual(result["metadata"]["resolved_stage"], "liquid_gamma")
        state = result["metadata"].get("phase_transition_state", {})
        self.assertEqual(state.get("transition_kind"), "crystallization")
        self.assertGreater(float(state.get("liquid_fraction", 0.0)), 0.0)
        self.assertLess(float(state.get("liquid_fraction", 1.0)), 1.0)

        masks = result.get("phase_masks")
        self.assertIsInstance(masks, dict)
        assert isinstance(masks, dict)
        self.assertIn("L", masks)
        self.assertIn("solid", masks)

        fractions = result["metadata"].get("phase_fractions", {})
        self.assertIn("L", fractions)
        self.assertIn("solid", fractions)


if __name__ == "__main__":
    unittest.main()
