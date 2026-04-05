from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.diagram_engine import available_diagram_systems, diagram_snapshot_params, render_diagram_snapshot


class DiagramEngineV2Tests(unittest.TestCase):
    def test_render_each_supported_system(self) -> None:
        compositions = {
            "fe-c": {"Fe": 99.2, "C": 0.8},
            "al-si": {"Al": 88.0, "Si": 12.0},
            "cu-zn": {"Cu": 68.0, "Zn": 32.0},
            "fe-si": {"Fe": 98.6, "Si": 1.4},
            "al-cu-mg": {"Al": 93.1, "Cu": 4.4, "Mg": 1.5},
        }
        for system in available_diagram_systems():
            with self.subTest(system=system):
                snapshot = render_diagram_snapshot(
                    composition=compositions.get(system, {"Fe": 99.0, "C": 1.0}),
                    processing=ProcessingState(temperature_c=700.0, aging_hours=8.0, aging_temperature_c=180.0),
                    requested_system=system,
                    size=(760, 360),
                )
                self.assertEqual(snapshot["image"].size, (760, 360))
                self.assertEqual(snapshot["used_system"], system)

    def test_custom_composition_fallback(self) -> None:
        params = diagram_snapshot_params(
            composition={"Ni": 70.0, "Cr": 30.0},
            processing=ProcessingState(temperature_c=900.0),
            requested_system="custom-multicomponent",
            inferred_system="custom-multicomponent",
        )
        self.assertTrue(params["is_fallback"])
        self.assertIn(params["used_system"], available_diagram_systems())

    def test_al_cu_mg_uses_aging_time_on_x_axis(self) -> None:
        params = diagram_snapshot_params(
            composition={"Al": 93.0, "Cu": 4.3, "Mg": 1.4},
            processing=ProcessingState(temperature_c=25.0, aging_hours=12.5, aging_temperature_c=185.0),
            requested_system="al-cu-mg",
            inferred_system="al-cu-mg",
            confidence=0.9,
        )
        self.assertAlmostEqual(params["current_point"]["x"], 12.5, places=6)
        self.assertAlmostEqual(params["current_point"]["y"], 185.0, places=6)


if __name__ == "__main__":
    unittest.main()
