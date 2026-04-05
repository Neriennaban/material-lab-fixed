from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessRoute, ProcessingOperation, ProcessingState
from core.route_validation import validate_process_route


class RouteValidationV2Tests(unittest.TestCase):
    def test_fe_c_quench_temper_valid(self) -> None:
        route = ProcessRoute(
            operations=[
                ProcessingOperation(method="quench_water", temperature_c=860.0, duration_min=30.0, cooling_mode="quenched"),
                ProcessingOperation(method="temper_medium", temperature_c=400.0, duration_min=90.0, cooling_mode="tempered"),
            ],
            route_name="qt",
        )
        rep = validate_process_route(route=route, inferred_system="fe-c", processing_context=ProcessingState())
        self.assertTrue(rep.is_valid)
        self.assertEqual(len(rep.errors), 0)

    def test_fe_c_temper_without_quench_invalid(self) -> None:
        route = ProcessRoute(
            operations=[
                ProcessingOperation(method="temper_medium", temperature_c=400.0, duration_min=90.0, cooling_mode="tempered"),
            ]
        )
        rep = validate_process_route(route=route, inferred_system="fe-c", processing_context=ProcessingState())
        self.assertFalse(rep.is_valid)
        self.assertTrue(any("requires one of previous operations" in e for e in rep.errors))

    def test_al_cu_mg_aging_without_solution_invalid(self) -> None:
        route = ProcessRoute(
            operations=[
                ProcessingOperation(method="age_artificial", aging_hours=8.0, aging_temperature_c=185.0, cooling_mode="aged"),
            ]
        )
        rep = validate_process_route(route=route, inferred_system="al-cu-mg", processing_context=ProcessingState())
        self.assertFalse(rep.is_valid)
        self.assertTrue(any("requires one of previous operations" in e for e in rep.errors))

    def test_global_bounds_fail(self) -> None:
        route = ProcessRoute(
            operations=[
                ProcessingOperation(method="cold_roll", deformation_pct=120.0, temperature_c=25.0),
            ]
        )
        rep = validate_process_route(route=route, inferred_system="cu-zn", processing_context=ProcessingState())
        self.assertFalse(rep.is_valid)
        self.assertTrue(any("outside" in e for e in rep.errors))

    def test_unknown_method_invalid(self) -> None:
        route = ProcessRoute(
            operations=[
                ProcessingOperation(method="unknown_magic", temperature_c=20.0),
            ]
        )
        rep = validate_process_route(route=route, inferred_system="fe-c", processing_context=ProcessingState())
        self.assertFalse(rep.is_valid)
        self.assertTrue(any("unknown method" in e for e in rep.errors))


if __name__ == "__main__":
    unittest.main()

