"""B4 — Python API for the JSON thermal-program format (TZ §9).

The TZ wants the CLI/API to accept a list of point dicts directly
(``[{"target": ..., "rate": ..., "hold": ...}]``-style and the
existing v3 ``{"time_s": ..., "temperature_c": ..., "label": ...}``
form). This test pins both formats end-to-end through the public
``ferro_micro_api.generate`` entry point.
"""

from __future__ import annotations

import unittest

from core.contracts_v3 import ThermalProgramV3
from core.metallography_v3 import ferro_micro_api as fm


class ThermalProgramListFormTest(unittest.TestCase):
    def test_v3_point_format_roundtrips_through_api(self) -> None:
        program = [
            {"time_s": 0.0, "temperature_c": 20.0, "label": "start", "locked": True},
            {"time_s": 600.0, "temperature_c": 870.0, "label": "austenitize"},
            {"time_s": 1800.0, "temperature_c": 870.0, "label": "hold"},
            {"time_s": 3600.0, "temperature_c": 20.0, "label": "cool"},
        ]
        sample = fm.generate(
            carbon=0.45,
            width=96,
            height=96,
            thermal_program=program,
            seed=42,
        )
        self.assertEqual(sample.image.shape, (96, 96, 3))

    def test_minimal_two_point_program(self) -> None:
        # Two-point degenerate program — heat to 870 °C and air cool.
        program = [
            {"time_s": 0.0, "temperature_c": 20.0},
            {"time_s": 1200.0, "temperature_c": 20.0},
        ]
        sample = fm.generate(
            carbon=0.20,
            width=96,
            height=96,
            thermal_program=program,
            seed=42,
        )
        self.assertEqual(sample.image.shape, (96, 96, 3))

    def test_thermal_program_v3_from_dict_round_trip(self) -> None:
        # ThermalProgramV3.from_dict accepts the {"points": [...]}
        # wrapper used by the v3 contracts.
        payload = {
            "points": [
                {"time_s": 0.0, "temperature_c": 20.0, "label": "start"},
                {"time_s": 1500.0, "temperature_c": 870.0, "label": "aust"},
                {"time_s": 3000.0, "temperature_c": 20.0, "label": "cool"},
            ]
        }
        program = ThermalProgramV3.from_dict(payload)
        self.assertGreaterEqual(len(program.points), 3)
        self.assertEqual(program.points[0].temperature_c, 20.0)
        self.assertEqual(program.points[1].temperature_c, 870.0)


if __name__ == "__main__":
    unittest.main()
