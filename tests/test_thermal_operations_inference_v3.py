from __future__ import annotations

import unittest

from core.contracts_v3 import QuenchSettingsV3, ThermalPointV3, ThermalProgramV3
from core.metallography_v3.thermal_program_v3 import (
    effective_processing_from_thermal,
    infer_operations_from_thermal_program,
    summarize_thermal_program,
)


class ThermalOperationsInferenceV3Tests(unittest.TestCase):
    def test_infer_austenitize_and_quench(self) -> None:
        program = ThermalProgramV3(
            points=[
                ThermalPointV3(time_s=0.0, temperature_c=20.0),
                ThermalPointV3(time_s=300.0, temperature_c=840.0),
                ThermalPointV3(time_s=540.0, temperature_c=840.0),
                ThermalPointV3(time_s=620.0, temperature_c=60.0),
            ],
            quench=QuenchSettingsV3(
                medium_code="water_20",
                quench_time_s=30.0,
                bath_temperature_c=20.0,
                sample_temperature_c=840.0,
            ),
        )
        summary = summarize_thermal_program(program)
        _, _, qsum = effective_processing_from_thermal(program)
        payload = infer_operations_from_thermal_program(program, summary=summary, quench_summary=qsum)
        ops = [str(x.get("code", "")) for x in payload.get("operations", []) if isinstance(x, dict)]
        self.assertIn("austenitization_heat", ops)
        self.assertIn("austenitization_hold", ops)
        self.assertIn("quench_cooling", ops)
        s = dict(payload.get("summary", {}))
        self.assertTrue(bool(s.get("has_austenitization")))
        self.assertTrue(bool(s.get("has_quench")))

    def test_infer_temper_after_quench(self) -> None:
        program = ThermalProgramV3(
            points=[
                ThermalPointV3(time_s=0.0, temperature_c=20.0),
                ThermalPointV3(time_s=300.0, temperature_c=840.0),
                ThermalPointV3(time_s=480.0, temperature_c=840.0),
                ThermalPointV3(time_s=560.0, temperature_c=90.0),
                ThermalPointV3(time_s=900.0, temperature_c=400.0),
                ThermalPointV3(time_s=1320.0, temperature_c=400.0),
                ThermalPointV3(time_s=1600.0, temperature_c=40.0),
            ],
            quench=QuenchSettingsV3(
                medium_code="water_20",
                quench_time_s=35.0,
                bath_temperature_c=20.0,
                sample_temperature_c=840.0,
            ),
        )
        summary = summarize_thermal_program(program)
        _, _, qsum = effective_processing_from_thermal(program)
        payload = infer_operations_from_thermal_program(program, summary=summary, quench_summary=qsum)
        ops = [str(x.get("code", "")) for x in payload.get("operations", []) if isinstance(x, dict)]
        self.assertIn("quench_cooling", ops)
        self.assertTrue(any(code.startswith("temper_") for code in ops))
        s = dict(payload.get("summary", {}))
        self.assertTrue(bool(s.get("has_temper")))


if __name__ == "__main__":
    unittest.main()
