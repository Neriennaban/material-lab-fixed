from __future__ import annotations

import unittest

from core.contracts_v3 import ThermalPointV3, ThermalProgramV3
from core.metallography_v3.thermal_program_v3 import infer_operations_from_thermal_program


class FeCStageTemperBandsV3Tests(unittest.TestCase):
    def _program_with_temper_peak(self, temper_peak_c: float) -> ThermalProgramV3:
        return ThermalProgramV3(
            points=[
                ThermalPointV3(time_s=0.0, temperature_c=20.0),
                ThermalPointV3(time_s=280.0, temperature_c=840.0),
                ThermalPointV3(time_s=460.0, temperature_c=840.0),
                ThermalPointV3(time_s=540.0, temperature_c=80.0),
                ThermalPointV3(time_s=860.0, temperature_c=float(temper_peak_c)),
                ThermalPointV3(time_s=1260.0, temperature_c=float(temper_peak_c)),
                ThermalPointV3(time_s=1520.0, temperature_c=35.0),
            ],
            sampling_mode="per_degree",
            degree_step_c=6.0,
            max_frames=240,
        )

    def test_detects_low_medium_high_temper_bands(self) -> None:
        cases = [
            (220.0, "low"),
            (400.0, "medium"),
            (600.0, "high"),
        ]
        for peak, band in cases:
            with self.subTest(peak=peak, band=band):
                payload = infer_operations_from_thermal_program(self._program_with_temper_peak(peak))
                summary = dict(payload.get("summary", {}))
                self.assertTrue(bool(summary.get("has_quench", False)))
                self.assertTrue(bool(summary.get("has_temper", False)))
                self.assertEqual(str(summary.get("temper_band_detected", "")), band)
                self.assertGreater(float(summary.get("temper_band_confidence", 0.0)), 0.45)


if __name__ == "__main__":
    unittest.main()
