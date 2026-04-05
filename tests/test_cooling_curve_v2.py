from __future__ import annotations

import unittest

from core.cooling_curve import normalize_cooling_curve_points, sample_cooling_curve


class CoolingCurveV2Tests(unittest.TestCase):
    def test_per_degree_sampling_contains_intermediate_temperatures(self) -> None:
        points = normalize_cooling_curve_points(
            [{"time_min": 0.0, "temperature_c": 100.0}, {"time_min": 10.0, "temperature_c": 95.0}],
            fallback_temperature_c=100.0,
        )
        series = sample_cooling_curve(points, mode="per_degree", degree_step=1.0, max_points=100, base_mode="equilibrium")
        temps = [round(float(x["temperature_c"]), 1) for x in series]
        self.assertIn(99.0, temps)
        self.assertIn(96.0, temps)
        self.assertEqual(round(float(series[0]["temperature_c"]), 1), 100.0)
        self.assertEqual(round(float(series[-1]["temperature_c"]), 1), 95.0)

    def test_points_mode_uses_only_support_points(self) -> None:
        points = normalize_cooling_curve_points(
            [{"time_min": 0.0, "temperature_c": 800.0}, {"time_min": 4.0, "temperature_c": 600.0}, {"time_min": 8.0, "temperature_c": 300.0}],
            fallback_temperature_c=800.0,
        )
        series = sample_cooling_curve(points, mode="points", degree_step=1.0, max_points=100, base_mode="equilibrium")
        self.assertEqual(len(series), 3)


if __name__ == "__main__":
    unittest.main()
