from __future__ import annotations

import unittest

from core.measurements import (
    astm_grain_size_from_intercept_length_cm,
    astm_grain_size_from_intercept_length_um,
    lamellar_spacing_metrics,
    line_density_from_plane_points,
    mean_free_path_from_volume_fraction,
    mean_lineal_intercept,
    surface_density_from_line_intersections,
)


class QuantitativeMetallographyTests(unittest.TestCase):
    def test_mean_lineal_intercept_returns_total_length_divided_by_intersections(
        self,
    ) -> None:
        self.assertAlmostEqual(mean_lineal_intercept(120.0, 6), 20.0)

    def test_astm_grain_size_from_intercept_matches_cm_formula(self) -> None:
        value_cm = astm_grain_size_from_intercept_length_cm(0.01)
        value_um = astm_grain_size_from_intercept_length_um(100.0)
        self.assertAlmostEqual(value_cm, 3.28, places=2)
        self.assertAlmostEqual(value_cm, value_um, places=6)

    def test_stereology_density_helpers_match_textbook_equalities(self) -> None:
        self.assertAlmostEqual(surface_density_from_line_intersections(12.5), 25.0)
        self.assertAlmostEqual(line_density_from_plane_points(4.5), 9.0)

    def test_lamellar_spacing_metrics_use_true_spacing_as_half_random_spacing(
        self,
    ) -> None:
        payload = lamellar_spacing_metrics(200.0, 20)
        self.assertAlmostEqual(payload["interlamellar_random_spacing_um"], 10.0)
        self.assertAlmostEqual(payload["interlamellar_true_spacing_um"], 5.0)
        self.assertAlmostEqual(payload["lamella_intersection_density_mm_inv"], 100.0)
        self.assertAlmostEqual(payload["lamella_surface_density_mm_inv"], 400.0)

    def test_mean_free_path_uses_one_minus_volume_fraction_over_nl(self) -> None:
        self.assertAlmostEqual(mean_free_path_from_volume_fraction(0.25, 8.0), 0.09375)


if __name__ == "__main__":
    unittest.main()
