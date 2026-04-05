import unittest

from core.microscope_measurements import (
    derive_um_per_px_100x,
    estimate_um_per_px_from_geometry,
    format_metric_area_um2,
    format_metric_length_um,
    line_measurement,
    polygon_area_measurement,
    scale_audit_report,
)
from core.imaging import extract_field_of_view


class MicroscopeMeasurementsTests(unittest.TestCase):
    def test_line_measurement_reports_length_and_angle(self) -> None:
        payload = line_measurement((10.0, 10.0), (13.0, 14.0), um_per_px=0.5)
        self.assertTrue(payload["valid"])
        self.assertAlmostEqual(float(payload["length_px"]), 5.0, places=6)
        self.assertAlmostEqual(float(payload["length_um"]), 2.5, places=6)
        self.assertAlmostEqual(float(payload["angle_deg"]), 53.130102, places=4)
        self.assertEqual(payload["label"], "2.50 мкм")

    def test_format_metric_length_um_switches_units(self) -> None:
        self.assertEqual(format_metric_length_um(0.42), "420 нм")
        self.assertEqual(format_metric_length_um(4.2), "4.20 мкм")
        self.assertEqual(format_metric_length_um(4200.0), "4.200 мм")

    def test_derive_um_per_px_100x_prefers_explicit_value(self) -> None:
        value, source = derive_um_per_px_100x(
            {
                "microscope_ready": {"um_per_px_100x": 0.82},
                "request_v3": {"microscope_profile": {"magnification": 300}},
            }
        )
        self.assertAlmostEqual(value, 0.82)
        self.assertEqual(source, "metadata.microscope_ready.um_per_px_100x")

    def test_derive_um_per_px_100x_recovers_from_native_scale(self) -> None:
        value, source = derive_um_per_px_100x(
            {
                "microscope_ready": {"native_um_per_px": 0.25},
                "request_v3": {"microscope_profile": {"magnification": 400}},
            }
        )
        self.assertAlmostEqual(value, 1.0)
        self.assertEqual(source, "metadata.microscope_ready.native_um_per_px")

    def test_estimate_um_per_px_from_geometry_matches_expected_ratio(self) -> None:
        um_per_px = estimate_um_per_px_from_geometry(
            um_per_px_100x=1.0,
            crop_size_px=(256, 256),
            output_size_px=(1024, 1024),
        )
        self.assertAlmostEqual(um_per_px, 0.25)

    def test_scale_audit_report_confirms_extract_field_of_view_geometry(self) -> None:
        sample = (255 * __import__("numpy").ones((1024, 1024), dtype="uint8"))
        crop, fov = extract_field_of_view(sample, magnification=400, output_size=(1024, 1024))
        self.assertEqual(crop.shape, (1024, 1024))
        actual_um_per_px = estimate_um_per_px_from_geometry(
            um_per_px_100x=1.0,
            crop_size_px=fov["crop_size_px"],
            output_size_px=(1024, 1024),
        )
        report = scale_audit_report(
            objective=400,
            source_size_px=(1024, 1024),
            crop_size_px=fov["crop_size_px"],
            output_size_px=(1024, 1024),
            um_per_px_100x=1.0,
            actual_um_per_px=actual_um_per_px,
        )
        self.assertTrue(report["ok"])
        self.assertTrue(report["scale_ok"])
        self.assertTrue(report["crop_ok"])
        self.assertAlmostEqual(float(report["expected_um_per_px"]), 0.25)
        self.assertEqual(int(report["expected_crop_w_px"]), 256)

    def test_scale_audit_report_detects_wrong_scale(self) -> None:
        report = scale_audit_report(
            objective=400,
            source_size_px=(1024, 1024),
            crop_size_px=(256, 256),
            output_size_px=(1024, 1024),
            um_per_px_100x=1.0,
            actual_um_per_px=0.5,
        )
        self.assertFalse(report["ok"])
        self.assertFalse(report["scale_ok"])
        self.assertTrue(report["crop_ok"])

    def test_polygon_area_measurement_reports_square_area_and_perimeter(self) -> None:
        payload = polygon_area_measurement(
            [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            um_per_px=0.5,
        )
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["kind"], "polygon_area")
        self.assertEqual(int(payload["vertex_count"]), 4)
        self.assertAlmostEqual(float(payload["area_px2"]), 16.0, places=6)
        self.assertAlmostEqual(float(payload["area_um2"]), 4.0, places=6)
        self.assertAlmostEqual(float(payload["perimeter_px"]), 16.0, places=6)
        self.assertAlmostEqual(float(payload["perimeter_um"]), 8.0, places=6)
        self.assertEqual(payload["label"], "4.00 мкм²")

    def test_polygon_area_measurement_reports_triangle_area(self) -> None:
        payload = polygon_area_measurement(
            [(0.0, 0.0), (6.0, 0.0), (0.0, 3.0)],
            um_per_px=1.0,
        )
        self.assertTrue(payload["valid"])
        self.assertAlmostEqual(float(payload["area_px2"]), 9.0, places=6)
        self.assertAlmostEqual(float(payload["area_um2"]), 9.0, places=6)

    def test_polygon_area_measurement_scales_with_um_per_px(self) -> None:
        payload = polygon_area_measurement(
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            um_per_px=0.2,
        )
        self.assertAlmostEqual(float(payload["area_px2"]), 100.0, places=6)
        self.assertAlmostEqual(float(payload["area_um2"]), 4.0, places=6)

    def test_polygon_area_measurement_is_invalid_for_less_than_three_vertices(self) -> None:
        payload = polygon_area_measurement(
            [(0.0, 0.0), (5.0, 0.0)],
            um_per_px=1.0,
        )
        self.assertFalse(payload["valid"])
        self.assertEqual(int(payload["vertex_count"]), 2)
        self.assertAlmostEqual(float(payload["area_um2"]), 0.0, places=6)

    def test_format_metric_area_um2_switches_to_mm2_for_large_values(self) -> None:
        self.assertEqual(format_metric_area_um2(12.5), "12.50 мкм²")
        self.assertEqual(format_metric_area_um2(1_500_000.0), "1.5000 мм²")


if __name__ == "__main__":
    unittest.main()
