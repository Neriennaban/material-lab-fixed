from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QApplication

from ui_qt.microscope_window import MicroscopeWindow, ZoomView
from ui_qt.modern_widgets import FlexibleDoubleSpinBox, parse_flexible_float


class QtNumericInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_parse_flexible_float_accepts_comma_and_dot(self) -> None:
        self.assertAlmostEqual(parse_flexible_float("1,25"), 1.25)
        self.assertAlmostEqual(parse_flexible_float("1.25"), 1.25)
        self.assertAlmostEqual(parse_flexible_float(" 2,5 "), 2.5)

    def test_flexible_spinbox_accepts_comma_and_dot(self) -> None:
        spin = FlexibleDoubleSpinBox()
        spin.setRange(-1000.0, 1000.0)
        spin.setSuffix(" mm")
        state_comma, _, _ = spin.validate("1,25 mm", 0)
        state_dot, _, _ = spin.validate("1.25 mm", 0)
        self.assertEqual(state_comma, QValidator.State.Acceptable)
        self.assertEqual(state_dot, QValidator.State.Acceptable)
        self.assertAlmostEqual(spin.valueFromText("1,25 mm"), 1.25)
        self.assertAlmostEqual(spin.valueFromText("1.25 mm"), 1.25)


class MicroscopeMeasurementPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MicroscopeWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()

    def test_area_history_stays_visible_when_tool_is_turned_off(self) -> None:
        self.window.area_measurement_history = [
            {
                "index": 1,
                "label": "S1",
                "area_px2": 128.0,
                "area_um2": 64.0,
                "perimeter_px": 48.0,
                "perimeter_um": 24.0,
                "vertex_count": 4,
                "vertices_px": [],
            }
        ]
        self.window._set_measurement_tool_mode(ZoomView.TOOL_POLYGON_AREA)
        self.assertEqual(self.window.area_measurement_table.rowCount(), 1)
        self.assertEqual(self.window.line_measurement_table.rowCount(), 0)

        self.window._set_measurement_tool_mode(ZoomView.TOOL_OFF)

        self.assertEqual(self.window.measurement_display_mode, ZoomView.TOOL_POLYGON_AREA)
        self.assertEqual(self.window.area_measurement_table.rowCount(), 1)
        self.assertEqual(self.window.line_measurement_table.rowCount(), 0)

    def test_clear_area_measurements_clears_only_area_table_when_tool_is_off(self) -> None:
        self.window.line_measurement_history = [
            {
                "index": 1,
                "label": "L1",
                "length_px": 42.0,
                "length_um": 21.0,
                "angle_deg": 15.0,
                "x0_px": 0.0,
                "y0_px": 0.0,
                "x1_px": 10.0,
                "y1_px": 10.0,
            }
        ]
        self.window.area_measurement_history = [
            {
                "index": 1,
                "label": "S1",
                "area_px2": 128.0,
                "area_um2": 64.0,
                "perimeter_px": 48.0,
                "perimeter_um": 24.0,
                "vertex_count": 4,
                "vertices_px": [],
            }
        ]
        self.window._refresh_measurement_tables()
        self.window._set_measurement_tool_mode(ZoomView.TOOL_POLYGON_AREA)
        self.window._set_measurement_tool_mode(ZoomView.TOOL_OFF)

        self.window._clear_area_measurements()

        self.assertEqual(len(self.window.line_measurement_history), 1)
        self.assertEqual(self.window.area_measurement_history, [])
        self.assertEqual(self.window.line_measurement_table.rowCount(), 1)
        self.assertEqual(self.window.area_measurement_table.rowCount(), 0)


if __name__ == "__main__":
    unittest.main()
