import importlib.util
import sys
import unittest
from pathlib import Path


if importlib.util.find_spec("PySide6") is not None:
    from PySide6.QtWidgets import QApplication
else:  # pragma: no cover
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is not available")
class MicroscopeMeasurementUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QApplication is not None and QApplication.instance() is None:
            cls.app = QApplication(sys.argv)

    def _make_window(self):
        from ui_qt.microscope_window import MicroscopeWindow

        win = MicroscopeWindow(samples_dir=Path.cwd())
        self.addCleanup(win.close)
        self.addCleanup(win.deleteLater)
        return win

    def test_measurement_controls_exist(self) -> None:
        win = self._make_window()
        self.assertTrue(hasattr(win, "measure_mode_btn"))
        self.assertTrue(hasattr(win, "area_measure_mode_btn"))
        self.assertTrue(hasattr(win, "measurement_table"))
        self.assertTrue(hasattr(win, "scale_toolbar_label"))

    def test_profile_payload_persists_measurement_tool_mode(self) -> None:
        win = self._make_window()
        win.area_measure_mode_btn.setChecked(True)
        payload = win._profile_payload_from_controls()
        self.assertEqual(payload["overlays"]["measurement_tool_mode"], "polygon_area")
        self.assertFalse(payload["overlays"]["measurement_ruler_enabled"])

    def test_clear_measurements_resets_active_history(self) -> None:
        win = self._make_window()
        win.measure_mode_btn.setChecked(True)
        win.line_measurement_history = [{"length_um": 12.0, "index": 1, "label": "x"}]
        win._line_measurement_counter = 1
        win._clear_measurements()
        self.assertEqual(win.line_measurement_history, [])
        self.assertEqual(win.measurement_table.rowCount(), 0)

    def test_measurement_modes_are_mutually_exclusive(self) -> None:
        win = self._make_window()
        win.measure_mode_btn.setChecked(True)
        self.assertEqual(win.measurement_tool_mode, "line")
        self.assertFalse(win.area_measure_mode_btn.isChecked())
        win.area_measure_mode_btn.setChecked(True)
        self.assertEqual(win.measurement_tool_mode, "polygon_area")
        self.assertFalse(win.measure_mode_btn.isChecked())

    def test_area_mode_uses_separate_history(self) -> None:
        win = self._make_window()
        win.measure_mode_btn.setChecked(True)
        win.line_measurement_history = [
            {"length_um": 12.0, "index": 1, "label": "line"}
        ]
        win.area_measure_mode_btn.setChecked(True)
        win.area_measurement_history = [
            {
                "area_um2": 20.0,
                "index": 1,
                "label": "area",
                "area_px2": 8.0,
                "vertex_count": 4,
            }
        ]
        win._refresh_measurement_table()
        self.assertEqual(win.measurement_table.rowCount(), 1)
        self.assertEqual(win.measurement_table.item(0, 1).text(), "area")
        self.assertEqual(win.line_measurement_history[0]["label"], "line")

    def test_measurement_average_reports_both_tables(self) -> None:
        win = self._make_window()
        win.line_measurement_history = [
            {"length_um": 12.0, "length_px": 4.0, "index": 1, "label": "line"}
        ]
        win.area_measurement_history = [
            {
                "area_um2": 20.0,
                "area_px2": 8.0,
                "index": 1,
                "label": "area",
                "vertex_count": 4,
            }
        ]
        win._update_measurement_average()
        text = win.measurement_average_label.text()
        self.assertIn("Линейка:", text)
        self.assertIn("Автолиния:", text)

    def test_apply_profile_payload_restores_measurement_tool_mode(self) -> None:
        win = self._make_window()
        win._apply_profile_payload(
            {
                "overlays": {
                    "measurement_tool_mode": "polygon_area",
                    "reticle_enabled": True,
                    "scale_bar_enabled": True,
                }
            }
        )
        self.assertEqual(win.measurement_tool_mode, "polygon_area")
        self.assertTrue(win.area_measure_mode_btn.isChecked())


if __name__ == "__main__":
    unittest.main()
