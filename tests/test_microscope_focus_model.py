import sys
import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication


class MicroscopeFocusModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)

    def _make_window(self):
        from ui_qt.microscope_window import MicroscopeWindow

        win = MicroscopeWindow(samples_dir=Path.cwd())
        self.addCleanup(win.close)
        self.addCleanup(win.deleteLater)
        return win

    def test_focus_target_changes_with_objective_and_xy(self) -> None:
        win = self._make_window()
        win._set_objective(200)
        target_a = win._focus_target_mm(200, 0.5, 0.5)
        target_b = win._focus_target_mm(500, 0.5, 0.5)
        target_c = win._focus_target_mm(500, 0.8, 0.2)

        self.assertNotEqual(target_a, target_b)
        self.assertNotEqual(target_b, target_c)

    def test_higher_magnification_is_more_sensitive(self) -> None:
        win = self._make_window()
        quality_200 = win._focus_quality_from_error(200, 0.25)
        quality_600 = win._focus_quality_from_error(600, 0.25)
        self.assertGreater(quality_200, quality_600)

    def test_focus_target_no_longer_depends_on_sample_identity(self) -> None:
        win = self._make_window()
        win.current_image_path = Path("sample_a.png")
        win.current_source_metadata = {"sample_id": "A", "final_stage": "ferrite"}
        target_a = win._focus_target_mm(400, 0.4, 0.6)
        win.current_image_path = Path("sample_b.png")
        win.current_source_metadata = {"sample_id": "B", "final_stage": "martensite"}
        target_b = win._focus_target_mm(400, 0.4, 0.6)
        self.assertAlmostEqual(target_a, target_b, places=6)
