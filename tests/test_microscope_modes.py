import sys
import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication


class TestMicroscopeModes(unittest.TestCase):
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

    def test_teacher_mode_state_is_removed_from_microscope(self):
        win = self._make_window()
        self.assertFalse(hasattr(win, "current_mode"))
        self.assertFalse(hasattr(win, "teacher_private_key_path"))
        self.assertFalse(hasattr(win, "teacher_answers"))

    def test_teacher_panel_and_switcher_are_not_built(self):
        win = self._make_window()
        self.assertFalse(hasattr(win, "teacher_panel"))
        self.assertFalse(hasattr(win, "mode_label"))
        self.assertFalse(hasattr(win, "mode_switch_btn"))
