from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


class QtWindowPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QApplication.instance() is None:
            cls.app = QApplication(sys.argv)

    def test_generator_window_saves_state_on_close(self) -> None:
        from ui_qt.sample_factory_window_v3 import SampleFactoryWindowV3

        win = SampleFactoryWindowV3()
        try:
            mock_manager = Mock()
            win.state_manager = mock_manager
            win.close()
            self.assertEqual(mock_manager.save_state.call_count, 1)
            self.assertIs(mock_manager.save_state.call_args.args[0], win)
        finally:
            win.deleteLater()

    def test_microscope_window_saves_state_on_close(self) -> None:
        from ui_qt.microscope_window import MicroscopeWindow

        win = MicroscopeWindow(samples_dir=Path("examples") / "factory_v3_output")
        try:
            mock_manager = Mock()
            win.state_manager = mock_manager
            win.close()
            self.assertEqual(mock_manager.save_state.call_count, 1)
            self.assertIs(mock_manager.save_state.call_args.args[0], win)
        finally:
            win.deleteLater()
