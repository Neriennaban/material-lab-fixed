from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from run_app_v2 import parse_args
from ui_qt.microscope_window import MicroscopeWindow
from ui_qt.sample_factory_window_v3 import SampleFactoryWindowV3


class AppDefaultPathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QApplication.instance() is None:
            cls.app = QApplication(sys.argv)

    def test_run_app_v2_defaults_to_factory_v3_output(self) -> None:
        with patch.object(sys, "argv", ["run_app_v2.py"]):
            args = parse_args()
        self.assertTrue(str(args.samples_dir).endswith(str(Path("examples") / "factory_v3_output")))

    def test_microscope_window_defaults_to_factory_v3_output(self) -> None:
        win = MicroscopeWindow()
        try:
            self.assertTrue(str(win.samples_dir).endswith(str(Path("examples") / "factory_v3_output")))
        finally:
            win.close()
            win.deleteLater()

    def test_generator_export_defaults_to_factory_v3_output(self) -> None:
        win = SampleFactoryWindowV3()
        try:
            self.assertTrue(win.export_dir_edit.text().endswith(str(Path("examples") / "factory_v3_output")))
        finally:
            win.close()
            win.deleteLater()
