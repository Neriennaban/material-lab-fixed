from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


class SampleFactoryPhaseAutofixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QApplication.instance() is None:
            cls.app = QApplication(sys.argv)

    def test_auto_fix_normalizes_phase_table_in_phase_mode(self) -> None:
        from ui_qt.sample_factory_window import SampleFactoryWindow

        win = SampleFactoryWindow()
        try:
            phase_idx = win.composition_mode_combo.findData("phase_fe_c")
            win.composition_mode_combo.setCurrentIndex(phase_idx)
            win._on_composition_mode_changed()
            win._fill_phase_table({"Ferrite": 30.0, "Pearlite": 10.0})

            with (
                patch("ui_qt.sample_factory_window.QMessageBox.information") as info,
                patch("ui_qt.sample_factory_window.QMessageBox.warning") as warning,
            ):
                win._auto_fix()

            self.assertAlmostEqual(
                sum(win._phase_table_values().values()), 100.0, places=6
            )
            self.assertEqual(info.call_count, 1)
            self.assertEqual(warning.call_count, 0)
        finally:
            win.close()
            win.deleteLater()
