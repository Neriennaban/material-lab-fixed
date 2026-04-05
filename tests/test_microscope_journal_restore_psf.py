from __future__ import annotations

import unittest
from pathlib import Path

try:
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None  # type: ignore


@unittest.skipIf(QApplication is None, "PySide6 is not available")
class MicroscopeJournalRestorePsfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_restore_journal_row_restores_psf_controls(self) -> None:
        from ui_qt.microscope_window import MicroscopeWindow

        win = MicroscopeWindow(samples_dir=Path.cwd())
        try:
            win.journal.insertRow(0)
            win.journal_records = [
                {
                    "controls": {
                        "objective": 400,
                        "focus_distance_mm": 18.5,
                        "stage_x": 0.42,
                        "stage_y": 0.61,
                        "psf_profile": "stir_sectioning",
                        "psf_strength": 0.9,
                        "sectioning_shear_deg": 40.0,
                        "hybrid_balance": 0.5,
                    }
                }
            ]
            win._restore_journal_row(0, 0)
            self.assertEqual(str(win.psf_profile_combo.currentData() or ""), "stir_sectioning")
            self.assertEqual(float(win.psf_strength_spin.value()), 0.9)
            self.assertEqual(float(win.sectioning_shear_spin.value()), 40.0)
            self.assertEqual(float(win.hybrid_balance_spin.value()), 0.5)
        finally:
            win.close()

    def test_append_journal_row_marks_research_optics(self) -> None:
        from ui_qt.microscope_window import MicroscopeWindow

        win = MicroscopeWindow(samples_dir=Path.cwd())
        try:
            win.current_capture_meta = {
                "controls_state": {
                    "objective": 400,
                    "focus_distance_mm": 18.5,
                    "stage_x": 0.42,
                    "stage_y": 0.61,
                    "psf_profile": "stir_sectioning",
                },
                "route_summary": {"final_stage": "pearlite"},
            }
            image_path = Path("dummy.png")
            meta_path = Path("dummy.json")
            win._append_journal_row(image_path=image_path, meta_path=meta_path)
            self.assertIn("research:stir_sectioning", win.journal.item(0, 5).text())
        finally:
            win.close()


if __name__ == "__main__":
    unittest.main()
