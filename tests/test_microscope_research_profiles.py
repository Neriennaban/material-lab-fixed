from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None  # type: ignore


class MicroscopeResearchProfilesTests(unittest.TestCase):
    def test_research_profile_files_exist_and_define_psf_profiles(self) -> None:
        expected = {
            "profiles/microscope_profile_research_bessel.json": "bessel_extended_dof",
            "profiles/microscope_profile_research_stir.json": "stir_sectioning",
            "profiles/microscope_profile_research_hybrid.json": "lens_axicon_hybrid",
        }
        for rel_path, profile in expected.items():
            path = Path(rel_path)
            self.assertTrue(path.exists(), msg=str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(str(payload.get("optics", {}).get("psf_profile", "")), profile)

    @unittest.skipIf(QApplication is None, "PySide6 is not available")
    def test_default_profile_stays_non_research_and_research_payload_restores(self) -> None:
        from ui_qt.microscope_window import MicroscopeWindow

        _app = QApplication.instance() or QApplication([])
        win = MicroscopeWindow(samples_dir=Path.cwd())
        try:
            payload = json.loads(Path("profiles/microscope_profile_research_stir.json").read_text(encoding="utf-8"))
            self.assertEqual(str(win.psf_profile_combo.currentData() or ""), "standard")
            win._apply_profile_payload(payload)
            self.assertEqual(str(win.psf_profile_combo.currentData() or ""), "stir_sectioning")
            self.assertGreater(float(win.psf_strength_spin.value()), 0.0)
        finally:
            win.close()


if __name__ == "__main__":
    unittest.main()
