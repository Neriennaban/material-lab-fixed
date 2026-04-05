from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.contracts_v3 import MetallographyRequestV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3

try:
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None  # type: ignore

from ui_qt.sample_factory_window_v3 import SampleFactoryWindowV3


@unittest.skipIf(QApplication is None, "PySide6 is not available")
class ExportManifestV3NoDiagramAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_manifest_keeps_deprecated_diagram_stub_without_asset(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v3_export_") as tmp:
            base = Path(tmp)
            pipeline = MetallographyPipelineV3()
            req = MetallographyRequestV3(
                sample_id="export_no_diagram",
                composition_wt={"Fe": 99.2, "C": 0.8},
                system_hint="fe-c",
                resolution=(96, 96),
                seed=4242,
            )
            req.thermal_program.points = [
                ThermalPointV3(time_s=0.0, temperature_c=20.0),
                ThermalPointV3(time_s=240.0, temperature_c=840.0),
                ThermalPointV3(time_s=420.0, temperature_c=840.0),
                ThermalPointV3(time_s=760.0, temperature_c=30.0),
            ]
            out = pipeline.generate(req)

            window = SampleFactoryWindowV3()
            try:
                window.current_request = req
                window.current_output = out
                window.export_dir_edit.setText(str(base))
                window.export_prefix_edit.setText("case")
                with patch("ui_qt.sample_factory_window_v3.QMessageBox.information", return_value=0), patch(
                    "ui_qt.sample_factory_window_v3.QMessageBox.warning",
                    return_value=0,
                ):
                    window._export_lab_package()
            finally:
                window.close()

            package_dirs = sorted(base.glob("case_*"))
            self.assertTrue(package_dirs)
            package_dir = package_dirs[-1]
            manifests = sorted(package_dir.glob("*_manifest.json"))
            self.assertTrue(manifests)
            payload = json.loads(manifests[-1].read_text(encoding="utf-8"))

            self.assertIsNone(payload.get("diagram"))
            self.assertTrue(bool(payload.get("diagram_deprecated", False)))
            self.assertFalse(any(package_dir.glob("*_diagram.png")))


if __name__ == "__main__":
    unittest.main()

