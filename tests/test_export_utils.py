from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from export.export_images import save_image_bundle
from export.export_tables import save_measurements_csv


class ExportUtilsTests(unittest.TestCase):
    def test_save_image_bundle_writes_requested_formats(self) -> None:
        image = np.full((8, 8, 3), 128, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            saved = save_image_bundle(
                image,
                output_dir=tmp,
                base_name="sample",
                formats=("png", "jpg"),
            )
            self.assertEqual({path.suffix for path in saved}, {".png", ".jpg"})
            for path in saved:
                self.assertTrue(path.exists())

    def test_save_measurements_csv_collects_keys_from_all_rows(self) -> None:
        rows = [
            {"name": "line_1", "length_um": 12.5},
            {"name": "area_1", "area_um2": 48.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = save_measurements_csv(rows, Path(tmp) / "measurements.csv")
            text = path.read_text(encoding="utf-8")
            self.assertIn("length_um", text)
            self.assertIn("area_um2", text)
