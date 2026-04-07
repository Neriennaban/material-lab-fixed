"""Tests for C2 — exporting per-phase masks for ML datasets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from export.export_images import save_phase_masks


class SavePhaseMasksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.masks = {
            "FERRITE": np.zeros((32, 32), dtype=np.uint8),
            "PEARLITE": np.zeros((32, 32), dtype=np.uint8),
            "CEMENTITE": np.zeros((32, 32), dtype=np.uint8),
        }
        self.masks["FERRITE"][:20, :] = 1  # 62.5 %
        self.masks["PEARLITE"][20:30, :] = 1  # 31.25 %
        self.masks["CEMENTITE"][30:, :] = 1  # 6.25 %

    def test_creates_one_png_per_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = save_phase_masks(self.masks, output_dir=tmp)
            saved = result["masks"]
            self.assertEqual(len(saved), 3)
            for path in saved.values():
                self.assertTrue(path.exists())
                with Image.open(path) as img:
                    self.assertEqual(img.size, (32, 32))
                    self.assertEqual(img.mode, "L")

    def test_legend_records_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = save_phase_masks(self.masks, output_dir=tmp)
            legend = result["legend"]
            self.assertEqual(legend["phase_count"], 3)
            entries = {e["phase"]: e for e in legend["phases"]}
            self.assertAlmostEqual(
                entries["FERRITE"]["coverage_fraction"], 0.625, places=4
            )
            self.assertAlmostEqual(
                entries["PEARLITE"]["coverage_fraction"], 0.3125, places=4
            )
            self.assertAlmostEqual(
                entries["CEMENTITE"]["coverage_fraction"], 0.0625, places=4
            )

    def test_legend_file_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = save_phase_masks(
                self.masks,
                output_dir=tmp,
                base_name="ferro_micro_mask",
            )
            legend_path = result["legend_path"]
            self.assertTrue(legend_path.exists())
            payload = json.loads(legend_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["base_name"], "ferro_micro_mask")
            self.assertEqual(payload["total_pixels"], 32 * 32)
            self.assertEqual(len(payload["phases"]), 3)

    def test_disable_legend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = save_phase_masks(
                self.masks, output_dir=tmp, write_legend=False
            )
            self.assertNotIn("legend_path", result)
            self.assertEqual(len(result["masks"]), 3)

    def test_skips_non_array_entries(self) -> None:
        masks = dict(self.masks)
        masks["NOT_AN_ARRAY"] = "ignored"  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as tmp:
            result = save_phase_masks(masks, output_dir=tmp)
            self.assertEqual(len(result["masks"]), 3)

    def test_round_trip_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = save_phase_masks(self.masks, output_dir=tmp)
            ferrite_path = result["masks"]["FERRITE"]
            with Image.open(ferrite_path) as img:
                arr = np.asarray(img)
            # Original mask was 0/1, file should be 0/255.
            recovered = (arr > 0).astype(np.uint8)
            self.assertTrue(np.array_equal(recovered, self.masks["FERRITE"]))


if __name__ == "__main__":
    unittest.main()
