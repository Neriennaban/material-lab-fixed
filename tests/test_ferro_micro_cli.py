"""Tests for the ``scripts.ferro_micro_cli`` argparse front-end (Phase B3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.ferro_micro_cli import main


class CLISingleRenderTest(unittest.TestCase):
    def test_single_render_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "steel_040.png"
            rc = main(
                [
                    "--carbon",
                    "0.4",
                    "--width",
                    "96",
                    "--height",
                    "96",
                    "-o",
                    str(out),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())
            with Image.open(out) as img:
                self.assertEqual(img.size, (96, 96))

    def test_color_mode_propagates_to_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "warm.png"
            rc = main(
                [
                    "--carbon",
                    "0.45",
                    "--width",
                    "96",
                    "--height",
                    "96",
                    "--color-mode",
                    "nital_warm",
                    "--magnification",
                    "400",
                    "-o",
                    str(out),
                ]
            )
            self.assertEqual(rc, 0)
            with Image.open(out) as img:
                # nital_warm produces a chromatic RGB image (mode RGB)
                self.assertIn(img.mode, ("RGB", "RGBA"))

    def test_thermal_program_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp) / "program.json"
            tp.write_text(
                json.dumps(
                    {
                        "points": [
                            {"time_s": 0.0, "temperature_c": 20.0, "label": "start"},
                            {"time_s": 600.0, "temperature_c": 870.0, "label": "aust"},
                            {"time_s": 1800.0, "temperature_c": 870.0, "label": "hold"},
                            {"time_s": 3600.0, "temperature_c": 20.0, "label": "cool"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "annealed.png"
            rc = main(
                [
                    "--carbon",
                    "0.8",
                    "--width",
                    "96",
                    "--height",
                    "96",
                    "--thermal-program",
                    str(tp),
                    "-o",
                    str(out),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())

    def test_missing_output_returns_error(self) -> None:
        rc = main(["--carbon", "0.4"])
        self.assertEqual(rc, 2)


class CLIAtlasTest(unittest.TestCase):
    def test_atlas_writes_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "atlas"
            rc = main(
                [
                    "--atlas",
                    "--carbon-range",
                    "0.0",
                    "0.6",
                    "0.2",
                    "--output-dir",
                    str(out_dir),
                    "--width",
                    "64",
                    "--height",
                    "64",
                ]
            )
            self.assertEqual(rc, 0)
            files = sorted(out_dir.glob("sample_*.png"))
            self.assertEqual(len(files), 4)  # 0.0, 0.2, 0.4, 0.6
            manifest = json.loads((out_dir / "atlas.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["samples"]), 4)
            self.assertEqual(manifest["samples"][0]["carbon_wt"], 0.0)

    def test_atlas_requires_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(["--atlas", "--output-dir", tmp])
            self.assertEqual(rc, 2)

    def test_atlas_requires_output_dir(self) -> None:
        rc = main(["--atlas", "--carbon-range", "0.0", "1.0", "0.5"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
