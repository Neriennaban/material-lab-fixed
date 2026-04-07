"""Tests for the C3 atlas script and the C5 blind-evaluation script."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.blind_eval_ferro_micro import main as blind_main
from scripts.generate_ferro_micro_atlas import (
    ATLAS_RECIPES,
    _expected_fractions_for_carbon,
    main as atlas_main,
)


class AtlasScriptTest(unittest.TestCase):
    def test_atlas_writes_18_images_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc = atlas_main(
                [
                    "--output-dir",
                    str(tmp),
                    "--width",
                    "64",
                    "--height",
                    "64",
                ]
            )
            self.assertEqual(rc, 0)
            files = sorted(Path(tmp).glob("*.png"))
            self.assertEqual(len(files), len(ATLAS_RECIPES))
            for f in files:
                with Image.open(f) as img:
                    self.assertEqual(img.size, (64, 64))
            manifest_path = Path(tmp) / "atlas_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["samples"]), len(ATLAS_RECIPES))

    def test_atlas_with_metrics_records_phase_fractions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc = atlas_main(
                [
                    "--output-dir",
                    str(tmp),
                    "--width",
                    "64",
                    "--height",
                    "64",
                    "--with-metrics",
                ]
            )
            self.assertEqual(rc, 0)
            manifest = json.loads(
                (Path(tmp) / "atlas_manifest.json").read_text(encoding="utf-8")
            )
            for sample in manifest["samples"]:
                if sample["carbon_wt"] >= 2.14:
                    continue  # cast iron has no lever-rule expectation
                self.assertIn("phase_fractions", sample)
                self.assertIn("max_relative_error_pct", sample)


class ExpectedFractionsTest(unittest.TestCase):
    def test_pure_iron_returns_only_ferrite(self) -> None:
        out = _expected_fractions_for_carbon(0.0)
        self.assertEqual(out, {"FERRITE": 1.0})

    def test_eutectoid_balance(self) -> None:
        out = _expected_fractions_for_carbon(0.40)
        # P = (0.40 - 0.02) / (0.77 - 0.02) ≈ 0.507
        self.assertAlmostEqual(out["PEARLITE"], 0.5067, places=3)
        self.assertAlmostEqual(out["FERRITE"], 1.0 - 0.5067, places=3)

    def test_hypereutectoid_returns_pearlite_and_cementite(self) -> None:
        out = _expected_fractions_for_carbon(1.20)
        self.assertIn("PEARLITE", out)
        self.assertIn("CEMENTITE", out)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=4)

    def test_cast_iron_returns_ledeburite(self) -> None:
        self.assertEqual(_expected_fractions_for_carbon(3.5), {"LEDEBURITE": 1.0})


class BlindEvalScriptTest(unittest.TestCase):
    def test_synthetic_only_run_writes_answer_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc = blind_main(
                [
                    "--output-dir",
                    str(tmp),
                    "--count",
                    "5",
                    "--width",
                    "64",
                    "--height",
                    "64",
                ]
            )
            self.assertEqual(rc, 0)
            files = sorted(Path(tmp).glob("*.png"))
            self.assertEqual(len(files), 5)
            answer_path = Path(tmp) / "answer_key.json"
            self.assertTrue(answer_path.exists())
            payload = json.loads(answer_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["items"]), 5)
            for item in payload["items"]:
                self.assertEqual(item["category"], "synthetic")
                self.assertIn("anon_id", item)
                self.assertIn("sha256", item)

    def test_run_with_empty_count_and_no_references_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc = blind_main(
                ["--output-dir", str(tmp), "--count", "0"]
            )
            self.assertEqual(rc, 2)

    def test_real_and_synthetic_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp) / "refs"
            ref_dir.mkdir()
            # Plant two fake "real" PNGs.
            for index in range(2):
                fake = ref_dir / f"real_{index}.png"
                Image.new("L", (32, 32), color=200).save(fake)

            out_dir = Path(tmp) / "blind"
            rc = blind_main(
                [
                    "--references",
                    str(ref_dir),
                    "--output-dir",
                    str(out_dir),
                    "--count",
                    "3",
                    "--width",
                    "64",
                    "--height",
                    "64",
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(
                (out_dir / "answer_key.json").read_text(encoding="utf-8")
            )
            categories = {item["category"] for item in payload["items"]}
            self.assertEqual(categories, {"real", "synthetic"})
            self.assertEqual(len(payload["items"]), 5)


if __name__ == "__main__":
    unittest.main()
