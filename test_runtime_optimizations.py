from __future__ import annotations

import json
import subprocess
import shutil
import tempfile
import time
import unittest
from pathlib import Path
import sys

from core.cache_manager import AdvancedCache
from core.materials import list_presets
from core.metallography_v3 import ferro_micro_api as fm
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3
from runtime_patches import apply_ui_runtime_patches


class AdvancedCacheRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.cache = AdvancedCache(
            cache_dir=self.temp_dir,
            max_memory_items=10,
            default_ttl=None,
            enable_disk=True,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lru_respects_recent_access(self) -> None:
        cache = AdvancedCache(
            cache_dir=None,
            max_memory_items=2,
            default_ttl=None,
            enable_disk=False,
        )
        cache.set("a", 1)
        cache.set("b", 2)
        self.assertEqual(cache.get("a"), 1)
        cache.set("c", 3)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("c"), 3)

    def test_disk_hit_does_not_trigger_rewrite(self) -> None:
        self.cache.set("key", {"value": 1})
        cold_cache = AdvancedCache(
            cache_dir=self.temp_dir,
            max_memory_items=10,
            default_ttl=None,
            enable_disk=True,
        )
        self.assertEqual(cold_cache.get("key"), {"value": 1})
        stats = cold_cache.stats()
        self.assertEqual(stats["disk_reads"], 1)
        self.assertEqual(stats["disk_writes"], 0)

    def test_disk_ttl_survives_restart(self) -> None:
        self.cache.set("key", "value", ttl=1)
        cold_cache = AdvancedCache(
            cache_dir=self.temp_dir,
            max_memory_items=10,
            default_ttl=None,
            enable_disk=True,
        )
        time.sleep(1.1)
        self.assertIsNone(cold_cache.get("key"))

    def test_direct_cache_manager_import_resolves_without_pytest_shims(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import core.cache_manager as cm; print(cm.AdvancedCache.__module__)",
            ],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(proc.stdout.strip(), "core.performance")


class FerroMicroRuntimeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        if hasattr(fm, "_get_pipeline"):
            fm._get_pipeline.cache_clear()

    def test_runtime_pipeline_cache_reuses_instance(self) -> None:
        if not hasattr(fm, "_get_pipeline"):
            self.skipTest("runtime pipeline cache shim is unavailable")
        presets_dir = str(fm.DEFAULT_PRESETS_DIR.resolve())
        profiles_dir = str(fm.DEFAULT_PROFILES_DIR.resolve())
        first = fm._get_pipeline(presets_dir, profiles_dir)
        second = fm._get_pipeline(presets_dir, profiles_dir)
        self.assertIs(first, second)

    def test_pipeline_load_preset_invalidates_on_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            presets_dir = Path(tmpdir)
            preset_path = presets_dir / "demo.json"
            preset_path.write_text(
                json.dumps({"seed": 1, "resolution": [64, 64]}),
                encoding="utf-8",
            )
            pipeline = MetallographyPipelineV3(
                presets_dir=presets_dir,
                profiles_dir=fm.DEFAULT_PROFILES_DIR,
            )
            first = pipeline.load_preset("demo")
            time.sleep(0.02)
            preset_path.write_text(
                json.dumps({"seed": 2, "resolution": [64, 64]}),
                encoding="utf-8",
            )
            second = pipeline.load_preset("demo")
            self.assertEqual(first["seed"], 1)
            self.assertEqual(second["seed"], 2)

    def test_list_presets_invalidates_on_directory_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.json").write_text("{}", encoding="utf-8")
            first = [p.name for p in list_presets(root)]
            time.sleep(0.02)
            (root / "b.json").write_text("{}", encoding="utf-8")
            second = [p.name for p in list_presets(root)]
            self.assertEqual(first, ["a.json"])
            self.assertEqual(second, ["a.json", "b.json"])


class UiRuntimeCacheTests(unittest.TestCase):
    def test_sample_factory_json_cache_invalidates_on_change(self) -> None:
        apply_ui_runtime_patches()
        from ui_qt.sample_factory_window_v3 import _json_load

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"
            path.write_text(json.dumps({"value": 1}), encoding="utf-8")
            first = _json_load(path)
            time.sleep(0.02)
            path.write_text(json.dumps({"value": 2}), encoding="utf-8")
            second = _json_load(path)
            self.assertEqual(first["value"], 1)
            self.assertEqual(second["value"], 2)

    def test_microscope_json_cache_invalidates_on_change(self) -> None:
        apply_ui_runtime_patches()
        import ui_qt.microscope_window as microscope_window

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            path.write_text(json.dumps({"image": "a.png"}), encoding="utf-8")
            first = microscope_window._load_json_dict(path)
            time.sleep(0.02)
            path.write_text(json.dumps({"image": "b.png"}), encoding="utf-8")
            second = microscope_window._load_json_dict(path)
            self.assertEqual(first["image"], "a.png")
            self.assertEqual(second["image"], "b.png")


if __name__ == "__main__":
    unittest.main()
