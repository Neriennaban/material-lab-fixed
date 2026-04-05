from __future__ import annotations

import json
import unittest
from pathlib import Path

from core.pipeline_v2 import GenerationPipelineV2


class PresetsAutoMigrationV2Tests(unittest.TestCase):
    def test_all_presets_are_auto_and_generate(self) -> None:
        pipeline = GenerationPipelineV2()
        presets_dir = Path("presets")
        preset_paths = sorted(presets_dir.glob("*.json"))
        self.assertTrue(preset_paths, "No presets found")

        for path in preset_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("generator"), "auto", f"Preset {path.name} is not migrated to auto")

            generation = payload.get("generation", {}) if isinstance(payload.get("generation"), dict) else {}
            auto_hint = generation.get("auto_hint", {}) if isinstance(generation.get("auto_hint"), dict) else {}
            self.assertIn("preferred_generator", auto_hint, f"Preset {path.name} misses auto_hint.preferred_generator")

            preset = pipeline.load_preset(path)
            request = pipeline.request_from_preset(preset=preset, resolution_override=(128, 128))
            output = pipeline.generate(request)
            auto_meta = output.metadata.get("auto_generator", {})
            self.assertTrue(bool(auto_meta.get("enabled")), f"Preset {path.name} did not run auto mode")


if __name__ == "__main__":
    unittest.main()
