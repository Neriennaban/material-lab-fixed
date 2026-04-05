from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ui_qt.theme_mirea import load_theme_mode, load_theme_settings, save_theme_mode


class ThemePersistenceProfilesTests(unittest.TestCase):
    def test_generator_and_microscope_profiles_are_independent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="theme_profiles_") as tmp:
            root = Path(tmp)
            generator_profile = root / "ui_theme_generator_v3.json"
            microscope_profile = root / "ui_theme_microscope_v3.json"

            save_theme_mode(generator_profile, "dark")
            save_theme_mode(microscope_profile, "light")

            self.assertEqual(load_theme_mode(generator_profile), "dark")
            self.assertEqual(load_theme_mode(microscope_profile), "light")
            self.assertEqual(load_theme_settings(generator_profile).get("style_profile"), "mirea_web_v1")
            self.assertEqual(load_theme_settings(microscope_profile).get("style_profile"), "mirea_web_v1")

    def test_invalid_payload_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="theme_profiles_") as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("{\"theme_mode\": 123}", encoding="utf-8")
            self.assertEqual(load_theme_mode(p, default="light"), "light")

    def test_legacy_profile_is_migrated_with_style_profile(self) -> None:
        with tempfile.TemporaryDirectory(prefix="theme_profiles_") as tmp:
            p = Path(tmp) / "legacy.json"
            p.write_text(json.dumps({"schema_version": 1, "theme_mode": "dark"}, ensure_ascii=False), encoding="utf-8")
            settings = load_theme_settings(p, default_mode="light")
            self.assertEqual(settings.get("theme_mode"), "dark")
            self.assertEqual(settings.get("style_profile"), "mirea_web_v1")
            payload = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(int(payload.get("schema_version", 0)), 2)
            self.assertEqual(str(payload.get("style_profile", "")), "mirea_web_v1")


if __name__ == "__main__":
    unittest.main()
