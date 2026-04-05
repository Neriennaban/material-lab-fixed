from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ui_qt.theme_mirea import load_theme_settings, save_theme_mode


class UiV3ThemeProfileRoundtripTests(unittest.TestCase):
    def test_roundtrip_for_generator_and_microscope_profiles(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ui_theme_roundtrip_") as tmp:
            root = Path(tmp)
            generator_profile = root / "ui_theme_generator_v3.json"
            microscope_profile = root / "ui_theme_microscope_v3.json"

            save_theme_mode(generator_profile, mode="dark", style_profile="mirea_web_v1")
            save_theme_mode(microscope_profile, mode="light", style_profile="mirea_web_v1")

            gen = load_theme_settings(generator_profile)
            mic = load_theme_settings(microscope_profile)

            self.assertEqual(str(gen.get("theme_mode")), "dark")
            self.assertEqual(str(mic.get("theme_mode")), "light")
            self.assertEqual(str(gen.get("style_profile")), "mirea_web_v1")
            self.assertEqual(str(mic.get("style_profile")), "mirea_web_v1")


if __name__ == "__main__":
    unittest.main()

