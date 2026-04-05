from __future__ import annotations

import re
import unittest

from ui_qt.theme_mirea import THEME_MODES, build_qss, theme_tokens


class ThemeTokensMireaTests(unittest.TestCase):
    def test_theme_modes_present(self) -> None:
        self.assertIn("light", THEME_MODES)
        self.assertIn("dark", THEME_MODES)

    def test_required_tokens_exist_and_have_hex_values(self) -> None:
        required = {
            "bg_base",
            "bg_surface",
            "text_primary",
            "text_secondary",
            "border",
            "primary",
            "primary_hover",
            "accent",
            "success",
            "warning",
            "error",
            "focus",
            "header_grad_start",
            "header_grad_end",
        }
        hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for mode in ("light", "dark"):
            tokens = theme_tokens(mode)
            self.assertTrue(required.issubset(tokens.keys()))
            for key in required:
                self.assertRegex(str(tokens[key]), hex_re)

    def test_qss_generated_for_both_modes(self) -> None:
        for mode in ("light", "dark"):
            qss = build_qss(mode)
            self.assertIsInstance(qss, str)
            self.assertIn("QWidget", qss)
            self.assertIn("QPushButton", qss)
            self.assertIn("QWidget#appHeader", qss)
            self.assertIn("QPushButton#primaryCta", qss)

    def test_web_reference_colors_are_used(self) -> None:
        light = theme_tokens("light")
        self.assertEqual(light.get("primary"), "#003D82")
        self.assertEqual(light.get("accent"), "#00A3E0")


if __name__ == "__main__":
    unittest.main()
