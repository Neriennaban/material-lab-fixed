from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ui_qt.theme_mirea import resolve_branding_logo


class ThemeAssetLoaderTests(unittest.TestCase):
    def test_preferred_logo_is_used_when_exists(self) -> None:
        with tempfile.TemporaryDirectory(prefix="theme_asset_") as tmp:
            preferred = Path(tmp) / "preferred_logo.png"
            preferred.write_bytes(b"\x89PNG\r\n\x1a\n")
            resolved = resolve_branding_logo(preferred)
            self.assertIsNotNone(resolved)
            self.assertEqual(Path(resolved or "").resolve(), preferred.resolve())

    def test_fallback_logo_resolution_does_not_fail(self) -> None:
        resolved = resolve_branding_logo("non_existing_logo.png")
        if resolved is not None:
            self.assertTrue(Path(resolved).exists())


if __name__ == "__main__":
    unittest.main()

