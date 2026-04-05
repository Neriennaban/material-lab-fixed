import unittest
from pathlib import Path


class FocusDocsTests(unittest.TestCase):
    def test_focus_guide_exists(self) -> None:
        guide = Path("docs/FOCUS_GUIDE_RU.md")
        self.assertTrue(guide.exists())

    def test_microscope_guide_links_to_focus_guide(self) -> None:
        microscope_guide = Path("docs/MICROSCOPE_GUIDE_RU.md")
        text = microscope_guide.read_text(encoding="utf-8")
        self.assertIn("docs/FOCUS_GUIDE_RU.md", text)


if __name__ == "__main__":
    unittest.main()
