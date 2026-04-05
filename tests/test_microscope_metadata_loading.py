import tempfile
import unittest
from pathlib import Path

from ui_qt.microscope_window import _candidate_metadata_paths


class MicroscopeMetadataLoadingTests(unittest.TestCase):
    def test_candidate_metadata_paths_include_export_variants(self) -> None:
        image = Path("sample.png")
        candidates = _candidate_metadata_paths(image)
        self.assertEqual(
            [p.name for p in candidates],
            ["sample.json", "sample_student.json", "sample_metadata.json"],
        )

    def test_candidate_metadata_paths_resolve_existing_student_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sample.png"
            image.write_bytes(b"fake")
            student = root / "sample_student.json"
            student.write_text("{}", encoding="utf-8")

            selected = next((candidate for candidate in _candidate_metadata_paths(image) if candidate.exists()), None)
            self.assertEqual(selected, student)


if __name__ == "__main__":
    unittest.main()
