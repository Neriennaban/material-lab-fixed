"""Tests for verification dialog UI."""

import unittest
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt

from core.security.verification import VerificationResult


class TestVerificationDialog(unittest.TestCase):
    """Test VerificationDialog UI components."""

    @classmethod
    def setUpClass(cls):
        """Create QApplication instance."""
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)

    def test_dialog_creation(self):
        """Test that dialog can be created."""
        from ui_qt.verification_dialog import VerificationDialog

        test_key_path = Path("keys/teacher_private_key.pem")
        dialog = VerificationDialog(None, test_key_path)

        self.assertIsNotNone(dialog)
        self.assertEqual(dialog.private_key_path, test_key_path)
        self.assertIsNone(dialog.image_path)
        self.assertIsNone(dialog.student_json_path)
        self.assertIsNone(dialog.answers_enc_path)

    def test_verify_button_initially_disabled(self):
        """Test that verify button is disabled when no files selected."""
        from ui_qt.verification_dialog import VerificationDialog

        dialog = VerificationDialog(None, Path("test.pem"))
        self.assertFalse(dialog.verify_btn.isEnabled())

    def test_file_selection_enables_verify_button(self):
        """Test that verify button enables when all files are selected."""
        from ui_qt.verification_dialog import VerificationDialog

        dialog = VerificationDialog(None, Path("test.pem"))

        # Initially disabled
        self.assertFalse(dialog.verify_btn.isEnabled())

        # Set file paths manually
        dialog.image_path = Path("test.png")
        dialog.student_json_path = Path("test_student.json")
        dialog.answers_enc_path = Path("test_answers.enc")

        # Trigger check
        dialog._check_files_selected()

        # Should be enabled now
        self.assertTrue(dialog.verify_btn.isEnabled())

    def test_partial_file_selection_keeps_button_disabled(self):
        """Test that verify button stays disabled with partial file selection."""
        from ui_qt.verification_dialog import VerificationDialog

        dialog = VerificationDialog(None, Path("test.pem"))

        # Set only image path
        dialog.image_path = Path("test.png")
        dialog._check_files_selected()
        self.assertFalse(dialog.verify_btn.isEnabled())

        # Set image and student json
        dialog.student_json_path = Path("test_student.json")
        dialog._check_files_selected()
        self.assertFalse(dialog.verify_btn.isEnabled())

        # Set all three
        dialog.answers_enc_path = Path("test_answers.enc")
        dialog._check_files_selected()
        self.assertTrue(dialog.verify_btn.isEnabled())

    def test_ui_components_exist(self):
        """Test that all UI components are created."""
        from ui_qt.verification_dialog import VerificationDialog

        dialog = VerificationDialog(None, Path("test.pem"))

        # Check file selection components
        self.assertIsNotNone(dialog.image_edit)
        self.assertIsNotNone(dialog.image_btn)
        self.assertIsNotNone(dialog.student_edit)
        self.assertIsNotNone(dialog.student_btn)
        self.assertIsNotNone(dialog.answers_edit)
        self.assertIsNotNone(dialog.answers_btn)

        # Check verify button
        self.assertIsNotNone(dialog.verify_btn)
        self.assertEqual(dialog.verify_btn.text(), "Проверить")

        # Check results area
        self.assertIsNotNone(dialog.results_text)
        self.assertTrue(dialog.results_text.isReadOnly())

    def test_window_properties(self):
        """Test dialog window properties."""
        from ui_qt.verification_dialog import VerificationDialog

        dialog = VerificationDialog(None, Path("test.pem"))

        self.assertEqual(dialog.windowTitle(), "Проверка целостности пакета ЛР")
        self.assertGreaterEqual(dialog.minimumWidth(), 700)
        self.assertGreaterEqual(dialog.minimumHeight(), 500)

    def test_display_results_formats_carbon_as_percent(self):
        """Test that carbon content is displayed in wt.% without 100x inflation."""
        from ui_qt.verification_dialog import VerificationDialog

        dialog = VerificationDialog(None, Path("test.pem"))
        result = VerificationResult(
            is_valid=True,
            image_authentic=True,
            data_authentic=True,
            signature_valid=True,
            hashes_match=True,
            error_message=None,
            answers={
                "sample_id": "sample_1",
                "steel_grade": "40",
                "carbon_content_calculated": 0.4,
                "phase_fractions": {"PEARLITE": 0.5},
                "inferred_system": "fe-c",
            },
        )

        dialog._display_results(result)

        text = dialog.results_text.toPlainText()
        self.assertIn("Содержание углерода: 0.4000%", text)
        self.assertNotIn("40.00%", text)


if __name__ == "__main__":
    unittest.main()
