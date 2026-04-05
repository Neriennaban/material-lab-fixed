"""Integration test for verification dialog workflow."""

import unittest
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication


class TestVerificationDialogWorkflow(unittest.TestCase):
    """Test verification dialog workflow."""

    @classmethod
    def setUpClass(cls):
        """Create QApplication instance."""
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)

    def test_dialog_opens_with_teacher_key(self):
        """Test that dialog can be opened with teacher key."""
        from ui_qt.verification_dialog import VerificationDialog

        test_key_path = Path("keys/teacher_private_key.pem")
        dialog = VerificationDialog(None, test_key_path)

        self.assertIsNotNone(dialog)
        self.assertEqual(dialog.private_key_path, test_key_path)

    def test_dialog_workflow_file_selection(self):
        """Test file selection workflow in dialog."""
        from ui_qt.verification_dialog import VerificationDialog

        dialog = VerificationDialog(None, Path("test.pem"))

        # Initially no files selected
        self.assertIsNone(dialog.image_path)
        self.assertIsNone(dialog.student_json_path)
        self.assertIsNone(dialog.answers_enc_path)
        self.assertFalse(dialog.verify_btn.isEnabled())

        # Simulate file selection
        dialog.image_path = Path("test.png")
        dialog.student_json_path = Path("test_student.json")
        dialog.answers_enc_path = Path("test_answers.enc")
        dialog._check_files_selected()

        # Verify button should be enabled
        self.assertTrue(dialog.verify_btn.isEnabled())

    def test_dialog_displays_results_area(self):
        """Test that dialog has results display area."""
        from ui_qt.verification_dialog import VerificationDialog

        dialog = VerificationDialog(None, Path("test.pem"))

        self.assertIsNotNone(dialog.results_text)
        self.assertTrue(dialog.results_text.isReadOnly())
        self.assertIn("результат", dialog.results_text.placeholderText().lower())


if __name__ == "__main__":
    unittest.main()
