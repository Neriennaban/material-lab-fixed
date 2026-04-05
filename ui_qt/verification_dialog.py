"""Verification dialog for checking student lab package integrity."""

from pathlib import Path
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QGroupBox,
    QMessageBox,
)
from PySide6.QtCore import Qt


class VerificationDialog(QDialog):
    """Dialog for verifying student lab package integrity."""

    def __init__(self, parent=None, private_key_path: Path | None = None):
        super().__init__(parent)
        self.private_key_path = private_key_path
        self.image_path: Path | None = None
        self.student_json_path: Path | None = None
        self.answers_enc_path: Path | None = None

        self.setWindowTitle("Проверка целостности пакета ЛР")
        self.setMinimumSize(700, 500)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # File selection group
        file_group = QGroupBox("Выбор файлов")
        file_layout = QVBoxLayout(file_group)

        # Image file
        image_layout = QHBoxLayout()
        image_layout.addWidget(QLabel("Изображение:"))
        self.image_edit = QLineEdit()
        self.image_edit.setReadOnly(True)
        self.image_edit.setPlaceholderText("Выберите файл .png")
        image_layout.addWidget(self.image_edit)
        self.image_btn = QPushButton("Обзор...")
        self.image_btn.clicked.connect(self._browse_image)
        image_layout.addWidget(self.image_btn)
        file_layout.addLayout(image_layout)

        # Student JSON file
        student_layout = QHBoxLayout()
        student_layout.addWidget(QLabel("student.json:"))
        self.student_edit = QLineEdit()
        self.student_edit.setReadOnly(True)
        self.student_edit.setPlaceholderText("Выберите файл student.json")
        student_layout.addWidget(self.student_edit)
        self.student_btn = QPushButton("Обзор...")
        self.student_btn.clicked.connect(self._browse_student_json)
        student_layout.addWidget(self.student_btn)
        file_layout.addLayout(student_layout)

        # Answers encrypted file
        answers_layout = QHBoxLayout()
        answers_layout.addWidget(QLabel("answers.enc:"))
        self.answers_edit = QLineEdit()
        self.answers_edit.setReadOnly(True)
        self.answers_edit.setPlaceholderText("Выберите файл answers.enc")
        answers_layout.addWidget(self.answers_edit)
        self.answers_btn = QPushButton("Обзор...")
        self.answers_btn.clicked.connect(self._browse_answers_enc)
        answers_layout.addWidget(self.answers_btn)
        file_layout.addLayout(answers_layout)

        layout.addWidget(file_group)

        # Verify button
        self.verify_btn = QPushButton("Проверить")
        self.verify_btn.setEnabled(False)
        self.verify_btn.clicked.connect(self._verify)
        self.verify_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        layout.addWidget(self.verify_btn)

        # Results area
        results_group = QGroupBox("Результаты проверки")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Результаты проверки появятся здесь...")
        results_layout.addWidget(self.results_text)
        layout.addWidget(results_group)

        # Close button
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _browse_image(self) -> None:
        """Browse for image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            str(Path.home()),
            "PNG Images (*.png);;All Files (*)",
        )
        if file_path:
            self.image_path = Path(file_path)
            self.image_edit.setText(str(self.image_path))
            self._check_files_selected()

    def _browse_student_json(self) -> None:
        """Browse for student.json file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите student.json",
            str(Path.home()),
            "JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            self.student_json_path = Path(file_path)
            self.student_edit.setText(str(self.student_json_path))
            self._check_files_selected()

    def _browse_answers_enc(self) -> None:
        """Browse for answers.enc file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите answers.enc",
            str(Path.home()),
            "Encrypted Files (*.enc);;All Files (*)",
        )
        if file_path:
            self.answers_enc_path = Path(file_path)
            self.answers_edit.setText(str(self.answers_enc_path))
            self._check_files_selected()

    def _check_files_selected(self) -> None:
        """Enable verify button only when all files are selected."""
        all_selected = (
            self.image_path is not None
            and self.student_json_path is not None
            and self.answers_enc_path is not None
        )
        self.verify_btn.setEnabled(all_selected)

    def _verify(self) -> None:
        """Verify the lab package integrity."""
        if not self.private_key_path:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Приватный ключ не настроен. Переключитесь в режим преподавателя.",
            )
            return

        if not self.private_key_path.exists():
            QMessageBox.critical(
                self, "Ошибка", f"Приватный ключ не найден:\n{self.private_key_path}"
            )
            return

        # Verify all files exist
        if not self.image_path.exists():
            QMessageBox.critical(self, "Ошибка", f"Файл не найден:\n{self.image_path}")
            return
        if not self.student_json_path.exists():
            QMessageBox.critical(
                self, "Ошибка", f"Файл не найден:\n{self.student_json_path}"
            )
            return
        if not self.answers_enc_path.exists():
            QMessageBox.critical(
                self, "Ошибка", f"Файл не найден:\n{self.answers_enc_path}"
            )
            return

        # Perform verification
        try:
            from core.security import verify_sample_integrity

            result = verify_sample_integrity(
                self.image_path,
                self.student_json_path,
                self.answers_enc_path,
                self.private_key_path,
            )

            # Display results
            self._display_results(result)

        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка проверки", f"Произошла ошибка при проверке:\n{str(e)}"
            )

    def _display_results(self, result) -> None:
        """Display verification results in the text area."""
        from core.security import VerificationResult

        if not isinstance(result, VerificationResult):
            self.results_text.setPlainText("Ошибка: неверный формат результата")
            return

        output = []

        # Header
        if result.is_valid:
            output.append("✅ ПРОВЕРКА ПРОЙДЕНА УСПЕШНО")
            output.append("=" * 50)
            output.append("")
        else:
            output.append("❌ ОБНАРУЖЕНА ПОДДЕЛКА")
            output.append("=" * 50)
            output.append("")

        # Detailed checks
        output.append("Детальная проверка:")
        output.append(
            f"  • Подпись данных: {'✅ Действительна' if result.signature_valid else '❌ Недействительна'}"
        )
        output.append(
            f"  • Целостность изображения: {'✅ Не изменено' if result.image_authentic else '❌ Изменено'}"
        )
        output.append(
            f"  • Целостность данных: {'✅ Не изменены' if result.data_authentic else '❌ Изменены'}"
        )
        output.append(
            f"  • Совпадение хешей: {'✅ Совпадают' if result.hashes_match else '❌ Не совпадают'}"
        )
        output.append("")

        # Error message or answers
        if not result.is_valid:
            output.append("Причина отказа:")
            output.append(f"  {result.error_message}")
            output.append("")
            output.append("⚠️ ВНИМАНИЕ: Данный пакет ЛР был изменен после создания!")
            output.append("Возможные причины:")
            output.append("  • Изображение было отредактировано")
            output.append("  • Файл student.json был изменен")
            output.append("  • Файлы не соответствуют друг другу")
            output.append("  • Использованы файлы из разных пакетов")
        else:
            output.append("📋 ПРАВИЛЬНЫЕ ОТВЕТЫ:")
            output.append("-" * 50)
            output.append("")

            answers = result.answers
            if answers:
                # Sample ID
                output.append(f"ID образца: {answers.get('sample_id', 'N/A')}")
                output.append("")

                # Steel grade and carbon content
                steel_grade = answers.get("steel_grade")
                carbon = answers.get("carbon_content_calculated")
                if steel_grade:
                    output.append(f"Марка стали: {steel_grade}")
                if carbon is not None:
                    output.append(f"Содержание углерода: {carbon:.4f}%")
                output.append("")

                # Phase fractions
                phase_fractions = answers.get("phase_fractions", {})
                if phase_fractions:
                    output.append("Фазовый состав:")
                    for phase, fraction in phase_fractions.items():
                        output.append(
                            f"  • {phase}: {fraction:.4f} (или {fraction * 100:.2f}%)"
                        )
                    output.append("")

                # System
                system = answers.get("inferred_system")
                if system:
                    output.append(f"Система: {system}")
                    output.append("")

                # Image hash
                image_hash = answers.get("image_sha256")
                if image_hash:
                    output.append(f"SHA256 изображения: {image_hash[:16]}...")

        # Set text with appropriate styling
        self.results_text.setPlainText("\n".join(output))

        # Set text color based on result
        if result.is_valid:
            self.results_text.setStyleSheet("QTextEdit { color: #2E7D32; }")
        else:
            self.results_text.setStyleSheet("QTextEdit { color: #C62828; }")
