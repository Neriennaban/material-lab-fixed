import sys
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from core.contracts_v2 import ValidationReport
from core.contracts_v3 import GenerationOutputV3


class GeneratorTeacherModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)

    def _make_output(self) -> GenerationOutputV3:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        gray = np.zeros((32, 32), dtype=np.uint8)
        return GenerationOutputV3(
            image_rgb=image,
            image_gray=gray,
            phase_masks=None,
            feature_masks=None,
            prep_maps=None,
            metadata={
                "sample_id": "sample_1",
                "final_stage": "alpha_pearlite",
                "high_resolution_render": {"requested_resolution": [2048, 2048]},
                "property_indicators": {"hv_estimate": 250},
                "phase_model": {"engine": "auto"},
                "phase_model_report": {
                    "blended_phase_fractions": {"FERRITE": 0.65, "PEARLITE": 0.35}
                },
                "composition_effect": {},
                "quality_metrics": {},
                "system_resolution": {},
                "system_generator": {},
                "textbook_profile": {},
            },
            validation_report=ValidationReport(is_valid=True),
        )

    def test_student_final_render_shows_only_three_lines(self) -> None:
        from ui_qt.sample_factory_window_v3 import SampleFactoryWindowV3

        win = SampleFactoryWindowV3()
        try:
            win.current_mode = "student"
            win.is_final_render = True
            win._show_output(self._make_output())

            lines = [
                line
                for line in win.info_text.toPlainText().splitlines()
                if line.strip()
            ]
            self.assertEqual(
                lines,
                [
                    "Название: Феррит-перлит",
                    "Разрешение: 2048 x 2048",
                    "Твердость: HV 250",
                ],
            )
            self.assertEqual(
                win.qc_text.toPlainText(), "Служебные данные скрыты в режиме студента."
            )
            self.assertFalse(win.intermediate_renders_group.isVisible())
        finally:
            win.close()
            win.deleteLater()

    def test_teacher_final_render_shows_full_block(self) -> None:
        from ui_qt.sample_factory_window_v3 import SampleFactoryWindowV3

        win = SampleFactoryWindowV3()
        try:
            win.current_mode = "teacher"
            win.is_final_render = True
            win._show_output(self._make_output())

            text = win.info_text.toPlainText()
            self.assertIn("Фазовая модель:", text)
            self.assertIn("Образец: sample_1", text)
        finally:
            win.close()
            win.deleteLater()
