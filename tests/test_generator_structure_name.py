import sys
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from core.contracts_v2 import ValidationReport
from core.contracts_v3 import GenerationOutputV3


class GeneratorStructureNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)

    def _window(self):
        from ui_qt.sample_factory_window_v3 import SampleFactoryWindowV3

        return SampleFactoryWindowV3()

    def _output(self, metadata):
        return GenerationOutputV3(
            image_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            image_gray=np.zeros((8, 8), dtype=np.uint8),
            phase_masks=None,
            feature_masks=None,
            prep_maps=None,
            metadata=metadata,
            validation_report=ValidationReport(is_valid=True),
        )

    def test_alpha_pearlite_maps_to_ferrite_pearlite(self):
        win = self._window()
        output = self._output({"final_stage": "alpha_pearlite"})
        self.assertEqual(win._resolve_student_structure_name(output), "Феррит-перлит")

    def test_martensite_maps_to_martensite(self):
        win = self._window()
        output = self._output({"final_stage": "martensite"})
        self.assertEqual(win._resolve_student_structure_name(output), "Мартенсит")

    def test_fallback_uses_dominant_phase_fractions(self):
        win = self._window()
        output = self._output(
            {
                "phase_model_report": {
                    "blended_phase_fractions": {
                        "FERRITE": 0.51,
                        "PEARLITE": 0.32,
                        "CEMENTITE": 0.08,
                    }
                }
            }
        )
        self.assertEqual(win._resolve_student_structure_name(output), "Феррит-перлит")
