from __future__ import annotations

import unittest
from pathlib import Path

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None  # type: ignore
    Qt = None  # type: ignore


@unittest.skipIf(QApplication is None, "PySide6 is not available")
class LR1RussianLabelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _find_preset_index(self, window, stem: str) -> int:
        for idx in range(window.preset_combo.count()):
            data = window.preset_combo.itemData(idx)
            if isinstance(data, str) and data and Path(data).stem == stem:
                return idx
        return -1

    def test_preset_combo_uses_short_lr1_labels_and_tooltips(self) -> None:
        from ui_qt.sample_factory_window_v3 import (
            CAST_IRON_LIBRARY_GROUP_LABEL,
            CURRICULUM_TEMPLATE_GROUP_LABEL,
            LR1_TEMPLATE_GROUP_LABEL,
            SampleFactoryWindowV3,
            STEEL_LIBRARY_GROUP_LABEL,
        )

        window = SampleFactoryWindowV3()
        try:
            texts = [window.preset_combo.itemText(idx) for idx in range(window.preset_combo.count())]
            self.assertIn(LR1_TEMPLATE_GROUP_LABEL, texts)
            self.assertIn(STEEL_LIBRARY_GROUP_LABEL, texts)
            self.assertIn(CAST_IRON_LIBRARY_GROUP_LABEL, texts)
            self.assertIn(CURRICULUM_TEMPLATE_GROUP_LABEL, texts)

            idx = self._find_preset_index(window, "LR1_ASTM5")
            self.assertGreaterEqual(idx, 0)
            self.assertEqual(window.preset_combo.itemText(idx), "ЛР1 ASTM 5")
            self.assertNotEqual(window.preset_combo.itemText(idx), "LR1_ASTM5")
            tooltip = str(window.preset_combo.itemData(idx, Qt.ItemDataRole.ToolTipRole) or "")
            self.assertIn("крупное зерно", tooltip)
            self.assertIn("90 мкм", tooltip)
        finally:
            window.close()

    def test_lab_template_combo_groups_lr1_sample_library_and_research(self) -> None:
        from ui_qt.sample_factory_window_v3 import (
            CAST_IRON_LIBRARY_GROUP_LABEL,
            CURRICULUM_TEMPLATE_GROUP_LABEL,
            LR1_TEMPLATE_GROUP_LABEL,
            RESEARCH_OPTICS_GROUP_LABEL,
            SampleFactoryWindowV3,
            STEEL_LIBRARY_GROUP_LABEL,
        )

        window = SampleFactoryWindowV3()
        try:
            texts = [window.lab_template_combo.itemText(idx) for idx in range(window.lab_template_combo.count())]
            self.assertIn(LR1_TEMPLATE_GROUP_LABEL, texts)
            self.assertIn(CURRICULUM_TEMPLATE_GROUP_LABEL, texts)
            self.assertIn(STEEL_LIBRARY_GROUP_LABEL, texts)
            self.assertIn(CAST_IRON_LIBRARY_GROUP_LABEL, texts)
            self.assertIn(RESEARCH_OPTICS_GROUP_LABEL, texts)

            idx_lr1 = window.lab_template_combo.findData("LR1_ASTM8")
            self.assertGreaterEqual(idx_lr1, 0)
            self.assertEqual(window.lab_template_combo.itemText(idx_lr1), "ЛР1 ASTM 8")
            tooltip_lr1 = str(window.lab_template_combo.itemData(idx_lr1, Qt.ItemDataRole.ToolTipRole) or "")
            self.assertIn("очень мелкое зерно", tooltip_lr1)
            self.assertIn("32 мкм", tooltip_lr1)

            idx_template = window.lab_template_combo.findData("fe_c_hypoeutectoid_textbook")
            self.assertGreaterEqual(idx_template, 0)
            self.assertEqual(window.lab_template_combo.itemText(idx_template), "ЛР1 доэвтект.")
            tooltip_template = str(window.lab_template_combo.itemData(idx_template, Qt.ItemDataRole.ToolTipRole) or "")
            self.assertIn("доэвтектоидная сталь", tooltip_template)
        finally:
            window.close()
