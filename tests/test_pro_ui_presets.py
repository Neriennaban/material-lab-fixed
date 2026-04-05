from __future__ import annotations

import json
import unittest
from pathlib import Path

from ui_qt.sample_factory_window_v3 import (
    CAST_IRON_SAMPLE_LIBRARY_PRESETS,
    GENERATION_MODE_OPTIONS,
    LAB_RESEARCH_TEMPLATE_PRESETS,
    LR1_SAMPLE_LIBRARY_PRESETS,
    LAB_TEMPLATE_PRESETS,
    OPTICAL_MODE_OPTIONS,
    STEEL_SAMPLE_LIBRARY_PRESETS,
)

try:
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None  # type: ignore


class ProUIPresetsTests(unittest.TestCase):
    def test_ui_lists_pro_realistic_generation_mode(self) -> None:
        keys = {key for key, _ in GENERATION_MODE_OPTIONS}
        self.assertIn("pro_realistic", keys)
        optical_keys = {key for key, _ in OPTICAL_MODE_OPTIONS}
        self.assertIn("brightfield", optical_keys)
        self.assertIn("darkfield", optical_keys)
        self.assertIn("polarized", optical_keys)
        self.assertIn("phase_contrast", optical_keys)
        self.assertIn("dic", optical_keys)
        self.assertIn("magnetic_etching", optical_keys)

    def test_lab_templates_include_pro_fe_c_presets(self) -> None:
        preset_keys = {key for _, key in LAB_TEMPLATE_PRESETS}
        self.assertIn("fe_c_hypoeutectoid_pro_realistic", preset_keys)
        self.assertIn("fe_c_eutectoid_pro_realistic", preset_keys)
        self.assertIn("fe_c_hypereutectoid_pro_realistic", preset_keys)

    def test_lab_templates_include_lr1_sample_library_presets(self) -> None:
        preset_keys = {key for _, key in LR1_SAMPLE_LIBRARY_PRESETS}
        self.assertEqual(
            preset_keys,
            {"LR1_ASTM5", "LR1_ASTM6", "LR1_ASTM7", "LR1_ASTM8"},
        )

    def test_lab_templates_include_sample_library_presets(self) -> None:
        steel_keys = {key for _, key in STEEL_SAMPLE_LIBRARY_PRESETS}
        cast_keys = {key for _, key in CAST_IRON_SAMPLE_LIBRARY_PRESETS}
        self.assertEqual(
            steel_keys,
            {"steel45_normalized_textbook", "steel45_improved_textbook", "steel_u8_tool_textbook"},
        )
        self.assertEqual(cast_keys, {"cast_iron_grey_textbook"})

    def test_lab_templates_include_research_optics_presets(self) -> None:
        preset_keys = {key for _, key in LAB_RESEARCH_TEMPLATE_PRESETS}
        self.assertIn("fe_c_eutectoid_research_optics_bessel", preset_keys)
        self.assertIn("fe_c_eutectoid_research_optics_stir", preset_keys)
        self.assertIn("fe_c_eutectoid_research_optics_hybrid", preset_keys)

    def test_pro_preset_files_exist_and_set_mode(self) -> None:
        for name in (
            "fe_c_hypoeutectoid_pro_realistic",
            "fe_c_eutectoid_pro_realistic",
            "fe_c_hypereutectoid_pro_realistic",
        ):
            path = Path("presets_v3") / f"{name}.json"
            self.assertTrue(path.exists(), msg=str(path))
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            self.assertEqual(
                str(payload.get("synthesis_profile", {}).get("generation_mode", "")),
                "pro_realistic",
            )

    def test_lr1_presets_use_safe_non_quench_defaults(self) -> None:
        preset_names = (
            "fe_c_hypoeutectoid_textbook",
            "fe_c_eutectoid_textbook",
            "fe_c_hypereutectoid_textbook",
            "fe_c_hypoeutectoid_pro_realistic",
            "fe_c_eutectoid_pro_realistic",
            "fe_c_hypereutectoid_pro_realistic",
            "alsi_eutectic_textbook",
            "brass_alpha_beta_textbook",
        )
        for name in preset_names:
            payload = json.loads((Path("presets_v3") / f"{name}.json").read_text(encoding="utf-8-sig"))
            quench = payload.get("thermal_program", {}).get("quench", {})
            self.assertEqual(str(quench.get("medium_code", "")), "air", msg=name)
            self.assertEqual(float(quench.get("quench_time_s", -1.0)), 0.0, msg=name)
            self.assertEqual(float(quench.get("bath_temperature_c", -1.0)), 25.0, msg=name)

    def test_sample_library_presets_have_russian_display_metadata(self) -> None:
        names = (
            "LR1_ASTM5",
            "steel45_normalized_textbook",
            "steel45_improved_textbook",
            "steel_u8_tool_textbook",
            "cast_iron_grey_textbook",
        )
        for name in names:
            payload = json.loads((Path("presets_v3") / f"{name}.json").read_text(encoding="utf-8-sig"))
            meta = dict(payload.get("metadata", {}))
            self.assertTrue(str(meta.get("display_name_ru", "")).strip(), msg=f"{name}: display_name_ru")
            self.assertTrue(str(meta.get("short_label_ru", "")).strip(), msg=f"{name}: short_label_ru")
            self.assertTrue(str(meta.get("group_ru", "")).strip(), msg=f"{name}: group_ru")
            self.assertTrue(str(meta.get("expected_properties_source", "")).strip(), msg=f"{name}: expected_properties_source")

    @unittest.skipIf(QApplication is None, "PySide6 is not available")
    def test_ui_defaults_to_lr1_safe_non_quench_medium(self) -> None:
        from ui_qt.sample_factory_window_v3 import SampleFactoryWindowV3

        _app = QApplication.instance() or QApplication([])
        window = SampleFactoryWindowV3()
        try:
            self.assertEqual(str(window.quench_medium_combo.currentData() or ""), "air")
            self.assertEqual(float(window.quench_time_spin.value()), 0.0)
            self.assertEqual(float(window.quench_bath_temp_spin.value()), 25.0)
            self.assertEqual(str(window.ms_psf_profile_combo.currentData() or ""), "standard")
            self.assertEqual(float(window.ms_psf_strength_spin.value()), 0.0)
        finally:
            window.close()

    @unittest.skipIf(QApplication is None, "PySide6 is not available")
    def test_applying_lr1_pro_template_keeps_textbook_safe_request(self) -> None:
        from ui_qt.sample_factory_window_v3 import SampleFactoryWindowV3

        _app = QApplication.instance() or QApplication([])
        window = SampleFactoryWindowV3()
        try:
            idx = window.lab_template_combo.findData("fe_c_eutectoid_pro_realistic")
            self.assertGreaterEqual(idx, 0)
            window.lab_template_combo.setCurrentIndex(idx)
            window._apply_lab_template()

            self.assertEqual(str(window.visual_standard_combo.currentData() or ""), "textbook_bw")
            self.assertEqual(str(window.synth_generation_mode_combo.currentData() or ""), "edu_engineering")
            self.assertEqual(str(window.quench_medium_combo.currentData() or ""), "air")
            self.assertEqual(float(window.quench_time_spin.value()), 0.0)
            self.assertEqual(float(window.quench_bath_temp_spin.value()), 25.0)

            request = window._collect_request(final_render=False, for_preview_only=False)
            self.assertEqual(str(request.synthesis_profile.generation_mode), "edu_engineering")
            self.assertEqual(str(request.synthesis_profile.system_generator_mode), "system_fe_c")
            self.assertEqual(str(request.synthesis_profile.profile_id), "textbook_steel_bw")
            self.assertEqual(str(request.microscope_profile.get("psf_profile", "")), "standard")
        finally:
            window.close()

    @unittest.skipIf(QApplication is None, "PySide6 is not available")
    def test_loading_lr1_astm_preset_preserves_hidden_material_context(self) -> None:
        from ui_qt.sample_factory_window_v3 import SampleFactoryWindowV3

        _app = QApplication.instance() or QApplication([])
        window = SampleFactoryWindowV3()
        try:
            idx = -1
            for row in range(window.preset_combo.count()):
                data = window.preset_combo.itemData(row)
                if isinstance(data, str) and data.endswith("LR1_ASTM5.json"):
                    idx = row
                    break
            self.assertGreaterEqual(idx, 0)
            window.preset_combo.setCurrentIndex(idx)
            window._load_selected_preset()

            request = window._collect_request(final_render=False, for_preview_only=False)
            self.assertEqual(str(request.lab_work), "LR1_grain_size")
            self.assertEqual(str(request.material_grade), "Учебный образец ЛР1")
            self.assertEqual(float(request.target_astm_grain_size or 0.0), 5.0)
            self.assertTrue(bool(request.expected_properties))
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
