from __future__ import annotations

import unittest

from core.materials_hybrid import (
    calculate_hybrid_heat_treatment,
    estimate_hybrid_properties,
    supports_hybrid_properties,
)


class MaterialsHybridIntegrationTests(unittest.TestCase):
    def test_supports_hybrid_for_fe_c_only(self) -> None:
        self.assertTrue(supports_hybrid_properties("fe-c", {"Fe": 99.2, "C": 0.8}))
        self.assertFalse(
            supports_hybrid_properties("al-cu-mg", {"Al": 93.1, "Cu": 4.4, "Mg": 1.5})
        )

    def test_alpha_pearlite_uses_textbook_reference_without_fallback(self) -> None:
        props = estimate_hybrid_properties(
            composition={"Fe": 99.55, "C": 0.45},
            final_stage="alpha_pearlite",
            effect={
                "grain_size_factor": 0.0,
                "elongation_factor": 0.0,
                "texture_strength": 0.0,
                "dislocation_proxy": 0.0,
                "precipitation_level": 0.0,
                "segregation_level": 0.0,
                "residual_stress": 0.0,
                "porosity_factor": 0.0,
            },
            overlay_rules={},
        )
        self.assertIsNotNone(props)
        assert props is not None
        self.assertEqual(
            props["property_model_source"], "hybrid_textbook_calculator_v1"
        )
        self.assertEqual(props["reference_dataset"], "textbook_material_properties")
        self.assertFalse(bool(props["compatibility_overlay_used"]))
        self.assertFalse(bool(props["fallback_used"]))
        self.assertGreater(float(props["hv_estimate"]), 150.0)
        self.assertLess(float(props["hv_estimate"]), 220.0)
        self.assertGreater(float(props["uts_estimate_mpa"]), 550.0)
        self.assertLess(float(props["uts_estimate_mpa"]), 750.0)

    def test_tempered_medium_uses_overlay_metadata(self) -> None:
        props = estimate_hybrid_properties(
            composition={"Fe": 99.2, "C": 0.8},
            final_stage="tempered_medium",
            effect={
                "grain_size_factor": -0.1,
                "elongation_factor": 0.0,
                "texture_strength": 0.0,
                "dislocation_proxy": 0.35,
                "precipitation_level": 0.12,
                "segregation_level": 0.0,
                "residual_stress": 0.08,
                "porosity_factor": 0.0,
            },
            overlay_rules={
                "defaults": {"ductility_base": "medium"},
                "systems": {
                    "fe-c": {
                        "hv_coeff": {
                            "dislocation_proxy": 75.0,
                            "precipitation_level": 45.0,
                            "grain_size_factor": -22.0,
                            "residual_stress": 15.0,
                        },
                        "stage_adjust": {
                            "tempered_medium": {"ductility": "medium-low"},
                        },
                    }
                },
                "ductility_thresholds_hv": [{"max_hv": 1000.0, "label": "low"}],
            },
        )
        self.assertIsNotNone(props)
        assert props is not None
        self.assertTrue(bool(props["compatibility_overlay_used"]))
        self.assertFalse(bool(props["fallback_used"]))
        self.assertEqual(props["ductility_class"], "medium-low")
        self.assertGreater(float(props["hv_estimate"]), 430.0)
        self.assertGreater(float(props["uts_estimate_mpa"]), 1400.0)

    def test_heat_treatment_tempering_labels_match_temperature_order(self) -> None:
        payload = calculate_hybrid_heat_treatment(composition={"Fe": 99.2, "C": 0.8})
        self.assertEqual(payload["tempering_low"]["name_ru"], "Низкий отпуск")
        self.assertEqual(payload["tempering_medium"]["name_ru"], "Средний отпуск")
        self.assertEqual(payload["tempering_high"]["name_ru"], "Высокий отпуск")

    def test_cast_iron_ledeburite_path_is_supported(self) -> None:
        props = estimate_hybrid_properties(
            composition={"Fe": 94.2, "C": 3.2, "Si": 2.0},
            final_stage="ledeburite",
            effect={
                "grain_size_factor": 0.0,
                "elongation_factor": 0.0,
                "texture_strength": 0.0,
                "dislocation_proxy": 0.0,
                "precipitation_level": 0.0,
                "segregation_level": 0.0,
                "residual_stress": 0.0,
                "porosity_factor": 0.0,
            },
            overlay_rules={},
        )
        self.assertIsNotNone(props)
        assert props is not None
        self.assertEqual(
            props["property_model_source"], "hybrid_textbook_calculator_v1"
        )
        self.assertFalse(bool(props["fallback_used"]))
        self.assertGreater(float(props["hv_estimate"]), 400.0)

    def test_unknown_stage_returns_none_for_processing_fallback(self) -> None:
        props = estimate_hybrid_properties(
            composition={"Fe": 99.2, "C": 0.8},
            final_stage="mystery_stage",
            effect={},
            overlay_rules={},
        )
        self.assertIsNone(props)


if __name__ == "__main__":
    unittest.main()
