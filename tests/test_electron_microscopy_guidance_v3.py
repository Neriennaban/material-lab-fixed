from __future__ import annotations

import unittest

from core.electron_microscopy_guidance import build_electron_microscopy_guidance


class ElectronMicroscopyGuidanceV3Tests(unittest.TestCase):
    def test_al_si_intermetallic_case_prefers_sem_bse(self) -> None:
        guidance = build_electron_microscopy_guidance(
            system="al-si",
            stage="eutectic",
            composition_wt={"Al": 87.4, "Si": 12.6},
            phase_fractions={"FCC_A1": 0.72, "SI": 0.20, "EUTECTIC_ALSI": 0.08},
            prep_summary={"relief_mean": 0.42, "contamination_level": 0.01},
            etch_summary={"reagent": "keller"},
            preset_metadata={},
            precipitation_state={},
        )
        self.assertEqual(str(guidance["primary_recommendation"]), "sem_bse")
        self.assertEqual(str(guidance["sem_guidance"]["preferred_mode"]), "backscattered_electron")
        self.assertTrue(bool(guidance["sem_guidance"]["avoid_etching_for_material_contrast"]))

    def test_pure_iron_case_stays_optical(self) -> None:
        guidance = build_electron_microscopy_guidance(
            system="fe-si",
            stage="recrystallized_ferrite",
            composition_wt={"Fe": 99.95},
            phase_fractions={"BCC_B2": 1.0},
            prep_summary={"relief_mean": 0.18, "contamination_level": 0.005},
            etch_summary={"reagent": "nital_2"},
            preset_metadata={},
            precipitation_state={},
        )
        self.assertEqual(str(guidance["primary_recommendation"]), "optical")
        self.assertFalse(bool(guidance["tem_guidance"]["recommended"]))

    def test_aged_al_cu_mg_case_marks_tem_candidate(self) -> None:
        guidance = build_electron_microscopy_guidance(
            system="al-cu-mg",
            stage="artificial_aged",
            composition_wt={"Al": 93.0, "Cu": 4.4, "Mg": 1.5},
            phase_fractions={"FCC_A1": 0.84, "THETA": 0.08, "S_PHASE": 0.08},
            prep_summary={"relief_mean": 0.24, "contamination_level": 0.01},
            etch_summary={"reagent": "keller"},
            preset_metadata={"sample_id": "AlCuMg_D16_aged_v3"},
            precipitation_state={"precipitate_family": "theta_s"},
        )
        self.assertTrue(bool(guidance["tem_guidance"]["recommended"]))
        self.assertTrue(bool(guidance["tem_guidance"]["parallel_face_thinning_required"]))
        self.assertTrue(bool(guidance["tem_guidance"]["ion_beam_final_thinning_candidate"]))

    def test_relief_dominant_case_prefers_sem_se(self) -> None:
        guidance = build_electron_microscopy_guidance(
            system="fe-c",
            stage="martensite",
            composition_wt={"Fe": 99.2, "C": 0.8},
            phase_fractions={"MARTENSITE": 0.88, "CEMENTITE": 0.12},
            prep_summary={
                "relief_mean": 0.68,
                "false_porosity_from_chipping_risk": 0.48,
                "outer_fragmented_layer_risk": 0.42,
                "contamination_level": 0.03,
            },
            etch_summary={"reagent": "nital_2"},
            preset_metadata={},
            precipitation_state={},
        )
        self.assertEqual(str(guidance["primary_recommendation"]), "sem_se")
        self.assertEqual(str(guidance["sem_guidance"]["preferred_mode"]), "secondary_electron")
        self.assertTrue(bool(guidance["sem_guidance"]["conductive_mounting_or_adhesive_helpful"]))


if __name__ == "__main__":
    unittest.main()
