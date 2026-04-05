from __future__ import annotations

import unittest

import numpy as np

from core.generator_calphad_phase import generate_calphad_phase_structure


class GeneratorCalphadPhaseTests(unittest.TestCase):
    def test_deterministic_by_seed_and_composition(self) -> None:
        kwargs = {
            "size": (96, 96),
            "seed": 1234,
            "system": "fe-si",
            "phase_fractions": {"BCC_B2": 1.0},
            "composition_wt": {"Fe": 99.5, "Si": 0.5},
            "composition_sensitivity_mode": "realistic",
        }
        a = generate_calphad_phase_structure(**kwargs)
        b = generate_calphad_phase_structure(**kwargs)
        self.assertTrue(np.array_equal(a["image"], b["image"]))

    def test_single_phase_changes_with_composition(self) -> None:
        base_kwargs = {
            "size": (128, 128),
            "seed": 17,
            "system": "fe-si",
            "phase_fractions": {"BCC_B2": 1.0},
            "composition_sensitivity_mode": "realistic",
        }
        low_si = generate_calphad_phase_structure(
            **base_kwargs,
            composition_wt={"Fe": 99.5, "Si": 0.5},
        )
        high_si = generate_calphad_phase_structure(
            **base_kwargs,
            composition_wt={"Fe": 95.0, "Si": 5.0},
        )
        mae = float(np.abs(low_si["image"].astype(np.float32) - high_si["image"].astype(np.float32)).mean())
        self.assertGreater(mae, 1.0)
        self.assertIn("composition_effect", low_si.get("metadata", {}))
        self.assertIn("composition_effect", high_si.get("metadata", {}))

    def test_sensitivity_mode_ordering(self) -> None:
        modes = ["realistic", "educational", "high_contrast"]
        maes: list[float] = []
        for mode in modes:
            low = generate_calphad_phase_structure(
                size=(128, 128),
                seed=991,
                system="fe-si",
                phase_fractions={"BCC_B2": 1.0},
                composition_wt={"Fe": 99.5, "Si": 0.5},
                composition_sensitivity_mode=mode,
            )
            high = generate_calphad_phase_structure(
                size=(128, 128),
                seed=991,
                system="fe-si",
                phase_fractions={"BCC_B2": 1.0},
                composition_wt={"Fe": 95.0, "Si": 5.0},
                composition_sensitivity_mode=mode,
            )
            maes.append(float(np.abs(low["image"].astype(np.float32) - high["image"].astype(np.float32)).mean()))

        self.assertGreater(maes[1], maes[0])
        self.assertGreater(maes[2], maes[1])

    def test_liquid_masks_present(self) -> None:
        out = generate_calphad_phase_structure(
            size=(80, 80),
            seed=77,
            system="al-si",
            phase_fractions={"LIQUID": 0.35, "FCC_A1": 0.65},
        )
        masks = out.get("phase_masks", {})
        self.assertIn("L", masks)
        self.assertIn("solid", masks)
        self.assertEqual(out["image"].shape, (80, 80))

    def test_edu_engineering_phase_visibility_and_tolerance_fe_c(self) -> None:
        out = generate_calphad_phase_structure(
            size=(128, 128),
            seed=2026,
            system="fe-c",
            phase_fractions={"FERRITE": 0.68, "PEARLITE": 0.32},
            generation_mode="edu_engineering",
            phase_emphasis_style="contrast_texture",
            phase_fraction_tolerance_pct=20.0,
        )
        vis = out.get("metadata", {}).get("phase_visibility_report", {})
        self.assertIsInstance(vis, dict)
        self.assertIn("separability_score", vis)
        self.assertGreater(float(vis.get("separability_score", 0.0)), 0.14)
        err = vis.get("fraction_error_pct", {})
        self.assertIsInstance(err, dict)
        self.assertTrue(err)
        self.assertLessEqual(max(float(v) for v in err.values()), 20.0 + 1e-6)

    def test_edu_engineering_textbook_profile_has_higher_readability(self) -> None:
        out = generate_calphad_phase_structure(
            size=(160, 160),
            seed=2227,
            system="fe-c",
            phase_fractions={"FERRITE": 0.58, "PEARLITE": 0.32, "CEMENTITE": 0.10},
            generation_mode="edu_engineering",
            phase_emphasis_style="contrast_texture",
            phase_fraction_tolerance_pct=20.0,
            visual_profile_id="textbook_steel_bw",
        )
        vis = out.get("metadata", {}).get("phase_visibility_report", {})
        self.assertIsInstance(vis, dict)
        self.assertGreater(float(vis.get("separability_score", 0.0)), 0.20)

    def test_edu_engineering_phase_visibility_and_tolerance_al_si(self) -> None:
        out = generate_calphad_phase_structure(
            size=(128, 128),
            seed=2027,
            system="al-si",
            phase_fractions={"FCC_A1": 0.82, "SI": 0.18},
            generation_mode="edu_engineering",
            phase_emphasis_style="contrast_texture",
            phase_fraction_tolerance_pct=20.0,
        )
        vis = out.get("metadata", {}).get("phase_visibility_report", {})
        self.assertIsInstance(vis, dict)
        self.assertGreater(float(vis.get("separability_score", 0.0)), 0.14)
        err = vis.get("fraction_error_pct", {})
        self.assertIsInstance(err, dict)
        self.assertTrue(err)
        self.assertLessEqual(max(float(v) for v in err.values()), 20.0 + 1e-6)

    def test_edu_engineering_is_deterministic(self) -> None:
        kwargs = {
            "size": (96, 96),
            "seed": 606,
            "system": "fe-c",
            "phase_fractions": {"FERRITE": 0.7, "PEARLITE": 0.3},
            "generation_mode": "edu_engineering",
            "phase_emphasis_style": "contrast_texture",
            "phase_fraction_tolerance_pct": 20.0,
        }
        a = generate_calphad_phase_structure(**kwargs)
        b = generate_calphad_phase_structure(**kwargs)
        self.assertTrue(np.array_equal(a["image"], b["image"]))


if __name__ == "__main__":
    unittest.main()
