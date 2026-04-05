from __future__ import annotations

import unittest

from core.generator_calphad_phase import generate_calphad_phase_structure


class PhaseVisibilityV3Tests(unittest.TestCase):
    def _assert_visibility(self, out: dict, tolerance_pct: float) -> None:
        meta = out.get("metadata", {})
        self.assertIsInstance(meta, dict)

        vis = meta.get("phase_visibility_report", {})
        self.assertIsInstance(vis, dict)
        self.assertGreater(float(vis.get("separability_score", 0.0)), 0.14)
        self.assertIn("within_tolerance", vis)

        err = vis.get("fraction_error_pct", {})
        self.assertIsInstance(err, dict)
        self.assertTrue(err)
        self.assertLessEqual(max(float(v) for v in err.values()), tolerance_pct + 1e-6)

        trace = meta.get("engineering_trace", {})
        self.assertIsInstance(trace, dict)
        self.assertEqual(trace.get("generation_mode"), "edu_engineering")
        self.assertIn("phase_emphasis_style", trace)
        self.assertEqual(float(trace.get("phase_fraction_tolerance_pct", -1.0)), tolerance_pct)

    def test_fe_c_visibility_edu_engineering(self) -> None:
        out = generate_calphad_phase_structure(
            size=(160, 160),
            seed=3201,
            system="fe-c",
            phase_fractions={"FERRITE": 0.7, "PEARLITE": 0.3},
            generation_mode="edu_engineering",
            phase_emphasis_style="contrast_texture",
            phase_fraction_tolerance_pct=20.0,
        )
        self._assert_visibility(out, tolerance_pct=20.0)

    def test_al_si_visibility_edu_engineering(self) -> None:
        out = generate_calphad_phase_structure(
            size=(160, 160),
            seed=3202,
            system="al-si",
            phase_fractions={"FCC_A1": 0.8, "SI": 0.2},
            generation_mode="edu_engineering",
            phase_emphasis_style="contrast_texture",
            phase_fraction_tolerance_pct=20.0,
        )
        self._assert_visibility(out, tolerance_pct=20.0)


if __name__ == "__main__":
    unittest.main()

