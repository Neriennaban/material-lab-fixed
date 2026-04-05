from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, SynthesisProfileV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


def _base_thermal(points: list[tuple[float, float]]) -> list[ThermalPointV3]:
    return [ThermalPointV3(time_s=t, temperature_c=temp) for t, temp in points]


class PipelineV3SystemGeneratorsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = MetallographyPipelineV3()

    def _assert_system_mode(
        self, req: MetallographyRequestV3, expected_mode: str
    ) -> None:
        out = self.pipeline.generate(req)
        meta = dict(out.metadata)
        info = dict(meta.get("system_generator", {}))
        self.assertEqual(info.get("requested_mode"), "system_auto")
        self.assertEqual(info.get("resolved_mode"), expected_mode)
        self.assertIn("resolved_system", info)
        self.assertIn("resolved_stage", info)
        if expected_mode == "system_fe_c":
            fe_c_unified = dict(info.get("fe_c_unified", {}))
            self.assertTrue(bool(fe_c_unified.get("enabled", False)))

    def test_auto_mode_fe_c(self) -> None:
        req = MetallographyRequestV3(
            sample_id="fe_c_case",
            composition_wt={"Fe": 99.2, "C": 0.8},
            synthesis_profile=SynthesisProfileV3(system_generator_mode="system_auto"),
            resolution=(96, 96),
            seed=11,
        )
        req.thermal_program.points = _base_thermal(
            [(0.0, 20.0), (220.0, 780.0), (420.0, 780.0), (740.0, 30.0)]
        )
        self._assert_system_mode(req, "system_fe_c")

    def test_auto_mode_fe_si(self) -> None:
        req = MetallographyRequestV3(
            sample_id="fe_si_case",
            composition_wt={"Fe": 98.6, "Si": 1.4},
            synthesis_profile=SynthesisProfileV3(system_generator_mode="system_auto"),
            resolution=(96, 96),
            seed=12,
        )
        req.thermal_program.points = _base_thermal(
            [(0.0, 20.0), (200.0, 820.0), (380.0, 820.0), (1200.0, 25.0)]
        )
        self._assert_system_mode(req, "system_fe_si")

    def test_pure_iron_recrystallized_ferrite_emits_bright_baseline_metadata(
        self,
    ) -> None:
        payload = self.pipeline.load_preset("fe_pure_brightfield_v3")
        req = MetallographyRequestV3.from_dict(payload)
        req.resolution = (128, 128)
        out = self.pipeline.generate(req)
        self.assertEqual(str(out.metadata.get("inferred_system", "")), "fe-si")
        self.assertEqual(
            str(out.metadata.get("final_stage", "")), "recrystallized_ferrite"
        )
        pure_iron = dict(out.metadata.get("pure_iron_baseline", {}))
        self.assertTrue(bool(pure_iron.get("applied", False)))
        self.assertGreater(float(pure_iron.get("cleanliness_score", 0.0)), 0.55)
        self.assertGreater(float(pure_iron.get("dark_defect_suppression", 0.0)), 0.45)
        recommendation = dict(out.metadata.get("pure_iron_optical_recommendation", {}))
        self.assertEqual(str(recommendation.get("default_mode", "")), "brightfield")
        self.assertEqual(
            str(recommendation.get("polarized_limit", "")),
            "cubic_isotropic_negative_control",
        )
        shared_recommendation = dict(out.metadata.get("optical_recommendation", {}))
        self.assertEqual(
            str(shared_recommendation.get("default_mode", "")), "brightfield"
        )
        self.assertEqual(
            str(shared_recommendation.get("secondary_mode", "")), "phase_contrast"
        )
        self.assertEqual(
            str(out.metadata.get("pure_iron_electropolish_profile", "")),
            "pure_iron_electropolish",
        )
        self.assertTrue(bool(out.metadata.get("single_phase_negative_control", False)))
        self.assertFalse(
            bool(out.metadata.get("multiphase_separability_applicable", True))
        )

    def test_zero_carbon_fe_c_ferrite_uses_power_voronoi_generator(self) -> None:
        req = MetallographyRequestV3(
            sample_id="fe_c_pure_ferrite",
            composition_wt={"Fe": 100.0, "C": 0.0},
            synthesis_profile=SynthesisProfileV3(system_generator_mode="system_auto"),
            resolution=(128, 128),
            seed=21,
        )
        req.thermal_program.points = _base_thermal(
            [(0.0, 20.0), (300.0, 820.0), (540.0, 820.0), (1200.0, 25.0)]
        )
        out = self.pipeline.generate(req)
        self.assertEqual(str(out.metadata.get("inferred_system", "")), "fe-c")
        self.assertEqual(str(out.metadata.get("final_stage", "")), "ferrite")
        self.assertTrue(
            bool(out.metadata.get("pure_iron_baseline", {}).get("applied", False))
        )
        morphology_trace = dict(
            out.metadata.get("fe_c_phase_render", {}).get("morphology_trace", {})
        )
        self.assertEqual(
            str(morphology_trace.get("generator", "")), "pure_ferrite_power_voronoi_v1"
        )
        self.assertEqual(
            str(morphology_trace.get("family", "")), "pure_ferrite_power_voronoi"
        )

    def test_auto_mode_al_si(self) -> None:
        req = MetallographyRequestV3(
            sample_id="al_si_case",
            composition_wt={"Al": 87.4, "Si": 12.6},
            synthesis_profile=SynthesisProfileV3(system_generator_mode="system_auto"),
            resolution=(96, 96),
            seed=13,
        )
        req.thermal_program.points = _base_thermal(
            [(0.0, 20.0), (180.0, 730.0), (260.0, 730.0), (900.0, 30.0)]
        )
        self._assert_system_mode(req, "system_al_si")

    def test_auto_mode_cu_zn(self) -> None:
        req = MetallographyRequestV3(
            sample_id="cu_zn_case",
            composition_wt={"Cu": 60.0, "Zn": 40.0},
            synthesis_profile=SynthesisProfileV3(system_generator_mode="system_auto"),
            resolution=(96, 96),
            seed=14,
        )
        req.thermal_program.points = _base_thermal(
            [(0.0, 20.0), (180.0, 700.0), (360.0, 700.0), (920.0, 30.0)]
        )
        self._assert_system_mode(req, "system_cu_zn")

    def test_auto_mode_al_cu_mg(self) -> None:
        req = MetallographyRequestV3(
            sample_id="al_cu_mg_case",
            composition_wt={"Al": 94.1, "Cu": 4.4, "Mg": 1.5},
            synthesis_profile=SynthesisProfileV3(system_generator_mode="system_auto"),
            resolution=(96, 96),
            seed=15,
        )
        req.thermal_program.points = _base_thermal(
            [
                (0.0, 20.0),
                (180.0, 530.0),
                (240.0, 530.0),
                (980.0, 180.0),
                (1300.0, 180.0),
                (1700.0, 30.0),
            ]
        )
        self._assert_system_mode(req, "system_al_cu_mg")

    def test_auto_mode_custom_fallback(self) -> None:
        req = MetallographyRequestV3(
            sample_id="custom_case",
            composition_wt={"Ni": 62.0, "Cr": 24.0, "Mo": 9.0, "W": 5.0},
            synthesis_profile=SynthesisProfileV3(system_generator_mode="system_auto"),
            resolution=(96, 96),
            seed=16,
        )
        req.thermal_program.points = _base_thermal(
            [(0.0, 20.0), (240.0, 980.0), (480.0, 980.0), (1300.0, 25.0)]
        )
        out = self.pipeline.generate(req)
        info = dict(out.metadata.get("system_generator", {}))
        self.assertEqual(info.get("resolved_mode"), "system_custom")
        self.assertTrue(bool(info.get("fallback_used")))


if __name__ == "__main__":
    unittest.main()
