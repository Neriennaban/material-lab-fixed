from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified
from core.metallography_v3.system_generators.generator_al_cu_mg import generate_al_cu_mg
from core.metallography_v3.system_generators.generator_al_si import generate_al_si
from core.metallography_v3.system_generators.generator_cu_zn import generate_cu_zn
from core.metallography_v3.system_generators.generator_fe_si import generate_fe_si
from core.metallography_v3.transformation_state import build_transformation_state


class TransformationStateV3Tests(unittest.TestCase):
    def test_fe_c_faster_cooling_shifts_to_martensite_and_refines_pearlite(self) -> None:
        slow = build_transformation_state(
            inferred_system="fe-c",
            stage="pearlite",
            composition_wt={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={
                "temperature_min_c": 20.0,
                "temperature_max_c": 860.0,
                "temperature_end_c": 20.0,
                "max_effective_cooling_rate_c_per_s": -1.2,
                "operation_inference": {"has_quench": False, "has_temper": False},
            },
            quench_summary={"effect_applied": False, "medium_code_resolved": "air"},
        )
        fast = build_transformation_state(
            inferred_system="fe-c",
            stage="martensite",
            composition_wt={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="water_quench"),
            thermal_summary={
                "temperature_min_c": 20.0,
                "temperature_max_c": 860.0,
                "temperature_end_c": 20.0,
                "max_effective_cooling_rate_c_per_s": -120.0,
                "operation_inference": {"has_quench": True, "has_temper": False},
            },
            quench_summary={
                "effect_applied": True,
                "medium_code_resolved": "water_20",
                "as_quenched_prediction": {"retained_austenite_fraction_est": 0.12},
            },
        )

        self.assertGreater(
            float(fast["transformation_trace"]["martensite_fraction"]),
            float(slow["transformation_trace"]["martensite_fraction"]),
        )
        self.assertLess(
            float(fast["morphology_state"]["interlamellar_spacing_px"]),
            float(slow["morphology_state"]["interlamellar_spacing_px"]),
        )
        self.assertLessEqual(
            float(fast["transformation_trace"]["ferrite_fraction"]),
            float(slow["transformation_trace"]["ferrite_fraction"]),
        )

    def test_fe_c_tempering_reduces_martensite_and_increases_carbide_scale(self) -> None:
        low_temper = build_transformation_state(
            inferred_system="fe-c",
            stage="tempered_medium",
            composition_wt={"Fe": 99.1, "C": 0.9},
            processing=ProcessingState(
                temperature_c=220.0,
                cooling_mode="tempered",
                aging_hours=1.0,
                aging_temperature_c=220.0,
            ),
            thermal_summary={
                "temperature_min_c": 20.0,
                "temperature_max_c": 860.0,
                "temperature_end_c": 220.0,
                "operation_inference": {
                    "has_quench": True,
                    "has_temper": True,
                    "temper_peak_temperature_c": 220.0,
                },
            },
            quench_summary={"effect_applied": True, "medium_code_resolved": "water_20"},
        )
        high_temper = build_transformation_state(
            inferred_system="fe-c",
            stage="tempered_medium",
            composition_wt={"Fe": 99.1, "C": 0.9},
            processing=ProcessingState(
                temperature_c=600.0,
                cooling_mode="tempered",
                aging_hours=16.0,
                aging_temperature_c=600.0,
            ),
            thermal_summary={
                "temperature_min_c": 20.0,
                "temperature_max_c": 860.0,
                "temperature_end_c": 600.0,
                "operation_inference": {
                    "has_quench": True,
                    "has_temper": True,
                    "temper_peak_temperature_c": 600.0,
                },
            },
            quench_summary={"effect_applied": True, "medium_code_resolved": "water_20"},
        )

        self.assertLess(
            float(high_temper["transformation_trace"]["martensite_fraction"]),
            float(low_temper["transformation_trace"]["martensite_fraction"]),
        )
        self.assertGreater(
            float(high_temper["precipitation_state"]["carbide_scale_px"]),
            float(low_temper["precipitation_state"]["carbide_scale_px"]),
        )
        self.assertGreater(
            float(high_temper["precipitation_state"]["recovery_level"]),
            float(low_temper["precipitation_state"]["recovery_level"]),
        )

    def test_al_si_cooling_and_modifier_effects_are_reflected(self) -> None:
        slow = build_transformation_state(
            inferred_system="al-si",
            stage="eutectic",
            composition_wt={"Al": 87.4, "Si": 12.6},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
        )
        fast_modified = build_transformation_state(
            inferred_system="al-si",
            stage="eutectic",
            composition_wt={"Al": 87.39, "Si": 12.6, "Sr": 0.01},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="water_quench"),
        )

        self.assertLess(
            float(fast_modified["morphology_state"]["sdas_px"]),
            float(slow["morphology_state"]["sdas_px"]),
        )
        self.assertLess(
            float(fast_modified["morphology_state"]["eutectic_scale_px"]),
            float(slow["morphology_state"]["eutectic_scale_px"]),
        )
        self.assertIn("modified", str(fast_modified["morphology_state"]["eutectic_si_modifier"]))

    def test_al_cu_mg_overaged_state_is_coarser_and_weaker_than_peak_aged(self) -> None:
        peak = build_transformation_state(
            inferred_system="al-cu-mg",
            stage="artificial_aged",
            composition_wt={"Al": 94.1, "Cu": 4.4, "Mg": 1.5},
            processing=ProcessingState(temperature_c=180.0, cooling_mode="aged", aging_hours=12.0, aging_temperature_c=180.0),
            quench_summary={"medium_code_resolved": "water_20"},
        )
        overaged = build_transformation_state(
            inferred_system="al-cu-mg",
            stage="overaged",
            composition_wt={"Al": 93.0, "Cu": 5.0, "Mg": 1.5, "Si": 0.5},
            processing=ProcessingState(temperature_c=220.0, cooling_mode="aged", aging_hours=36.0, aging_temperature_c=220.0),
            quench_summary={"medium_code_resolved": "oil_20_80"},
        )

        self.assertGreater(
            float(overaged["precipitation_state"]["precipitate_scale_px"]),
            float(peak["precipitation_state"]["precipitate_scale_px"]),
        )
        self.assertGreaterEqual(
            float(overaged["precipitation_state"]["pfz_width_px"]),
            float(peak["precipitation_state"]["pfz_width_px"]),
        )
        self.assertLess(
            float(overaged["precipitation_state"]["peak_strength_fraction"]),
            float(peak["precipitation_state"]["peak_strength_fraction"]),
        )

    def test_cu_zn_and_fe_si_state_tracks_deformation_and_recrystallization(self) -> None:
        brass_cold = build_transformation_state(
            inferred_system="cu-zn",
            stage="cold_worked",
            composition_wt={"Cu": 60.0, "Zn": 40.0},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium", deformation_pct=40.0),
        )
        brass_annealed = build_transformation_state(
            inferred_system="cu-zn",
            stage="alpha_beta",
            composition_wt={"Cu": 60.0, "Zn": 40.0},
            processing=ProcessingState(temperature_c=650.0, cooling_mode="equilibrium", deformation_pct=5.0),
        )
        fe_si_cold = build_transformation_state(
            inferred_system="fe-si",
            stage="cold_worked_ferrite",
            composition_wt={"Fe": 97.0, "Si": 3.0},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium", deformation_pct=35.0),
        )
        fe_si_rex = build_transformation_state(
            inferred_system="fe-si",
            stage="recrystallized_ferrite",
            composition_wt={"Fe": 97.0, "Si": 3.0},
            processing=ProcessingState(temperature_c=780.0, cooling_mode="equilibrium", deformation_pct=5.0),
        )

        self.assertGreater(
            float(brass_cold["morphology_state"]["deformation_band_density"]),
            float(brass_annealed["morphology_state"]["deformation_band_density"]),
        )
        self.assertLess(
            float(brass_cold["morphology_state"]["recrystallized_fraction"]),
            float(brass_annealed["morphology_state"]["recrystallized_fraction"]),
        )
        self.assertGreater(
            float(fe_si_cold["morphology_state"]["cold_work_band_density"]),
            float(fe_si_rex["morphology_state"]["cold_work_band_density"]),
        )
        self.assertLess(
            float(fe_si_cold["morphology_state"]["recrystallized_fraction"]),
            float(fe_si_rex["morphology_state"]["recrystallized_fraction"]),
        )


class TransformationMetadataIntegrationTests(unittest.TestCase):
    def test_generators_emit_transformation_metadata_blocks(self) -> None:
        fe_c_state = build_transformation_state(
            inferred_system="fe-c",
            stage="pearlite",
            composition_wt={"Fe": 99.3, "C": 0.7},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
        )
        fe_c = render_fe_c_unified(
            SystemGenerationContext(
                size=(96, 96),
                seed=1,
                inferred_system="fe-c",
                stage="pearlite",
                phase_fractions={"PEARLITE": 0.8, "FERRITE": 0.2},
                composition_wt={"Fe": 99.3, "C": 0.7},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
                transformation_state=fe_c_state,
            )
        )
        self.assertIn("transformation_trace", fe_c.metadata)
        self.assertIn("kinetics_model", fe_c.metadata)
        self.assertIn("morphology_state", fe_c.metadata)
        self.assertIn("precipitation_state", fe_c.metadata)
        self.assertIn("validation_against_rules", fe_c.metadata)

        al_si_state = build_transformation_state(
            inferred_system="al-si",
            stage="eutectic",
            composition_wt={"Al": 87.4, "Si": 12.6, "Sr": 0.01},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="water_quench"),
        )
        al_si = generate_al_si(
            SystemGenerationContext(
                size=(96, 96),
                seed=2,
                inferred_system="al-si",
                stage="eutectic",
                phase_fractions={"FCC_A1": 0.2, "EUTECTIC_ALSI": 0.7, "SI": 0.1},
                composition_wt={"Al": 87.4, "Si": 12.6, "Sr": 0.01},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="water_quench"),
                transformation_state=al_si_state,
            )
        )
        self.assertIn("transformation_trace", al_si.metadata)
        self.assertIn("eutectic_si_modifier", al_si.metadata["system_generator_extra"]["al_si_morphology"])

        al_cu_mg_state = build_transformation_state(
            inferred_system="al-cu-mg",
            stage="artificial_aged",
            composition_wt={"Al": 94.1, "Cu": 4.4, "Mg": 1.5},
            processing=ProcessingState(temperature_c=180.0, cooling_mode="aged", aging_hours=12.0, aging_temperature_c=180.0),
        )
        al_cu_mg = generate_al_cu_mg(
            SystemGenerationContext(
                size=(96, 96),
                seed=3,
                inferred_system="al-cu-mg",
                stage="artificial_aged",
                phase_fractions={"FCC_A1": 0.84, "THETA": 0.10, "S_PHASE": 0.06},
                composition_wt={"Al": 94.1, "Cu": 4.4, "Mg": 1.5},
                processing=ProcessingState(temperature_c=180.0, cooling_mode="aged", aging_hours=12.0, aging_temperature_c=180.0),
                transformation_state=al_cu_mg_state,
            )
        )
        self.assertIn("precipitation_state", al_cu_mg.metadata)
        self.assertIn("peak_strength_fraction", al_cu_mg.metadata["system_generator_extra"]["al_cu_mg_morphology"])

        cu_zn_state = build_transformation_state(
            inferred_system="cu-zn",
            stage="alpha_beta",
            composition_wt={"Cu": 60.0, "Zn": 40.0},
            processing=ProcessingState(temperature_c=650.0, cooling_mode="equilibrium", deformation_pct=5.0),
        )
        cu_zn = generate_cu_zn(
            SystemGenerationContext(
                size=(96, 96),
                seed=4,
                inferred_system="cu-zn",
                stage="alpha_beta",
                phase_fractions={"ALPHA": 0.65, "BETA": 0.35},
                composition_wt={"Cu": 60.0, "Zn": 40.0},
                processing=ProcessingState(temperature_c=650.0, cooling_mode="equilibrium", deformation_pct=5.0),
                transformation_state=cu_zn_state,
            )
        )
        self.assertIn("recrystallized_fraction", cu_zn.metadata["system_generator_extra"]["cu_zn_morphology"])

        fe_si_state = build_transformation_state(
            inferred_system="fe-si",
            stage="cold_worked_ferrite",
            composition_wt={"Fe": 97.0, "Si": 3.0},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium", deformation_pct=30.0),
        )
        fe_si = generate_fe_si(
            SystemGenerationContext(
                size=(96, 96),
                seed=5,
                inferred_system="fe-si",
                stage="cold_worked_ferrite",
                phase_fractions={"BCC_B2": 1.0},
                composition_wt={"Fe": 97.0, "Si": 3.0},
                processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium", deformation_pct=30.0),
                transformation_state=fe_si_state,
            )
        )
        self.assertIn("morphology_state", fe_si.metadata)

    def test_pipeline_v3_surfaces_transformation_metadata(self) -> None:
        pipeline = MetallographyPipelineV3(generator_version="v3.0.0+transformation-tests")
        payload = pipeline.load_preset("steel_tempered_400_textbook")
        payload["resolution"] = [256, 256]
        payload["sample_id"] = "transformation_state_pipeline"
        payload["synthesis_profile"]["phase_topology_mode"] = "physics_guided_hybrid"
        out = pipeline.generate(pipeline.request_from_preset(payload))
        for key in (
            "transformation_trace",
            "kinetics_model",
            "morphology_state",
            "precipitation_state",
            "validation_against_rules",
        ):
            self.assertIn(key, out.metadata)
            self.assertIsInstance(out.metadata[key], dict)


if __name__ == "__main__":
    unittest.main()
