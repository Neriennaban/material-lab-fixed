from __future__ import annotations

import unittest

import numpy as np

from core.contracts_v2 import ProcessingState
from core.contracts_v3 import (
    EtchProfileV3,
    PrepOperationV3,
    SamplePrepRouteV3,
    SynthesisProfileV3,
)
from core.metallography_pro.morphology_fe_c import build_spatial_morphology_state
from core.metallography_pro.reflected_light import render_reflected_light
from core.metallography_pro.surface_state import build_surface_state
from core.metallography_pro.transformation_fe_c import (
    build_continuous_transformation_state,
)
from core.metallography_pro.validation_pro import run_pro_validation


class ProSurfaceAndReflectedLightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prep = SamplePrepRouteV3(
            steps=[
                PrepOperationV3(
                    method="grinding",
                    duration_s=120.0,
                    abrasive_um=15.0,
                    load_n=30.0,
                    rpm=240.0,
                    direction_deg=18.0,
                ),
                PrepOperationV3(
                    method="polishing",
                    duration_s=90.0,
                    abrasive_um=3.0,
                    load_n=18.0,
                    rpm=160.0,
                ),
            ],
            roughness_target_um=0.05,
        )

    def test_surface_state_tracks_etch_selectivity_and_damage(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 98.9, "C": 1.1},
            stage="pearlite_cementite",
            phase_fractions={"PEARLITE": 0.72, "CEMENTITE": 0.28},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 820.0, "hold_time_s": 120.0},
        )
        morph = build_spatial_morphology_state(
            size=(128, 128),
            seed=201,
            stage="pearlite_cementite",
            phase_fractions={"PEARLITE": 0.72, "CEMENTITE": 0.28},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        surface, prep_maps, _, prep_summary, etch_summary, _, etch_maps = (
            build_surface_state(
                morphology_state=morph,
                transformation_state=state,
                prep_route=self.prep,
                etch_profile=EtchProfileV3(reagent="picral"),
                seed=202,
                native_um_per_px=0.5,
            )
        )
        self.assertEqual(surface.height_um.shape, (128, 128))
        self.assertGreater(float(surface.etch_depth_um.mean()), 0.0)
        self.assertTrue(bool(prep_summary.get("phase_coupling_applied")))
        self.assertTrue(bool(etch_summary.get("prep_coupling_applied")))
        self.assertIn("etch_rate", etch_maps)
        self.assertIn("scratch_trace_revelation_risk", surface.summary)
        self.assertIn("false_porosity_pullout_risk", surface.summary)
        self.assertIn("relief_dominance_risk", surface.summary)
        self.assertIn("stain_deposit_contrast_dominance_risk", surface.summary)
        cementite_zone = morph.phase_masks["CEMENTITE"] > 0
        pearlite_zone = morph.phase_masks["PEARLITE"] > 0
        etch_rate = etch_maps["selectivity"].astype(np.float32)
        self.assertGreater(
            float(etch_rate[cementite_zone].mean()),
            float(etch_rate[pearlite_zone].mean()),
        )
        self.assertIn("topography", prep_maps)

    def test_reflected_light_produces_nonflat_image(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.2, "C": 0.8},
            stage="pearlite",
            phase_fractions={"PEARLITE": 0.99, "CEMENTITE": 0.01},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 840.0, "hold_time_s": 180.0},
        )
        morph = build_spatial_morphology_state(
            size=(128, 128),
            seed=203,
            stage="pearlite",
            phase_fractions={"PEARLITE": 0.99, "CEMENTITE": 0.01},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=morph,
            transformation_state=state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=204,
            native_um_per_px=0.5,
        )
        image_gray, meta = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "brightfield"},
            seed=205,
            native_um_per_px=0.5,
        )
        self.assertEqual(image_gray.dtype, np.uint8)
        self.assertGreater(float(image_gray.std()), 5.0)
        self.assertEqual(meta.get("model"), "reflected_light_explicit_surface_v1")
        self.assertEqual(meta.get("optical_mode"), "brightfield")

        validation = run_pro_validation(
            image_gray=image_gray,
            phase_masks=morph.phase_masks,
            morphology_state=morph,
            surface_state=surface,
            transformation_state=state,
            native_um_per_px=0.5,
        )
        self.assertIn("grain_size_astm_number_proxy", validation)
        self.assertIn("phase_fraction_grid_point_count_proxy", validation)
        self.assertIn("phase_fraction_lineal_fraction_proxy", validation)
        self.assertIn("pl_grain_boundaries_mm_inv", validation)
        self.assertIn("sv_interphase_boundaries_mm_inv", validation)
        self.assertIn("interlamellar_random_spacing_um", validation)
        self.assertIn("interlamellar_true_spacing_um", validation)
        self.assertIn("boundary_ferrite_coverage", validation)
        self.assertIn("two_point_corr_curve", validation)
        self.assertIn("directional_artifact_anisotropy_score", validation)
        self.assertIn("scratch_trace_revelation_risk", validation)
        self.assertIn("false_porosity_pullout_risk", validation)
        self.assertIn("relief_dominance_risk", validation)
        self.assertIn("stain_deposit_contrast_dominance_risk", validation)
        self.assertIn("artifact_risk_scores", validation)
        self.assertIn("surface_roughness_ra_um", validation)
        self.assertGreaterEqual(float(validation["surface_roughness_ra_um"]), 0.0)
        self.assertGreaterEqual(float(validation["scratch_trace_revelation_risk"]), 0.0)
        self.assertLessEqual(float(validation["scratch_trace_revelation_risk"]), 1.0)
        self.assertGreaterEqual(float(validation["false_porosity_pullout_risk"]), 0.0)
        self.assertLessEqual(float(validation["false_porosity_pullout_risk"]), 1.0)
        self.assertIn("trigger_ratio", validation["artifact_risk_scores"])
        self.assertIn("dominant_driver", validation["artifact_risk_scores"])

    def test_optical_modes_change_reflected_light_output(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.45, "C": 0.55},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 830.0, "hold_time_s": 120.0},
        )
        morph = build_spatial_morphology_state(
            size=(128, 128),
            seed=206,
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=morph,
            transformation_state=state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=207,
            native_um_per_px=0.5,
        )
        bright, _ = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "brightfield"},
            seed=208,
            native_um_per_px=0.5,
        )
        dark, dark_meta = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "darkfield"},
            seed=208,
            native_um_per_px=0.5,
        )
        self.assertEqual(dark_meta.get("optical_mode"), "darkfield")
        self.assertGreater(
            float(np.mean(np.abs(bright.astype(np.float32) - dark.astype(np.float32)))),
            2.0,
        )
        dic, dic_meta = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "dic"},
            seed=208,
            native_um_per_px=0.5,
        )
        self.assertEqual(dic_meta.get("optical_mode"), "dic")
        self.assertIn("interference_gradient", dic_meta.get("contrast_mechanisms", []))
        self.assertGreater(
            float(np.mean(np.abs(bright.astype(np.float32) - dic.astype(np.float32)))),
            2.0,
        )

    def test_polarized_mode_extinguishes_isotropic_ferritic_case(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.95},
            stage="ferrite",
            phase_fractions={"FERRITE": 1.0},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="air"),
            thermal_summary={"temperature_max_c": 780.0, "hold_time_s": 180.0},
        )
        morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=280,
            stage="ferrite",
            phase_fractions={"FERRITE": 1.0},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=morph,
            transformation_state=state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=281,
            native_um_per_px=0.5,
        )
        bright, _ = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "brightfield"},
            seed=282,
            native_um_per_px=0.5,
        )
        polarized, meta = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={
                "optical_mode": "polarized",
                "optical_mode_parameters": {"crossed_polars": True},
            },
            seed=282,
            native_um_per_px=0.5,
        )
        self.assertEqual(str(meta.get("optical_mode", "")), "polarized")
        self.assertTrue(bool(meta.get("crossed_polars", False)))
        self.assertLess(float(polarized.mean()), float(bright.mean()) * 0.45)
        self.assertLessEqual(float(meta.get("anisotropy_coverage", 1.0)), 0.05)

    def test_phase_contrast_plate_sign_changes_output(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.45, "C": 0.55},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 830.0, "hold_time_s": 120.0},
        )
        morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=283,
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=morph,
            transformation_state=state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=284,
            native_um_per_px=0.5,
        )
        pos, pos_meta = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={
                "optical_mode": "phase_contrast",
                "optical_mode_parameters": {"phase_plate_type": "positive"},
            },
            seed=285,
            native_um_per_px=0.5,
        )
        neg, neg_meta = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={
                "optical_mode": "phase_contrast",
                "optical_mode_parameters": {"phase_plate_type": "negative"},
            },
            seed=285,
            native_um_per_px=0.5,
        )
        self.assertEqual(str(pos_meta.get("phase_plate_type", "")), "positive")
        self.assertEqual(str(neg_meta.get("phase_plate_type", "")), "negative")
        self.assertGreater(
            float(np.mean(np.abs(pos.astype(np.float32) - neg.astype(np.float32)))), 1.0
        )

    def test_pure_ferrite_negative_control_stays_bright_in_pro_surface_path(
        self,
    ) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 100.0, "C": 0.0},
            stage="ferrite",
            phase_fractions={"FERRITE": 1.0},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="air"),
            thermal_summary={"temperature_max_c": 840.0, "hold_time_s": 180.0},
        )
        morph = build_spatial_morphology_state(
            size=(256, 256),
            seed=401,
            stage="ferrite",
            phase_fractions={"FERRITE": 1.0},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        surface, _, _, prep_summary, etch_summary, _, _ = build_surface_state(
            morphology_state=morph,
            transformation_state=state,
            prep_route=SamplePrepRouteV3(),
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=402,
            native_um_per_px=0.5,
            system="fe-c",
            composition_wt={"Fe": 100.0, "C": 0.0},
            artifact_level=0.2,
        )
        image_gray, meta = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(
                generation_mode="pro_realistic", artifact_level=0.2
            ),
            microscope_profile={
                "optical_mode": "brightfield",
                "pure_iron_baseline": {"applied": True},
            },
            seed=403,
            native_um_per_px=0.5,
        )
        arr = image_gray.astype(np.float32)
        self.assertTrue(bool(prep_summary.get("pure_iron_baseline_applied", False)))
        self.assertTrue(bool(etch_summary.get("pure_iron_baseline_applied", False)))
        self.assertGreater(float(arr.mean()), 170.0)
        self.assertGreater(float(np.quantile(arr, 0.05)), 95.0)
        self.assertEqual(str(meta.get("optical_mode", "")), "brightfield")

    def test_artifact_level_changes_pro_surface_relief_budget(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.45, "C": 0.55},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 830.0, "hold_time_s": 120.0},
        )
        morph = build_spatial_morphology_state(
            size=(128, 128),
            seed=404,
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        low_surface, _, _, low_prep, low_etch, _, _ = build_surface_state(
            morphology_state=morph,
            transformation_state=state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=405,
            native_um_per_px=0.5,
            system="fe-c",
            composition_wt={"Fe": 99.45, "C": 0.55},
            artifact_level=0.05,
        )
        high_surface, _, _, high_prep, high_etch, _, _ = build_surface_state(
            morphology_state=morph,
            transformation_state=state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=405,
            native_um_per_px=0.5,
            system="fe-c",
            composition_wt={"Fe": 99.45, "C": 0.55},
            artifact_level=0.65,
        )
        self.assertLess(
            float(np.ptp(low_surface.height_um)), float(np.ptp(high_surface.height_um))
        )
        self.assertLess(
            float(low_etch.get("stain_level_mean", 0.0)),
            float(high_etch.get("stain_level_mean", 1.0)),
        )
        self.assertLessEqual(
            float(low_prep.get("contamination_mean", 0.0)),
            float(high_prep.get("contamination_mean", 1.0)),
        )

    def test_magnetic_etching_highlights_ferromagnetic_phase_over_austenite(
        self,
    ) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.3, "C": 0.7},
            stage="alpha_gamma",
            phase_fractions={"FERRITE": 0.55, "AUSTENITE": 0.45},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="air"),
            thermal_summary={"temperature_max_c": 820.0, "hold_time_s": 160.0},
        )
        morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=286,
            stage="alpha_gamma",
            phase_fractions={"FERRITE": 0.55, "AUSTENITE": 0.45},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=morph,
            transformation_state=state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=287,
            native_um_per_px=0.5,
        )
        magnetic, meta = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "magnetic_etching"},
            seed=288,
            native_um_per_px=0.5,
        )
        ferrite_zone = morph.phase_masks["FERRITE"] > 0
        austenite_zone = morph.phase_masks["AUSTENITE"] > 0
        self.assertEqual(str(meta.get("optical_mode", "")), "magnetic_etching")
        self.assertTrue(bool(meta.get("magnetic_field_active", False)))
        self.assertGreater(float(meta.get("ferromagnetic_fraction", 0.0)), 0.4)
        self.assertGreater(float(meta.get("magnetic_signal_fraction", 0.0)), 0.02)
        self.assertLess(
            float(magnetic[ferrite_zone].mean()), float(magnetic[austenite_zone].mean())
        )

    def test_static_psf_profiles_change_render_and_emit_metadata(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.45, "C": 0.55},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={"temperature_max_c": 830.0, "hold_time_s": 120.0},
        )
        morph = build_spatial_morphology_state(
            size=(128, 128),
            seed=240,
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=morph,
            transformation_state=state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=241,
            native_um_per_px=0.5,
        )
        standard, meta_standard = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={
                "optical_mode": "brightfield",
                "psf_profile": "standard",
                "psf_strength": 0.0,
            },
            seed=242,
            native_um_per_px=0.5,
        )
        bessel, meta_bessel = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={
                "optical_mode": "brightfield",
                "psf_profile": "bessel_extended_dof",
                "psf_strength": 0.9,
            },
            seed=242,
            native_um_per_px=0.5,
        )
        stir, meta_stir = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={
                "optical_mode": "brightfield",
                "psf_profile": "stir_sectioning",
                "psf_strength": 0.9,
            },
            seed=242,
            native_um_per_px=0.5,
        )
        self.assertGreater(
            float(
                np.mean(np.abs(standard.astype(np.float32) - bessel.astype(np.float32)))
            ),
            1.0,
        )
        self.assertGreater(
            float(
                np.mean(np.abs(standard.astype(np.float32) - stir.astype(np.float32)))
            ),
            1.0,
        )
        self.assertEqual(str(meta_bessel["psf_profile"]), "bessel_extended_dof")
        self.assertGreater(float(meta_bessel["effective_dof_factor"]), 1.0)
        self.assertEqual(str(meta_stir["psf_profile"]), "stir_sectioning")
        self.assertTrue(bool(meta_stir["sectioning_active"]))

    def test_family_aware_visual_differentiation_sets_expected_flags(self) -> None:
        pearlite_state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.2, "C": 0.8},
            stage="pearlite",
            phase_fractions={"PEARLITE": 0.99, "CEMENTITE": 0.01},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={
                "temperature_max_c": 840.0,
                "temperature_end_c": 20.0,
                "hold_time_s": 180.0,
            },
        )
        pearlite_morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=212,
            stage="pearlite",
            phase_fractions={"PEARLITE": 0.99, "CEMENTITE": 0.01},
            transformation_state=pearlite_state,
            native_um_per_px=0.5,
        )
        pearlite_surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=pearlite_morph,
            transformation_state=pearlite_state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=213,
            native_um_per_px=0.5,
        )
        _, pearlite_meta = render_reflected_light(
            surface_state=pearlite_surface,
            morphology_state=pearlite_morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "brightfield"},
            seed=214,
            native_um_per_px=0.5,
        )

        bainite_state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.15, "C": 0.85},
            stage="bainite",
            phase_fractions={"BAINITE": 0.78, "CEMENTITE": 0.14, "AUSTENITE": 0.08},
            processing=ProcessingState(temperature_c=320.0, cooling_mode="oil_quench"),
            thermal_summary={
                "temperature_max_c": 860.0,
                "temperature_end_c": 320.0,
                "hold_time_s": 600.0,
            },
            quench_summary={"effect_applied": True},
        )
        bainite_morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=215,
            stage="bainite",
            phase_fractions={"BAINITE": 0.78, "CEMENTITE": 0.14, "AUSTENITE": 0.08},
            transformation_state=bainite_state,
            native_um_per_px=0.5,
        )
        bainite_surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=bainite_morph,
            transformation_state=bainite_state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=216,
            native_um_per_px=0.5,
        )
        _, bainite_meta = render_reflected_light(
            surface_state=bainite_surface,
            morphology_state=bainite_morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "brightfield"},
            seed=217,
            native_um_per_px=0.5,
        )

        wid_state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.45, "C": 0.55},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="normalized"),
            thermal_summary={
                "temperature_max_c": 830.0,
                "temperature_end_c": 20.0,
                "hold_time_s": 120.0,
            },
        )
        wid_morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=218,
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.35, "PEARLITE": 0.65},
            transformation_state=wid_state,
            native_um_per_px=0.5,
        )
        wid_surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=wid_morph,
            transformation_state=wid_state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=219,
            native_um_per_px=0.5,
        )
        _, wid_meta = render_reflected_light(
            surface_state=wid_surface,
            morphology_state=wid_morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "brightfield"},
            seed=220,
            native_um_per_px=0.5,
        )

        self.assertTrue(bool(pearlite_meta.get("lamella_modulation_applied", False)))
        self.assertFalse(bool(pearlite_meta.get("bainite_modulation_applied", False)))
        self.assertTrue(bool(bainite_meta.get("bainite_modulation_applied", False)))
        self.assertTrue(bool(wid_meta.get("widmanstatten_modulation_applied", False)))
        if str(wid_state.ferrite_morphology_family) == "allotriomorphic":
            self.assertTrue(
                bool(wid_meta.get("allotriomorphic_modulation_applied", False))
            )

    def test_bainite_family_validation_emits_sheaf_metrics(self) -> None:
        state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.15, "C": 0.85},
            stage="bainite",
            phase_fractions={"BAINITE": 0.78, "CEMENTITE": 0.14, "AUSTENITE": 0.08},
            processing=ProcessingState(temperature_c=320.0, cooling_mode="oil_quench"),
            thermal_summary={
                "temperature_max_c": 860.0,
                "temperature_end_c": 320.0,
                "hold_time_s": 600.0,
            },
            quench_summary={"effect_applied": True},
        )
        morph = build_spatial_morphology_state(
            size=(128, 128),
            seed=209,
            stage="bainite",
            phase_fractions={"BAINITE": 0.78, "CEMENTITE": 0.14, "AUSTENITE": 0.08},
            transformation_state=state,
            native_um_per_px=0.5,
        )
        surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=morph,
            transformation_state=state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=210,
            native_um_per_px=0.5,
        )
        image_gray, _ = render_reflected_light(
            surface_state=surface,
            morphology_state=morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "brightfield"},
            seed=211,
            native_um_per_px=0.5,
        )
        validation = run_pro_validation(
            image_gray=image_gray,
            phase_masks=morph.phase_masks,
            morphology_state=morph,
            surface_state=surface,
            transformation_state=state,
            native_um_per_px=0.5,
            reflected_light_model={
                "psf_profile": "bessel_extended_dof",
                "effective_dof_factor": 1.5,
                "extended_dof_retention_score": 0.4,
            },
        )
        self.assertGreater(float(validation["bainite_sheaf_area_fraction"]), 0.0)
        self.assertGreater(float(validation["bainite_fraction_exact"]), 0.0)
        self.assertGreater(float(validation["bainite_sheaf_density_na_mm2"]), 0.0)
        self.assertGreater(float(validation["mean_sheaf_length_um"]), 0.0)
        self.assertGreaterEqual(float(validation["sheaf_aspect_ratio_proxy"]), 1.0)
        self.assertIn("bainite_family_split_label", validation)
        self.assertGreaterEqual(
            float(validation["bainite_family_split_area_fraction"]), 0.0
        )
        self.assertEqual(str(validation["psf_profile_family"]), "bessel_extended_dof")

    def test_allotriomorphic_and_bainite_split_metrics_propagate(self) -> None:
        ferrite_state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.55, "C": 0.45},
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.55, "PEARLITE": 0.45},
            processing=ProcessingState(temperature_c=20.0, cooling_mode="equilibrium"),
            thermal_summary={
                "temperature_max_c": 840.0,
                "temperature_end_c": 770.0,
                "hold_time_s": 240.0,
            },
        )
        ferrite_morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=221,
            stage="alpha_pearlite",
            phase_fractions={"FERRITE": 0.55, "PEARLITE": 0.45},
            transformation_state=ferrite_state,
            native_um_per_px=0.5,
        )
        ferrite_surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=ferrite_morph,
            transformation_state=ferrite_state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=222,
            native_um_per_px=0.5,
        )
        ferrite_img, ferrite_meta = render_reflected_light(
            surface_state=ferrite_surface,
            morphology_state=ferrite_morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "brightfield"},
            seed=223,
            native_um_per_px=0.5,
        )
        ferrite_val = run_pro_validation(
            image_gray=ferrite_img,
            phase_masks=ferrite_morph.phase_masks,
            morphology_state=ferrite_morph,
            surface_state=ferrite_surface,
            transformation_state=ferrite_state,
            native_um_per_px=0.5,
        )
        self.assertIn("allotriomorphic_ferrite_area_fraction", ferrite_val)
        self.assertGreaterEqual(
            float(ferrite_val["allotriomorphic_ferrite_area_fraction"]), 0.0
        )
        self.assertIn("bright_ferritic_baseline_score", ferrite_val)
        self.assertIn("dark_defect_field_dominance", ferrite_val)
        if str(ferrite_state.ferrite_morphology_family) == "allotriomorphic":
            self.assertTrue(
                bool(ferrite_meta.get("allotriomorphic_modulation_applied", False))
            )

        upper_state = build_continuous_transformation_state(
            composition_wt={"Fe": 99.15, "C": 0.85},
            stage="bainite",
            phase_fractions={"BAINITE": 0.74, "CEMENTITE": 0.16, "AUSTENITE": 0.10},
            processing=ProcessingState(temperature_c=380.0, cooling_mode="oil_quench"),
            thermal_summary={
                "temperature_max_c": 860.0,
                "temperature_end_c": 380.0,
                "hold_time_s": 480.0,
            },
            quench_summary={"effect_applied": True},
        )
        upper_morph = build_spatial_morphology_state(
            size=(96, 96),
            seed=224,
            stage="bainite",
            phase_fractions={"BAINITE": 0.74, "CEMENTITE": 0.16, "AUSTENITE": 0.10},
            transformation_state=upper_state,
            native_um_per_px=0.5,
        )
        upper_surface, _, _, _, _, _, _ = build_surface_state(
            morphology_state=upper_morph,
            transformation_state=upper_state,
            prep_route=self.prep,
            etch_profile=EtchProfileV3(reagent="nital_2"),
            seed=225,
            native_um_per_px=0.5,
        )
        upper_img, _ = render_reflected_light(
            surface_state=upper_surface,
            morphology_state=upper_morph,
            synthesis_profile=SynthesisProfileV3(generation_mode="pro_realistic"),
            microscope_profile={"optical_mode": "brightfield"},
            seed=226,
            native_um_per_px=0.5,
        )
        upper_val = run_pro_validation(
            image_gray=upper_img,
            phase_masks=upper_morph.phase_masks,
            morphology_state=upper_morph,
            surface_state=upper_surface,
            transformation_state=upper_state,
            native_um_per_px=0.5,
        )
        self.assertEqual(
            str(upper_val["bainite_family_split_label"]), "upper_bainite_sheaves"
        )
        self.assertGreaterEqual(
            float(upper_val["upper_bainite_sheaf_area_fraction"]),
            float(upper_val["bainite_family_split_area_fraction"]),
        )


if __name__ == "__main__":
    unittest.main()
