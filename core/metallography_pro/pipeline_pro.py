from __future__ import annotations

from typing import Any

import numpy as np

from core.contracts_v2 import ProcessingState
from core.contracts_v3 import (
    EtchProfileV3,
    PrepOperationV3,
    SamplePrepRouteV3,
    SynthesisProfileV3,
)
from core.metallography_v3.system_generators.base import (
    build_composition_effect,
    build_phase_visibility_report,
)

from .morphology_fe_c import build_spatial_morphology_state
from .reflected_light import render_reflected_light
from .surface_state import build_surface_state
from .transformation_fe_c import build_continuous_transformation_state
from .validation_pro import run_pro_validation

_SUPPORTED_FE_C_STAGES = {
    "ferrite",
    "alpha_pearlite",
    "pearlite",
    "pearlite_cementite",
    "bainite",
    "martensite",
    "martensite_tetragonal",
    "martensite_cubic",
    "troostite_quench",
    "troostite_temper",
    "sorbite_quench",
    "sorbite_temper",
    "tempered_low",
    "tempered_medium",
    "tempered_high",
}


def supports_pro_realistic_fe_c_stage(stage: str) -> bool:
    return str(stage or "").strip().lower() in _SUPPORTED_FE_C_STAGES


def _composition_fraction(composition_wt: dict[str, float] | None, key: str) -> float:
    if not isinstance(composition_wt, dict):
        return 0.0
    total = 0.0
    cleaned: dict[str, float] = {}
    for name, value in composition_wt.items():
        try:
            vv = float(value)
        except Exception:
            continue
        if vv <= 0.0:
            continue
        cleaned[str(name).strip()] = vv
        total += vv
    if total <= 1e-12:
        return 0.0
    return float(cleaned.get(key, 0.0) / total * 100.0)


def _is_pure_iron_like(
    *, stage: str, phase_fractions: dict[str, float], composition_wt: dict[str, float]
) -> bool:
    stage_name = str(stage or "").strip().lower()
    ferrite = float(
        phase_fractions.get("FERRITE", 0.0) + phase_fractions.get("DELTA_FERRITE", 0.0)
    )
    fe_pct = _composition_fraction(composition_wt, "Fe")
    c_pct = _composition_fraction(composition_wt, "C")
    si_pct = _composition_fraction(composition_wt, "Si")
    return bool(
        stage_name == "ferrite"
        and ferrite >= 0.95
        and fe_pct >= 99.8
        and c_pct <= 0.03
        and si_pct <= 0.25
    )


def _default_pro_prep_route(*, pure_iron_like: bool) -> SamplePrepRouteV3:
    if pure_iron_like:
        return SamplePrepRouteV3(
            steps=[
                PrepOperationV3(
                    method="polishing_3um",
                    duration_s=75.0,
                    abrasive_um=3.0,
                    load_n=10.0,
                    rpm=110.0,
                    cloth_type="short_nap",
                    slurry_type="diamond",
                    cleaning_between_steps=True,
                ),
                PrepOperationV3(
                    method="polishing_1um",
                    duration_s=90.0,
                    abrasive_um=1.0,
                    load_n=7.0,
                    rpm=95.0,
                    cloth_type="napless",
                    slurry_type="colloidal_silica",
                    cleaning_between_steps=True,
                ),
            ],
            roughness_target_um=0.03,
            relief_mode="hardness_coupled",
            contamination_level=0.003,
        )
    return SamplePrepRouteV3(
        steps=[
            PrepOperationV3(
                method="grinding_1200",
                duration_s=45.0,
                abrasive_um=9.0,
                load_n=12.0,
                rpm=140.0,
                coolant="alcohol",
                cloth_type="rigid_pad",
                cleaning_between_steps=True,
            ),
            PrepOperationV3(
                method="polishing_1um",
                duration_s=75.0,
                abrasive_um=1.0,
                load_n=8.0,
                rpm=105.0,
                cloth_type="napless",
                slurry_type="diamond",
                cleaning_between_steps=True,
            ),
        ],
        roughness_target_um=0.04,
        relief_mode="hardness_coupled",
        contamination_level=0.006,
    )


def generate_pro_realistic_fe_c(
    *,
    size: tuple[int, int],
    seed: int,
    stage: str,
    phase_fractions: dict[str, float],
    composition_wt: dict[str, float],
    processing: ProcessingState,
    prep_route: SamplePrepRouteV3,
    etch_profile: EtchProfileV3,
    synthesis_profile: SynthesisProfileV3,
    microscope_profile: dict[str, Any],
    thermal_summary: dict[str, Any] | None = None,
    quench_summary: dict[str, Any] | None = None,
    phase_fraction_source: str = "default_formula",
    phase_calibration_mode: str = "default_formula",
) -> dict[str, Any]:
    magnification = int(microscope_profile.get("magnification", 200))
    native_um_per_px = float(max(0.05, 1.0 / (float(magnification) / 100.0)))
    continuous_state = build_continuous_transformation_state(
        composition_wt=composition_wt,
        stage=stage,
        phase_fractions=phase_fractions,
        processing=processing,
        thermal_summary=thermal_summary,
        quench_summary=quench_summary,
    )
    morphology_state = build_spatial_morphology_state(
        size=size,
        seed=seed,
        stage=stage,
        phase_fractions=phase_fractions,
        transformation_state=continuous_state,
        native_um_per_px=native_um_per_px,
    )
    pure_iron_like = _is_pure_iron_like(
        stage=stage,
        phase_fractions=dict(continuous_state.phase_fractions),
        composition_wt=composition_wt,
    )
    implicit_baseline_route_applied = False
    effective_prep_route = prep_route
    if not list(prep_route.steps or []):
        effective_prep_route = _default_pro_prep_route(pure_iron_like=pure_iron_like)
        implicit_baseline_route_applied = True
    (
        surface_state,
        prep_maps,
        prep_timeline,
        prep_summary,
        etch_summary,
        etch_concentration,
        etch_maps,
    ) = build_surface_state(
        morphology_state=morphology_state,
        transformation_state=continuous_state,
        prep_route=effective_prep_route,
        etch_profile=etch_profile,
        seed=seed + 77,
        native_um_per_px=native_um_per_px,
        system="fe-c",
        composition_wt=composition_wt,
        artifact_level=float(synthesis_profile.artifact_level),
    )
    prep_summary = {
        **dict(prep_summary),
        "implicit_baseline_route_applied": bool(implicit_baseline_route_applied),
    }
    microscope_profile_effective = dict(microscope_profile)
    if pure_iron_like:
        microscope_profile_effective["pure_iron_baseline"] = {"applied": True}
    image_gray, reflected_light_model = render_reflected_light(
        surface_state=surface_state,
        morphology_state=morphology_state,
        synthesis_profile=synthesis_profile,
        microscope_profile=microscope_profile_effective,
        seed=seed + 131,
        native_um_per_px=native_um_per_px,
    )
    phase_visibility_report = build_phase_visibility_report(
        image_gray=image_gray,
        phase_masks=morphology_state.phase_masks,
        phase_fractions=phase_fractions,
        tolerance_pct=float(synthesis_profile.phase_fraction_tolerance_pct),
    )
    validation_pro = run_pro_validation(
        image_gray=image_gray,
        phase_masks=morphology_state.phase_masks,
        morphology_state=morphology_state,
        surface_state=surface_state,
        transformation_state=continuous_state,
        native_um_per_px=native_um_per_px,
        reflected_light_model=reflected_light_model,
    )
    pure_iron_baseline = {
        "applied": bool(pure_iron_like),
        "cleanliness_score": float(
            max(
                float(prep_summary.get("pure_iron_cleanliness_score", 0.0) or 0.0),
                float(etch_summary.get("pure_iron_cleanliness_score", 0.0) or 0.0),
                1.0
                - float(validation_pro.get("dark_defect_field_dominance", 0.0) or 0.0),
            )
        ),
        "dark_defect_suppression": float(
            max(
                float(
                    prep_summary.get("pure_iron_dark_defect_suppression", 0.0) or 0.0
                ),
                float(
                    etch_summary.get("pure_iron_dark_defect_suppression", 0.0) or 0.0
                ),
                1.0
                - float(validation_pro.get("dark_defect_field_dominance", 0.0) or 0.0),
            )
        ),
        "boundary_visibility_score": float(
            max(
                float(
                    prep_summary.get("pure_iron_boundary_visibility_score", 0.0) or 0.0
                ),
                float(
                    etch_summary.get("pure_iron_boundary_visibility_score", 0.0) or 0.0
                ),
                float(validation_pro.get("bright_ferritic_baseline_score", 0.0) or 0.0),
            )
        ),
    }
    composition_effect = build_composition_effect(
        system="fe-c",
        composition_wt=composition_wt,
        mode=str(synthesis_profile.composition_sensitivity_mode),
        seed=seed,
        single_phase_compensation=len(phase_fractions) <= 1,
    )
    morphology_trace = {
        "family": "continuous_pro",
        "transformation_family": str(continuous_state.transformation_family),
        "ferrite_morphology_family": str(continuous_state.ferrite_morphology_family),
        "bainite_morphology_family": str(continuous_state.bainite_morphology_family),
        "martensite_morphology_family": str(
            continuous_state.martensite_morphology_family
        ),
        "pearlite_morphology_family": str(continuous_state.pearlite_morphology_family),
        "prior_austenite_grain_count": int(
            morphology_state.summary.get("prior_austenite_grain_count", 0)
        ),
        "prior_austenite_grain_size_um": float(
            continuous_state.prior_austenite_grain_size_um
        ),
        "colony_size_um_mean": float(continuous_state.colony_size_um_mean),
        "interlamellar_spacing_um_mean": float(
            continuous_state.interlamellar_spacing_um_mean
        ),
        "bainite_sheaf_length_um": float(continuous_state.bainite_sheaf_length_um),
        "bainite_sheaf_thickness_um": float(
            continuous_state.bainite_sheaf_thickness_um
        ),
        "ae1_temperature_c": float(continuous_state.ae1_temperature_c),
        "ae3_temperature_c": float(continuous_state.ae3_temperature_c),
        "bs_temperature_c": float(continuous_state.bs_temperature_c),
        "ms_temperature_c": float(continuous_state.ms_temperature_c),
        "t0_temperature_c": float(continuous_state.t0_temperature_c),
        "austenitization_hold_s": float(continuous_state.austenitization_hold_s),
        "time_in_upper_c_window_s": float(continuous_state.time_in_upper_c_window_s),
        "time_in_lower_c_window_s": float(continuous_state.time_in_lower_c_window_s),
        "time_below_ms_s": float(continuous_state.time_below_ms_s),
        "time_in_bainite_hold_s": float(continuous_state.time_in_bainite_hold_s),
        "ferrite_effective_exposure_s": float(
            continuous_state.ferrite_effective_exposure_s
        ),
        "pearlite_effective_exposure_s": float(
            continuous_state.pearlite_effective_exposure_s
        ),
        "bainite_effective_exposure_s": float(
            continuous_state.bainite_effective_exposure_s
        ),
        "martensite_effective_exposure_s": float(
            continuous_state.martensite_effective_exposure_s
        ),
        "diffusional_equivalent_time_s": float(
            continuous_state.diffusional_equivalent_time_s
        ),
        "hardenability_factor": float(continuous_state.hardenability_factor),
        "continuous_cooling_shift_factor": float(
            continuous_state.continuous_cooling_shift_factor
        ),
        "ferrite_nucleation_drive": float(continuous_state.ferrite_nucleation_drive),
        "pearlite_nucleation_drive": float(continuous_state.pearlite_nucleation_drive),
        "bainite_nucleation_drive": float(continuous_state.bainite_nucleation_drive),
        "ferrite_progress": float(continuous_state.ferrite_progress),
        "pearlite_progress": float(continuous_state.pearlite_progress),
        "ferrite_pearlite_competition_index": float(
            continuous_state.ferrite_pearlite_competition_index
        ),
        "bainite_activation_progress": float(
            continuous_state.bainite_activation_progress
        ),
        "martensite_conversion_progress": float(
            continuous_state.martensite_conversion_progress
        ),
        "proeutectoid_phase": (
            "FERRITE"
            if str(stage).strip().lower() == "alpha_pearlite"
            else (
                "CEMENTITE"
                if str(stage).strip().lower() == "pearlite_cementite"
                else ""
            )
        ),
        "boundary_phase_bias": float(continuous_state.proeutectoid_boundary_bias),
        "native_um_per_px": float(native_um_per_px),
    }
    # C1.1 — apply the post-process colour palette here as well so
    # the pro-realistic path supports ``color_mode != grayscale_nital``.
    # When the synthesis profile keeps the default mode the value
    # below stays ``None`` and ``pipeline_v3`` falls back to the
    # legacy ``_to_rgb`` stack on its side.
    _pro_image_gray = image_gray.astype(np.uint8)
    _pro_color_mode = str(
        getattr(synthesis_profile, "color_mode", "grayscale_nital")
        or "grayscale_nital"
    )
    _pro_image_rgb: np.ndarray | None = None
    if _pro_color_mode != "grayscale_nital":
        try:
            from core.metallography_v3.fe_c_color_palette import apply_color_palette

            _pro_image_rgb = apply_color_palette(
                image_gray=_pro_image_gray,
                phase_masks=morphology_state.phase_masks,
                color_mode=_pro_color_mode,
                seed=int(seed),
                labels=None,
            )
        except Exception:
            _pro_image_rgb = None
    return {
        "image_gray": _pro_image_gray,
        "image_rgb": _pro_image_rgb,
        "phase_masks": morphology_state.phase_masks,
        "feature_masks": dict(morphology_state.feature_maps),
        "prep_maps": {**prep_maps, **etch_maps},
        "prep_timeline": prep_timeline,
        "prep_summary": prep_summary,
        "etch_summary": etch_summary,
        "etch_concentration": etch_concentration,
        "etch_maps": etch_maps,
        "texture_profile": {
            "profile_id": str(synthesis_profile.profile_id),
            "phase_topology_mode": "pro_explicit_surface",
            "system_generator_mode": "pro_fe_c",
            "contrast_target": float(synthesis_profile.contrast_target),
            "boundary_sharpness": float(synthesis_profile.boundary_sharpness),
            "artifact_level": float(synthesis_profile.artifact_level),
            "composition_effect": composition_effect,
            "phase_visibility_report": phase_visibility_report,
            "engineering_trace": {
                "generation_mode": str(synthesis_profile.generation_mode),
                "phase_emphasis_style": str(synthesis_profile.phase_emphasis_style),
                "phase_fraction_tolerance_pct": float(
                    synthesis_profile.phase_fraction_tolerance_pct
                ),
                "homogeneity_level": "continuous_state",
            },
        },
        "composition_effect": composition_effect,
        "phase_visibility_report": phase_visibility_report,
        "engineering_trace": {
            "generation_mode": str(synthesis_profile.generation_mode),
            "phase_emphasis_style": str(synthesis_profile.phase_emphasis_style),
            "phase_fraction_tolerance_pct": float(
                synthesis_profile.phase_fraction_tolerance_pct
            ),
            "backend": "pro_realistic_fe_c_v1",
            "implicit_baseline_route_applied": bool(implicit_baseline_route_applied),
            "pure_iron_baseline_applied": bool(pure_iron_like),
        },
        "system_generator": {
            "requested_mode": "pro_realistic",
            "resolved_mode": "pro_fe_c",
            "resolved_system": "fe-c",
            "resolved_stage": str(stage),
            "fallback_used": False,
            "selection_reason": "supported_pro_fe_c_stage",
            "confidence": float(max(continuous_state.confidence.values() or [0.0])),
        },
        "fe_c_phase_render": {
            "input_phase_fractions": dict(phase_fractions),
            "normalized_phase_fractions": dict(continuous_state.phase_fractions),
            "rendered_phase_layers": list(morphology_state.phase_masks.keys()),
            "phase_masks_present": True,
            "medium_influence_applied": bool(
                dict(quench_summary or {}).get("effect_applied", False)
            ),
            "quench_effect_applied": bool(
                dict(quench_summary or {}).get("effect_applied", False)
            ),
            "temper_shift_applied": dict(
                dict(quench_summary or {}).get("temper_shift_c", {})
            )
            if isinstance(dict(quench_summary or {}).get("temper_shift_c", {}), dict)
            else {},
            "retained_austenite_used": float(
                continuous_state.retained_austenite_fraction
            ),
            "homogeneity_mode": "continuous_state",
            "specialized_realism_mode": "continuous_state_surface_optics",
            "fragment_filter_mode": "spatial_morphology",
            "fraction_source": str(phase_fraction_source),
            "table_locked": bool(str(phase_calibration_mode) == "table_interpolated"),
            "morphology_trace": morphology_trace,
        },
        "transformation_trace": {
            "ferrite_fraction": float(continuous_state.ferrite_fraction),
            "pearlite_fraction": float(continuous_state.pearlite_fraction),
            "cementite_fraction": float(continuous_state.cementite_fraction),
            "martensite_fraction": float(continuous_state.martensite_fraction),
            "bainite_fraction": float(continuous_state.bainite_fraction),
            "retained_austenite_fraction": float(
                continuous_state.retained_austenite_fraction
            ),
        },
        "kinetics_model": {
            "family": "continuous_transformation_state_v1",
            "sources": [
                "S4",
                "S5",
                "S8",
                "S9",
                "S11",
                "S14",
                "S15",
                "S17",
                "Bhadeshia_Steels",
                "Porter_Easterling_ch1",
                "Porter_Easterling_ch2",
                "Porter_Easterling_ch3",
                "Porter_Easterling_ch5",
                "Porter_Easterling_ch6",
            ],
            "phase_fraction_source": str(phase_fraction_source),
            "phase_calibration_mode": str(phase_calibration_mode),
            "growth_mode": str(continuous_state.growth_mode),
            "partitioning_mode": str(continuous_state.partitioning_mode),
            "incomplete_transformation_limit_active": bool(
                continuous_state.incomplete_transformation_limit_active
            ),
            "ferrite_effective_exposure_s": float(
                continuous_state.ferrite_effective_exposure_s
            ),
            "pearlite_effective_exposure_s": float(
                continuous_state.pearlite_effective_exposure_s
            ),
            "bainite_effective_exposure_s": float(
                continuous_state.bainite_effective_exposure_s
            ),
            "martensite_effective_exposure_s": float(
                continuous_state.martensite_effective_exposure_s
            ),
            "diffusional_equivalent_time_s": float(
                continuous_state.diffusional_equivalent_time_s
            ),
            "hardenability_factor": float(continuous_state.hardenability_factor),
            "continuous_cooling_shift_factor": float(
                continuous_state.continuous_cooling_shift_factor
            ),
            "ferrite_nucleation_drive": float(
                continuous_state.ferrite_nucleation_drive
            ),
            "pearlite_nucleation_drive": float(
                continuous_state.pearlite_nucleation_drive
            ),
            "bainite_nucleation_drive": float(
                continuous_state.bainite_nucleation_drive
            ),
            "ferrite_progress": float(continuous_state.ferrite_progress),
            "pearlite_progress": float(continuous_state.pearlite_progress),
            "ferrite_pearlite_competition_index": float(
                continuous_state.ferrite_pearlite_competition_index
            ),
            "bainite_activation_progress": float(
                continuous_state.bainite_activation_progress
            ),
            "martensite_conversion_progress": float(
                continuous_state.martensite_conversion_progress
            ),
        },
        "morphology_state": {
            "transformation_family": str(continuous_state.transformation_family),
            "ferrite_morphology_family": str(
                continuous_state.ferrite_morphology_family
            ),
            "bainite_morphology_family": str(
                continuous_state.bainite_morphology_family
            ),
            "martensite_morphology_family": str(
                continuous_state.martensite_morphology_family
            ),
            "pearlite_morphology_family": str(
                continuous_state.pearlite_morphology_family
            ),
            "prior_austenite_grain_size_um": float(
                continuous_state.prior_austenite_grain_size_um
            ),
            "colony_size_um_mean": float(continuous_state.colony_size_um_mean),
            "interlamellar_spacing_um_mean": float(
                continuous_state.interlamellar_spacing_um_mean
            ),
            "martensite_packet_size_um": float(
                continuous_state.martensite_packet_size_um
            ),
            "bainite_sheaf_length_um": float(continuous_state.bainite_sheaf_length_um),
            "bainite_sheaf_thickness_um": float(
                continuous_state.bainite_sheaf_thickness_um
            ),
            "ae1_temperature_c": float(continuous_state.ae1_temperature_c),
            "ae3_temperature_c": float(continuous_state.ae3_temperature_c),
            "bs_temperature_c": float(continuous_state.bs_temperature_c),
            "ms_temperature_c": float(continuous_state.ms_temperature_c),
            "t0_temperature_c": float(continuous_state.t0_temperature_c),
            "austenitization_hold_s": float(continuous_state.austenitization_hold_s),
            "time_in_upper_c_window_s": float(
                continuous_state.time_in_upper_c_window_s
            ),
            "time_in_lower_c_window_s": float(
                continuous_state.time_in_lower_c_window_s
            ),
            "time_below_ms_s": float(continuous_state.time_below_ms_s),
            "time_in_bainite_hold_s": float(continuous_state.time_in_bainite_hold_s),
            "ferrite_effective_exposure_s": float(
                continuous_state.ferrite_effective_exposure_s
            ),
            "pearlite_effective_exposure_s": float(
                continuous_state.pearlite_effective_exposure_s
            ),
            "bainite_effective_exposure_s": float(
                continuous_state.bainite_effective_exposure_s
            ),
            "martensite_effective_exposure_s": float(
                continuous_state.martensite_effective_exposure_s
            ),
            "diffusional_equivalent_time_s": float(
                continuous_state.diffusional_equivalent_time_s
            ),
            "hardenability_factor": float(continuous_state.hardenability_factor),
            "continuous_cooling_shift_factor": float(
                continuous_state.continuous_cooling_shift_factor
            ),
            "ferrite_nucleation_drive": float(
                continuous_state.ferrite_nucleation_drive
            ),
            "pearlite_nucleation_drive": float(
                continuous_state.pearlite_nucleation_drive
            ),
            "bainite_nucleation_drive": float(
                continuous_state.bainite_nucleation_drive
            ),
            "ferrite_progress": float(continuous_state.ferrite_progress),
            "pearlite_progress": float(continuous_state.pearlite_progress),
            "ferrite_pearlite_competition_index": float(
                continuous_state.ferrite_pearlite_competition_index
            ),
            "bainite_activation_progress": float(
                continuous_state.bainite_activation_progress
            ),
            "martensite_conversion_progress": float(
                continuous_state.martensite_conversion_progress
            ),
            "native_um_per_px": float(native_um_per_px),
        },
        "precipitation_state": {
            "carbide_size_um": float(continuous_state.carbide_size_um),
            "recovery_level": float(continuous_state.recovery_level),
        },
        "validation_against_rules": {
            "supported_stage": True,
            "resolved_stage": str(stage),
            "phase_fraction_source": str(phase_fraction_source),
        },
        "pure_iron_baseline": pure_iron_baseline,
        "continuous_transformation_state": continuous_state.to_metadata(),
        "spatial_morphology_state": morphology_state.to_metadata(),
        "surface_state_summary": surface_state.to_metadata(),
        "reflected_light_model": reflected_light_model,
        "validation_pro": validation_pro,
    }
