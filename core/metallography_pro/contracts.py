from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ContinuousTransformationState:
    system: str
    resolved_stage: str
    transformation_family: str
    growth_mode: str
    partitioning_mode: str
    incomplete_transformation_limit_active: bool
    ferrite_morphology_family: str
    bainite_morphology_family: str
    martensite_morphology_family: str
    pearlite_morphology_family: str
    family_weights: dict[str, float]
    phase_fractions: dict[str, float]
    ferrite_fraction: float
    pearlite_fraction: float
    cementite_fraction: float
    martensite_fraction: float
    bainite_fraction: float
    retained_austenite_fraction: float
    ae1_temperature_c: float
    ae3_temperature_c: float
    bs_temperature_c: float
    ms_temperature_c: float
    t0_temperature_c: float
    austenitization_hold_s: float
    time_in_upper_c_window_s: float
    time_in_lower_c_window_s: float
    time_below_ms_s: float
    time_in_bainite_hold_s: float
    ferrite_effective_exposure_s: float
    pearlite_effective_exposure_s: float
    bainite_effective_exposure_s: float
    martensite_effective_exposure_s: float
    diffusional_equivalent_time_s: float
    hardenability_factor: float
    continuous_cooling_shift_factor: float
    ferrite_nucleation_drive: float
    pearlite_nucleation_drive: float
    bainite_nucleation_drive: float
    ferrite_progress: float
    pearlite_progress: float
    ferrite_pearlite_competition_index: float
    bainite_activation_progress: float
    martensite_conversion_progress: float
    prior_austenite_grain_size_um: float
    colony_size_um_mean: float
    colony_size_um_std: float
    interlamellar_spacing_um_mean: float
    interlamellar_spacing_um_std: float
    proeutectoid_boundary_bias: float
    martensite_packet_size_um: float
    bainite_sheaf_length_um: float
    bainite_sheaf_thickness_um: float
    bainite_sheaf_density: float
    carbide_size_um: float
    recovery_level: float
    confidence: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "system": str(self.system),
            "resolved_stage": str(self.resolved_stage),
            "transformation_family": str(self.transformation_family),
            "growth_mode": str(self.growth_mode),
            "partitioning_mode": str(self.partitioning_mode),
            "incomplete_transformation_limit_active": bool(self.incomplete_transformation_limit_active),
            "ferrite_morphology_family": str(self.ferrite_morphology_family),
            "bainite_morphology_family": str(self.bainite_morphology_family),
            "martensite_morphology_family": str(self.martensite_morphology_family),
            "pearlite_morphology_family": str(self.pearlite_morphology_family),
            "family_weights": dict(self.family_weights),
            "phase_fractions": dict(self.phase_fractions),
            "ferrite_fraction": float(self.ferrite_fraction),
            "pearlite_fraction": float(self.pearlite_fraction),
            "cementite_fraction": float(self.cementite_fraction),
            "martensite_fraction": float(self.martensite_fraction),
            "bainite_fraction": float(self.bainite_fraction),
            "retained_austenite_fraction": float(self.retained_austenite_fraction),
            "ae1_temperature_c": float(self.ae1_temperature_c),
            "ae3_temperature_c": float(self.ae3_temperature_c),
            "bs_temperature_c": float(self.bs_temperature_c),
            "ms_temperature_c": float(self.ms_temperature_c),
            "t0_temperature_c": float(self.t0_temperature_c),
            "austenitization_hold_s": float(self.austenitization_hold_s),
            "time_in_upper_c_window_s": float(self.time_in_upper_c_window_s),
            "time_in_lower_c_window_s": float(self.time_in_lower_c_window_s),
            "time_below_ms_s": float(self.time_below_ms_s),
            "time_in_bainite_hold_s": float(self.time_in_bainite_hold_s),
            "ferrite_effective_exposure_s": float(self.ferrite_effective_exposure_s),
            "pearlite_effective_exposure_s": float(self.pearlite_effective_exposure_s),
            "bainite_effective_exposure_s": float(self.bainite_effective_exposure_s),
            "martensite_effective_exposure_s": float(self.martensite_effective_exposure_s),
            "diffusional_equivalent_time_s": float(self.diffusional_equivalent_time_s),
            "hardenability_factor": float(self.hardenability_factor),
            "continuous_cooling_shift_factor": float(self.continuous_cooling_shift_factor),
            "ferrite_nucleation_drive": float(self.ferrite_nucleation_drive),
            "pearlite_nucleation_drive": float(self.pearlite_nucleation_drive),
            "bainite_nucleation_drive": float(self.bainite_nucleation_drive),
            "ferrite_progress": float(self.ferrite_progress),
            "pearlite_progress": float(self.pearlite_progress),
            "ferrite_pearlite_competition_index": float(self.ferrite_pearlite_competition_index),
            "bainite_activation_progress": float(self.bainite_activation_progress),
            "martensite_conversion_progress": float(self.martensite_conversion_progress),
            "prior_austenite_grain_size_um": float(self.prior_austenite_grain_size_um),
            "colony_size_um_mean": float(self.colony_size_um_mean),
            "colony_size_um_std": float(self.colony_size_um_std),
            "interlamellar_spacing_um_mean": float(self.interlamellar_spacing_um_mean),
            "interlamellar_spacing_um_std": float(self.interlamellar_spacing_um_std),
            "proeutectoid_boundary_bias": float(self.proeutectoid_boundary_bias),
            "martensite_packet_size_um": float(self.martensite_packet_size_um),
            "bainite_sheaf_length_um": float(self.bainite_sheaf_length_um),
            "bainite_sheaf_thickness_um": float(self.bainite_sheaf_thickness_um),
            "bainite_sheaf_density": float(self.bainite_sheaf_density),
            "carbide_size_um": float(self.carbide_size_um),
            "recovery_level": float(self.recovery_level),
            "confidence": dict(self.confidence),
            "provenance": dict(self.provenance),
        }


@dataclass(slots=True)
class SpatialMorphologyState:
    phase_label_map: np.ndarray
    phase_masks: dict[str, np.ndarray]
    pag_id_map: np.ndarray
    colony_id_map: np.ndarray
    packet_id_map: np.ndarray | None
    orientation_rad: np.ndarray
    lamella_field: np.ndarray | None
    packet_field: np.ndarray | None
    boundary_class_map: np.ndarray
    feature_maps: dict[str, np.ndarray]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return dict(self.summary)


@dataclass(slots=True)
class SurfaceState:
    height_um: np.ndarray
    etch_depth_um: np.ndarray
    reflectance_base: np.ndarray
    damage_layer: np.ndarray
    smear_map: np.ndarray
    pullout_map: np.ndarray
    contamination_map: np.ndarray
    stain_map: np.ndarray
    roughness_um: np.ndarray
    summary: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return dict(self.summary)
