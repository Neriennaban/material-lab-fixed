from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .contracts_v2 import ValidationReport


@dataclass(slots=True)
class ThermalTransitionV3:
    model: str = "linear"  # linear | sigmoid | power | cosine
    curvature: float = 1.0
    segment_medium_code: str = "inherit"  # inherit | water_20 | water_100 | brine_20_30 | oil_20_80 | polymer | air | furnace | custom
    segment_medium_factor: float | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ThermalTransitionV3":
        data = payload or {}
        factor = data.get("segment_medium_factor")
        return cls(
            model=str(data.get("model", "linear")),
            curvature=float(data.get("curvature", 1.0)),
            segment_medium_code=str(data.get("segment_medium_code", "inherit")),
            segment_medium_factor=(None if factor is None else float(factor)),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ThermalPointV3:
    time_s: float
    temperature_c: float
    label: str = ""
    locked: bool = False
    transition_to_next: ThermalTransitionV3 = field(default_factory=ThermalTransitionV3)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ThermalPointV3":
        data = payload or {}
        return cls(
            time_s=float(data.get("time_s", 0.0)),
            temperature_c=float(data.get("temperature_c", 20.0)),
            label=str(data.get("label", "")),
            locked=bool(data.get("locked", False)),
            transition_to_next=ThermalTransitionV3.from_dict(data.get("transition_to_next")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QuenchSettingsV3:
    medium_code: str = "water_20"
    quench_time_s: float = 30.0
    bath_temperature_c: float = 20.0
    sample_temperature_c: float = 840.0
    custom_medium_name: str = ""
    custom_severity_factor: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "QuenchSettingsV3":
        data = payload or {}
        return cls(
            medium_code=str(data.get("medium_code", "water_20")),
            quench_time_s=float(data.get("quench_time_s", 30.0)),
            bath_temperature_c=float(data.get("bath_temperature_c", 20.0)),
            sample_temperature_c=float(data.get("sample_temperature_c", 840.0)),
            custom_medium_name=str(data.get("custom_medium_name", "")),
            custom_severity_factor=float(data.get("custom_severity_factor", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ThermalProgramV3:
    points: list[ThermalPointV3] = field(
        default_factory=lambda: [
            ThermalPointV3(time_s=0.0, temperature_c=20.0, label="Старт"),
            ThermalPointV3(time_s=600.0, temperature_c=840.0, label="Нагрев"),
            ThermalPointV3(time_s=720.0, temperature_c=840.0, label="Выдержка"),
            ThermalPointV3(time_s=900.0, temperature_c=20.0, label="Охлаждение"),
        ]
    )
    quench: QuenchSettingsV3 = field(default_factory=QuenchSettingsV3)
    sampling_mode: str = "per_degree"  # per_degree | points
    degree_step_c: float = 1.0
    max_frames: int = 320

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ThermalProgramV3":
        data = payload or {}
        raw_points = data.get("points", [])
        points: list[ThermalPointV3] = []
        if isinstance(raw_points, list):
            for item in raw_points:
                if isinstance(item, dict):
                    points.append(ThermalPointV3.from_dict(item))
        if len(points) < 2:
            points = cls().points
        return cls(
            points=points,
            quench=QuenchSettingsV3.from_dict(data.get("quench")),
            sampling_mode=str(data.get("sampling_mode", "per_degree")),
            degree_step_c=float(data.get("degree_step_c", 1.0)),
            max_frames=int(data.get("max_frames", 320)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [point.to_dict() for point in self.points],
            "quench": self.quench.to_dict(),
            "sampling_mode": str(self.sampling_mode),
            "degree_step_c": float(self.degree_step_c),
            "max_frames": int(self.max_frames),
        }


@dataclass(slots=True)
class PrepOperationV3:
    method: str
    duration_s: float = 0.0
    abrasive_um: float | None = None
    load_n: float | None = None
    rpm: float | None = None
    coolant: str | None = None
    note: str = ""
    direction_deg: float = 0.0
    load_profile: str = "constant"  # constant | ramp_up | ramp_down | pulse
    cloth_type: str = "standard"
    slurry_type: str = "diamond"
    lubricant_flow_ml_min: float = 0.0
    cleaning_between_steps: bool = False
    oscillation_hz: float = 0.0
    path_pattern: str = "linear"  # linear | circular | figure8 | random
    electrolyte_code: str | None = None
    voltage_v: float | None = None
    current_density_a_cm2: float | None = None
    electrolyte_temperature_c: float | None = None
    probe_tip_radius_mm: float | None = None
    spot_diameter_mm: float | None = None
    electrolyte_refresh_interval_s: float | None = None
    movement_pattern: str = "none"
    post_polish_followup: str = "none"  # none | chemical_etch | electrolytic_etch

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PrepOperationV3":
        data = payload or {}
        return cls(
            method=str(data.get("method", "")).strip(),
            duration_s=float(data.get("duration_s", 0.0)),
            abrasive_um=(None if data.get("abrasive_um") is None else float(data.get("abrasive_um"))),
            load_n=(None if data.get("load_n") is None else float(data.get("load_n"))),
            rpm=(None if data.get("rpm") is None else float(data.get("rpm"))),
            coolant=(None if data.get("coolant") in (None, "") else str(data.get("coolant"))),
            note=str(data.get("note", "")),
            direction_deg=float(data.get("direction_deg", 0.0)),
            load_profile=str(data.get("load_profile", "constant")),
            cloth_type=str(data.get("cloth_type", "standard")),
            slurry_type=str(data.get("slurry_type", "diamond")),
            lubricant_flow_ml_min=float(data.get("lubricant_flow_ml_min", 0.0)),
            cleaning_between_steps=bool(data.get("cleaning_between_steps", False)),
            oscillation_hz=float(data.get("oscillation_hz", 0.0)),
            path_pattern=str(data.get("path_pattern", "linear")),
            electrolyte_code=(None if data.get("electrolyte_code") in (None, "") else str(data.get("electrolyte_code"))),
            voltage_v=(None if data.get("voltage_v") is None else float(data.get("voltage_v"))),
            current_density_a_cm2=(
                None if data.get("current_density_a_cm2") is None else float(data.get("current_density_a_cm2"))
            ),
            electrolyte_temperature_c=(
                None if data.get("electrolyte_temperature_c") is None else float(data.get("electrolyte_temperature_c"))
            ),
            probe_tip_radius_mm=(None if data.get("probe_tip_radius_mm") is None else float(data.get("probe_tip_radius_mm"))),
            spot_diameter_mm=(None if data.get("spot_diameter_mm") is None else float(data.get("spot_diameter_mm"))),
            electrolyte_refresh_interval_s=(
                None if data.get("electrolyte_refresh_interval_s") is None else float(data.get("electrolyte_refresh_interval_s"))
            ),
            movement_pattern=str(data.get("movement_pattern", "none")),
            post_polish_followup=str(data.get("post_polish_followup", "none")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SamplePrepRouteV3:
    steps: list[PrepOperationV3] = field(default_factory=list)
    roughness_target_um: float = 0.05
    relief_mode: str = "hardness_coupled"  # hardness_coupled | phase_coupled
    contamination_level: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SamplePrepRouteV3":
        data = payload or {}
        raw_steps = data.get("steps", [])
        steps: list[PrepOperationV3] = []
        if isinstance(raw_steps, list):
            for item in raw_steps:
                if isinstance(item, dict):
                    steps.append(PrepOperationV3.from_dict(item))
        return cls(
            steps=steps,
            roughness_target_um=float(data.get("roughness_target_um", 0.05)),
            relief_mode=str(data.get("relief_mode", "hardness_coupled")),
            contamination_level=float(data.get("contamination_level", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "roughness_target_um": float(self.roughness_target_um),
            "relief_mode": str(self.relief_mode),
            "contamination_level": float(self.contamination_level),
        }


@dataclass(slots=True)
class EtchProfileV3:
    reagent: str = "nital_2"
    etch_mode: str = "chemical"  # chemical | electrolytic
    time_s: float = 8.0
    temperature_c: float = 22.0
    agitation: str = "gentle"  # none | gentle | active
    overetch_factor: float = 1.0
    concentration_value: float = 2.0
    concentration_unit: str = "wt_pct"  # wt_pct | mol_l
    concentration_wt_pct: float = 2.0
    concentration_mol_l: float = 0.4
    electrolyte_code: str | None = None
    voltage_v: float | None = None
    voltage_ratio_to_polish: float | None = None
    current_density_a_cm2: float | None = None
    area_mode: str = "global"  # global | local
    requires_prior_electropolish: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "EtchProfileV3":
        data = payload or {}
        return cls(
            reagent=str(data.get("reagent", "nital_2")),
            etch_mode=str(data.get("etch_mode", "chemical")),
            time_s=float(data.get("time_s", 8.0)),
            temperature_c=float(data.get("temperature_c", 22.0)),
            agitation=str(data.get("agitation", "gentle")),
            overetch_factor=float(data.get("overetch_factor", 1.0)),
            concentration_value=float(data.get("concentration_value", data.get("concentration_wt_pct", 2.0))),
            concentration_unit=str(data.get("concentration_unit", "wt_pct")),
            concentration_wt_pct=float(data.get("concentration_wt_pct", 2.0)),
            concentration_mol_l=float(data.get("concentration_mol_l", 0.4)),
            electrolyte_code=(None if data.get("electrolyte_code") in (None, "") else str(data.get("electrolyte_code"))),
            voltage_v=(None if data.get("voltage_v") is None else float(data.get("voltage_v"))),
            voltage_ratio_to_polish=(
                None if data.get("voltage_ratio_to_polish") is None else float(data.get("voltage_ratio_to_polish"))
            ),
            current_density_a_cm2=(
                None if data.get("current_density_a_cm2") is None else float(data.get("current_density_a_cm2"))
            ),
            area_mode=str(data.get("area_mode", "global")),
            requires_prior_electropolish=bool(data.get("requires_prior_electropolish", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SynthesisProfileV3:
    profile_id: str = "textbook_steel_bw"
    phase_topology_mode: str = "auto"
    system_generator_mode: str = "system_auto"
    contrast_target: float = 1.0
    boundary_sharpness: float = 1.0
    artifact_level: float = 0.35
    composition_sensitivity_mode: str = "realistic"
    generation_mode: str = "edu_engineering"  # realistic_visual | edu_engineering | pro_realistic
    phase_emphasis_style: str = "contrast_texture"  # contrast_texture | max_contrast | morphology_only
    phase_fraction_tolerance_pct: float = 20.0
    # A10.0 — colour mode selects the downstream palette that the
    # post-process colourer applies to the final grayscale frame.
    # Default ``"grayscale_nital"`` preserves the legacy one-channel
    # output; other values (``nital_warm``, ``dic_polarized``,
    # ``tint_etch_blue_yellow``) switch on RGB rendering in
    # ``fe_c_color_palette.apply_color_palette``. Not supported when
    # ``generation_mode == "pro_realistic"``.
    color_mode: str = "grayscale_nital"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SynthesisProfileV3":
        data = payload or {}
        return cls(
            profile_id=str(data.get("profile_id", "textbook_steel_bw")),
            phase_topology_mode=str(data.get("phase_topology_mode", "auto")),
            system_generator_mode=str(data.get("system_generator_mode", "system_auto")),
            contrast_target=float(data.get("contrast_target", 1.0)),
            boundary_sharpness=float(data.get("boundary_sharpness", 1.0)),
            artifact_level=float(data.get("artifact_level", 0.35)),
            composition_sensitivity_mode=str(data.get("composition_sensitivity_mode", "realistic")),
            generation_mode=str(data.get("generation_mode", "edu_engineering")),
            phase_emphasis_style=str(data.get("phase_emphasis_style", "contrast_texture")),
            phase_fraction_tolerance_pct=float(data.get("phase_fraction_tolerance_pct", 20.0)),
            color_mode=str(data.get("color_mode", "grayscale_nital")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PhaseModelConfigV3:
    engine: str = "explicit_rules_v3"
    phase_control_mode: str = "auto_with_override"  # auto_with_override | auto_only | manual_only
    manual_phase_fractions: dict[str, float] = field(default_factory=dict)
    manual_override_weight: float = 0.35
    allow_custom_fallback: bool = True
    phase_balance_tolerance_pct: float = 20.0
    # A0.1 — opt-in stage override. When set to one of the new
    # specialised identifiers (``white_cast_iron_*`` or
    # ``bainite_upper`` / ``bainite_lower``) the pipeline routes the
    # render through the dedicated dispatcher in ``fe_c_unified``,
    # bypassing the auto-resolver. Empty string and ``"auto"`` are
    # treated as no override.
    requested_stage: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PhaseModelConfigV3":
        data = payload or {}
        manual: dict[str, float] = {}
        for k, v in dict(data.get("manual_phase_fractions", {})).items():
            try:
                vv = float(v)
            except Exception:
                continue
            if vv > 0.0:
                manual[str(k)] = vv
        return cls(
            engine=str(data.get("engine", "explicit_rules_v3")),
            phase_control_mode=str(data.get("phase_control_mode", "auto_with_override")),
            manual_phase_fractions=manual,
            manual_override_weight=float(data.get("manual_override_weight", 0.35)),
            allow_custom_fallback=bool(data.get("allow_custom_fallback", True)),
            phase_balance_tolerance_pct=float(data.get("phase_balance_tolerance_pct", 20.0)),
            requested_stage=str(data.get("requested_stage", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MetallographyRequestV3:
    sample_id: str = "sample_v3"
    composition_wt: dict[str, float] = field(default_factory=dict)
    system_hint: str | None = None
    material_grade: str | None = None
    material_class_ru: str | None = None
    lab_work: str | None = None
    target_astm_grain_size: float | None = None
    mean_grain_diameter_um: float | None = None
    expected_properties: dict[str, Any] = field(default_factory=dict)
    preset_metadata: dict[str, Any] = field(default_factory=dict)
    thermal_program: ThermalProgramV3 = field(default_factory=ThermalProgramV3)
    prep_route: SamplePrepRouteV3 = field(default_factory=SamplePrepRouteV3)
    etch_profile: EtchProfileV3 = field(default_factory=EtchProfileV3)
    synthesis_profile: SynthesisProfileV3 = field(default_factory=SynthesisProfileV3)
    phase_model: PhaseModelConfigV3 = field(default_factory=PhaseModelConfigV3)
    microscope_profile: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    resolution: tuple[int, int] = (1024, 1024)  # HxW
    strict_validation: bool = True
    reference_profile_id: str | None = None
    generate_intermediate_renders: bool = False  # Генерировать промежуточные рендеры для каждой точки термопрограммы

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetallographyRequestV3":
        data = dict(payload or {})
        if "thermo" in data:
            raise ValueError("LEGACY_FIELD_REMOVED: field 'thermo' is not supported in V3 request.")
        if "process_route" in data:
            raise ValueError("LEGACY_FIELD_REMOVED: field 'process_route' is not supported in V3 request.")
        resolution = data.get("resolution", [1024, 1024])
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            h, w = int(resolution[0]), int(resolution[1])
        else:
            h, w = 1024, 1024
        thermal_program = ThermalProgramV3.from_dict(data.get("thermal_program"))
        prep_route = SamplePrepRouteV3.from_dict(data.get("prep_route"))
        etch = EtchProfileV3.from_dict(data.get("etch_profile"))
        synthesis = SynthesisProfileV3.from_dict(data.get("synthesis_profile"))
        phase_model = PhaseModelConfigV3.from_dict(data.get("phase_model"))
        return cls(
            sample_id=str(data.get("sample_id", "sample_v3")),
            composition_wt={str(k): float(v) for k, v in dict(data.get("composition_wt", {})).items()},
            system_hint=(None if data.get("system_hint") in (None, "") else str(data.get("system_hint"))),
            material_grade=(None if data.get("material_grade") in (None, "") else str(data.get("material_grade"))),
            material_class_ru=(
                None if data.get("material_class_ru") in (None, "") else str(data.get("material_class_ru"))
            ),
            lab_work=(None if data.get("lab_work") in (None, "") else str(data.get("lab_work"))),
            target_astm_grain_size=(
                None
                if data.get("target_astm_grain_size") in (None, "")
                else float(data.get("target_astm_grain_size"))
            ),
            mean_grain_diameter_um=(
                None
                if data.get("mean_grain_diameter_um") in (None, "")
                else float(data.get("mean_grain_diameter_um"))
            ),
            expected_properties=dict(data.get("expected_properties", {})),
            preset_metadata=dict(data.get("metadata", {})),
            thermal_program=thermal_program,
            prep_route=prep_route,
            etch_profile=etch,
            synthesis_profile=synthesis,
            phase_model=phase_model,
            microscope_profile=dict(data.get("microscope_profile", {})),
            seed=int(data.get("seed", 42)),
            resolution=(h, w),
            strict_validation=bool(data.get("strict_validation", True)),
            reference_profile_id=(
                None if data.get("reference_profile_id") in (None, "") else str(data.get("reference_profile_id"))
            ),
            generate_intermediate_renders=bool(data.get("generate_intermediate_renders", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "composition_wt": dict(self.composition_wt),
            "system_hint": self.system_hint,
            "material_grade": self.material_grade,
            "material_class_ru": self.material_class_ru,
            "lab_work": self.lab_work,
            "target_astm_grain_size": self.target_astm_grain_size,
            "mean_grain_diameter_um": self.mean_grain_diameter_um,
            "expected_properties": dict(self.expected_properties),
            "metadata": dict(self.preset_metadata),
            "thermal_program": self.thermal_program.to_dict(),
            "prep_route": self.prep_route.to_dict(),
            "etch_profile": self.etch_profile.to_dict(),
            "synthesis_profile": self.synthesis_profile.to_dict(),
            "phase_model": self.phase_model.to_dict(),
            "microscope_profile": dict(self.microscope_profile),
            "seed": int(self.seed),
            "resolution": [int(self.resolution[0]), int(self.resolution[1])],
            "strict_validation": bool(self.strict_validation),
            "reference_profile_id": self.reference_profile_id,
            "generate_intermediate_renders": bool(self.generate_intermediate_renders),
        }


@dataclass(slots=True)
class IntermediateRenderV3:
    """Промежуточный рендер для одной точки термопрограммы."""
    point_index: int
    time_s: float
    temperature_c: float
    label: str
    image_rgb: np.ndarray
    image_gray: np.ndarray
    phase_info: dict[str, Any]  # Информация о фазах на этой точке


@dataclass(slots=True)
class GenerationOutputV3:
    image_rgb: np.ndarray
    image_gray: np.ndarray
    phase_masks: dict[str, np.ndarray] | None
    feature_masks: dict[str, np.ndarray] | None
    prep_maps: dict[str, np.ndarray] | None
    metadata: dict[str, Any]
    validation_report: ValidationReport
    intermediate_renders: list[IntermediateRenderV3] = field(default_factory=list)

    def metadata_json_safe(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload["validation_report"] = self.validation_report.to_dict()
        payload["intermediate_renders_count"] = len(self.intermediate_renders)
        return payload


@dataclass
class StudentDataV3:
    """Data visible to students in lab package."""
    sample_id: str
    timestamp: str
    composition_wt: dict[str, float]
    thermal_program: dict[str, Any]
    prep_route: dict[str, Any]
    etch_profile: dict[str, Any]
    seed: int
    resolution: tuple[int, int]
    image_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "timestamp": self.timestamp,
            "composition_wt": self.composition_wt,
            "thermal_program": self.thermal_program,
            "prep_route": self.prep_route,
            "etch_profile": self.etch_profile,
            "seed": self.seed,
            "resolution": list(self.resolution),
            "image_sha256": self.image_sha256,
        }


@dataclass
class TeacherAnswersV3:
    """Protected answers for teachers only."""
    sample_id: str
    image_sha256: str
    phase_fractions: dict[str, float]
    inferred_system: str
    steel_grade: str | None
    carbon_content_calculated: float | None
    verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "image_sha256": self.image_sha256,
            "phase_fractions": self.phase_fractions,
            "inferred_system": self.inferred_system,
            "steel_grade": self.steel_grade,
            "carbon_content_calculated": self.carbon_content_calculated,
            "verification": self.verification or {},
        }
