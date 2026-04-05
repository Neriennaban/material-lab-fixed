from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .cooling_modes import canonicalize_cooling_mode

AlloyComposition = dict[str, float]


@dataclass(slots=True)
class ThermoBackendConfig:
    backend: str = "calphad_py"
    strict_mode: bool = True
    db_profile_path: str | None = None
    db_overrides: dict[str, str] = field(default_factory=dict)
    cache_dir: str | None = None
    cache_policy: str = "hybrid"
    equilibrium_model: str = "global_min"
    scheil_enabled: bool = True
    kinetics_enabled: bool = True
    pressure_pa: float = 101325.0
    t_grid_step_c: float = 10.0
    scheil_dt_c: float = 5.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ThermoBackendConfig":
        data = payload or {}
        return cls(
            backend=str(data.get("backend", "calphad_py")),
            strict_mode=bool(data.get("strict_mode", True)),
            db_profile_path=(None if data.get("db_profile_path") in (None, "") else str(data.get("db_profile_path"))),
            db_overrides={str(k): str(v) for k, v in dict(data.get("db_overrides", {})).items()},
            cache_dir=(None if data.get("cache_dir") in (None, "") else str(data.get("cache_dir"))),
            cache_policy=str(data.get("cache_policy", "hybrid")),
            equilibrium_model=str(data.get("equilibrium_model", "global_min")),
            scheil_enabled=bool(data.get("scheil_enabled", True)),
            kinetics_enabled=bool(data.get("kinetics_enabled", True)),
            pressure_pa=float(data.get("pressure_pa", 101325.0)),
            t_grid_step_c=float(data.get("t_grid_step_c", 10.0)),
            scheil_dt_c=float(data.get("scheil_dt_c", 5.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcessingState:
    temperature_c: float = 20.0
    cooling_mode: str = "equilibrium"
    deformation_pct: float = 0.0
    aging_hours: float = 0.0
    aging_temperature_c: float = 20.0
    pressure_mpa: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        self.cooling_mode = canonicalize_cooling_mode(self.cooling_mode)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ProcessingState":
        data = payload or {}
        return cls(
            temperature_c=float(data.get("temperature_c", 20.0)),
            cooling_mode=str(data.get("cooling_mode", "equilibrium")),
            deformation_pct=float(data.get("deformation_pct", 0.0)),
            aging_hours=float(data.get("aging_hours", 0.0)),
            aging_temperature_c=float(data.get("aging_temperature_c", data.get("temperature_c", 20.0))),
            pressure_mpa=(None if data.get("pressure_mpa") is None else float(data.get("pressure_mpa"))),
            note=str(data.get("note", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcessingOperation:
    method: str
    temperature_c: float = 20.0
    duration_min: float = 0.0
    cooling_mode: str = "equilibrium"
    deformation_pct: float = 0.0
    strain_rate_s: float = 0.0
    aging_hours: float = 0.0
    aging_temperature_c: float = 20.0
    pressure_mpa: float | None = None
    atmosphere: str = "air"
    note: str = ""

    def __post_init__(self) -> None:
        self.cooling_mode = canonicalize_cooling_mode(self.cooling_mode)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ProcessingOperation":
        data = payload or {}
        return cls(
            method=str(data.get("method", "")).strip(),
            temperature_c=float(data.get("temperature_c", 20.0)),
            duration_min=float(data.get("duration_min", 0.0)),
            cooling_mode=str(data.get("cooling_mode", "equilibrium")),
            deformation_pct=float(data.get("deformation_pct", 0.0)),
            strain_rate_s=float(data.get("strain_rate_s", 0.0)),
            aging_hours=float(data.get("aging_hours", 0.0)),
            aging_temperature_c=float(data.get("aging_temperature_c", data.get("temperature_c", 20.0))),
            pressure_mpa=(None if data.get("pressure_mpa") is None else float(data.get("pressure_mpa"))),
            atmosphere=str(data.get("atmosphere", "air")),
            note=str(data.get("note", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcessRoute:
    operations: list[ProcessingOperation] = field(default_factory=list)
    route_name: str = "route"
    route_notes: str = ""
    step_preview_enabled: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ProcessRoute":
        data = payload or {}
        operations_raw = data.get("operations", [])
        operations: list[ProcessingOperation] = []
        if isinstance(operations_raw, list):
            for item in operations_raw:
                if isinstance(item, dict):
                    operations.append(ProcessingOperation.from_dict(item))
        return cls(
            operations=operations,
            route_name=str(data.get("route_name", "route")),
            route_notes=str(data.get("route_notes", "")),
            step_preview_enabled=bool(data.get("step_preview_enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operations": [op.to_dict() for op in self.operations],
            "route_name": self.route_name,
            "route_notes": self.route_notes,
            "step_preview_enabled": self.step_preview_enabled,
        }


@dataclass(slots=True)
class ValidationReport:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    normalized_composition: AlloyComposition = field(default_factory=dict)
    inferred_system: str = "custom-multicomponent"
    confidence: float = 0.0
    raw_sum_wt: float = 0.0
    normalized_sum_wt: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationRequestV2:
    mode: str = "direct"
    composition: AlloyComposition = field(default_factory=dict)
    processing: ProcessingState = field(default_factory=ProcessingState)
    generator: str = "dendritic_cast"
    generator_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    resolution: tuple[int, int] = (1024, 1024)  # HxW
    microscope_params: dict[str, Any] = field(default_factory=dict)
    preset_name: str | None = None
    auto_normalize: bool = True
    strict_validation: bool = True
    thermo: ThermoBackendConfig | None = None
    process_route: ProcessRoute | None = None
    route_policy: str = "single_state"  # single_state | route_driven
    preview_step_index: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationRequestV2":
        resolution = payload.get("resolution", [1024, 1024])
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            h, w = int(resolution[0]), int(resolution[1])
        else:
            h, w = 1024, 1024

        process_route_raw = payload.get("process_route")
        process_route = None
        if isinstance(process_route_raw, dict):
            process_route = ProcessRoute.from_dict(process_route_raw)

        thermo_raw = payload.get("thermo")
        thermo = None
        if isinstance(thermo_raw, dict):
            thermo = ThermoBackendConfig.from_dict(thermo_raw)

        preview_step = payload.get("preview_step_index")
        preview_step_index = None if preview_step is None else int(preview_step)

        generator_name = str(payload.get("generator", "dendritic_cast")).strip().lower()
        if generator_name == "universal_auto":
            generator_name = "auto"

        return cls(
            mode=str(payload.get("mode", "direct")),
            composition={str(k): float(v) for k, v in dict(payload.get("composition", {})).items()},
            processing=ProcessingState.from_dict(payload.get("processing")),
            generator=generator_name,
            generator_params=dict(payload.get("generator_params", {})),
            seed=int(payload.get("seed", 42)),
            resolution=(h, w),
            microscope_params=dict(payload.get("microscope_params", {})),
            preset_name=(None if payload.get("preset_name") is None else str(payload.get("preset_name"))),
            auto_normalize=bool(payload.get("auto_normalize", True)),
            strict_validation=bool(payload.get("strict_validation", True)),
            thermo=thermo,
            process_route=process_route,
            route_policy=str(payload.get("route_policy", "single_state")),
            preview_step_index=preview_step_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "composition": self.composition,
            "processing": self.processing.to_dict(),
            "generator": self.generator,
            "generator_params": self.generator_params,
            "seed": self.seed,
            "resolution": [self.resolution[0], self.resolution[1]],
            "microscope_params": self.microscope_params,
            "preset_name": self.preset_name,
            "auto_normalize": self.auto_normalize,
            "strict_validation": self.strict_validation,
            "thermo": (None if self.thermo is None else self.thermo.to_dict()),
            "process_route": (None if self.process_route is None else self.process_route.to_dict()),
            "route_policy": self.route_policy,
            "preview_step_index": self.preview_step_index,
        }


@dataclass(slots=True)
class GenerationOutputV2:
    image_rgb: np.ndarray
    image_gray: np.ndarray
    phase_masks: dict[str, np.ndarray] | None
    metadata: dict[str, Any]
    validation_report: ValidationReport

    def metadata_json_safe(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload["validation_report"] = self.validation_report.to_dict()
        return payload
