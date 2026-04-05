from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from export.export_images import save_image
from export.export_tables import save_json, save_measurements_csv

from .auto_generator import AutoGeneratorDecision, select_auto_generator
from .alloy_validation import normalize_composition_keys, validate_alloy
from .cooling_curve import normalize_cooling_curve_points, sample_cooling_curve
from .cooling_modes import canonicalize_cooling_mode, resolve_auto_cooling_mode
from .contracts_v2 import (
    GenerationOutputV2,
    GenerationRequestV2,
    ProcessRoute,
    ProcessingState,
    ThermoBackendConfig,
    ValidationReport,
)
from .diagram_engine import diagram_snapshot_params
from .generator_phase_map import normalize_system, resolve_phase_transition_state
from .generator_registry import GeneratorRegistry
from .imaging import simulate_microscope_view
from .materials import MaterialPreset, list_presets, load_preset
from .processing_simulation import RouteSimulationResult, simulate_process_route
from .route_validation import RouteValidationResult
from .calphad.cache import CalphadCache
from .calphad.db_manager import (
    CALPHAD_SUPPORTED_SYSTEMS,
    CalphadDBReference,
    resolve_database_reference,
    validate_database_reference,
)
from .calphad.engine_pycalphad import run_equilibrium
from .calphad.kinetics import run_jmak_lsw
from .calphad.scheil import run_scheil


@dataclass(slots=True)
class BatchResult:
    rows: list[dict[str, Any]]
    csv_index_path: Path


def _composition_hash(composition: dict[str, float]) -> str:
    normalized = {
        str(k): float(v)
        for k, v in sorted(composition.items(), key=lambda item: item[0])
    }
    payload = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _ensure_uint8_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3].mean(axis=2).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.uint8, copy=False)
        return np.stack([gray] * 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3].astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _phase_fraction_estimate(
    phase_masks: dict[str, np.ndarray] | None, metadata: dict[str, Any]
) -> dict[str, float]:
    if isinstance(phase_masks, dict) and phase_masks:
        out: dict[str, float] = {}
        for key, mask in phase_masks.items():
            if isinstance(mask, np.ndarray):
                out[str(key)] = float((mask > 0).mean())
        if out:
            return out

    for candidate_key in ("phase_fractions", "fractions"):
        candidate = metadata.get(candidate_key)
        if isinstance(candidate, dict):
            return {str(k): float(v) for k, v in candidate.items()}
    return {}


def _blank_route_validation() -> RouteValidationResult:
    return RouteValidationResult(
        is_valid=True, errors=[], warnings=[], normalized_operations=[]
    )


def _resolve_processing_auto_mode(
    processing: ProcessingState, inferred_system: str
) -> ProcessingState:
    mode = canonicalize_cooling_mode(processing.cooling_mode)
    if mode != "auto":
        return ProcessingState(
            temperature_c=processing.temperature_c,
            cooling_mode=mode,
            deformation_pct=processing.deformation_pct,
            aging_hours=processing.aging_hours,
            aging_temperature_c=processing.aging_temperature_c,
            pressure_mpa=processing.pressure_mpa,
            note=processing.note,
        )

    resolved_mode = resolve_auto_cooling_mode(
        inferred_system=inferred_system, processing=processing
    )
    return ProcessingState(
        temperature_c=processing.temperature_c,
        cooling_mode=resolved_mode,
        deformation_pct=processing.deformation_pct,
        aging_hours=processing.aging_hours,
        aging_temperature_c=processing.aging_temperature_c,
        pressure_mpa=processing.pressure_mpa,
        note=processing.note,
    )


def _quantize(value: float, step: float) -> float:
    if step <= 1e-9:
        return float(value)
    return round(float(value) / float(step)) * float(step)


def _transition_events(track: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not track:
        return events

    idx = 0
    while idx < len(track):
        item = track[idx]
        kind = str(item.get("transition_kind", "none"))
        if kind == "none":
            idx += 1
            continue

        start = idx
        end = idx
        while (
            end + 1 < len(track)
            and str(track[end + 1].get("transition_kind", "none")) == kind
        ):
            end += 1

        segment = track[start : end + 1]
        if kind == "melting":
            peak = max(segment, key=lambda x: float(x.get("liquid_fraction", 0.0)))
        else:
            peak = min(segment, key=lambda x: float(x.get("liquid_fraction", 0.0)))

        for event_type, source in (
            ("start", track[start]),
            ("peak", peak),
            ("end", track[end]),
        ):
            events.append(
                {
                    "event": event_type,
                    "transition_kind": kind,
                    "index": int(source.get("index", start)),
                    "time_min": float(source.get("time_min", 0.0)),
                    "temperature_c": float(source.get("temperature_c", 0.0)),
                    "stage": str(source.get("stage", "")),
                    "liquid_fraction": float(source.get("liquid_fraction", 0.0)),
                }
            )

        idx = end + 1

    return events


def _supported_by_calphad(inferred_system: str) -> bool:
    return normalize_system(inferred_system) in set(CALPHAD_SUPPORTED_SYSTEMS)


def _resolve_calphad_system(inferred_system: str, composition: dict[str, float]) -> str:
    system = normalize_system(inferred_system)
    if _supported_by_calphad(system):
        return system

    comp = {str(k): max(0.0, float(v)) for k, v in composition.items()}
    if not comp:
        return system
    total = max(1e-9, float(sum(comp.values())))
    if total <= 0.0:
        return system

    si = float(comp.get("Si", 0.0))
    fe = float(comp.get("Fe", 0.0))
    al = float(comp.get("Al", 0.0))
    cu = float(comp.get("Cu", 0.0))
    zn = float(comp.get("Zn", 0.0))
    mg = float(comp.get("Mg", 0.0))
    c = float(comp.get("C", 0.0))

    scores = {
        "fe-c": (fe + 2.4 * c) if (fe > 0.0 or c > 0.0) else 0.0,
        "fe-si": (fe + 1.5 * si) if (fe > 0.0 or si > 0.0) else 0.0,
        "al-si": (al + 1.6 * si) if (al > 0.0 or si > 0.0) else 0.0,
        "cu-zn": (cu + 1.5 * zn) if (cu > 0.0 or zn > 0.0) else 0.0,
        "al-cu-mg": (al + 1.2 * cu + 1.1 * mg)
        if (al > 0.0 and (cu > 0.0 or mg > 0.0))
        else 0.0,
    }
    best = max(scores, key=scores.get)
    best_share = float(scores[best]) / total
    if best_share < 0.35:
        return "custom-multicomponent"
    return best


def _phase_transition_from_calphad(
    *,
    liquid_fraction: float,
    thermal_slope: float | None,
) -> dict[str, Any]:
    liq = float(max(0.0, min(1.0, liquid_fraction)))
    slope = 0.0 if thermal_slope is None else float(thermal_slope)
    direction = "steady"
    if slope < -1e-9:
        direction = "cooling"
    elif slope > 1e-9:
        direction = "heating"

    transition = "none"
    if 1e-6 < liq < 0.999999:
        if direction == "cooling":
            transition = "crystallization"
        elif direction == "heating":
            transition = "melting"

    return {
        "transition_kind": transition,
        "liquid_fraction": liq,
        "solid_fraction": float(max(0.0, min(1.0, 1.0 - liq))),
        "thermal_direction": direction,
    }


class GenerationPipelineV2:
    def __init__(
        self,
        presets_dir: str | Path | None = None,
        generator_registry: GeneratorRegistry | None = None,
        generator_version: str = "v2.1.0",
        calphad_profile_path: str | Path | None = None,
        calphad_tdb_dir: str | Path | None = None,
        calphad_cache_dir: str | Path | None = None,
        v3_proxy_enabled: bool = False,
    ) -> None:
        self.generator_registry = generator_registry or GeneratorRegistry()
        self.generator_version = generator_version
        if presets_dir is None:
            self.presets_dir = Path(__file__).resolve().parents[1] / "presets"
        else:
            self.presets_dir = Path(presets_dir)
        self.calphad_profile_path = (
            Path(calphad_profile_path)
            if calphad_profile_path is not None
            else Path(__file__).resolve().parents[1]
            / "profiles"
            / "calphad_profile_v2.json"
        )
        self.calphad_tdb_dir = (
            None if calphad_tdb_dir is None else Path(calphad_tdb_dir)
        )
        self.calphad_cache_dir = (
            None if calphad_cache_dir is None else Path(calphad_cache_dir)
        )
        self.calphad_cache = CalphadCache(
            cache_dir=self.calphad_cache_dir
            or (Path(__file__).resolve().parents[1] / ".cache" / "calphad"),
            policy="hybrid",
        )
        self.v3_proxy_enabled = bool(v3_proxy_enabled)
        self._v3_pipeline: Any | None = None

    def _ensure_v3_pipeline(self) -> Any:
        if self._v3_pipeline is not None:
            return self._v3_pipeline
        from .metallography_v3.pipeline_v3 import MetallographyPipelineV3

        self._v3_pipeline = MetallographyPipelineV3(
            presets_dir=Path(__file__).resolve().parents[1] / "presets_v3",
            profiles_dir=Path(__file__).resolve().parents[1] / "profiles_v3",
        )
        return self._v3_pipeline

    def _convert_v2_request_to_v3(self, request: GenerationRequestV2) -> Any:
        from .contracts_v3 import (
            EtchProfileV3,
            MetallographyRequestV3,
            PrepOperationV3,
            QuenchSettingsV3,
            SamplePrepRouteV3,
            SynthesisProfileV3,
            ThermalPointV3,
            ThermalProgramV3,
        )

        prep_steps = [
            PrepOperationV3(
                method="grinding_800",
                duration_s=90.0,
                abrasive_um=18.0,
                load_n=22.0,
                rpm=180.0,
            ),
            PrepOperationV3(
                method="polishing_3um",
                duration_s=120.0,
                abrasive_um=3.0,
                load_n=14.0,
                rpm=140.0,
            ),
            PrepOperationV3(
                method="polishing_1um",
                duration_s=90.0,
                abrasive_um=1.0,
                load_n=10.0,
                rpm=120.0,
            ),
        ]
        prep_route = SamplePrepRouteV3(
            steps=prep_steps,
            roughness_target_um=0.05,
            relief_mode="hardness_coupled",
            contamination_level=0.02,
        )

        start_temp = float(
            max(
                20.0,
                request.processing.temperature_c
                if request.processing.temperature_c > 0
                else 20.0,
            )
        )
        thermal_points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0, label="A"),
            ThermalPointV3(time_s=420.0, temperature_c=start_temp, label="B"),
            ThermalPointV3(time_s=540.0, temperature_c=start_temp, label="C"),
            ThermalPointV3(time_s=900.0, temperature_c=20.0, label="D"),
        ]
        cool_mode = str(request.processing.cooling_mode or "").strip().lower()
        medium_code = "water_20" if cool_mode in {"quench", "quenched"} else "air"
        quench = QuenchSettingsV3(
            medium_code=medium_code,
            quench_time_s=30.0,
            bath_temperature_c=20.0,
            sample_temperature_c=start_temp,
            custom_medium_name="",
            custom_severity_factor=1.0,
        )
        thermal_program = ThermalProgramV3(
            points=thermal_points,
            quench=quench,
            sampling_mode="per_degree",
            degree_step_c=1.0,
            max_frames=320,
        )
        etch_profile = EtchProfileV3(
            reagent=str(request.generator_params.get("etch_reagent", "nital_2")),
            time_s=float(request.generator_params.get("etch_time_s", 8.0)),
            temperature_c=float(
                request.processing.temperature_c
                if request.processing.temperature_c > 0
                else 22.0
            ),
            agitation=str(request.generator_params.get("etch_agitation", "gentle")),
            overetch_factor=float(request.generator_params.get("overetch_factor", 1.0)),
        )
        synthesis = SynthesisProfileV3(profile_id="balanced_realism")
        return MetallographyRequestV3(
            sample_id=str(request.preset_name or "v2_proxy_sample"),
            composition_wt={str(k): float(v) for k, v in request.composition.items()},
            system_hint=None,
            thermal_program=thermal_program,
            prep_route=prep_route,
            etch_profile=etch_profile,
            synthesis_profile=synthesis,
            microscope_profile=dict(request.microscope_params),
            seed=int(request.seed),
            resolution=(int(request.resolution[0]), int(request.resolution[1])),
            strict_validation=bool(request.strict_validation),
            reference_profile_id=None,
        )

    def generate_via_v3_proxy(self, request: GenerationRequestV2) -> GenerationOutputV2:
        v3 = self._ensure_v3_pipeline()
        v3_request = self._convert_v2_request_to_v3(request)
        v3_output = v3.generate(v3_request)
        metadata = dict(v3_output.metadata)
        metadata["request"] = request.to_dict()
        metadata["v3_proxy"] = True
        return GenerationOutputV2(
            image_rgb=v3_output.image_rgb,
            image_gray=v3_output.image_gray,
            phase_masks=v3_output.phase_masks,
            metadata=metadata,
            validation_report=v3_output.validation_report,
        )

    def list_preset_paths(self) -> list[Path]:
        return list_presets(self.presets_dir)

    def load_preset(self, name_or_path: str | Path) -> MaterialPreset:
        candidate = Path(name_or_path)
        if candidate.exists():
            return load_preset(candidate)
        by_name = self.presets_dir / f"{name_or_path}.json"
        if by_name.exists():
            return load_preset(by_name)
        raise FileNotFoundError(f"Preset not found: {name_or_path}")

    def request_from_preset(
        self,
        preset: MaterialPreset,
        seed_override: int | None = None,
        resolution_override: tuple[int, int] | None = None,
    ) -> GenerationRequestV2:
        generation = dict(preset.generation)
        processing = ProcessingState.from_dict(
            {
                "temperature_c": generation.pop("temperature_c", 20.0),
                "cooling_mode": generation.pop("cooling_mode", "equilibrium"),
                "deformation_pct": generation.pop("deformation_pct", 0.0),
                "aging_hours": generation.pop("aging_hours", 0.0),
                "aging_temperature_c": generation.pop("aging_temperature_c", 20.0),
                "pressure_mpa": generation.pop("pressure_mpa", None),
                "note": generation.pop("note", ""),
            }
        )
        resolution = (
            resolution_override
            if resolution_override is not None
            else preset.image_size
        )

        process_route = None
        raw_route = generation.pop("process_route", None)
        if isinstance(raw_route, dict):
            process_route = ProcessRoute.from_dict(raw_route)

        route_policy = str(generation.pop("route_policy", "single_state"))
        preview_step = generation.pop("preview_step_index", None)
        preview_step_index = None if preview_step is None else int(preview_step)

        return GenerationRequestV2(
            mode="preset",
            composition={str(k): float(v) for k, v in preset.composition.items()},
            processing=processing,
            generator=preset.generator,
            generator_params=generation,
            seed=int(seed_override if seed_override is not None else preset.seed),
            resolution=(int(resolution[0]), int(resolution[1])),
            microscope_params=dict(preset.microscope),
            preset_name=preset.name,
            auto_normalize=True,
            strict_validation=True,
            process_route=process_route,
            route_policy=route_policy,
            preview_step_index=preview_step_index,
        )

    def generate_from_preset(
        self,
        name_or_path: str | Path,
        seed_override: int | None = None,
        resolution_override: tuple[int, int] | None = None,
    ) -> GenerationOutputV2:
        preset = self.load_preset(name_or_path)
        request = self.request_from_preset(
            preset=preset,
            seed_override=seed_override,
            resolution_override=resolution_override,
        )
        return self.generate(request)

    def _effective_thermo(
        self, thermo: ThermoBackendConfig | None
    ) -> ThermoBackendConfig:
        cfg = ThermoBackendConfig.from_dict(
            (None if thermo is None else thermo.to_dict())
        )
        if not cfg.db_profile_path:
            cfg.db_profile_path = str(self.calphad_profile_path)
        if not cfg.cache_dir:
            cfg.cache_dir = str(
                self.calphad_cache_dir
                or (Path(__file__).resolve().parents[1] / ".cache" / "calphad")
            )
        if cfg.cache_policy not in {"memory", "disk", "hybrid"}:
            cfg.cache_policy = "hybrid"
        return cfg

    def validate_calphad_setup(
        self, thermo: ThermoBackendConfig | None = None
    ) -> dict[str, Any]:
        cfg = self._effective_thermo(thermo)
        report: dict[str, Any] = {
            "backend": cfg.backend,
            "strict_mode": cfg.strict_mode,
            "systems": {},
        }
        for system in CALPHAD_SUPPORTED_SYSTEMS:
            try:
                db_ref = resolve_database_reference(
                    system=system,
                    thermo=cfg,
                    profile_path=cfg.db_profile_path,
                    tdb_dir=self.calphad_tdb_dir,
                )
                validate_database_reference(db_ref)
                report["systems"][system] = {
                    "ok": True,
                    "path": str(db_ref.path),
                    "source": db_ref.source,
                    "sha256": db_ref.sha256,
                }
            except Exception as exc:
                report["systems"][system] = {"ok": False, "error": str(exc)}
        report["is_valid"] = all(bool(v.get("ok")) for v in report["systems"].values())
        return report

    def _resolve_calphad_reference(
        self,
        *,
        inferred_system: str,
        thermo: ThermoBackendConfig,
    ) -> CalphadDBReference | None:
        system = normalize_system(inferred_system)
        if system not in set(CALPHAD_SUPPORTED_SYSTEMS):
            return None
        db_ref = resolve_database_reference(
            system=system,
            thermo=thermo,
            profile_path=thermo.db_profile_path,
            tdb_dir=self.calphad_tdb_dir,
        )
        validate_database_reference(db_ref)
        return db_ref

    def _run_calphad_suite(
        self,
        *,
        composition: dict[str, float],
        inferred_system: str,
        processing: ProcessingState,
        thermo: ThermoBackendConfig,
        generator_params: dict[str, Any],
        request: GenerationRequestV2,
    ) -> dict[str, Any]:
        system = normalize_system(inferred_system)
        db_ref = self._resolve_calphad_reference(inferred_system=system, thermo=thermo)
        if db_ref is None:
            raise ValueError(f"SYSTEM_UNSUPPORTED: {inferred_system}")

        eq = run_equilibrium(
            db_ref=db_ref,
            system=system,
            composition=composition,
            temperature_c=float(processing.temperature_c),
            pressure_pa=float(thermo.pressure_pa),
            equilibrium_model=thermo.equilibrium_model,
            cache=self.calphad_cache,
        )

        scheil_result: dict[str, Any] | None = None
        has_cast_ops = False
        if request.process_route is not None:
            for op in request.process_route.operations:
                method = str(op.method).strip().lower()
                if method.startswith("cast_") or method == "directional_solidification":
                    has_cast_ops = True
                    break
        if thermo.scheil_enabled and (
            bool(generator_params.get("cooling_curve_enabled"))
            or has_cast_ops
            or eq.liquid_fraction > 0.05
        ):
            points = normalize_cooling_curve_points(
                generator_params.get("cooling_curve", []),
                fallback_temperature_c=float(processing.temperature_c),
            )
            if points and len(points) >= 2:
                t_start = float(points[0]["temperature_c"])
                t_end = float(points[-1]["temperature_c"])
            else:
                t_start = float(max(processing.temperature_c, 1200.0))
                t_end = float(min(processing.temperature_c, 20.0))
            scheil_result = run_scheil(
                db_ref=db_ref,
                system=system,
                composition=composition,
                t_start_c=t_start,
                t_end_c=t_end,
                d_t_c=float(max(0.1, thermo.scheil_dt_c)),
                pressure_pa=float(thermo.pressure_pa),
                equilibrium_model=thermo.equilibrium_model,
                cache=self.calphad_cache,
            )

        kinetics_result: dict[str, Any] | None = None
        if thermo.kinetics_enabled and float(processing.aging_hours) > 0.0:
            kinetics_result = run_jmak_lsw(
                system=system,
                temperature_c=float(processing.aging_temperature_c),
                aging_hours=float(processing.aging_hours),
                base_phase_fractions=eq.stable_phases,
            )

        calphad_payload = {
            "backend": thermo.backend,
            "strict_mode": bool(thermo.strict_mode),
            "database_used": str(db_ref.path),
            "database_hash": db_ref.sha256,
            "solver_status": eq.solver_status,
            "compute_time_ms": float(eq.compute_time_ms),
            "equilibrium_result": eq.to_dict(),
            "scheil_result": (scheil_result or {"enabled": False}),
            "kinetics_result": (kinetics_result or {"enabled": False}),
            "calculation_confidence": (
                0.92
                if eq.solver_status == "ok"
                else 0.62
                if eq.solver_status == "approx_fallback"
                else 0.45
            ),
        }
        return calphad_payload

    def _resolve_route(
        self,
        request: GenerationRequestV2,
        composition: dict[str, float],
        inferred_system: str,
        initial_processing: ProcessingState,
    ) -> RouteSimulationResult | None:
        route = request.process_route
        if route is None:
            return None
        if request.route_policy.strip().lower() != "route_driven":
            return None
        if not route.operations:
            return None

        return simulate_process_route(
            composition=composition,
            inferred_system=inferred_system,
            route=route,
            initial_processing=initial_processing,
            generator=request.generator,
            base_seed=request.seed,
            step_preview_index=request.preview_step_index,
        )

    def _extract_cooling_curve_config(
        self, request: GenerationRequestV2
    ) -> dict[str, Any]:
        params = dict(request.generator_params or {})
        enabled = bool(params.get("cooling_curve_enabled", False))
        raw_points = params.get("cooling_curve")
        if not enabled or not isinstance(raw_points, list):
            return {"enabled": False}

        points = normalize_cooling_curve_points(
            raw_points,
            fallback_temperature_c=float(request.processing.temperature_c),
        )
        if len(points) < 2:
            return {"enabled": False}

        mode = str(params.get("cooling_curve_mode", "per_degree")).strip().lower()
        if mode not in {"per_degree", "points"}:
            mode = "per_degree"
        degree_step = float(params.get("cooling_curve_degree_step", 1.0))
        max_points = int(params.get("cooling_curve_max_points", 220))
        return {
            "enabled": True,
            "points": points,
            "mode": mode,
            "degree_step": max(0.1, degree_step),
            "max_points": max(5, max_points),
        }

    def generate_cooling_curve_series(
        self, request: GenerationRequestV2
    ) -> list[GenerationOutputV2]:
        config = self._extract_cooling_curve_config(request)
        if not config.get("enabled"):
            return []

        samples = sample_cooling_curve(
            config["points"],
            mode=str(config["mode"]),
            degree_step=float(config["degree_step"]),
            max_points=int(config["max_points"]),
            base_mode=request.processing.cooling_mode,
        )
        if not samples:
            return []

        validation = validate_alloy(
            composition=request.composition,
            processing=request.processing,
            auto_normalize=request.auto_normalize,
            strict_custom_limits=request.strict_validation,
        )
        composition = (
            dict(validation.normalized_composition)
            if validation.normalized_composition
            else normalize_composition_keys(request.composition)[0]
        )
        inferred_system = normalize_system(validation.inferred_system)
        thermo = self._effective_thermo(request.thermo)
        calphad_system = _resolve_calphad_system(inferred_system, composition)
        calphad_supported = _supported_by_calphad(calphad_system)
        requested_stage = str((request.generator_params or {}).get("stage", "auto"))
        microscope_signature = json.dumps(
            request.microscope_params,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        auto_strict_calphad = (
            request.generator.strip().lower() == "auto"
            and bool(thermo.strict_mode)
            and calphad_supported
        )
        phase_cache_enabled = (
            request.route_policy.strip().lower() != "route_driven"
            and request.process_route is None
            and request.preview_step_index is None
            and request.generator.strip().lower() in {"auto", "phase_map"}
            and inferred_system in {"fe-c", "al-si", "cu-zn", "fe-si"}
        )

        outputs: list[GenerationOutputV2] = []
        track: list[dict[str, Any]] = []
        render_cache: dict[str, GenerationOutputV2] = {}
        cache_hits = 0

        for sample in samples:
            sample_processing = ProcessingState(
                temperature_c=float(sample["temperature_c"]),
                cooling_mode=str(sample["cooling_mode"]),
                deformation_pct=float(request.processing.deformation_pct),
                aging_hours=float(request.processing.aging_hours),
                aging_temperature_c=float(request.processing.aging_temperature_c),
                pressure_mpa=request.processing.pressure_mpa,
                note=request.processing.note,
            )
            phase_state = resolve_phase_transition_state(
                system=inferred_system,
                composition=composition,
                processing=sample_processing,
                thermal_slope=float(sample["slope_c_per_min"]),
                requested_stage=requested_stage,
            )

            liquid_fraction = float(phase_state.get("liquid_fraction", 0.0))
            stage = str(phase_state.get("stage", "unknown"))
            if phase_cache_enabled:
                liq_quant = _quantize(liquid_fraction, 0.04)
                key_payload = {
                    "mode": "phase_state_cache",
                    "generator": "phase_map",
                    "system": inferred_system,
                    "stage": stage,
                    "liquid_fraction_q": liq_quant,
                    "resolution": list(request.resolution),
                    "seed": int(request.seed),
                    "microscope": microscope_signature,
                }
            else:
                key_payload = {
                    "mode": "full_request",
                    "generator": request.generator,
                    "temperature_c": float(sample_processing.temperature_c),
                    "cooling_mode": str(sample_processing.cooling_mode),
                    "resolution": list(request.resolution),
                    "seed": int(request.seed),
                    "microscope": microscope_signature,
                }
            render_key = json.dumps(
                key_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )

            if render_key in render_cache:
                template = render_cache[render_key]
                cache_hits += 1
            else:
                req = GenerationRequestV2.from_dict(request.to_dict())
                req.route_policy = "single_state"
                req.process_route = None
                req.preview_step_index = None
                req.processing = sample_processing
                if phase_cache_enabled:
                    req.generator = "phase_map"
                    params = dict(req.generator_params or {})
                    params["system"] = inferred_system
                    params["stage"] = stage
                    params["liquid_fraction"] = float(_quantize(liquid_fraction, 0.04))
                    params["thermal_slope"] = float(sample["slope_c_per_min"])
                    params["temperature_c"] = float(sample_processing.temperature_c)
                    params["cooling_mode"] = str(sample_processing.cooling_mode)
                    req.generator_params = params
                template = self.generate(req)
                render_cache[render_key] = template

            out = GenerationOutputV2(
                image_rgb=template.image_rgb,
                image_gray=template.image_gray,
                phase_masks=template.phase_masks,
                metadata=dict(template.metadata),
                validation_report=template.validation_report,
            )
            request_meta = out.metadata.get("request")
            if isinstance(request_meta, dict):
                req_proc = request_meta.get("processing")
                if isinstance(req_proc, dict):
                    req_proc["temperature_c"] = float(sample_processing.temperature_c)
                    req_proc["cooling_mode"] = str(sample_processing.cooling_mode)
            out.metadata["diagram_snapshot_params"] = diagram_snapshot_params(
                composition=composition,
                processing=sample_processing,
                requested_system=(
                    (request.generator_params or {}).get("system")
                    if isinstance(request.generator_params, dict)
                    else None
                ),
                inferred_system=inferred_system,
                confidence=validation.confidence,
            )
            out.metadata["stage"] = stage
            point_payload = {
                "index": int(sample["index"]),
                "time_min": float(sample["time_min"]),
                "temperature_c": float(sample["temperature_c"]),
                "cooling_mode": str(sample["cooling_mode"]),
                "slope_c_per_min": float(sample["slope_c_per_min"]),
            }
            out.metadata["cooling_curve_point"] = point_payload
            transition_payload = {
                "stage": stage,
                "transition_kind": str(phase_state.get("transition_kind", "none")),
                "liquid_fraction": float(phase_state.get("liquid_fraction", 0.0)),
                "solid_fraction": float(phase_state.get("solid_fraction", 1.0)),
                "thermal_direction": str(
                    phase_state.get("thermal_direction", "steady")
                ),
            }
            calphad_meta = out.metadata.get("calphad")
            if (
                phase_cache_enabled
                and calphad_supported
                and not isinstance(calphad_meta, dict)
            ):
                liquid_fraction = float(transition_payload["liquid_fraction"])
                solid_fraction = float(transition_payload["solid_fraction"])
                stage_key = str(stage or "solid").upper()
                stable_phases = {
                    ("LIQUID" if liquid_fraction >= 0.999 else stage_key): (
                        1.0 if liquid_fraction >= 0.999 else solid_fraction
                    )
                }
                if 0.0 < liquid_fraction < 1.0:
                    stable_phases = {
                        stage_key: solid_fraction,
                        "LIQUID": liquid_fraction,
                    }
                calphad_meta = {
                    "equilibrium_result": {
                        "system": calphad_system,
                        "stable_phases": stable_phases,
                        "liquid_fraction": liquid_fraction,
                        "solid_fraction": solid_fraction,
                        "chemical_potentials": {},
                        "solver_status": "series_phase_cache",
                        "compute_time_ms": 0.0,
                        "temperature_c": float(sample_processing.temperature_c),
                        "pressure_pa": float(thermo.pressure_pa),
                    },
                    "series_cached": True,
                }
                out.metadata["calphad"] = calphad_meta
            if isinstance(calphad_meta, dict):
                eq = calphad_meta.get("equilibrium_result")
                if isinstance(eq, dict):
                    liq = float(
                        eq.get("liquid_fraction", transition_payload["liquid_fraction"])
                    )
                    from_calphad = _phase_transition_from_calphad(
                        liquid_fraction=liq,
                        thermal_slope=float(sample["slope_c_per_min"]),
                    )
                    transition_payload.update(from_calphad)
            out.metadata["phase_transition_state"] = transition_payload
            outputs.append(out)
            track.append(
                {
                    "index": int(sample["index"]),
                    "time_min": float(sample["time_min"]),
                    "temperature_c": float(sample["temperature_c"]),
                    "cooling_mode": str(sample["cooling_mode"]),
                    "slope_c_per_min": float(sample["slope_c_per_min"]),
                    "stage": str(out.metadata.get("stage", stage)),
                    "transition_kind": str(
                        transition_payload.get("transition_kind", "none")
                    ),
                    "thermal_direction": str(
                        transition_payload.get("thermal_direction", "steady")
                    ),
                    "liquid_fraction": float(
                        transition_payload.get("liquid_fraction", 0.0)
                    ),
                    "solid_fraction": float(
                        transition_payload.get("solid_fraction", 1.0)
                    ),
                }
            )

        transition_events = _transition_events(track)
        unique_renders = len(render_cache)
        summary = {
            "mode": str(config["mode"]),
            "point_count": len(outputs),
            "temperature_start_c": float(track[0]["temperature_c"])
            if track
            else float(request.processing.temperature_c),
            "temperature_end_c": float(track[-1]["temperature_c"])
            if track
            else float(request.processing.temperature_c),
            "render_cache": {
                "unique_renders": int(unique_renders),
                "cache_hits": int(cache_hits),
                "cache_mode": "phase_state_cache"
                if phase_cache_enabled
                else "full_request",
            },
        }
        for out in outputs:
            out.metadata["cooling_curve_series"] = {
                "enabled": True,
                "config": {
                    "mode": str(config["mode"]),
                    "degree_step": float(config["degree_step"]),
                    "max_points": int(config["max_points"]),
                },
                "points": config["points"],
                "summary": summary,
            }
            out.metadata["phase_transition_track"] = track
            out.metadata["phase_transition_events"] = transition_events
        return outputs

    def generate(self, request: GenerationRequestV2) -> GenerationOutputV2:
        if self.v3_proxy_enabled:
            return self.generate_via_v3_proxy(request)

        validation = validate_alloy(
            composition=request.composition,
            processing=request.processing,
            auto_normalize=request.auto_normalize,
            strict_custom_limits=request.strict_validation,
        )

        if request.strict_validation and not validation.is_valid:
            joined = (
                "; ".join(validation.errors)
                if validation.errors
                else "Validation failed"
            )
            raise ValueError(joined)

        if validation.normalized_composition:
            composition = dict(validation.normalized_composition)
        else:
            composition, _ = normalize_composition_keys(request.composition)

        inferred_system = normalize_system(validation.inferred_system)
        calphad_system = _resolve_calphad_system(inferred_system, composition)
        thermo = self._effective_thermo(request.thermo)

        requested_generator = self.generator_registry.canonical_name(request.generator)
        strict_needs_calphad = bool(thermo.strict_mode) and requested_generator in {
            "auto",
            "calphad_phase",
        }
        calphad_available = False
        calphad_preflight_error = ""
        if _supported_by_calphad(calphad_system):
            try:
                self._resolve_calphad_reference(
                    inferred_system=calphad_system, thermo=thermo
                )
                calphad_available = True
            except Exception as exc:
                calphad_preflight_error = str(exc)
        else:
            calphad_preflight_error = (
                f"SYSTEM_UNSUPPORTED: inferred system '{validation.inferred_system}' is not in "
                f"{list(CALPHAD_SUPPORTED_SYSTEMS)}"
            )

        if strict_needs_calphad and not calphad_available:
            raise ValueError(calphad_preflight_error or "SYSTEM_UNSUPPORTED")

        base_processing = _resolve_processing_auto_mode(
            processing=request.processing,
            inferred_system=inferred_system,
        )

        route_sim = self._resolve_route(
            request=request,
            composition=composition,
            inferred_system=inferred_system,
            initial_processing=base_processing,
        )
        route_validation = (
            _blank_route_validation()
            if route_sim is None
            else route_sim.route_validation
        )

        if (
            request.strict_validation
            and route_sim is not None
            and not route_validation.is_valid
        ):
            raise ValueError("; ".join(route_validation.errors))

        effective_processing = (
            base_processing if route_sim is None else route_sim.final_processing
        )
        user_params = dict(request.generator_params or {})
        route_overrides = (
            dict(route_sim.generator_param_overrides) if route_sim is not None else {}
        )
        route_context: dict[str, Any] | None = None
        if route_sim is not None:
            route_context = {
                "resolved_stage": route_sim.final_stage,
                "inferred_system": inferred_system,
                "preview_step_index": request.preview_step_index,
            }

        selected_generator = requested_generator
        effective_params = dict(user_params)
        effective_params.update(route_overrides)
        calphad_meta: dict[str, Any] | None = None

        auto_meta = {
            "enabled": False,
            "selected_generator": selected_generator,
            "selection_reason": "Manual generator mode.",
            "selection_confidence": 1.0,
            "coverage_mode": "supported",
        }

        if requested_generator == "auto":
            decision: AutoGeneratorDecision = select_auto_generator(
                composition=composition,
                inferred_system=(
                    calphad_system if calphad_available else inferred_system
                ),
                processing=effective_processing,
                route_sim=route_sim,
                user_params=user_params,
                route_overrides=route_overrides,
                calphad_available=calphad_available,
            )
            selected_generator = decision.selected_generator
            effective_params = dict(decision.resolved_params)
            auto_meta = {
                "enabled": True,
                "selected_generator": decision.selected_generator,
                "selection_reason": decision.selection_reason,
                "selection_confidence": float(decision.selection_confidence),
                "coverage_mode": decision.coverage_mode,
            }
            if bool(thermo.strict_mode) and selected_generator != "calphad_phase":
                raise ValueError(
                    "SYSTEM_UNSUPPORTED: strict CALPHAD-only mode requires a CALPHAD-backed generator "
                    f"(resolved '{selected_generator}')"
                )
            if route_sim is not None:
                route_sim = simulate_process_route(
                    composition=composition,
                    inferred_system=inferred_system,
                    route=request.process_route,
                    initial_processing=base_processing,
                    generator=selected_generator,
                    base_seed=request.seed,
                    step_preview_index=request.preview_step_index,
                )
                route_validation = route_sim.route_validation
                if request.strict_validation and not route_validation.is_valid:
                    raise ValueError("; ".join(route_validation.errors))
                effective_processing = route_sim.final_processing
                route_overrides = dict(route_sim.generator_param_overrides)
                effective_params = dict(decision.resolved_params)
                effective_params.update(route_overrides)
                route_context = {
                    "resolved_stage": route_sim.final_stage,
                    "inferred_system": inferred_system,
                    "preview_step_index": request.preview_step_index,
                }

        if selected_generator == "calphad_phase":
            if not calphad_available:
                raise ValueError(calphad_preflight_error or "SYSTEM_UNSUPPORTED")
            calphad_meta = self._run_calphad_suite(
                composition=composition,
                inferred_system=calphad_system,
                processing=effective_processing,
                thermo=thermo,
                generator_params=effective_params,
                request=request,
            )
            if "composition_sensitivity_mode" not in effective_params:
                effective_params["composition_sensitivity_mode"] = "realistic"
            eq_result = dict(calphad_meta.get("equilibrium_result", {}))
            stable = eq_result.get("stable_phases", {})
            if not isinstance(stable, dict) or not stable:
                raise ValueError("SOLVER_FAIL: empty stable phase set")
            transition_payload = _phase_transition_from_calphad(
                liquid_fraction=float(eq_result.get("liquid_fraction", 0.0)),
                thermal_slope=(
                    None
                    if effective_params.get("thermal_slope") is None
                    else float(effective_params.get("thermal_slope"))
                ),
            )
            route_stage = route_sim.final_stage if route_sim is not None else ""
            transition_payload["stage"] = str(
                effective_params.get("stage") or route_stage or "calphad_equilibrium"
            )
            effective_params["system"] = calphad_system
            effective_params["calphad_result"] = eq_result
            effective_params["phase_transition_state"] = transition_payload
            kinetics_payload = calphad_meta.get("kinetics_result")
            if isinstance(kinetics_payload, dict) and kinetics_payload.get("enabled"):
                effective_params["kinetics_result"] = kinetics_payload
            if route_context is None:
                route_context = {"inferred_system": calphad_system}
            else:
                route_context["inferred_system"] = calphad_system
        elif strict_needs_calphad:
            raise ValueError(
                "SYSTEM_UNSUPPORTED: strict CALPHAD-only mode requires generator 'calphad_phase' "
                f"(resolved '{selected_generator}')"
            )

        generated = self.generator_registry.generate(
            name=selected_generator,
            size=request.resolution,
            seed=request.seed,
            composition=composition,
            processing=effective_processing,
            generator_params=effective_params,
            inferred_system=(
                calphad_system
                if selected_generator == "calphad_phase"
                else inferred_system
            ),
            route_context=route_context,
        )

        gray = _ensure_uint8_gray(generated["image_gray"])
        rgb = _ensure_uint8_rgb(generated["image_rgb"])
        phase_masks = generated.get("phase_masks")
        metadata = dict(generated.get("metadata", {}))

        microscope = dict(request.microscope_params or {})
        view_meta: dict[str, Any] | None = None
        if microscope:
            output_size = microscope.pop("output_size", request.resolution)
            if isinstance(output_size, list) and len(output_size) == 2:
                out_h, out_w = int(output_size[0]), int(output_size[1])
            elif isinstance(output_size, tuple) and len(output_size) == 2:
                out_h, out_w = int(output_size[0]), int(output_size[1])
            else:
                out_h, out_w = int(request.resolution[0]), int(request.resolution[1])

            view_gray, view_meta = simulate_microscope_view(
                sample=gray,
                magnification=int(microscope.get("magnification", 200)),
                pan_x=float(microscope.get("pan_x", 0.5)),
                pan_y=float(microscope.get("pan_y", 0.5)),
                output_size=(out_h, out_w),
                focus=float(microscope.get("focus", 1.0)),
                brightness=float(microscope.get("brightness", 1.0)),
                contrast=float(microscope.get("contrast", 1.0)),
                vignette_strength=float(microscope.get("vignette_strength", 0.15)),
                uneven_strength=float(microscope.get("uneven_strength", 0.08)),
                noise_sigma=float(microscope.get("noise_sigma", 3.0)),
                add_dust=bool(microscope.get("add_dust", False)),
                add_scratches=bool(microscope.get("add_scratches", False)),
                etch_uneven=float(microscope.get("etch_uneven", 0.0)),
                seed=int(microscope.get("seed", request.seed + 1000)),
            )
            gray = view_gray
            rgb = np.stack([view_gray] * 3, axis=2).astype(np.uint8)

        phase_fraction = _phase_fraction_estimate(
            phase_masks=phase_masks, metadata=metadata
        )
        diagram_meta = diagram_snapshot_params(
            composition=composition,
            processing=effective_processing,
            requested_system=effective_params.get("system")
            if isinstance(effective_params, dict)
            else None,
            inferred_system=(
                calphad_system
                if selected_generator == "calphad_phase"
                else inferred_system
            ),
            confidence=validation.confidence,
        )

        stage = str(
            metadata.get("resolved_stage")
            or metadata.get("requested_stage")
            or metadata.get("style")
            or (route_sim.final_stage if route_sim is not None else "")
            or metadata.get("generator_name")
            or "generated"
        )

        curve_cfg = self._extract_cooling_curve_config(request)
        cooling_curve_meta: dict[str, Any] | None = None
        if curve_cfg.get("enabled"):
            cooling_curve_meta = {
                "enabled": True,
                "mode": str(curve_cfg.get("mode", "per_degree")),
                "degree_step": float(curve_cfg.get("degree_step", 1.0)),
                "max_points": int(curve_cfg.get("max_points", 220)),
                "points": list(curve_cfg.get("points", [])),
            }

        metadata.update(
            {
                "request": request.to_dict(),
                "composition_normalized": composition,
                "inferred_system": inferred_system,
                "generator_version": self.generator_version,
                "diagram_snapshot_params": diagram_meta,
                "phase_fraction_estimate": phase_fraction,
                "stage": stage,
                "validation_report": validation.to_dict(),
                "route_validation": route_validation.to_dict(),
                "auto_generator": auto_meta,
            }
        )
        if cooling_curve_meta is not None:
            metadata["cooling_curve"] = cooling_curve_meta
        if calphad_meta is not None:
            metadata["calphad"] = calphad_meta

        if route_sim is not None:
            metadata.update(
                {
                    "process_route": request.process_route.to_dict()
                    if request.process_route
                    else None,
                    **route_sim.to_metadata(),
                }
            )

        if view_meta is not None:
            metadata["microscope_view"] = view_meta

        return GenerationOutputV2(
            image_rgb=rgb,
            image_gray=gray,
            phase_masks=phase_masks,
            metadata=metadata,
            validation_report=validation,
        )

    def _save_step_series_if_needed(
        self,
        request: GenerationRequestV2,
        output: GenerationOutputV2,
        out_dir: Path,
        sample_id: str,
    ) -> list[str]:
        timeline = output.metadata.get("route_timeline")
        process_route = request.process_route
        if not isinstance(timeline, list):
            return []
        if process_route is None or not process_route.step_preview_enabled:
            return []

        saved_paths: list[str] = []
        for idx, _step in enumerate(timeline):
            req_dict = request.to_dict()
            req_dict["preview_step_index"] = idx
            req_dict["microscope_params"] = {}
            step_request = GenerationRequestV2.from_dict(req_dict)
            step_output = self.generate(step_request)
            step_path = out_dir / f"{sample_id}_step_{idx + 1:02d}.png"
            save_image(step_output.image_rgb, step_path)
            saved_paths.append(str(step_path))
        return saved_paths

    def _save_cooling_curve_series_if_needed(
        self,
        request: GenerationRequestV2,
        out_dir: Path,
        sample_id: str,
    ) -> list[str]:
        config = self._extract_cooling_curve_config(request)
        if not config.get("enabled"):
            return []
        series = self.generate_cooling_curve_series(request)
        if not series:
            return []

        saved: list[str] = []
        for idx, frame in enumerate(series, start=1):
            point = frame.metadata.get("cooling_curve_point", {})
            temp = int(
                round(
                    float(point.get("temperature_c", request.processing.temperature_c))
                )
            )
            path = out_dir / f"{sample_id}_curve_{idx:03d}_T{temp:+04d}.png"
            save_image(frame.image_rgb, path)
            saved.append(str(path))
        return saved

    def generate_batch(
        self,
        requests: list[GenerationRequestV2],
        output_dir: str | Path,
        file_prefix: str = "sample",
    ) -> BatchResult:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for idx, request in enumerate(requests, start=1):
            sample_id = f"{file_prefix}_{idx:03d}"
            comp_hash = _composition_hash(
                {str(k): float(v) for k, v in request.composition.items()}
            )

            try:
                output = self.generate(request)
                image_path = out / f"{sample_id}.png"
                metadata_path = out / f"{sample_id}.json"
                save_image(output.image_rgb, image_path)

                step_paths = self._save_step_series_if_needed(
                    request=request, output=output, out_dir=out, sample_id=sample_id
                )
                curve_paths = self._save_cooling_curve_series_if_needed(
                    request=request, out_dir=out, sample_id=sample_id
                )
                payload = output.metadata_json_safe()
                if step_paths:
                    payload["step_series_images"] = step_paths
                if curve_paths:
                    payload["cooling_curve_series_images"] = curve_paths
                save_json(payload, metadata_path)

                route_name = ""
                step_count = 0
                if request.process_route is not None:
                    route_name = request.process_route.route_name
                    step_count = len(request.process_route.operations)

                props = output.metadata.get("property_indicators", {})
                hv = (
                    float(props.get("hv_estimate", 0.0))
                    if isinstance(props, dict)
                    else 0.0
                )
                uts = (
                    float(props.get("uts_estimate_mpa", 0.0))
                    if isinstance(props, dict)
                    else 0.0
                )
                stage = str(output.metadata.get("stage", "generated"))

                rows.append(
                    {
                        "sample_id": sample_id,
                        "mode": request.mode,
                        "seed": int(request.seed),
                        "composition_hash": comp_hash,
                        "validation_passed": True,
                        "system": output.validation_report.inferred_system,
                        "stage": stage,
                        "route_name": route_name,
                        "step_count": step_count,
                        "final_stage": stage,
                        "hv_estimate": hv,
                        "uts_estimate_mpa": uts,
                        "image_path": str(image_path),
                        "metadata_path": str(metadata_path),
                        "error": "",
                    }
                )
            except Exception as exc:
                report: ValidationReport = validate_alloy(
                    composition=request.composition,
                    processing=request.processing,
                    auto_normalize=request.auto_normalize,
                    strict_custom_limits=request.strict_validation,
                )
                route_name = (
                    request.process_route.route_name
                    if request.process_route is not None
                    else ""
                )
                step_count = (
                    len(request.process_route.operations)
                    if request.process_route is not None
                    else 0
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "mode": request.mode,
                        "seed": int(request.seed),
                        "composition_hash": comp_hash,
                        "validation_passed": bool(report.is_valid),
                        "system": report.inferred_system,
                        "stage": "",
                        "route_name": route_name,
                        "step_count": step_count,
                        "final_stage": "",
                        "hv_estimate": 0.0,
                        "uts_estimate_mpa": 0.0,
                        "image_path": "",
                        "metadata_path": "",
                        "error": str(exc),
                    }
                )

        index_path = out / f"{file_prefix}_index.csv"
        save_measurements_csv(rows, index_path)
        return BatchResult(rows=rows, csv_index_path=index_path)
