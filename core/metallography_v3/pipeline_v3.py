from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from core.alloy_validation import normalize_composition_keys, validate_alloy
from core.contracts_v2 import ValidationReport
from core.contracts_v3 import (
    GenerationOutputV3,
    IntermediateRenderV3,
    MetallographyRequestV3,
)
from core.imaging import simulate_microscope_view
from core.electron_microscopy_guidance import build_electron_microscopy_guidance
from core.optical_mode_transfer import build_ferromagnetic_mask
from core.materials_hybrid import (
    calculate_hybrid_heat_treatment,
    calculate_hybrid_properties,
    supports_hybrid_properties,
    validate_expected_properties,
)
from core.metallography_pro.pipeline_pro import (
    generate_pro_realistic_fe_c,
    supports_pro_realistic_fe_c_stage,
)
from core.metallography_v3.etch_simulator import apply_etch
from core.metallography_v3.microstructure_state import build_microstructure_state
from core.metallography_v3.morphology_engine import generate_phase_topology
from core.metallography_v3.hardness_brinell import (
    hbw_estimate_from_microstructure,
    hbw_from_indent,
)
from core.metallography_v3.prep_simulator import apply_prep_route
from core.metallography_v3.quality_control import run_quality_checks
from core.metallography_v3.reference_calibration import (
    load_builtin_profiles,
    load_reference_profile,
    resolve_reference_style,
)
from core.metallography_v3.phase_orchestrator import (
    build_phase_bundle,
    infer_training_system,
)
from core.metallography_v3.thermal_program_v3 import (
    effective_processing_from_thermal,
    infer_operations_from_thermal_program,
    sample_thermal_program,
    summarize_thermal_program,
    validate_thermal_program,
)
from export.export_images import save_image
from export.export_tables import save_json, save_measurements_csv


@dataclass(slots=True)
class BatchResultV3:
    rows: list[dict[str, Any]]
    csv_index_path: Path


_TEXTBOOK_RULES_PATH = (
    Path(__file__).resolve().parents[1] / "rulebook" / "textbook_visual_rules_v3.json"
)


def _load_textbook_rules() -> dict[str, Any]:
    if not _TEXTBOOK_RULES_PATH.exists():
        return {}
    try:
        return json.loads(_TEXTBOOK_RULES_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


_TEXTBOOK_RULES = _load_textbook_rules()
_PROPERTIES_RULES_PATH = (
    Path(__file__).resolve().parents[1] / "rulebook" / "properties_rules.json"
)
_PROPERTIES_RULES = (
    json.loads(_PROPERTIES_RULES_PATH.read_text(encoding="utf-8-sig"))
    if _PROPERTIES_RULES_PATH.exists()
    else {}
)
_MAX_INTERNAL_RENDER_SIDE = 4096

_PHASE_LABEL_ALIASES: dict[str, str] = {
    "L": "LIQUID",
    "FE3C": "CEMENTITE",
    "MARTENSITE_T": "MARTENSITE_TETRAGONAL",
    "MARTENSITE_C": "MARTENSITE_CUBIC",
    "ALPHA_AL": "FCC_A1",
    "ALPHA_SSS": "FCC_A1",
    "PRIMARY_SI": "SI",
    "EUTECTIC": "EUTECTIC_ALSI",
    "PRECIPITATES": "PRECIPITATE",
}


def _lift_small_dark_blobs(
    image_gray: np.ndarray,
    *,
    threshold: float = 40.0,
    max_pixels: int = 48,
) -> np.ndarray:
    if ndimage is None:
        return image_gray.astype(np.uint8, copy=False)
    arr = image_gray.astype(np.float32, copy=False)
    mask = arr < float(threshold)
    labels, count = ndimage.label(mask.astype(np.uint8))
    if int(count) <= 0:
        return image_gray.astype(np.uint8, copy=False)
    local = ndimage.gaussian_filter(arr, sigma=1.05)
    out = arr.copy()
    for label in range(1, int(count) + 1):
        zone = labels == label
        if int(zone.sum()) <= int(max_pixels):
            out[zone] = 0.84 * local[zone] + 0.16 * out[zone]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _brighten_pure_ferrite_baseline(image_gray: np.ndarray) -> np.ndarray:
    arr = image_gray.astype(np.float32, copy=False)
    if ndimage is not None:
        arr = ndimage.gaussian_filter(arr, sigma=0.3)
    q01 = float(np.quantile(arr, 0.01))
    q05 = float(np.quantile(arr, 0.05))
    arr += max(0.0, 88.0 - q01)
    arr += max(0.0, 126.0 - q05) * 0.8
    out = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return _lift_small_dark_blobs(
        out,
        threshold=44.0,
        max_pixels=max(24, int(out.size // 32768)),
    )


def _resolve_morph_pure_iron_baseline(morph: dict[str, Any]) -> dict[str, Any]:
    payload = morph.get("pure_iron_baseline")
    if isinstance(payload, dict):
        return dict(payload)
    system_extra = dict(morph.get("system_generator_extra", {}))
    nested = system_extra.get("pure_iron_baseline")
    if isinstance(nested, dict):
        return dict(nested)
    return {}


def _apply_textbook_brightfield_policy(
    image_gray: np.ndarray,
    *,
    generation_mode: str,
    profile_id: str,
    optical_mode: str,
    pure_iron_baseline_applied: bool,
) -> np.ndarray:
    if str(optical_mode or "brightfield").strip().lower() != "brightfield":
        return image_gray.astype(np.uint8, copy=False)
    if str(generation_mode or "").strip().lower() != "edu_engineering":
        return image_gray.astype(np.uint8, copy=False)
    if str(profile_id or "").strip().lower() != "textbook_steel_bw":
        return image_gray.astype(np.uint8, copy=False)

    out = _lift_small_dark_blobs(
        image_gray,
        threshold=42.0,
        max_pixels=max(24, int(image_gray.size // 32768)),
    )
    if bool(pure_iron_baseline_applied):
        out = _brighten_pure_ferrite_baseline(out)
    return out.astype(np.uint8, copy=False)


def _build_optical_recommendation(
    *,
    system: str,
    stage: str,
    phase_fractions: dict[str, float],
    preset_metadata: dict[str, Any],
    pure_iron_baseline_applied: bool,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "default_mode": "brightfield",
        "diagnostic_mode": "darkfield",
        "secondary_mode": "brightfield",
        "polarized_recommended": False,
        "dic_recommended": False,
        "rationale": [],
    }
    if pure_iron_baseline_applied:
        rec.update(
            {
                "default_mode": "brightfield",
                "diagnostic_mode": "darkfield",
                "secondary_mode": "phase_contrast",
                "polarized_recommended": False,
                "dic_recommended": False,
                "rationale": ["cubic ferritic negative control under crossed polars"],
            }
        )
        return rec

    mart = float(
        phase_fractions.get("MARTENSITE", 0.0)
        + phase_fractions.get("MARTENSITE_TETRAGONAL", 0.0)
        + phase_fractions.get("MARTENSITE_CUBIC", 0.0)
        + phase_fractions.get("TROOSTITE", 0.0)
        + phase_fractions.get("SORBITE", 0.0)
    )
    bainite = float(phase_fractions.get("BAINITE", 0.0))
    delta_ferrite = float(phase_fractions.get("DELTA_FERRITE", 0.0))
    graphite_meta = " ".join(
        [
            str(preset_metadata.get("graphite_form", "")),
            str(preset_metadata.get("graphite_morphology", "")),
            str(preset_metadata.get("target_microstructure", "")),
        ]
    ).lower()

    if mart >= 0.35:
        rec.update(
            {
                "secondary_mode": "polarized",
                "polarized_recommended": True,
                "dic_recommended": True,
            }
        )
        rec["rationale"].append(
            "crossed polars can reveal packet/lath martensitic structure"
        )
    elif bainite >= 0.25:
        rec.update({"secondary_mode": "dic", "dic_recommended": True})
        rec["rationale"].append("DIC emphasizes relief-sensitive bainitic morphology")
    elif "graphite" in graphite_meta and (
        "nodul" in graphite_meta or "шаров" in graphite_meta
    ):
        rec.update({"secondary_mode": "polarized", "polarized_recommended": True})
        rec["rationale"].append(
            "polarized light is especially useful for graphite nodule internal structure"
        )
    elif "graphite" in graphite_meta:
        rec.update({"secondary_mode": "dic", "dic_recommended": True})
        rec["rationale"].append(
            "DIC helps read relief around graphite and matrix boundaries"
        )

    if delta_ferrite > 0.05 and float(phase_fractions.get("AUSTENITE", 0.0)) > 0.05:
        rec["diagnostic_mode"] = "magnetic_etching"
        rec["rationale"].append(
            "magnetic etching can separate ferromagnetic delta-ferrite from austenite"
        )

    if not rec["rationale"]:
        rec["rationale"].append(
            f"{system}:{stage} best starts from brightfield then darkfield/DIC checks"
        )
    return rec


_STAGE_MICROCONSTITUENTS: dict[str, dict[str, list[str]]] = {
    "fe-c": {
        "liquid": ["LIQUID"],
        "liquid_gamma": ["LIQUID", "AUSTENITE"],
        "delta_ferrite": ["DELTA_FERRITE", "AUSTENITE"],
        "austenite": ["AUSTENITE"],
        "ferrite": ["FERRITE"],
        "alpha_gamma": ["FERRITE", "AUSTENITE"],
        "gamma_cementite": ["AUSTENITE", "CEMENTITE"],
        "alpha_pearlite": ["FERRITE", "PEARLITE"],
        "pearlite": ["PEARLITE", "FERRITE"],
        "pearlite_cementite": ["PEARLITE", "CEMENTITE"],
        "ledeburite": ["LEDEBURITE", "PEARLITE", "CEMENTITE"],
        "martensite": ["MARTENSITE", "CEMENTITE"],
        "martensite_tetragonal": ["MARTENSITE_TETRAGONAL", "CEMENTITE"],
        "martensite_cubic": ["MARTENSITE_CUBIC", "CEMENTITE"],
        "troostite_quench": ["TROOSTITE", "CEMENTITE"],
        "troostite_temper": ["TROOSTITE", "CEMENTITE", "FERRITE"],
        "sorbite_quench": ["SORBITE", "CEMENTITE"],
        "sorbite_temper": ["SORBITE", "CEMENTITE", "FERRITE"],
        "bainite": ["BAINITE", "CEMENTITE"],
        "tempered_low": ["TROOSTITE", "MARTENSITE", "CEMENTITE"],
        "tempered_medium": ["SORBITE", "MARTENSITE", "CEMENTITE", "FERRITE"],
        "tempered_high": ["SORBITE", "FERRITE", "CEMENTITE"],
    },
    "al-si": {
        "liquid": ["LIQUID"],
        "liquid_alpha": ["LIQUID", "FCC_A1"],
        "liquid_si": ["LIQUID", "SI"],
        "alpha_eutectic": ["FCC_A1", "EUTECTIC_ALSI"],
        "eutectic": ["EUTECTIC_ALSI", "FCC_A1", "SI"],
        "primary_si_eutectic": ["SI", "EUTECTIC_ALSI", "FCC_A1"],
        "supersaturated": ["FCC_A1", "PRECIPITATE"],
        "aged": ["FCC_A1", "PRECIPITATE"],
    },
    "cu-zn": {
        "alpha": ["ALPHA"],
        "alpha_beta": ["ALPHA", "BETA"],
        "beta": ["BETA"],
        "beta_prime": ["BETA_PRIME", "BETA"],
        "cold_worked": ["ALPHA", "BETA", "DEFORMATION_BANDS"],
    },
    "fe-si": {
        "liquid": ["LIQUID"],
        "liquid_ferrite": ["LIQUID", "BCC_B2"],
        "hot_ferrite": ["BCC_B2"],
        "recrystallized_ferrite": ["BCC_B2"],
        "cold_worked_ferrite": ["BCC_B2", "DEFORMATION_BANDS"],
    },
    "al-cu-mg": {
        "solutionized": ["FCC_A1", "THETA"],
        "quenched": ["FCC_A1", "THETA"],
        "natural_aged": ["FCC_A1", "THETA"],
        "artificial_aged": ["FCC_A1", "THETA", "S_PHASE"],
        "overaged": ["FCC_A1", "THETA", "S_PHASE", "QPHASE"],
    },
}


def _resolve_textbook_targets(
    system: str, profile_id: str
) -> tuple[dict[str, float], list[str]]:
    defaults = (
        _TEXTBOOK_RULES.get("defaults", {}) if isinstance(_TEXTBOOK_RULES, dict) else {}
    )
    systems = (
        _TEXTBOOK_RULES.get("systems", {}) if isinstance(_TEXTBOOK_RULES, dict) else {}
    )
    profiles = (
        _TEXTBOOK_RULES.get("profiles", {}) if isinstance(_TEXTBOOK_RULES, dict) else {}
    )

    readability: dict[str, float] = {}
    default_r = (
        defaults.get("readability_thresholds", {}) if isinstance(defaults, dict) else {}
    )
    if isinstance(default_r, dict):
        readability = {str(k): float(v) for k, v in default_r.items()}

    sys_payload = (
        systems.get(str(system).strip().lower(), {})
        if isinstance(systems, dict)
        else {}
    )
    if isinstance(sys_payload, dict):
        sys_r = sys_payload.get("readability_thresholds", {})
        if isinstance(sys_r, dict):
            for key, value in sys_r.items():
                readability[str(key)] = float(value)

    target_micro: list[str] = []
    profile_payload = (
        profiles.get(str(profile_id), {}) if isinstance(profiles, dict) else {}
    )
    if isinstance(profile_payload, dict):
        raw = profile_payload.get("target_microconstituents", [])
        if isinstance(raw, list):
            target_micro = [str(x) for x in raw if str(x).strip()]
    return readability, target_micro


def _normalize_microconstituent_name(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    canon = token.upper().replace("-", "_").replace(" ", "_")
    return _PHASE_LABEL_ALIASES.get(canon, canon)


def _stage_microconstituents(system: str, stage: str) -> list[str]:
    sys_name = str(system or "").strip().lower()
    stage_name = str(stage or "").strip().lower()
    payload = _STAGE_MICROCONSTITUENTS.get(sys_name, {})
    if not isinstance(payload, dict):
        return []
    raw = payload.get(stage_name, [])
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        name = _normalize_microconstituent_name(item)
        if name:
            out.append(name)
    return out


def _resolve_target_microconstituents(
    *,
    system: str,
    stage: str,
    profile_targets: list[str],
    phase_fractions: dict[str, float],
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    def _push(name: Any) -> None:
        token = _normalize_microconstituent_name(name)
        if token and token not in seen:
            seen.add(token)
            merged.append(token)

    for item in profile_targets:
        _push(item)
    for item in _stage_microconstituents(system=system, stage=stage):
        _push(item)
    for phase_name, fraction in sorted(
        phase_fractions.items(), key=lambda item: float(item[1]), reverse=True
    ):
        if float(fraction) <= 1e-6:
            continue
        _push(phase_name)

    if not merged:
        merged.append("MATRIX")
    return merged


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        arr = gray.astype(np.uint8, copy=False)
        h, w = arr.shape
        return np.broadcast_to(arr[:, :, None], (h, w, 3))
    if gray.ndim == 3 and gray.shape[2] >= 3:
        return gray[:, :, :3].astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image shape: {gray.shape}")


def _resolve_internal_resolution(
    requested: tuple[int, int],
) -> tuple[tuple[int, int], dict[str, Any]]:
    h = max(64, int(requested[0]))
    w = max(64, int(requested[1]))
    max_side = max(h, w)
    if max_side <= _MAX_INTERNAL_RENDER_SIDE:
        payload = {
            "enabled": False,
            "requested_resolution": [h, w],
            "internal_resolution": [h, w],
            "upscale_method": "none",
            "mask_resolution": [h, w],
        }
        return (h, w), payload

    scale = float(max_side) / float(_MAX_INTERNAL_RENDER_SIDE)
    internal_h = max(512, int(round(h / scale)))
    internal_w = max(512, int(round(w / scale)))
    payload = {
        "enabled": True,
        "requested_resolution": [h, w],
        "internal_resolution": [internal_h, internal_w],
        "upscale_method": "lanczos",
        "mask_resolution": [internal_h, internal_w],
    }
    return (internal_h, internal_w), payload


def _resize_gray_to(image_gray: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_h = max(1, int(target_size[0]))
    target_w = max(1, int(target_size[1]))
    if image_gray.shape[0] == target_h and image_gray.shape[1] == target_w:
        return image_gray.astype(np.uint8, copy=False)
    pil = Image.fromarray(image_gray.astype(np.uint8, copy=False), mode="L")
    resized = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def _is_critical_validation_error(message: str) -> bool:
    text = str(message or "").strip().lower()
    critical_tokens = (
        "unknown element",
        "invalid element",
        "nan",
        "inf",
    )
    return any(token in text for token in critical_tokens)


class MetallographyPipelineV3:
    def __init__(
        self,
        presets_dir: str | Path | None = None,
        profiles_dir: str | Path | None = None,
        generator_version: str = "v3.0.0",
    ) -> None:
        self.presets_dir = (
            Path(presets_dir) if presets_dir is not None else Path("presets_v3")
        )
        self.profiles_dir = (
            Path(profiles_dir) if profiles_dir is not None else Path("profiles_v3")
        )
        self.generator_version = generator_version
        self._builtin_profiles = load_builtin_profiles(self.profiles_dir)

    def list_preset_paths(self) -> list[Path]:
        if not self.presets_dir.exists():
            return []
        return sorted(self.presets_dir.glob("*.json"))

    def load_preset(self, name_or_path: str | Path) -> dict[str, Any]:
        candidate = Path(name_or_path)
        if not candidate.exists():
            candidate = self.presets_dir / f"{name_or_path}.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Preset not found: {name_or_path}")
        payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError(f"Preset payload must be object: {candidate}")
        return payload

    def request_from_preset(
        self, preset_payload: dict[str, Any]
    ) -> MetallographyRequestV3:
        return MetallographyRequestV3.from_dict(preset_payload)

    def generate_from_preset(self, name_or_path: str | Path) -> GenerationOutputV3:
        payload = self.load_preset(name_or_path)
        request = self.request_from_preset(payload)
        return self.generate(request)

    def generate(self, request: MetallographyRequestV3) -> GenerationOutputV3:
        composition, parse_errors = normalize_composition_keys(request.composition_wt)
        if parse_errors:
            raise ValueError("; ".join(parse_errors))
        thermal_program = request.thermal_program
        thermal_validation = validate_thermal_program(thermal_program)
        if not bool(thermal_validation.get("is_valid", True)):
            raise ValueError(
                "; ".join([str(x) for x in thermal_validation.get("errors", [])])
            )

        thermal_summary = summarize_thermal_program(thermal_program)
        sampled_curve = sample_thermal_program(thermal_program)
        process_probe, thermal_summary_runtime, quench_summary = (
            effective_processing_from_thermal(thermal_program)
        )
        thermal_summary.update(dict(thermal_summary_runtime))
        operations_from_curve = infer_operations_from_thermal_program(
            thermal_program,
            summary=thermal_summary,
            quench_summary=quench_summary,
        )
        thermal_summary["operation_inference"] = dict(
            operations_from_curve.get("summary", {})
        )

        report: ValidationReport = validate_alloy(
            composition=composition,
            processing=process_probe,
            auto_normalize=True,
            strict_custom_limits=bool(request.strict_validation),
        )
        critical_validation_errors = [
            err for err in report.errors if _is_critical_validation_error(err)
        ]
        soft_validation_errors = [
            err for err in report.errors if err not in critical_validation_errors
        ]
        validation_severity_summary = {
            "critical_errors": list(critical_validation_errors),
            "soft_errors": list(soft_validation_errors),
            "warnings": list(report.warnings),
        }
        if request.strict_validation and critical_validation_errors:
            raise ValueError("; ".join(critical_validation_errors))
        if soft_validation_errors:
            report.warnings.extend(
                [f"Soft validation: {err}" for err in soft_validation_errors]
            )

        composition_norm = (
            dict(report.normalized_composition)
            if report.normalized_composition
            else {str(k): float(v) for k, v in composition.items()}
        )
        system, system_confidence, _ = infer_training_system(
            composition=composition_norm,
            system_hint=request.system_hint or report.inferred_system,
        )

        micro_state = build_microstructure_state(
            composition=composition_norm,
            inferred_system=system,
            processing=process_probe,
            thermal_summary=thermal_summary,
            operations_from_curve=operations_from_curve,
            quench_summary=quench_summary,
            seed=request.seed,
        )
        final_processing = process_probe
        requested_resolution = (int(request.resolution[0]), int(request.resolution[1]))
        internal_resolution, hires_meta = _resolve_internal_resolution(
            requested_resolution
        )

        phase_bundle = build_phase_bundle(
            composition=composition_norm,
            processing=final_processing,
            system_hint=system,
            phase_model=request.phase_model,
            thermal_summary=thermal_summary,
            quench_summary=quench_summary,
        )
        hybrid_heat_treatment: dict[str, Any] = {}
        expected_properties_validation: dict[str, Any] = {}
        if supports_hybrid_properties(str(system), composition_norm):
            hybrid_props = calculate_hybrid_properties(
                composition=composition_norm,
                inferred_system=str(system),
                final_stage=str(phase_bundle.stage),
                phase_fractions=dict(phase_bundle.phase_fractions),
                material_grade=request.material_grade,
                material_class_ru=request.material_class_ru,
                effect_vector=dict(micro_state.effect_vector),
                grain_size_um=request.mean_grain_diameter_um,
                overlay_rules=_PROPERTIES_RULES,
            )
            micro_state.property_indicators = dict(hybrid_props)
            expected_properties_validation = validate_expected_properties(
                request.expected_properties,
                hybrid_props,
            )
            hybrid_heat_treatment = calculate_hybrid_heat_treatment(
                composition=composition_norm,
                material_grade=request.material_grade,
            )

        ref_style = None
        if request.reference_profile_id:
            ref_style = load_reference_profile(
                profile_id=request.reference_profile_id,
                profiles_root=self.profiles_dir,
            )
        if ref_style is None:
            ref_style = resolve_reference_style(
                profile_id=request.synthesis_profile.profile_id,
                profiles_root=self.profiles_dir,
            )

        use_pro_realistic = (
            str(request.synthesis_profile.generation_mode).strip().lower()
            == "pro_realistic"
            and str(phase_bundle.system).strip().lower() == "fe-c"
            and supports_pro_realistic_fe_c_stage(str(phase_bundle.stage))
        )

        if use_pro_realistic:
            morph = generate_pro_realistic_fe_c(
                size=internal_resolution,
                seed=request.seed,
                stage=str(phase_bundle.stage),
                phase_fractions=dict(phase_bundle.phase_fractions),
                composition_wt=composition_norm,
                processing=final_processing,
                prep_route=request.prep_route,
                etch_profile=request.etch_profile,
                synthesis_profile=request.synthesis_profile,
                microscope_profile=dict(request.microscope_profile),
                thermal_summary=thermal_summary,
                quench_summary=quench_summary,
                phase_fraction_source=str(
                    phase_bundle.phase_model_report.get(
                        "fraction_source", "default_formula"
                    )
                ),
                phase_calibration_mode=str(
                    phase_bundle.phase_model_report.get(
                        "calibration_mode", "default_formula"
                    )
                ),
            )
            prep = {
                "prep_maps": dict(morph["prep_maps"]),
                "prep_timeline": list(morph.get("prep_timeline", [])),
                "prep_summary": dict(morph.get("prep_summary", {})),
            }
            etched = {
                "image_gray": np.asarray(morph["image_gray"], dtype=np.uint8),
                "etch_maps": dict(morph.get("etch_maps", {})),
                "etch_rate_map": np.asarray(
                    dict(morph.get("etch_maps", {})).get(
                        "etch_rate", np.zeros(internal_resolution, dtype=np.uint8)
                    ),
                    dtype=np.uint8,
                ),
                "etch_summary": dict(morph.get("etch_summary", {})),
                "etch_concentration": dict(morph.get("etch_concentration", {})),
            }
        else:
            morph = generate_phase_topology(
                size=internal_resolution,
                seed=request.seed,
                phase_bundle=phase_bundle,
                micro_state=micro_state,
                synthesis_profile=request.synthesis_profile,
                reference_style=ref_style,
                composition_wt=composition_norm,
                composition_sensitivity_mode=str(
                    request.synthesis_profile.composition_sensitivity_mode
                ),
                generation_mode=str(request.synthesis_profile.generation_mode),
                phase_emphasis_style=str(
                    request.synthesis_profile.phase_emphasis_style
                ),
                phase_fraction_tolerance_pct=float(
                    request.synthesis_profile.phase_fraction_tolerance_pct
                ),
                thermal_summary=thermal_summary,
                quench_summary=quench_summary,
            )

            prep = apply_prep_route(
                image_gray=morph["image_gray"],
                prep_route=request.prep_route,
                seed=request.seed + 77,
                phase_masks=morph["phase_masks"],
                system=str(phase_bundle.system),
                composition_wt=composition_norm,
                effect_vector=dict(micro_state.effect_vector),
            )
            etched = apply_etch(
                image_gray=prep["image_gray"],
                phase_masks=morph["phase_masks"],
                etch_profile=request.etch_profile,
                seed=request.seed + 131,
                prep_maps=prep["prep_maps"],
                system=str(phase_bundle.system),
                composition_wt=composition_norm,
                effect_vector=dict(micro_state.effect_vector),
            )

        image_gray_internal = etched["image_gray"].astype(np.uint8)
        morph_pure_iron = _resolve_morph_pure_iron_baseline(morph)
        pure_iron_baseline_applied = bool(
            bool(morph_pure_iron.get("applied", False))
            or bool(prep["prep_summary"].get("pure_iron_baseline_applied", False))
            or bool(etched["etch_summary"].get("pure_iron_baseline_applied", False))
        )
        preview_optics: dict[str, Any] = {}
        if request.microscope_profile.get("simulate_preview", False):
            preview_ferromagnetic_mask, _ = build_ferromagnetic_mask(
                morph["phase_masks"],
                pure_iron_like=bool(pure_iron_baseline_applied),
                size=image_gray_internal.shape,
            )
            sim, preview_optics = simulate_microscope_view(
                sample=image_gray_internal,
                optical_ferromagnetic_mask_sample=np.clip(
                    preview_ferromagnetic_mask * 255.0, 0.0, 255.0
                ).astype(np.uint8),
                magnification=int(request.microscope_profile.get("magnification", 200)),
                output_size=internal_resolution,
                focus=float(request.microscope_profile.get("focus", 1.0)),
                brightness=float(request.microscope_profile.get("brightness", 1.0)),
                contrast=float(request.microscope_profile.get("contrast", 1.0)),
                vignette_strength=float(
                    request.microscope_profile.get("vignette_strength", 0.08)
                ),
                uneven_strength=float(
                    request.microscope_profile.get("uneven_strength", 0.06)
                ),
                noise_sigma=float(request.microscope_profile.get("noise_sigma", 1.2)),
                add_dust=bool(request.microscope_profile.get("add_dust", False)),
                add_scratches=bool(
                    request.microscope_profile.get("add_scratches", False)
                ),
                etch_uneven=float(request.microscope_profile.get("etch_uneven", 0.0)),
                optical_mode=str(
                    request.microscope_profile.get("optical_mode", "brightfield")
                ),
                optical_mode_parameters=dict(
                    request.microscope_profile.get("optical_mode_parameters", {})
                ),
                optical_context={
                    "inferred_system": str(phase_bundle.system),
                    "final_stage": str(phase_bundle.stage),
                    "pure_iron_baseline_applied": bool(pure_iron_baseline_applied),
                },
                psf_profile=str(
                    request.microscope_profile.get("psf_profile", "standard")
                ),
                psf_strength=float(request.microscope_profile.get("psf_strength", 0.0)),
                sectioning_shear_deg=float(
                    request.microscope_profile.get("sectioning_shear_deg", 35.0)
                ),
                hybrid_balance=float(
                    request.microscope_profile.get("hybrid_balance", 0.5)
                ),
                seed=request.seed + 999,
            )
            image_gray_internal = sim

        image_gray_internal = _apply_textbook_brightfield_policy(
            image_gray_internal,
            generation_mode=str(request.synthesis_profile.generation_mode),
            profile_id=str(request.synthesis_profile.profile_id),
            optical_mode=str(
                request.microscope_profile.get("optical_mode", "brightfield")
            ),
            pure_iron_baseline_applied=bool(pure_iron_baseline_applied),
        )

        image_gray = _resize_gray_to(image_gray_internal, requested_resolution)

        image_rgb = _to_rgb(image_gray)
        prep_maps = dict(prep["prep_maps"])
        if isinstance(etched.get("etch_maps"), dict):
            prep_maps.update(dict(etched["etch_maps"]))
        else:
            prep_maps["etch_rate"] = etched["etch_rate_map"]

        qc = run_quality_checks(
            image_gray=image_gray_internal,
            phase_masks=morph["phase_masks"],
            feature_masks=morph["feature_masks"],
            prep_maps=prep_maps,
            phase_visibility_report=dict(morph.get("phase_visibility_report", {})),
            generation_mode=str(request.synthesis_profile.generation_mode),
            profile_id=str(request.synthesis_profile.profile_id),
            pure_iron_baseline_applied=bool(pure_iron_baseline_applied),
        )
        brinell_estimated = hbw_estimate_from_microstructure(
            system=str(phase_bundle.system),
            stage=str(phase_bundle.stage),
            phase_fractions=dict(phase_bundle.phase_fractions),
            effect_vector=dict(micro_state.effect_vector),
        )
        brinell_direct: dict[str, Any] | None = None
        direct_payload = dict(request.microscope_profile.get("brinell_direct", {}))
        if direct_payload:
            try:
                brinell_direct = hbw_from_indent(
                    load_kgf=float(
                        direct_payload.get(
                            "P_kgf", direct_payload.get("load_kgf", 187.5)
                        )
                    ),
                    ball_d_mm=float(
                        direct_payload.get("D_mm", direct_payload.get("ball_d_mm", 2.5))
                    ),
                    indent_d_mm=float(
                        direct_payload.get(
                            "d_mm", direct_payload.get("indent_d_mm", 0.9)
                        )
                    ),
                )
            except Exception as exc:
                brinell_direct = {"mode": "direct", "error": str(exc)}
        readability_targets, profile_targets = _resolve_textbook_targets(
            system=phase_bundle.system,
            profile_id=str(request.synthesis_profile.profile_id),
        )
        target_microconstituents = _resolve_target_microconstituents(
            system=str(phase_bundle.system),
            stage=str(phase_bundle.stage),
            profile_targets=profile_targets,
            phase_fractions=dict(phase_bundle.phase_fractions),
        )
        visibility = dict(morph.get("phase_visibility_report", {}))
        significant_phase_count = sum(
            1 for _, frac in phase_bundle.phase_fractions.items() if float(frac) >= 0.05
        )
        multi_phase_readability = significant_phase_count >= 2
        dominant_fraction = max(
            [float(v) for v in phase_bundle.phase_fractions.values()] or [1.0]
        )
        single_phase_negative_control = bool(
            not multi_phase_readability and pure_iron_baseline_applied
        )
        achieved_readability = {
            "separability_score": float(qc.get("phase_separability_score", 0.0)),
            "dynamic_range_p05_p95": float(qc.get("dynamic_range_p05_p95", 0.0)),
            "within_tolerance": bool(
                True
                if single_phase_negative_control
                else visibility.get("within_tolerance", True)
            ),
            "multi_phase": multi_phase_readability,
            "dominant_fraction": float(dominant_fraction),
            "single_phase_negative_control": single_phase_negative_control,
            "multiphase_separability_applicable": bool(multi_phase_readability),
        }
        sep_min = float(readability_targets.get("separability_score_min", 0.0))
        dyn_min = float(readability_targets.get("dynamic_range_min", 0.0))
        sep_ok = achieved_readability["separability_score"] >= sep_min
        dyn_ok = achieved_readability["dynamic_range_p05_p95"] >= dyn_min
        if not multi_phase_readability or dominant_fraction >= 0.78:
            sep_ok = True
        if dominant_fraction >= 0.78:
            dyn_ok = True
        textbook_pass = bool(
            sep_ok and dyn_ok and bool(achieved_readability["within_tolerance"])
        )
        textbook_profile = {
            "profile_id": str(request.synthesis_profile.profile_id),
            "target_microconstituents": target_microconstituents,
            "readability_targets": readability_targets,
            "achieved_readability": achieved_readability,
            "pass": textbook_pass,
        }
        pure_iron_optical_recommendation: dict[str, Any] = {}
        pure_iron_electropolish_profile = ""
        pure_iron_polarized_extinction_score = 0.0
        if pure_iron_baseline_applied:
            cleanliness = float(
                max(
                    prep["prep_summary"].get("pure_iron_cleanliness_score", 0.0),
                    etched["etch_summary"].get("pure_iron_cleanliness_score", 0.0),
                )
            )
            boundary = float(
                max(
                    prep["prep_summary"].get(
                        "pure_iron_boundary_visibility_score", 0.0
                    ),
                    etched["etch_summary"].get(
                        "pure_iron_boundary_visibility_score", 0.0
                    ),
                )
            )
            scratch_mean = float(prep["prep_summary"].get("scratch_mean", 0.0))
            stain_level = float(etched["etch_summary"].get("stain_level_mean", 0.0))
            recommended_profiles = dict(
                request.preset_metadata.get("recommended_electropolish_options", {})
            )
            full_recipe_profiles = recommended_profiles.get("full_recipe_profiles", [])
            if isinstance(full_recipe_profiles, list) and full_recipe_profiles:
                pure_iron_electropolish_profile = str(full_recipe_profiles[0])
            else:
                pure_iron_electropolish_profile = "pure_iron_electropolish"
            pure_iron_polarized_extinction_score = float(
                np.clip(0.82 + 0.18 * cleanliness, 0.0, 1.0)
            )
            pure_iron_optical_recommendation = {
                "default_mode": "brightfield",
                "diagnostic_mode": "darkfield"
                if max(scratch_mean, stain_level) >= 0.14
                else "brightfield",
                "secondary_mode": "phase_contrast"
                if cleanliness >= 0.7 and boundary <= 0.75
                else "brightfield",
                "polarized_recommended": False,
                "polarized_limit": "cubic_isotropic_negative_control",
                "recommended_electropolish_profile": pure_iron_electropolish_profile,
                "single_phase_negative_control": True,
                "multiphase_separability_applicable": False,
            }
        optical_recommendation = _build_optical_recommendation(
            system=str(phase_bundle.system),
            stage=str(phase_bundle.stage),
            phase_fractions=dict(phase_bundle.phase_fractions),
            preset_metadata=dict(request.preset_metadata),
            pure_iron_baseline_applied=bool(pure_iron_baseline_applied),
        )
        electron_microscopy_guidance = build_electron_microscopy_guidance(
            system=str(phase_bundle.system),
            stage=str(phase_bundle.stage),
            composition_wt=dict(composition_norm),
            phase_fractions=dict(phase_bundle.phase_fractions),
            prep_summary=dict(prep["prep_summary"]),
            etch_summary=dict(etched["etch_summary"]),
            preset_metadata=dict(request.preset_metadata),
            precipitation_state=dict(morph.get("precipitation_state", {})),
        )

        render_scale = float(
            max(
                requested_resolution[0] / max(1, internal_resolution[0]),
                requested_resolution[1] / max(1, internal_resolution[1]),
            )
        )
        base_um_per_px = float(
            max(
                0.02,
                1.0
                / (float(request.microscope_profile.get("magnification", 200)) / 100.0),
            )
        )
        microscope_ready = {
            "um_per_px": float(base_um_per_px / max(1.0, render_scale)),
            "native_um_per_px": float(base_um_per_px),
            "render_scale": float(render_scale),
            "um_per_px_100x": 1.0,
            "recommended_magnifications": [100, 200, 400, 600],
        }

        metadata: dict[str, Any] = {
            "sample_id": request.sample_id,
            "request_v3": request.to_dict(),
            "inferred_system": phase_bundle.system,
            "generator_version": self.generator_version,
            "phase_model": {
                "engine": str(request.phase_model.engine),
                "phase_control_mode": str(request.phase_model.phase_control_mode),
                "override_weight": float(request.phase_model.manual_override_weight),
                "allow_custom_fallback": bool(
                    request.phase_model.allow_custom_fallback
                ),
                "phase_balance_tolerance_pct": float(
                    request.phase_model.phase_balance_tolerance_pct
                ),
            },
            "phase_model_report": dict(phase_bundle.phase_model_report),
            "system_resolution": {
                "inferred_system": phase_bundle.system,
                "stage": phase_bundle.stage,
                "confidence": float(
                    phase_bundle.confidence
                    if phase_bundle.confidence > 0
                    else system_confidence
                ),
            },
            "thermal_program_summary": {
                **dict(thermal_summary),
                "sampling_mode": str(thermal_program.sampling_mode),
                "degree_step_c": float(thermal_program.degree_step_c),
                "max_frames": int(thermal_program.max_frames),
                "sampled_points_count": int(len(sampled_curve)),
                "operation_inference": dict(operations_from_curve.get("summary", {})),
                "stage_inference_profile": str(
                    dict(thermal_summary).get(
                        "stage_inference_profile",
                        dict(operations_from_curve.get("summary", {})).get(
                            "stage_inference_profile", "fe_c_temper_curve_v2"
                        ),
                    )
                ),
            },
            "segment_transition_report": list(
                dict(thermal_summary).get(
                    "segment_transition_report",
                    dict(operations_from_curve.get("summary", {})).get(
                        "segment_transition_report", []
                    ),
                )
            ),
            "thermal_curve_samples": list(sampled_curve),
            "operations_from_curve": dict(operations_from_curve),
            "curve_plot": None,
            "curve_csv": None,
            "quench_summary": dict(quench_summary),
            "quench_medium_profile": {
                "medium_code_resolved": str(
                    quench_summary.get(
                        "medium_code_resolved", quench_summary.get("medium_code", "")
                    )
                ),
                "cooling_rate_band_800_400": list(
                    quench_summary.get("cooling_rate_band_800_400", [])
                ),
                "hardness_hrc_as_quenched_range": list(
                    quench_summary.get("hardness_hrc_as_quenched_range", [])
                ),
                "stress_mpa_range": list(quench_summary.get("stress_mpa_range", [])),
                "harden_depth_mm_range": list(
                    quench_summary.get("harden_depth_mm_range", [])
                ),
                "defect_risk": str(quench_summary.get("defect_risk", "")),
            },
            "temper_adjustment": {
                "shift_c": float(
                    dict(operations_from_curve.get("summary", {})).get(
                        "recommended_temper_shift_c", 0.0
                    )
                ),
                "source_medium": str(
                    quench_summary.get(
                        "medium_code_resolved", quench_summary.get("medium_code", "")
                    )
                ),
                "applied_to_ranges": dict(quench_summary.get("temper_shift_c", {})),
            },
            "as_quenched_prediction": dict(
                quench_summary.get("as_quenched_prediction", {})
            ),
            "operation_guidance": dict(quench_summary.get("operation_guidance", {})),
            "process_timeline": list(micro_state.process_timeline),
            "resolved_stage_by_step": list(micro_state.resolved_stage_by_step),
            "stage_rule_source": str(
                phase_bundle.phase_model_report.get(
                    "stage_rule_source", "explicit_phase_rules_v3.json"
                )
            ),
            "prep_timeline": list(prep["prep_timeline"]),
            "etch_summary": dict(etched["etch_summary"]),
            "etch_concentration": dict(etched.get("etch_concentration", {})),
            "texture_profile": dict(morph["texture_profile"]),
            "composition_effect": dict(morph.get("composition_effect", {})),
            "phase_visibility_report": dict(morph.get("phase_visibility_report", {})),
            "system_generator": dict(morph.get("system_generator", {})),
            "fe_c_phase_render": dict(morph.get("fe_c_phase_render", {})),
            "transformation_trace": dict(morph.get("transformation_trace", {})),
            "kinetics_model": dict(morph.get("kinetics_model", {})),
            "morphology_state": dict(morph.get("morphology_state", {})),
            "precipitation_state": dict(morph.get("precipitation_state", {})),
            "validation_against_rules": dict(morph.get("validation_against_rules", {})),
            "continuous_transformation_state": dict(
                morph.get("continuous_transformation_state", {})
            ),
            "spatial_morphology_state": dict(morph.get("spatial_morphology_state", {})),
            "surface_state_summary": dict(morph.get("surface_state_summary", {})),
            "reflected_light_model": dict(morph.get("reflected_light_model", {})),
            "validation_pro": dict(morph.get("validation_pro", {})),
            "engineering_trace": {
                **dict(morph.get("engineering_trace", {})),
                "validation_severity_summary": validation_severity_summary,
            },
            "quality_metrics": {
                **qc,
                "computed_at_resolution": [
                    int(internal_resolution[0]),
                    int(internal_resolution[1]),
                ],
            },
            "textbook_profile": textbook_profile,
            "diagram_style": {
                "deprecated": True,
                "removed": True,
                "value": None,
            },
            "diagram_style_report": {
                "deprecated": True,
                "removed": True,
                "value": None,
            },
            "microscope_ready": microscope_ready,
            "preview_optics": dict(preview_optics),
            "high_resolution_render": hires_meta,
            "phase_fraction_estimate": {
                str(k): float((v > 0).mean()) for k, v in morph["phase_masks"].items()
            },
            "brinell": {
                "estimated": brinell_estimated,
                "direct": brinell_direct,
            },
            "property_indicators": dict(micro_state.property_indicators),
            "material_grade": request.material_grade,
            "material_class_ru": request.material_class_ru,
            "lab_work": request.lab_work,
            "target_astm_grain_size": request.target_astm_grain_size,
            "mean_grain_diameter_um": request.mean_grain_diameter_um,
            "expected_properties": dict(request.expected_properties),
            "expected_properties_validation": expected_properties_validation,
            "preset_metadata": dict(request.preset_metadata),
            "heat_treatment_guidance": hybrid_heat_treatment,
            "technology_influence": dict(micro_state.technology_influence),
            "final_stage": str(phase_bundle.stage),
            "prep_summary": dict(prep["prep_summary"]),
            "pure_iron_baseline": {
                "applied": pure_iron_baseline_applied,
                "cleanliness_score": float(
                    max(
                        float(morph_pure_iron.get("cleanliness_score", 0.0) or 0.0),
                        float(
                            prep["prep_summary"].get("pure_iron_cleanliness_score", 0.0)
                            or 0.0
                        ),
                        float(
                            etched["etch_summary"].get(
                                "pure_iron_cleanliness_score", 0.0
                            )
                            or 0.0
                        ),
                    )
                ),
                "dark_defect_suppression": float(
                    max(
                        float(
                            morph_pure_iron.get("dark_defect_suppression", 0.0) or 0.0
                        ),
                        float(
                            prep["prep_summary"].get(
                                "pure_iron_dark_defect_suppression", 0.0
                            )
                            or 0.0
                        ),
                        float(
                            etched["etch_summary"].get(
                                "pure_iron_dark_defect_suppression", 0.0
                            )
                            or 0.0
                        ),
                    )
                ),
                "boundary_visibility_score": float(
                    max(
                        float(
                            morph_pure_iron.get("boundary_visibility_score", 0.0) or 0.0
                        ),
                        float(
                            prep["prep_summary"].get(
                                "pure_iron_boundary_visibility_score", 0.0
                            )
                            or 0.0
                        ),
                        float(
                            etched["etch_summary"].get(
                                "pure_iron_boundary_visibility_score", 0.0
                            )
                            or 0.0
                        ),
                    )
                ),
            },
            "pure_iron_optical_recommendation": pure_iron_optical_recommendation,
            "optical_recommendation": optical_recommendation,
            "electron_microscopy_guidance": electron_microscopy_guidance,
            "pure_iron_electropolish_profile": pure_iron_electropolish_profile,
            "pure_iron_polarized_extinction_score": float(
                pure_iron_polarized_extinction_score
            ),
            "single_phase_negative_control": bool(single_phase_negative_control),
            "multiphase_separability_applicable": bool(multi_phase_readability),
            "reference_profile_id": request.reference_profile_id,
            "validation_report": report.to_dict(),
        }

        # Генерация промежуточных рендеров для каждой точки термопрограммы
        intermediate_renders = []
        if (
            request.generate_intermediate_renders
            and len(thermal_program.points) > 1
            and not use_pro_realistic
        ):
            intermediate_renders = self._generate_intermediate_renders(
                request=request,
                composition_norm=composition_norm,
                system=system,
                internal_resolution=internal_resolution,
                requested_resolution=requested_resolution,
            )
        elif request.generate_intermediate_renders and use_pro_realistic:
            metadata["pro_intermediate_renders_skipped"] = True

        return GenerationOutputV3(
            image_rgb=image_rgb,
            image_gray=image_gray,
            phase_masks=morph["phase_masks"],
            feature_masks=morph["feature_masks"],
            prep_maps=prep_maps,
            metadata=metadata,
            validation_report=report,
            intermediate_renders=intermediate_renders,
        )

    def _generate_intermediate_renders(
        self,
        request: MetallographyRequestV3,
        composition_norm: dict[str, float],
        system: str,
        internal_resolution: tuple[int, int],
        requested_resolution: tuple[int, int],
    ) -> list:
        """
        Генерирует промежуточные рендеры для каждой ключевой точки термопрограммы.

        Для каждой точки создается упрощенный рендер микроструктуры,
        показывающий состояние материала на этом этапе термообработки.
        """
        from core.contracts_v3 import (
            IntermediateRenderV3,
            ThermalProgramV3,
            ThermalPointV3,
        )

        intermediate_renders = []
        points = request.thermal_program.points

        # Используем пониженное разрешение для промежуточных рендеров (для производительности)
        intermediate_res = (
            min(512, internal_resolution[0]),
            min(512, internal_resolution[1]),
        )

        for idx, point in enumerate(points):
            try:
                # Создаем термопрограмму, которая заканчивается на текущей точке
                truncated_points = points[: idx + 1]

                # Если это не последняя точка, добавляем быструю выдержку
                if idx < len(points) - 1:
                    # Копируем точку и обнуляем transition_to_next для последней точки
                    final_point = ThermalPointV3(
                        time_s=point.time_s,
                        temperature_c=point.temperature_c,
                        label=point.label,
                        locked=point.locked,
                    )
                    truncated_points = truncated_points[:-1] + [final_point]

                truncated_program = ThermalProgramV3(
                    points=truncated_points,
                    quench=request.thermal_program.quench,
                    sampling_mode=request.thermal_program.sampling_mode,
                    degree_step_c=request.thermal_program.degree_step_c,
                    max_frames=request.thermal_program.max_frames,
                )

                # Создаем упрощенный запрос для этой точки
                intermediate_request = MetallographyRequestV3(
                    sample_id=f"{request.sample_id}_point_{idx}",
                    composition_wt=composition_norm,
                    system_hint=system,
                    thermal_program=truncated_program,
                    prep_route=request.prep_route,
                    etch_profile=request.etch_profile,
                    synthesis_profile=request.synthesis_profile,
                    phase_model=request.phase_model,
                    microscope_profile={
                        **request.microscope_profile,
                        "simulate_preview": False,
                    },
                    seed=request.seed + idx * 1000,  # Разный seed для каждой точки
                    resolution=intermediate_res,
                    strict_validation=False,  # Не строгая валидация для промежуточных
                    reference_profile_id=request.reference_profile_id,
                    generate_intermediate_renders=False,  # Избегаем рекурсии
                )

                # Генерируем микроструктуру для этой точки
                thermal_summary = summarize_thermal_program(truncated_program)
                process_probe, _, quench_summary = effective_processing_from_thermal(
                    truncated_program
                )

                micro_state = build_microstructure_state(
                    composition=composition_norm,
                    inferred_system=system,
                    processing=process_probe,
                    thermal_summary=thermal_summary,
                    operations_from_curve=infer_operations_from_thermal_program(
                        truncated_program,
                        summary=thermal_summary,
                        quench_summary=quench_summary,
                    ),
                    quench_summary=quench_summary,
                    seed=intermediate_request.seed,
                )

                phase_bundle = build_phase_bundle(
                    composition=composition_norm,
                    processing=process_probe,
                    system_hint=system,
                    phase_model=intermediate_request.phase_model,
                    thermal_summary=thermal_summary,
                    quench_summary=quench_summary,
                )

                ref_style = None
                if intermediate_request.reference_profile_id:
                    ref_style = load_reference_profile(
                        profile_id=intermediate_request.reference_profile_id,
                        profiles_root=self.profiles_dir,
                    )
                if ref_style is None:
                    ref_style = resolve_reference_style(
                        profile_id=intermediate_request.synthesis_profile.profile_id,
                        profiles_root=self.profiles_dir,
                    )

                morph = generate_phase_topology(
                    size=intermediate_res,
                    seed=intermediate_request.seed,
                    phase_bundle=phase_bundle,
                    micro_state=micro_state,
                    synthesis_profile=intermediate_request.synthesis_profile,
                    reference_style=ref_style,
                    composition_wt=composition_norm,
                    composition_sensitivity_mode=str(
                        intermediate_request.synthesis_profile.composition_sensitivity_mode
                    ),
                    generation_mode=str(
                        intermediate_request.synthesis_profile.generation_mode
                    ),
                    phase_emphasis_style=str(
                        intermediate_request.synthesis_profile.phase_emphasis_style
                    ),
                    phase_fraction_tolerance_pct=float(
                        intermediate_request.synthesis_profile.phase_fraction_tolerance_pct
                    ),
                    thermal_summary=thermal_summary,
                    quench_summary=quench_summary,
                )

                prep = apply_prep_route(
                    image_gray=morph["image_gray"],
                    prep_route=intermediate_request.prep_route,
                    seed=intermediate_request.seed + 77,
                    phase_masks=morph["phase_masks"],
                    system=str(phase_bundle.system),
                    composition_wt=composition_norm,
                    effect_vector=dict(micro_state.effect_vector),
                )

                etched = apply_etch(
                    image_gray=prep["image_gray"],
                    phase_masks=morph["phase_masks"],
                    etch_profile=intermediate_request.etch_profile,
                    seed=intermediate_request.seed + 131,
                    prep_maps=prep["prep_maps"],
                    system=str(phase_bundle.system),
                    composition_wt=composition_norm,
                    effect_vector=dict(micro_state.effect_vector),
                )

                image_gray_intermediate = etched["image_gray"].astype(np.uint8)
                image_rgb_intermediate = _to_rgb(image_gray_intermediate)

                # Создаем объект промежуточного рендера
                intermediate_render = IntermediateRenderV3(
                    point_index=idx,
                    time_s=float(point.time_s),
                    temperature_c=float(point.temperature_c),
                    label=str(point.label or f"Точка {idx + 1}"),
                    image_rgb=image_rgb_intermediate,
                    image_gray=image_gray_intermediate,
                    phase_info={
                        "stage": str(phase_bundle.stage),
                        "phase_fractions": {
                            k: float(v) for k, v in phase_bundle.phase_fractions.items()
                        },
                        "system": str(phase_bundle.system),
                    },
                )

                intermediate_renders.append(intermediate_render)

            except Exception as exc:
                # Если не удалось сгенерировать промежуточный рендер, пропускаем его
                # но не прерываем весь процесс
                print(
                    f"Warning: Failed to generate intermediate render for point {idx}: {exc}"
                )
                continue

        return intermediate_renders

    def generate_batch(
        self,
        requests: list[MetallographyRequestV3],
        output_dir: str | Path,
        file_prefix: str = "sample_v3",
    ) -> BatchResultV3:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for idx, req in enumerate(requests, start=1):
            sample_id = f"{file_prefix}_{idx:03d}"
            try:
                result = self.generate(req)
                image_path = out / f"{sample_id}.png"
                meta_path = out / f"{sample_id}.json"
                save_image(result.image_rgb, image_path)
                save_json(result.metadata_json_safe(), meta_path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "system": result.metadata.get("inferred_system", ""),
                        "stage": result.metadata.get("final_stage", ""),
                        "system_generator": dict(
                            result.metadata.get("system_generator", {})
                        ).get("resolved_mode", ""),
                        "textbook_pass": bool(
                            dict(result.metadata.get("textbook_profile", {})).get(
                                "pass", False
                            )
                        ),
                        "validation_passed": True,
                        "image_path": str(image_path),
                        "metadata_path": str(meta_path),
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "system": "",
                        "stage": "",
                        "system_generator": "",
                        "textbook_pass": False,
                        "validation_passed": False,
                        "image_path": "",
                        "metadata_path": "",
                        "error": str(exc),
                    }
                )

        index_path = out / f"{file_prefix}_index.csv"
        save_measurements_csv(rows, index_path)
        return BatchResultV3(rows=rows, csv_index_path=index_path)
