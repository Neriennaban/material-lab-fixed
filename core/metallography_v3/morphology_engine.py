from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from core.contracts_v3 import SynthesisProfileV3
from core.generator_grains import generate_grain_structure
from core.metallography_v3.microstructure_state import MicrostructureStateV3
from core.metallography_v3.phase_orchestrator import PhaseBundleV3
from core.metallography_v3.transformation_state import build_transformation_state
from core.metallography_v3.system_generators.base import (
    SystemGenerationContext,
    build_phase_visibility_report,
)
from core.metallography_v3.system_generators.registry import SystemGeneratorRegistryV3

_SYSTEM_GEN_REGISTRY: SystemGeneratorRegistryV3 | None = None
_TOPOLOGY_CACHE_MAX = 3
_TOPOLOGY_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _physics_guided_realism_enabled(topology_mode: str) -> bool:
    mode = str(topology_mode or "").strip().lower()
    if mode in {"physics_guided", "physics_guided_hybrid"}:
        return True
    env = str(os.environ.get("ML_ENABLE_PHYSICS_GUIDED_REALISM", "")).strip().lower()
    return env in {"1", "true", "yes", "on"}


def _system_registry() -> SystemGeneratorRegistryV3:
    global _SYSTEM_GEN_REGISTRY
    if _SYSTEM_GEN_REGISTRY is None:
        _SYSTEM_GEN_REGISTRY = SystemGeneratorRegistryV3()
    return _SYSTEM_GEN_REGISTRY


def _normalize_u8(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo + 1e-9:
        return np.zeros_like(image, dtype=np.uint8)
    out = (arr - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _lift_small_dark_blobs(
    image_gray: np.ndarray,
    *,
    threshold: float = 40.0,
    max_pixels: int = 48,
) -> np.ndarray:
    """Lift dark blobs smaller than ``max_pixels`` toward the local mean.

    Phase D.2 — the legacy implementation iterated over every
    connected component and rebuilt a boolean mask with
    ``labels == label`` inside the loop, giving an O(K × N) cost
    that dominated the pipeline at 1K+ resolution (2.0 s of a 6 s
    budget at 1024×1024). The vectorised version below processes
    every blob in a single pass using ``np.bincount`` and
    ``np.isin`` — O(N + K) instead of O(N × K).
    """
    if ndimage is None:
        return image_gray.astype(np.uint8, copy=False)
    arr = image_gray.astype(np.float32, copy=False)
    mask = arr < float(threshold)
    labels, count = ndimage.label(mask.astype(np.uint8))
    if int(count) <= 0:
        return image_gray.astype(np.uint8, copy=False)
    # Component sizes in one pass (bincount includes background as 0).
    sizes = np.bincount(labels.ravel())
    tiny_ids = np.where(sizes <= int(max_pixels))[0]
    # Drop the background label (0) which always wins the size check.
    if tiny_ids.size > 0 and tiny_ids[0] == 0:
        tiny_ids = tiny_ids[1:]
    if tiny_ids.size == 0:
        return image_gray.astype(np.uint8, copy=False)
    tiny_mask = np.isin(labels, tiny_ids, assume_unique=False)
    if not tiny_mask.any():
        return image_gray.astype(np.uint8, copy=False)
    local = ndimage.gaussian_filter(arr, sigma=1.05)
    out = arr.copy()
    out[tiny_mask] = 0.84 * local[tiny_mask] + 0.16 * out[tiny_mask]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _phase_boundaries(
    phase_masks: dict[str, np.ndarray], size: tuple[int, int]
) -> np.ndarray:
    label = np.zeros(size, dtype=np.int32)
    idx = 1
    for _, mask in phase_masks.items():
        if isinstance(mask, np.ndarray):
            label[mask > 0] = idx
            idx += 1
    borders = np.zeros(size, dtype=np.uint8)
    borders[:-1, :] |= (label[:-1, :] != label[1:, :]).astype(np.uint8)
    borders[:, :-1] |= (label[:, :-1] != label[:, 1:]).astype(np.uint8)
    if ndimage is not None:
        borders = ndimage.binary_dilation(borders > 0, iterations=1).astype(np.uint8)
    return borders


def _stable_json(value: Any) -> str:
    def _default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, set):
            return sorted([str(x) for x in obj])
        if isinstance(obj, tuple):
            return list(obj)
        return str(obj)

    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=_default,
        )
    except Exception:
        return str(value)


def _topology_cache_key(
    *,
    size: tuple[int, int],
    seed: int,
    phase_bundle: PhaseBundleV3,
    micro_state: MicrostructureStateV3,
    synthesis_profile: SynthesisProfileV3,
    reference_style: dict[str, Any] | None,
    composition_wt: dict[str, float] | None,
    composition_sensitivity_mode: str,
    generation_mode: str,
    phase_emphasis_style: str,
    phase_fraction_tolerance_pct: float,
    thermal_summary: dict[str, Any] | None,
    quench_summary: dict[str, Any] | None,
) -> str:
    phase_items = [
        [str(k), round(float(v), 8)]
        for k, v in sorted(
            dict(phase_bundle.phase_fractions).items(), key=lambda item: str(item[0])
        )
    ]
    comp_items = [
        [str(k), round(float(v), 8)]
        for k, v in sorted(
            dict(composition_wt or {}).items(), key=lambda item: str(item[0])
        )
    ]
    effect_items = [
        [str(k), round(float(v), 8)]
        for k, v in sorted(
            dict(micro_state.effect_vector).items(), key=lambda item: str(item[0])
        )
    ]
    payload = {
        "size": [int(size[0]), int(size[1])],
        "seed": int(seed),
        "system": str(phase_bundle.system),
        "stage": str(phase_bundle.stage),
        "phase_fractions": phase_items,
        "confidence": round(float(phase_bundle.confidence), 8),
        "final_stage": str(micro_state.final_stage),
        "effect_vector": effect_items,
        "synthesis_profile": {
            "profile_id": str(synthesis_profile.profile_id),
            "phase_topology_mode": str(synthesis_profile.phase_topology_mode),
            "system_generator_mode": str(
                getattr(synthesis_profile, "system_generator_mode", "system_auto")
            ),
            "contrast_target": round(float(synthesis_profile.contrast_target), 8),
            "boundary_sharpness": round(float(synthesis_profile.boundary_sharpness), 8),
            "artifact_level": round(float(synthesis_profile.artifact_level), 8),
        },
        "reference_style": reference_style or {},
        "composition": comp_items,
        "composition_sensitivity_mode": str(composition_sensitivity_mode),
        "generation_mode": str(generation_mode),
        "phase_emphasis_style": str(phase_emphasis_style),
        "phase_fraction_tolerance_pct": round(float(phase_fraction_tolerance_pct), 8),
        "thermal_summary": thermal_summary or {},
        "quench_summary": quench_summary or {},
    }
    digest = hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()
    return digest


def _clone_topology_payload(payload: dict[str, Any]) -> dict[str, Any]:
    phase_masks = {
        str(k): np.array(v, copy=True)
        for k, v in dict(payload.get("phase_masks", {})).items()
        if isinstance(v, np.ndarray)
    }
    feature_masks = {
        str(k): np.array(v, copy=True)
        for k, v in dict(payload.get("feature_masks", {})).items()
        if isinstance(v, np.ndarray)
    }
    raw_labels = payload.get("grain_labels")
    grain_labels = (
        np.array(raw_labels, copy=True) if isinstance(raw_labels, np.ndarray) else None
    )
    return {
        "image_gray": np.array(payload.get("image_gray"), copy=True),
        "phase_masks": phase_masks,
        "feature_masks": feature_masks,
        "texture_profile": dict(payload.get("texture_profile", {})),
        "composition_effect": dict(payload.get("composition_effect", {})),
        "phase_visibility_report": dict(payload.get("phase_visibility_report", {})),
        "engineering_trace": dict(payload.get("engineering_trace", {})),
        "system_generator": dict(payload.get("system_generator", {})),
        "fe_c_phase_render": dict(payload.get("fe_c_phase_render", {})),
        "transformation_trace": dict(payload.get("transformation_trace", {})),
        "kinetics_model": dict(payload.get("kinetics_model", {})),
        "morphology_state": dict(payload.get("morphology_state", {})),
        "precipitation_state": dict(payload.get("precipitation_state", {})),
        "validation_against_rules": dict(payload.get("validation_against_rules", {})),
        "grain_labels": grain_labels,
    }


def _apply_style(
    image: np.ndarray,
    synthesis: SynthesisProfileV3,
    effect_vector: dict[str, float],
    ref_style: dict[str, Any] | None,
    *,
    generation_mode: str,
    phase_emphasis_style: str,
) -> np.ndarray:
    arr = image.astype(np.float32)
    contrast = float(max(0.5, min(2.2, synthesis.contrast_target)))
    sharpness = float(max(0.4, min(2.5, synthesis.boundary_sharpness)))
    artifact_level = float(max(0.0, min(1.0, synthesis.artifact_level)))
    profile_id = str(synthesis.profile_id or "").strip().lower()
    disloc = float(max(0.0, effect_vector.get("dislocation_proxy", 0.0)))
    segregation = float(max(0.0, effect_vector.get("segregation_level", 0.0)))

    edu_mode = str(generation_mode).strip().lower() == "edu_engineering"
    textbook_steel_bw = profile_id == "textbook_steel_bw"
    if edu_mode:
        # In educational-engineering mode keep phase readability prioritized over decorative post-effects.
        artifact_level = min(artifact_level, 0.18)
        style = str(phase_emphasis_style).strip().lower()
        if style == "max_contrast":
            contrast = min(2.6, contrast * 1.25)
            sharpness = min(2.8, sharpness * 1.16)
        elif style == "contrast_texture":
            contrast = min(2.45, contrast * 1.14)
            sharpness = min(2.7, sharpness * 1.10)
        elif style == "morphology_only":
            contrast = min(1.2, contrast)

    arr = (arr - 127.5) * contrast + 127.5
    if ndimage is not None:
        blurred = ndimage.gaussian_filter(arr, sigma=max(0.15, 1.2 / sharpness))
        arr = arr + (arr - blurred) * 0.8 * sharpness
    grain_noise = (
        np.random.default_rng(42).normal(0.0, 1.0, size=image.shape).astype(np.float32)
    )
    if ndimage is not None:
        grain_noise = ndimage.gaussian_filter(
            grain_noise, sigma=1.2 + 4.5 * (1.0 - artifact_level)
        )
    noise_gain = 0.64 if edu_mode else 1.0
    if textbook_steel_bw and edu_mode:
        noise_gain *= 0.55
    arr += grain_noise * noise_gain * (6.0 + 10.0 * disloc + 8.0 * segregation)

    if isinstance(ref_style, dict):
        target_std = float(ref_style.get("std", 48.0))
        cur_std = float(arr.std()) + 1e-6
        arr = (arr - arr.mean()) * (target_std / cur_std) + arr.mean()

    out = np.clip(arr, 0, 255).astype(np.uint8)
    if textbook_steel_bw and edu_mode:
        q01 = float(np.quantile(out.astype(np.float32), 0.01))
        if q01 < 44.0:
            lifted = out.astype(np.float32) + (44.0 - q01)
            out = np.clip(lifted, 0.0, 255.0).astype(np.uint8)
        out = _lift_small_dark_blobs(
            out,
            threshold=42.0,
            max_pixels=max(24, int(out.size // 32768)),
        )
    return out


def generate_phase_topology(
    *,
    size: tuple[int, int],
    seed: int,
    phase_bundle: PhaseBundleV3,
    micro_state: MicrostructureStateV3,
    synthesis_profile: SynthesisProfileV3,
    reference_style: dict[str, Any] | None,
    composition_wt: dict[str, float] | None = None,
    composition_sensitivity_mode: str = "realistic",
    generation_mode: str = "realistic_visual",
    phase_emphasis_style: str = "contrast_texture",
    phase_fraction_tolerance_pct: float = 20.0,
    thermal_summary: dict[str, Any] | None = None,
    quench_summary: dict[str, Any] | None = None,
    microscope_profile: dict[str, Any] | None = None,
    color_mode: str = "grayscale_nital",
) -> dict[str, Any]:
    # Compute microscope-derived context (A0.2 magnification propagation).
    # Default to 200× (0.5 µm/px) when no microscope profile is supplied so
    # the behaviour matches the pre-A0.2 baseline for all existing presets.
    _magnification = 200.0
    if isinstance(microscope_profile, dict):
        try:
            raw_mag = float(microscope_profile.get("magnification", 200.0))
        except Exception:
            raw_mag = 200.0
        if raw_mag > 0.0:
            _magnification = raw_mag
    _native_um_per_px = 1.0 / max(1e-3, _magnification / 100.0)

    cache_key = _topology_cache_key(
        size=size,
        seed=int(seed),
        phase_bundle=phase_bundle,
        micro_state=micro_state,
        synthesis_profile=synthesis_profile,
        reference_style=reference_style,
        composition_wt=composition_wt,
        composition_sensitivity_mode=composition_sensitivity_mode,
        generation_mode=generation_mode,
        phase_emphasis_style=phase_emphasis_style,
        phase_fraction_tolerance_pct=phase_fraction_tolerance_pct,
        thermal_summary=thermal_summary,
        quench_summary=quench_summary,
    )
    cached = _TOPOLOGY_CACHE.get(cache_key)
    if cached is not None:
        _TOPOLOGY_CACHE.move_to_end(cache_key)
        return _clone_topology_payload(cached)

    phases = dict(phase_bundle.phase_fractions)
    topology_mode = str(synthesis_profile.phase_topology_mode or "auto").strip().lower()
    if topology_mode.startswith("v2_"):
        raise ValueError(
            "LEGACY_FIELD_REMOVED: V2 compatibility topology modes are not supported in V3."
        )
    system_generator_mode = str(
        getattr(synthesis_profile, "system_generator_mode", "system_auto")
        or "system_auto"
    )
    composition_effect: dict[str, Any] = {
        "mode": str(composition_sensitivity_mode),
        "solute_index": 0.0,
        "composition_hash": "none",
        "seed_offset": 0,
        "single_phase_compensation": False,
    }
    phase_visibility_report: dict[str, Any] = {
        "target_phase_fractions": {},
        "achieved_phase_fractions": {},
        "fraction_error_pct": {},
        "within_tolerance": True,
        "separability_score": 0.0,
    }
    engineering_trace: dict[str, Any] = {
        "generation_mode": str(generation_mode),
        "phase_emphasis_style": str(phase_emphasis_style),
        "phase_fraction_tolerance_pct": float(phase_fraction_tolerance_pct),
        "homogeneity_level": "light",
    }
    system_generator_meta: dict[str, Any] = {
        "requested_mode": system_generator_mode,
        "resolved_mode": "",
        "resolved_system": str(phase_bundle.system),
        "resolved_stage": str(phase_bundle.stage),
        "fallback_used": False,
        "selection_reason": "",
        "confidence": float(phase_bundle.confidence),
    }
    fe_c_phase_render: dict[str, Any] = {}
    transformation_trace: dict[str, Any] = {}
    kinetics_model: dict[str, Any] = {}
    morphology_state: dict[str, Any] = {}
    precipitation_state: dict[str, Any] = {}
    validation_against_rules: dict[str, Any] = {}
    grain_labels_from_meta: np.ndarray | None = None
    if not phases:
        fallback = generate_grain_structure(
            size=size, seed=seed, mean_grain_size_px=52.0
        )
        image = fallback["image"]
        phase_masks: dict[str, np.ndarray] = {"solid": np.ones(size, dtype=np.uint8)}
        phase_visibility_report = build_phase_visibility_report(
            image_gray=image.astype(np.uint8),
            phase_masks=phase_masks,
            phase_fractions={"solid": 1.0},
            tolerance_pct=float(phase_fraction_tolerance_pct),
        )
        system_generator_meta.update(
            {
                "resolved_mode": "system_custom",
                "fallback_used": True,
                "selection_reason": "empty_phase_fractions",
            }
        )
    else:
        transformation_state: dict[str, Any] = {}
        if _physics_guided_realism_enabled(topology_mode):
            transformation_state = build_transformation_state(
                inferred_system=str(phase_bundle.system),
                stage=str(phase_bundle.stage or micro_state.final_stage),
                composition_wt=dict(composition_wt or {}),
                processing=micro_state.final_processing,
                effect_vector=dict(micro_state.effect_vector),
                thermal_summary=dict(thermal_summary or {}),
                quench_summary=dict(quench_summary or {}),
            )
        context = SystemGenerationContext(
            size=size,
            seed=int(seed),
            inferred_system=str(phase_bundle.system),
            stage=str(phase_bundle.stage or micro_state.final_stage),
            phase_fractions=dict(phases),
            composition_wt=dict(composition_wt or {}),
            processing=micro_state.final_processing,
            effect_vector=dict(micro_state.effect_vector),
            thermal_summary=dict(thermal_summary or {}),
            quench_summary=dict(quench_summary or {}),
            composition_sensitivity_mode=str(composition_sensitivity_mode),
            generation_mode=str(generation_mode),
            phase_emphasis_style=str(phase_emphasis_style),
            phase_fraction_tolerance_pct=float(phase_fraction_tolerance_pct),
            visual_profile_id=str(synthesis_profile.profile_id),
            confidence=float(phase_bundle.confidence),
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
            transformation_state=dict(transformation_state),
            magnification=float(_magnification),
            native_um_per_px=float(_native_um_per_px),
            color_mode=str(color_mode),
        )
        generated, selection = _system_registry().generate(
            context=context,
            requested_mode=system_generator_mode,
        )
        image = generated.image_gray.astype(np.uint8)
        phase_masks = {
            str(name): (mask > 0).astype(np.uint8)
            for name, mask in dict(generated.phase_masks).items()
            if isinstance(mask, np.ndarray)
        }
        meta = dict(generated.metadata)
        # A10.3 — extract the per-grain label map through the morph
        # dict so ``apply_color_palette`` can use it for DIC/polarised
        # renderings. The value is stored under the private
        # ``_grain_labels`` key so we can ``pop`` it immediately and
        # keep ``metadata`` JSON-serialisable.
        grain_labels_from_meta = None
        if isinstance(meta, dict):
            raw_labels = meta.pop("_grain_labels", None)
            if isinstance(raw_labels, np.ndarray):
                grain_labels_from_meta = raw_labels
            # Also scrub the legacy key name in case older callers
            # still stashed it there (defensive; should be a no-op).
            meta.pop("grain_labels", None)
            # Morphology trace may carry a copy of the same ndarray;
            # strip it before the trace is handed to fe_c_phase_render.
            fe_c_render_meta = meta.get("fe_c_phase_render")
            if isinstance(fe_c_render_meta, dict):
                mt = fe_c_render_meta.get("morphology_trace")
                if isinstance(mt, dict):
                    mt.pop("grain_labels", None)
        if isinstance(meta, dict):
            if isinstance(meta.get("composition_effect"), dict):
                composition_effect = dict(meta["composition_effect"])
            if isinstance(meta.get("phase_visibility_report"), dict):
                phase_visibility_report = dict(meta["phase_visibility_report"])
            else:
                phase_visibility_report = build_phase_visibility_report(
                    image_gray=image,
                    phase_masks=phase_masks,
                    phase_fractions=phases,
                    tolerance_pct=float(phase_fraction_tolerance_pct),
                )
            if isinstance(meta.get("engineering_trace"), dict):
                engineering_trace = dict(meta["engineering_trace"])
            if isinstance(meta.get("system_generator_extra"), dict):
                system_generator_meta.update(dict(meta["system_generator_extra"]))
            if isinstance(meta.get("fe_c_phase_render"), dict):
                fe_c_phase_render = dict(meta["fe_c_phase_render"])
            if isinstance(meta.get("transformation_trace"), dict):
                transformation_trace = dict(meta["transformation_trace"])
            if isinstance(meta.get("kinetics_model"), dict):
                kinetics_model = dict(meta["kinetics_model"])
            if isinstance(meta.get("morphology_state"), dict):
                morphology_state = dict(meta["morphology_state"])
            if isinstance(meta.get("precipitation_state"), dict):
                precipitation_state = dict(meta["precipitation_state"])
            if isinstance(meta.get("validation_against_rules"), dict):
                validation_against_rules = dict(meta["validation_against_rules"])
        system_generator_meta.update(selection.to_dict())

    styled = _apply_style(
        image=image,
        synthesis=synthesis_profile,
        effect_vector=micro_state.effect_vector,
        ref_style=reference_style,
        generation_mode=str(generation_mode),
        phase_emphasis_style=str(phase_emphasis_style),
    )
    feature_masks: dict[str, np.ndarray] = {}
    feature_masks["phase_boundaries"] = _phase_boundaries(phase_masks, size=size)
    feature_masks["high_contrast"] = (styled > int(np.quantile(styled, 0.85))).astype(
        np.uint8
    )
    feature_masks["low_contrast"] = (styled < int(np.quantile(styled, 0.15))).astype(
        np.uint8
    )

    result = {
        "image_gray": _normalize_u8(styled),
        "phase_masks": phase_masks,
        "feature_masks": feature_masks,
        "texture_profile": {
            "profile_id": synthesis_profile.profile_id,
            "phase_topology_mode": synthesis_profile.phase_topology_mode,
            "system_generator_mode": system_generator_mode,
            "system_generator": system_generator_meta,
            "contrast_target": synthesis_profile.contrast_target,
            "boundary_sharpness": synthesis_profile.boundary_sharpness,
            "artifact_level": synthesis_profile.artifact_level,
            "composition_effect": composition_effect,
            "phase_visibility_report": phase_visibility_report,
            "engineering_trace": engineering_trace,
        },
        "composition_effect": composition_effect,
        "phase_visibility_report": phase_visibility_report,
        "engineering_trace": engineering_trace,
        "system_generator": system_generator_meta,
        "fe_c_phase_render": fe_c_phase_render,
        "transformation_trace": transformation_trace,
        "kinetics_model": kinetics_model,
        "morphology_state": morphology_state,
        "precipitation_state": precipitation_state,
        "validation_against_rules": validation_against_rules,
        # A10.3 — in-memory label map for DIC / per-grain colouring.
        # ``None`` when the underlying generator did not produce one.
        "grain_labels": (
            grain_labels_from_meta
            if isinstance(grain_labels_from_meta, np.ndarray)
            else None
        ),
    }
    _TOPOLOGY_CACHE[cache_key] = _clone_topology_payload(result)
    while len(_TOPOLOGY_CACHE) > _TOPOLOGY_CACHE_MAX:
        _TOPOLOGY_CACHE.popitem(last=False)
    return result
