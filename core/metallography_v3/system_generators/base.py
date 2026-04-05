from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.contracts_v2 import ProcessingState

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


@dataclass(slots=True)
class SystemGenerationContext:
    size: tuple[int, int]
    seed: int
    inferred_system: str
    stage: str
    phase_fractions: dict[str, float]
    composition_wt: dict[str, float]
    processing: ProcessingState
    effect_vector: dict[str, float] = field(default_factory=dict)
    thermal_summary: dict[str, Any] = field(default_factory=dict)
    quench_summary: dict[str, Any] = field(default_factory=dict)
    composition_sensitivity_mode: str = "realistic"
    generation_mode: str = "edu_engineering"
    phase_emphasis_style: str = "contrast_texture"
    phase_fraction_tolerance_pct: float = 20.0
    visual_profile_id: str = ""
    confidence: float = 0.0
    phase_fraction_source: str = "default_formula"
    phase_calibration_mode: str = "default_formula"
    transformation_state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SystemGenerationResult:
    image_gray: np.ndarray
    phase_masks: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


def ensure_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    arr = image.astype(np.float32)
    return np.clip(arr, 0, 255).astype(np.uint8)


def normalize_phase_fractions(phase_fractions: dict[str, float]) -> dict[str, float]:
    cleaned = {
        str(name): float(max(0.0, value))
        for name, value in phase_fractions.items()
        if float(value) > 1e-9
    }
    total = float(sum(cleaned.values()))
    if total <= 1e-12:
        return {"solid": 1.0}
    return {name: val / total for name, val in cleaned.items()}


def build_phase_masks_from_intensity(
    image_gray: np.ndarray,
    phase_fractions: dict[str, float],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    phases = normalize_phase_fractions(phase_fractions)
    names = [name for name, _ in sorted(phases.items(), key=lambda item: item[1], reverse=True)]
    if len(names) <= 1:
        return {"solid": np.ones_like(image_gray, dtype=np.uint8)}

    probs = np.asarray([float(phases[name]) for name in names], dtype=np.float64)
    probs /= float(probs.sum())
    rng = np.random.default_rng(int(seed) + 5179)
    field = image_gray.astype(np.float32) + rng.normal(0.0, 0.001, size=image_gray.shape).astype(np.float32)
    flat = field.ravel()
    order = np.argsort(flat)
    labels = np.zeros(flat.size, dtype=np.int32)
    start = 0
    for idx, prob in enumerate(probs):
        if idx == len(probs) - 1:
            end = flat.size
        else:
            end = min(flat.size, start + int(round(float(prob) * flat.size)))
        labels[order[start:end]] = idx
        start = end
    labels2d = labels.reshape(image_gray.shape)
    masks: dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        masks[name] = (labels2d == idx).astype(np.uint8)
    return masks


def _phase_separability_score(image_gray: np.ndarray, phase_masks: dict[str, np.ndarray]) -> float:
    means: list[float] = []
    stds: list[float] = []
    for name, mask in phase_masks.items():
        if str(name) in {"L", "solid"}:
            continue
        if not isinstance(mask, np.ndarray):
            continue
        pix = image_gray[mask > 0]
        if pix.size < 32:
            continue
        means.append(float(np.mean(pix)))
        stds.append(float(np.std(pix)) + 1e-6)
    if len(means) < 2:
        return 0.0
    ratios: list[float] = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            ratios.append(abs(means[i] - means[j]) / (stds[i] + stds[j]))
    if not ratios:
        return 0.0
    raw = float(np.mean(ratios))
    return float(np.clip(1.0 - np.exp(-0.9 * raw), 0.0, 1.0))


def build_phase_visibility_report(
    *,
    image_gray: np.ndarray,
    phase_masks: dict[str, np.ndarray],
    phase_fractions: dict[str, float],
    tolerance_pct: float,
) -> dict[str, Any]:
    target = normalize_phase_fractions(phase_fractions)
    actual: dict[str, float] = {}
    err: dict[str, float] = {}
    for phase_name, target_frac in target.items():
        mask = phase_masks.get(phase_name)
        if isinstance(mask, np.ndarray):
            actual_frac = float((mask > 0).mean())
        else:
            actual_frac = 0.0
        actual[phase_name] = actual_frac
        denom = max(1e-6, float(target_frac))
        err[phase_name] = float(abs(actual_frac - float(target_frac)) / denom * 100.0)

    return {
        "target_phase_fractions": target,
        "achieved_phase_fractions": actual,
        "fraction_error_pct": err,
        "within_tolerance": bool(max(err.values() or [0.0]) <= float(max(0.0, tolerance_pct))),
        "separability_score": float(_phase_separability_score(image_gray=image_gray, phase_masks=phase_masks)),
    }


def _normalize_composition(composition_wt: dict[str, float] | None) -> dict[str, float]:
    if not isinstance(composition_wt, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in composition_wt.items():
        k = str(key).strip()
        if not k:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        if v > 0.0:
            out[k] = v
    total = float(sum(out.values()))
    if total <= 1e-12:
        return {}
    return {k: 100.0 * float(v) / total for k, v in out.items()}


def _composition_signature(composition_wt: dict[str, float] | None) -> tuple[str, int]:
    norm = _normalize_composition(composition_wt)
    if not norm:
        return "none", 0
    payload = json.dumps(
        {k: float(v) for k, v in sorted(norm.items(), key=lambda x: x[0])},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return digest[:16], int(digest[:8], 16)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _solute_index(system: str, composition_wt: dict[str, float] | None) -> float:
    comp = _normalize_composition(composition_wt)
    if not comp:
        return 0.0
    sys_name = str(system).strip().lower()
    if sys_name == "fe-c":
        return _clip01(float(comp.get("C", 0.0)) / 2.1)
    if sys_name == "fe-si":
        return _clip01(float(comp.get("Si", 0.0)) / 6.0)
    if sys_name == "al-si":
        return _clip01(float(comp.get("Si", 0.0)) / 25.0)
    if sys_name == "cu-zn":
        return _clip01(float(comp.get("Zn", 0.0)) / 45.0)
    if sys_name == "al-cu-mg":
        cu = float(comp.get("Cu", 0.0))
        mg = float(comp.get("Mg", 0.0))
        return _clip01((cu + 0.7 * mg) / 8.0)
    matrix_el = "Fe" if "Fe" in comp else ("Al" if "Al" in comp else ("Cu" if "Cu" in comp else ""))
    if matrix_el:
        alloying = max(0.0, 100.0 - float(comp.get(matrix_el, 0.0)))
        return _clip01(alloying / 30.0)
    return _clip01(float(sum(comp.values())) / 100.0)


def build_composition_effect(
    *,
    system: str,
    composition_wt: dict[str, float] | None,
    mode: str,
    seed: int,
    single_phase_compensation: bool = False,
) -> dict[str, Any]:
    comp_hash, seed_offset = _composition_signature(composition_wt)
    return {
        "mode": str(mode),
        "solute_index": float(round(_solute_index(system=system, composition_wt=composition_wt), 6)),
        "composition_hash": comp_hash,
        "seed_offset": int(seed_offset ^ int(seed)),
        "single_phase_compensation": bool(single_phase_compensation),
    }


def soft_unsharp(image_gray: np.ndarray, amount: float = 0.45) -> np.ndarray:
    arr = image_gray.astype(np.float32)
    if ndimage is None:
        return ensure_u8(arr)
    blur = ndimage.gaussian_filter(arr, sigma=1.0)
    out = arr + (arr - blur) * float(max(0.0, amount))
    return ensure_u8(out)
