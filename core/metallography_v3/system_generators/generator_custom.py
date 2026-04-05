from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from core.generator_eutectic import generate_aged_aluminum_structure, generate_eutectic_al_si
from core.generator_grains import generate_grain_structure

from .base import (
    SystemGenerationContext,
    SystemGenerationResult,
    build_composition_effect,
    build_phase_visibility_report,
    ensure_u8,
    normalize_phase_fractions,
    soft_unsharp,
)


def _liquid_texture(size: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    field = rng.normal(0.0, 1.0, size=size).astype(np.float32)
    if ndimage is not None:
        field = ndimage.gaussian_filter(field, sigma=8.0)
    swirl = np.sin(xx / max(1, w) * 6.0) + np.cos(yy / max(1, h) * 4.5)
    image = 152.0 + field * 20.0 + swirl * 6.0
    return ensure_u8(image)


def _labels_from_fractions(size: tuple[int, int], seed: int, fractions: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    phases = normalize_phase_fractions(fractions)
    names = [name for name, _ in sorted(phases.items(), key=lambda item: item[1], reverse=True)]
    probs = np.asarray([float(phases[name]) for name in names], dtype=np.float64)
    probs = probs / float(probs.sum())
    rng = np.random.default_rng(int(seed) + 7331)
    field = rng.normal(0.0, 1.0, size=size).astype(np.float32)
    if ndimage is not None:
        field = ndimage.gaussian_filter(field, sigma=max(2.0, min(size) / 35.0))
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
    return labels.reshape(size), names


def generate_custom(ctx: SystemGenerationContext) -> SystemGenerationResult:
    phases = normalize_phase_fractions(dict(ctx.phase_fractions))
    if not phases:
        phases = {"MATRIX": 0.72, "SECONDARY": 0.18, "PRECIPITATES": 0.10}
    labels, names = _labels_from_fractions(ctx.size, int(ctx.seed), phases)

    matrix_tex = generate_grain_structure(
        size=ctx.size,
        seed=int(ctx.seed) + 11,
        mean_grain_size_px=58.0,
        grain_size_jitter=0.18,
        boundary_width_px=2,
        boundary_contrast=0.44,
    )["image"]
    secondary_tex = generate_eutectic_al_si(
        size=ctx.size,
        seed=int(ctx.seed) + 17,
        si_phase_fraction=0.35,
        morphology="branched",
    )["image"]
    precip_tex = generate_aged_aluminum_structure(
        size=ctx.size,
        seed=int(ctx.seed) + 23,
        precipitate_fraction=0.12,
        precipitate_scale_px=2.2,
    )["image"]
    liquid_tex = _liquid_texture(ctx.size, int(ctx.seed) + 29)

    image = np.zeros(ctx.size, dtype=np.float32)
    phase_masks: dict[str, np.ndarray] = {}
    for idx, phase_name in enumerate(names):
        mask = labels == idx
        phase_masks[str(phase_name)] = mask.astype(np.uint8)
        token = str(phase_name).upper()
        if token in {"L", "LIQUID"}:
            tex = liquid_tex
        elif "PRECIP" in token:
            tex = precip_tex
        elif "SECOND" in token or "INTER" in token:
            tex = secondary_tex
        else:
            tex = matrix_tex
        image[mask] = tex[mask]

    image_gray = soft_unsharp(ensure_u8(image), amount=0.36)
    visibility = build_phase_visibility_report(
        image_gray=image_gray,
        phase_masks=phase_masks,
        phase_fractions=phases,
        tolerance_pct=float(ctx.phase_fraction_tolerance_pct),
    )
    composition_effect = build_composition_effect(
        system="custom-multicomponent",
        composition_wt=ctx.composition_wt,
        mode=str(ctx.composition_sensitivity_mode),
        seed=int(ctx.seed),
        single_phase_compensation=bool(len(phase_masks) <= 1),
    )
    metadata: dict[str, Any] = {
        "system_generator_name": "system_custom",
        "resolved_stage": str(ctx.stage),
        "phase_transition_state": {
            "stage": str(ctx.stage),
            "transition_kind": "none",
            "liquid_fraction": float(phases.get("LIQUID", phases.get("L", 0.0))),
            "solid_fraction": float(max(0.0, 1.0 - float(phases.get("LIQUID", phases.get("L", 0.0)))),
            ),
            "thermal_direction": "steady",
        },
        "composition_effect": composition_effect,
        "phase_visibility_report": visibility,
        "engineering_trace": {
            "generation_mode": str(ctx.generation_mode),
            "phase_emphasis_style": str(ctx.phase_emphasis_style),
            "phase_fraction_tolerance_pct": float(ctx.phase_fraction_tolerance_pct),
            "system_generator_name": "system_custom",
        },
    }
    return SystemGenerationResult(image_gray=image_gray, phase_masks=phase_masks, metadata=metadata)
