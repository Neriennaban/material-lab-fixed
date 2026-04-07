from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from core.generator_grains import generate_grain_structure
from core.metallography_v3.realism_utils import (
    allocate_phase_masks,
    boundary_mask_from_labels,
    clamp,
    cooling_index,
    distance_to_mask,
    low_frequency_field,
    multiscale_noise,
    normalize01,
    rescale_to_u8,
    select_fraction_mask,
)
from core.metallography_v3.pure_ferrite_generator import (
    generate_pure_ferrite_micrograph,
)

from .base import (
    SystemGenerationContext,
    SystemGenerationResult,
    build_composition_effect,
    build_phase_visibility_report,
    ensure_u8,
    normalize_phase_fractions,
    soft_unsharp,
)
from core.metallography_v3.transformation_state import (
    metadata_blocks_from_transformation_state,
)
from .fe_c_textures import (
    fe_c_texture_map,
    texture_sorbite_quench,
    texture_sorbite_temper,
    texture_troostite_quench,
    texture_troostite_temper,
)

_PHASE_ALIASES: dict[str, str] = {
    "L": "LIQUID",
    "LIQUID": "LIQUID",
    "ALPHA": "FERRITE",
    "FERRITE": "FERRITE",
    "DELTA_FERRITE": "DELTA_FERRITE",
    "GAMMA": "AUSTENITE",
    "AUSTENITE": "AUSTENITE",
    "PEARLITE": "PEARLITE",
    "FE3C": "CEMENTITE",
    "CEMENTITE": "CEMENTITE",
    "CARBIDE": "CEMENTITE",
    "MARTENSITE": "MARTENSITE",
    "MARTENSITE_T": "MARTENSITE_TETRAGONAL",
    "MARTENSITE_TETRAGONAL": "MARTENSITE_TETRAGONAL",
    "MARTENSITE_C": "MARTENSITE_CUBIC",
    "MARTENSITE_CUBIC": "MARTENSITE_CUBIC",
    "TROOSTITE": "TROOSTITE",
    "SORBITE": "SORBITE",
    "BAINITE": "BAINITE",
    "LEDEBURITE": "LEDEBURITE",
}

_STAGE_DEFAULT_FRACTIONS: dict[str, dict[str, float]] = {
    "liquid": {"LIQUID": 1.0},
    "liquid_gamma": {"LIQUID": 0.62, "AUSTENITE": 0.38},
    "delta_ferrite": {"DELTA_FERRITE": 0.82, "AUSTENITE": 0.18},
    "austenite": {"AUSTENITE": 1.0},
    "ferrite": {"FERRITE": 1.0},
    "alpha_gamma": {"FERRITE": 0.55, "AUSTENITE": 0.45},
    "gamma_cementite": {"AUSTENITE": 0.72, "CEMENTITE": 0.28},
    "alpha_pearlite": {"FERRITE": 0.5, "PEARLITE": 0.5},
    "pearlite": {"PEARLITE": 1.0},
    "pearlite_cementite": {"PEARLITE": 0.82, "CEMENTITE": 0.18},
    "ledeburite": {"LEDEBURITE": 0.62, "PEARLITE": 0.28, "CEMENTITE": 0.1},
    # White cast iron variants (Fe-C 2.14-6.67 %, carbides not graphite).
    # Hypoeutectic: primary austenite dendrites (now pearlite) in ledeburite matrix.
    "white_cast_iron_hypoeutectic": {
        "PEARLITE": 0.42,
        "LEDEBURITE": 0.52,
        "CEMENTITE": 0.06,
    },
    # Eutectic: mostly ledeburite, little free phase.
    "white_cast_iron_eutectic": {"LEDEBURITE": 0.92, "CEMENTITE": 0.08},
    # Hypereutectic: primary cementite needles in ledeburite matrix.
    "white_cast_iron_hypereutectic": {
        "CEMENTITE_PRIMARY": 0.28,
        "LEDEBURITE": 0.64,
        "CEMENTITE": 0.08,
    },
    "martensite": {"MARTENSITE": 0.9, "CEMENTITE": 0.1},
    "martensite_tetragonal": {"MARTENSITE_TETRAGONAL": 0.9, "CEMENTITE": 0.1},
    "martensite_cubic": {"MARTENSITE_CUBIC": 0.94, "CEMENTITE": 0.06},
    "troostite_quench": {"TROOSTITE": 0.88, "CEMENTITE": 0.12},
    "troostite_temper": {"TROOSTITE": 0.66, "CEMENTITE": 0.2, "FERRITE": 0.14},
    "sorbite_quench": {"SORBITE": 0.84, "CEMENTITE": 0.16},
    "sorbite_temper": {"SORBITE": 0.62, "CEMENTITE": 0.22, "FERRITE": 0.16},
    "bainite": {"BAINITE": 0.82, "CEMENTITE": 0.18},
    # Upper bainite (350-550 °C): feathery packets of parallel ferrite
    # plates with Fe3C between them. Slightly coarser, more carbide content.
    "bainite_upper": {"BAINITE": 0.78, "CEMENTITE": 0.22},
    # Lower bainite (200-350 °C): needle-like ferrite with Fe3C precipitates
    # inside the laths at ~55-60°. Finer, harder, less free cementite.
    "bainite_lower": {"BAINITE": 0.85, "CEMENTITE": 0.15},
    "tempered_low": {"MARTENSITE": 0.6, "TROOSTITE": 0.2, "CEMENTITE": 0.2},
    "tempered_medium": {
        "TROOSTITE": 0.5,
        "MARTENSITE": 0.18,
        "CEMENTITE": 0.2,
        "FERRITE": 0.12,
    },
    "tempered_high": {"SORBITE": 0.42, "FERRITE": 0.4, "CEMENTITE": 0.18},
}

_TRANSITION_STAGES: set[str] = {
    "liquid_gamma",
    "delta_ferrite",
    "alpha_gamma",
    "gamma_cementite",
    "alpha_pearlite",
    "pearlite_cementite",
    "ledeburite",
    "white_cast_iron_hypoeutectic",
    "white_cast_iron_eutectic",
    "white_cast_iron_hypereutectic",
    "troostite_temper",
    "sorbite_temper",
    "bainite",
    "bainite_upper",
    "bainite_lower",
    "tempered_low",
    "tempered_medium",
    "tempered_high",
}

_SPECIALIZED_PEARLITIC_STAGES = {"alpha_pearlite", "pearlite", "pearlite_cementite"}
_SPECIALIZED_MARTENSITIC_STAGES = {
    "martensite",
    "martensite_tetragonal",
    "martensite_cubic",
    "troostite_quench",
    "troostite_temper",
    "sorbite_quench",
    "sorbite_temper",
    "bainite",
    "tempered_low",
    "tempered_medium",
    "tempered_high",
}
# New taxonomy for A0.1: white cast iron and explicit upper/lower bainite.
# These sets are consulted by `render_fe_c_unified` to dispatch to new build
# functions. In phase A0 the sets are declared but not yet wired — the
# dispatch is added in phase A1/A6 when the specialised render functions
# land. Until then, these stages fall through to `_generic_render`, which
# uses the phase templates above.
_SPECIALIZED_CAST_IRON_STAGES = {
    "white_cast_iron_hypoeutectic",
    "white_cast_iron_eutectic",
    "white_cast_iron_hypereutectic",
}
_SPECIALIZED_BAINITIC_STAGES = {"bainite_upper", "bainite_lower"}


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
    *,
    stage: str,
    phase_fractions: dict[str, float],
    composition_wt: dict[str, float] | None,
) -> bool:
    stage_name = str(stage or "").strip().lower()
    ferritic_fraction = float(
        phase_fractions.get("FERRITE", 0.0) + phase_fractions.get("DELTA_FERRITE", 0.0)
    )
    fe_pct = _composition_fraction(composition_wt, "Fe")
    c_pct = _composition_fraction(composition_wt, "C")
    si_pct = _composition_fraction(composition_wt, "Si")
    return bool(
        stage_name == "ferrite"
        and ferritic_fraction >= 0.95
        and fe_pct >= 99.8
        and c_pct <= 0.03
        and si_pct <= 0.25
    )


def _lift_small_dark_defects(
    image_gray: np.ndarray, *, max_pixels: int = 18
) -> np.ndarray:
    if ndimage is None:
        return image_gray
    arr = image_gray.astype(np.float32)
    threshold = float(np.quantile(arr, 0.08))
    mask = arr < threshold
    labels, count = ndimage.label(mask)
    if count <= 0:
        return image_gray
    local = ndimage.gaussian_filter(arr, sigma=1.1)
    out = arr.copy()
    for label in range(1, int(count) + 1):
        zone = labels == label
        if int(zone.sum()) <= int(max_pixels):
            out[zone] = 0.84 * local[zone] + 0.16 * out[zone]
    return ensure_u8(out)


def _brighten_pure_ferrite_baseline(image_gray: np.ndarray) -> np.ndarray:
    arr = image_gray.astype(np.float32)
    if ndimage is not None:
        arr = ndimage.gaussian_filter(arr, sigma=0.35)
    lo = float(np.quantile(arr, 0.02))
    hi = float(np.quantile(arr, 0.985))
    if hi > lo + 1e-6:
        arr = (arr - lo) / (hi - lo)
        arr = arr * 76.0 + 152.0
    dark_floor = float(np.quantile(arr, 0.07))
    arr += max(0.0, 138.0 - dark_floor)
    arr = np.clip(arr, 132.0, 246.0)
    bright = ensure_u8(arr)
    bright = _lift_small_dark_defects(bright, max_pixels=20)
    if ndimage is not None:
        smooth = ndimage.gaussian_filter(bright.astype(np.float32), sigma=1.0)
        bright = ensure_u8(smooth + (bright.astype(np.float32) - smooth) * 0.10)
    return bright


def _pure_ferrite_render(
    *,
    context: SystemGenerationContext,
    seed_split: dict[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], int, dict[str, Any]]:
    morph = (
        dict(context.transformation_state.get("morphology_state", {}))
        if isinstance(context.transformation_state, dict)
        else {}
    )
    prior_px = float(morph.get("prior_austenite_grain_size_px", 92.0))
    mean_eq_d_px = clamp(prior_px * 0.78, 42.0, 96.0)
    render = generate_pure_ferrite_micrograph(
        size=context.size,
        seed=seed_split["seed_topology"],
        mean_eq_d_px=float(mean_eq_d_px),
        size_sigma=0.22,
        relax_iter=1,
        boundary_width_px=2.2,
        boundary_depth=0.13,
        blur_sigma_px=0.6,
    )
    image_gray = render["image_gray"].astype(np.uint8)
    phase_masks = {"FERRITE": np.ones(context.size, dtype=np.uint8)}
    rendered_layers = ["FERRITE"]
    fragment_area = int(max(48.0, math.pi * (mean_eq_d_px * 0.5) ** 2))
    trace = {
        "family": "pure_ferrite_power_voronoi",
        **dict(render.get("metadata", {})),
    }
    # A10.3 — expose the per-grain label map so downstream palettes
    # (DIC polarised, tint etching) can colour each grain individually.
    raw_labels = render.get("labels")
    if isinstance(raw_labels, np.ndarray):
        trace["grain_labels"] = raw_labels.astype(np.int32)
    return image_gray, phase_masks, rendered_layers, fragment_area, trace


def _canon_phase_name(value: str) -> str:
    key = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    return _PHASE_ALIASES.get(key, key)


def _normalize_input_fractions(phase_fractions: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for name, value in dict(phase_fractions).items():
        phase = _canon_phase_name(str(name))
        if not phase:
            continue
        try:
            val = float(value)
        except Exception:
            continue
        if val <= 0.0:
            continue
        merged[phase] = float(merged.get(phase, 0.0) + val)
    return normalize_phase_fractions(merged)


def _stabilize_fractions(
    *,
    stage: str,
    input_fractions: dict[str, float],
    min_frac: float = 0.02,
    table_locked: bool = False,
) -> tuple[dict[str, float], bool]:
    stage_name = str(stage).strip().lower()
    defaults = normalize_phase_fractions(
        _STAGE_DEFAULT_FRACTIONS.get(stage_name, {"FERRITE": 1.0})
    )
    inp = normalize_phase_fractions(input_fractions)
    stage_coverage_pass = stage_name in _STAGE_DEFAULT_FRACTIONS

    if inp:
        blended: dict[str, float] = {}
        default_weight = 0.03 if bool(table_locked) else 0.2
        input_weight = 1.0 - default_weight
        for name, value in defaults.items():
            blended[name] = blended.get(name, 0.0) + float(value) * default_weight
        for name, value in inp.items():
            blended[name] = blended.get(name, 0.0) + float(value) * input_weight
        out = normalize_phase_fractions(blended)
    else:
        out = dict(defaults)

    significant = {k: float(v) for k, v in out.items() if float(v) >= float(min_frac)}
    if not significant:
        significant = {max(out.items(), key=lambda item: float(item[1]))[0]: 1.0}
    out = normalize_phase_fractions(significant)

    if stage_name in _TRANSITION_STAGES and len(out) < 2:
        for phase_name, phase_val in sorted(
            defaults.items(), key=lambda item: float(item[1]), reverse=True
        ):
            if phase_name not in out:
                out[phase_name] = max(float(min_frac), float(phase_val) * 0.35)
            if len(out) >= 2:
                break
        out = normalize_phase_fractions(out)
    return out, stage_coverage_pass


def _labels_from_fractions(
    *, size: tuple[int, int], seed: int, fractions: dict[str, float]
) -> tuple[np.ndarray, list[str]]:
    names = [
        name
        for name, _ in sorted(
            fractions.items(), key=lambda item: float(item[1]), reverse=True
        )
    ]
    probs = np.asarray([float(fractions[name]) for name in names], dtype=np.float64)
    probs /= float(np.sum(probs))

    rng = np.random.default_rng(int(seed))

    def _upsample_repeat(
        arr: np.ndarray, scale: int, target: tuple[int, int]
    ) -> np.ndarray:
        up = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
        return up[: target[0], : target[1]]

    min_side = int(min(size))
    downscale = 1
    if min_side >= 1600:
        downscale = 4
    elif min_side >= 900:
        downscale = 3
    elif min_side >= 500:
        downscale = 2

    if downscale > 1:
        coarse_h = max(64, int(np.ceil(size[0] / downscale)))
        coarse_w = max(64, int(np.ceil(size[1] / downscale)))
        coarse_shape = (coarse_h, coarse_w)
        low = rng.normal(0.0, 1.0, size=coarse_shape).astype(np.float32)
        mid = rng.normal(0.0, 1.0, size=coarse_shape).astype(np.float32)
        if ndimage is not None:
            low = ndimage.gaussian_filter(
                low, sigma=max(2.0, float(min(coarse_shape)) / 12.0)
            )
            mid = ndimage.gaussian_filter(
                mid, sigma=max(1.2, float(min(coarse_shape)) / 30.0)
            )
        low = _upsample_repeat(low, downscale, size)
        mid = _upsample_repeat(mid, downscale, size)
        fine = rng.normal(0.0, 1.0, size=size).astype(np.float32)
        if ndimage is not None:
            fine = ndimage.gaussian_filter(
                fine, sigma=max(0.8, float(min(size)) / 140.0)
            )
    else:
        low = rng.normal(0.0, 1.0, size=size).astype(np.float32)
        mid = rng.normal(0.0, 1.0, size=size).astype(np.float32)
        fine = rng.normal(0.0, 1.0, size=size).astype(np.float32)
        if ndimage is not None:
            low = ndimage.gaussian_filter(low, sigma=max(5.0, float(min(size)) / 16.0))
            mid = ndimage.gaussian_filter(mid, sigma=max(2.5, float(min(size)) / 45.0))
            fine = ndimage.gaussian_filter(
                fine, sigma=max(1.2, float(min(size)) / 110.0)
            )
    field = low * 0.90 + mid * 0.08 + fine * 0.02
    field = (field - float(np.mean(field))) / float(np.std(field) + 1e-6)

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


def _coarsen_phase_labels(labels: np.ndarray, *, min_fragment_area: int) -> np.ndarray:
    if ndimage is None:
        return labels
    out = labels.astype(np.int32, copy=True)
    area_thr = max(1, int(min_fragment_area))
    unique_ids = [int(v) for v in np.unique(out)]
    if len(unique_ids) <= 1:
        return out

    for _ in range(3):
        small_mask_total = np.zeros_like(out, dtype=bool)
        for phase_id in unique_ids:
            phase_mask = out == int(phase_id)
            if not np.any(phase_mask):
                continue
            cc, num = ndimage.label(phase_mask.astype(np.uint8))
            if int(num) <= 1:
                continue
            counts = np.bincount(cc.ravel())
            if counts.size <= 1:
                continue
            largest_comp = int(np.argmax(counts[1:]) + 1)
            comp_ids = np.arange(counts.size, dtype=np.int32)
            tiny_ids = comp_ids[
                (comp_ids > 0) & (comp_ids != largest_comp) & (counts < area_thr)
            ]
            if tiny_ids.size == 0:
                continue
            small_mask_total |= np.isin(cc, tiny_ids, assume_unique=False)

        if not np.any(small_mask_total):
            break
        stable_mask = ~small_mask_total
        if not np.any(stable_mask):
            break
        _, nearest_idx = ndimage.distance_transform_edt(
            small_mask_total, return_indices=True
        )
        nearest_labels = out[tuple(nearest_idx)]
        out[small_mask_total] = nearest_labels[small_mask_total]
        unique_ids = [int(v) for v in np.unique(out)]
        if len(unique_ids) <= 1:
            break
    return out


def _phase_boundaries(labels: np.ndarray) -> np.ndarray:
    borders = np.zeros_like(labels, dtype=np.uint8)
    borders[:-1, :] |= (labels[:-1, :] != labels[1:, :]).astype(np.uint8)
    borders[:, :-1] |= (labels[:, :-1] != labels[:, 1:]).astype(np.uint8)
    if ndimage is not None:
        borders = ndimage.binary_dilation(borders > 0, iterations=1).astype(np.uint8)
    return borders


def _distance_from_boundaries(mask: np.ndarray, *, max_steps: int = 48) -> np.ndarray:
    if ndimage is not None:
        return distance_to_mask(mask)

    boundary = mask.astype(bool, copy=False)
    dist = np.full(mask.shape, float(max_steps), dtype=np.float32)
    dist[boundary] = 0.0
    frontier = boundary.copy()
    visited = boundary.copy()

    for step in range(1, max(1, int(max_steps)) + 1):
        grown = np.zeros_like(frontier, dtype=bool)
        grown[1:, :] |= frontier[:-1, :]
        grown[:-1, :] |= frontier[1:, :]
        grown[:, 1:] |= frontier[:, :-1]
        grown[:, :-1] |= frontier[:, 1:]
        frontier = grown & ~visited
        if not np.any(frontier):
            break
        dist[frontier] = float(step)
        visited |= frontier
    return dist


def _suppress_small_inclusions(image_gray: np.ndarray) -> np.ndarray:
    if ndimage is None:
        return image_gray
    arr = image_gray.astype(np.float32)
    h, w = arr.shape
    area = max(1, int(h * w))
    max_comp_area = max(8, min(72, area // 3500))
    local_base = ndimage.gaussian_filter(arr, sigma=1.3)
    local_base_wide = ndimage.gaussian_filter(arr, sigma=2.2)
    dark_thr = float(np.quantile(arr, 0.03))
    bright_thr = float(np.quantile(arr, 0.97))

    def _replace_tiny(mask: np.ndarray, area_limit: int) -> None:
        if not np.any(mask):
            return
        labels, num = ndimage.label(mask.astype(np.uint8))
        if int(num) <= 0:
            return
        counts = np.bincount(labels.ravel())
        if counts.size <= 1:
            return
        comp_ids = np.arange(counts.size, dtype=np.int32)
        tiny_ids = comp_ids[(comp_ids > 0) & (counts <= int(area_limit))]
        if tiny_ids.size == 0:
            return
        tiny_mask = np.isin(labels, tiny_ids, assume_unique=False)
        arr[tiny_mask] = (
            local_base[tiny_mask] * 0.72 + local_base_wide[tiny_mask] * 0.28
        )

    local_contrast = arr - local_base
    dark_mask = (arr <= dark_thr) & (local_contrast <= -10.0)
    bright_mask = (arr >= bright_thr) & (local_contrast >= 10.0)
    _replace_tiny(dark_mask, max_comp_area)
    _replace_tiny(bright_mask, max_comp_area)

    dark_thr2 = float(np.quantile(arr, 0.02))
    bright_thr2 = float(np.quantile(arr, 0.98))
    local_contrast2 = arr - local_base_wide
    dark_mask2 = (arr <= dark_thr2) & (local_contrast2 <= -8.0)
    bright_mask2 = (arr >= bright_thr2) & (local_contrast2 >= 8.0)
    _replace_tiny(dark_mask2, int(max_comp_area * 1.6))
    _replace_tiny(bright_mask2, int(max_comp_area * 1.6))
    return ensure_u8(arr)


def _texture_for_phase(
    *,
    phase_name: str,
    stage_name: str,
    size: tuple[int, int],
    seed: int,
    textures: dict[str, Any],
) -> np.ndarray:
    phase = _canon_phase_name(phase_name)
    stage = str(stage_name).strip().lower()
    if phase == "TROOSTITE":
        return (
            texture_troostite_quench(size=size, seed=seed)
            if stage == "troostite_quench"
            else texture_troostite_temper(size=size, seed=seed)
        )
    if phase == "SORBITE":
        return (
            texture_sorbite_quench(size=size, seed=seed)
            if stage == "sorbite_quench"
            else texture_sorbite_temper(size=size, seed=seed)
        )
    fn = textures.get(phase) or textures.get("FERRITE")
    return ensure_u8(fn(size, int(seed)))


def _grain_map(
    size: tuple[int, int], seed: int, mean_grain_size_px: float, elongation: float = 1.0
) -> dict[str, Any]:
    return generate_grain_structure(
        size=size,
        seed=seed,
        mean_grain_size_px=max(18.0, float(mean_grain_size_px)),
        grain_size_jitter=0.22,
        boundary_width_px=1,
        boundary_contrast=0.0,
        elongation=max(0.85, float(elongation)),
    )


def _phase_field_from_labels(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = labels.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(int(seed))
    count = int(labels.max()) + 1
    theta = rng.uniform(0.0, math.pi, size=count).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=count).astype(np.float32)
    spacing = rng.normal(1.0, 0.15, size=count).astype(np.float32)
    proj = xx * np.cos(theta[labels]) + yy * np.sin(theta[labels])
    return (
        proj,
        theta[labels],
        phase[labels],
        spacing[labels],
        rng.normal(0.0, 1.0, size=labels.shape).astype(np.float32),
    )


def _pearlite_image(
    *,
    labels: np.ndarray,
    seed: int,
    lamella_period_px: float,
    colony_size_px: float,
    ferrite_tone: float = 184.0,
    cementite_tone: float = 82.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = labels.shape
    proj, theta_map, phase_map, spacing_factor, _ = _phase_field_from_labels(
        labels, seed + 19
    )
    curvature = (
        (
            multiscale_noise(
                size=(h, w), seed=seed + 23, scales=((26.0, 0.7), (9.0, 0.3))
            )
            - 0.5
        )
        * lamella_period_px
        * 1.2
    )
    local_spacing = np.clip(lamella_period_px * spacing_factor, 2.0, 12.0)
    wave = np.sin((2.0 * math.pi / local_spacing) * (proj + curvature) + phase_map)
    lamella_soft = normalize01(wave)
    cementite = wave > 0.36
    ferritic_lamellae = wave < -0.70

    image = np.full((h, w), 105.0, dtype=np.float32)
    image[ferritic_lamellae] = ferrite_tone
    image[cementite] = cementite_tone
    image += (
        multiscale_noise(
            size=(h, w), seed=seed + 31, scales=((22.0, 0.65), (5.0, 0.35))
        )
        - 0.5
    ) * 11.0
    boundaries = boundary_mask_from_labels(labels, width=2)
    image[boundaries] -= 10.0
    if ndimage is not None:
        image = ndimage.gaussian_filter(image, sigma=0.55)
    return np.clip(image, 0.0, 255.0).astype(np.uint8), {
        "colony_count": int(labels.max()) + 1,
        "colony_size_px": float(colony_size_px),
        "interlamellar_spacing_px": float(lamella_period_px),
        "lamella_fill_fraction": float(cementite.mean()),
    }


def _martensite_style(c_wt: float) -> str:
    if c_wt < 0.25:
        return "lath_dominant"
    if c_wt < 0.55:
        return "mixed_lath_plate"
    return "plate_dominant"


def _structural_rank(name: str) -> int:
    key = _canon_phase_name(name)
    order = {
        "MARTENSITE": 10,
        "MARTENSITE_TETRAGONAL": 10,
        "MARTENSITE_CUBIC": 9,
        "BAINITE": 8,
        "TROOSTITE": 7,
        "SORBITE": 6,
        "PEARLITE": 5,
        "FERRITE": 4,
        "AUSTENITE": 3,
        "CEMENTITE": 2,
    }
    return int(order.get(key, 1))


def _dominant_structure(phase_fractions: dict[str, float]) -> str:
    if not phase_fractions:
        return "FERRITE"
    return max(
        phase_fractions.items(),
        key=lambda item: (float(item[1]), _structural_rank(item[0])),
    )[0]


def _build_pearlitic_render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    size = context.size
    c_wt = float(context.composition_wt.get("C", 0.0))
    cool_idx = cooling_index(getattr(context.processing, "cooling_mode", "equilibrium"))
    morphology_state = dict(context.transformation_state.get("morphology_state", {}))
    grain_scale = clamp(
        float(
            morphology_state.get(
                "prior_austenite_grain_size_px",
                126.0
                - 52.0 * cool_idx
                + 22.0 * float(context.effect_vector.get("grain_size_factor", 0.0)),
            )
        ),
        42.0,
        168.0,
    )
    grain = _grain_map(
        size=size, seed=seed_split["seed_topology"], mean_grain_size_px=grain_scale
    )
    labels = grain["labels"]
    boundaries = boundary_mask_from_labels(labels, width=2)
    dist = _distance_from_boundaries(
        boundaries, max_steps=max(24, int(round(grain_scale * 0.35)))
    )
    boundary_scale = clamp(
        5.8 - 2.1 * cool_idx + max(0.0, c_wt - 0.77) * 2.4,
        1.6,
        8.5,
    )
    boundary_pref = np.exp(-((dist / boundary_scale) ** 1.35)).astype(np.float32)
    boundary_pref = normalize01(boundary_pref)
    boundary_rank = normalize01(float(np.max(dist)) - dist.astype(np.float32))
    low = low_frequency_field(size, seed_split["seed_noise"], sigma=26.0)
    noise = multiscale_noise(
        size=size,
        seed=seed_split["seed_particles"],
        scales=((20.0, 0.62), (8.0, 0.25), (2.0, 0.13)),
    )

    lamella_period = clamp(
        float(
            morphology_state.get(
                "interlamellar_spacing_px", 8.7 - 4.5 * cool_idx - 0.8 * min(c_wt, 1.2)
            )
        ),
        2.0,
        9.6,
    )
    colony_size_px = clamp(
        float(
            morphology_state.get(
                "colony_size_px", grain_scale * (0.82 + 0.18 * (1.0 - cool_idx))
            )
        ),
        28.0,
        164.0,
    )
    pearlite_image, pearlite_meta = _pearlite_image(
        labels=labels,
        seed=seed_split["seed_lamella"],
        lamella_period_px=lamella_period,
        colony_size_px=colony_size_px,
    )
    ferrite_grain = _grain_map(
        size=size,
        seed=seed_split["seed_boundary"],
        mean_grain_size_px=grain_scale * 0.9,
    )
    # Проэвтектоидный феррит: светлый, с тёмными канавками по границам.
    ferrite_img = ferrite_grain["image"].astype(np.float32) * 0.12 + 205.0
    ferrite_img[boundaries] -= 40.0
    ferrite_img += (noise - 0.5) * 8.0
    ferrite_img = np.clip(ferrite_img, 0.0, 255.0).astype(np.uint8)

    # Проэвтектоидный цементит для учебного режима делаем тёмным и читаемым.
    cementite_img = 70.0 + (noise - 0.5) * 12.0 - boundary_pref * 18.0
    cementite_img = np.clip(cementite_img, 0.0, 255.0).astype(np.uint8)

    proeutectoid = (
        "FERRITE"
        if stage == "alpha_pearlite"
        else ("CEMENTITE" if stage == "pearlite_cementite" else "")
    )
    proeutectoid_frac = float(phase_fractions.get(proeutectoid, 0.0)) if proeutectoid else 0.0
    if stage == "alpha_pearlite":
        ferr_field = normalize01(boundary_rank * 0.95 + low * 0.03 + noise * 0.02)
        ferr_mask = select_fraction_mask(
            field=ferr_field,
            available=np.ones(size, dtype=bool),
            fraction_total=proeutectoid_frac,
        )
        phase_masks = {
            "FERRITE": ferr_mask.astype(np.uint8),
            "PEARLITE": (~ferr_mask).astype(np.uint8),
        }
    elif stage == "pearlite_cementite":
        cem_field = normalize01(boundary_rank * 0.97 + low * 0.02 + noise * 0.01)
        cem_mask = select_fraction_mask(
            field=cem_field,
            available=np.ones(size, dtype=bool),
            fraction_total=proeutectoid_frac,
        )
        phase_masks = {
            "CEMENTITE": cem_mask.astype(np.uint8),
            "PEARLITE": (~cem_mask).astype(np.uint8),
        }
    else:
        ordered_fields: list[tuple[str, np.ndarray]] = []
        if (
            "CEMENTITE" in phase_fractions
            and float(phase_fractions.get("CEMENTITE", 0.0)) > 0.0
        ):
            cem_field = normalize01(boundary_rank * 0.84 + noise * 0.16)
            ordered_fields.append(("CEMENTITE", cem_field))
        ordered_fields.append(
            ("PEARLITE", normalize01(low * 0.55 + (1.0 - boundary_pref) * 0.45))
        )

        dominant = _dominant_structure(phase_fractions)
        phase_masks = allocate_phase_masks(
            size=size,
            phase_fractions=phase_fractions,
            ordered_fields=ordered_fields,
            remainder_name=dominant,
        )

    canvas = pearlite_image.astype(np.float32)
    if "PEARLITE" in phase_masks:
        mask = phase_masks["PEARLITE"] > 0
        canvas[mask] = pearlite_image[mask].astype(np.float32)
    if "FERRITE" in phase_masks:
        mask = phase_masks["FERRITE"] > 0
        canvas[mask] = ferrite_img[mask].astype(np.float32)
    if "CEMENTITE" in phase_masks:
        mask = phase_masks["CEMENTITE"] > 0
        canvas[mask] = cementite_img[mask].astype(np.float32)

    canvas[boundaries] -= 6.0
    canvas += (
        multiscale_noise(
            size=size,
            seed=seed_split["seed_noise"] + 5,
            scales=((14.0, 0.65), (3.4, 0.35)),
        )
        - 0.5
    ) * 4.0
    image_gray = soft_unsharp(
        _suppress_small_inclusions(rescale_to_u8(canvas, lo=25.0, hi=245.0)),
        amount=0.38,
    )

    boundary_bias = 0.0
    if (
        proeutectoid
        and proeutectoid in phase_masks
        and np.any(phase_masks[proeutectoid] > 0)
    ):
        boundary_bias = float(boundary_pref[phase_masks[proeutectoid] > 0].mean())
    elif "CEMENTITE" in phase_masks and np.any(phase_masks["CEMENTITE"] > 0):
        boundary_bias = float(boundary_pref[phase_masks["CEMENTITE"] > 0].mean())

    trace = {
        "family": "pearlitic",
        "prior_austenite_grain_count": int(labels.max()) + 1,
        "prior_austenite_grain_size_px": float(grain_scale),
        "colony_size_px": float(pearlite_meta["colony_size_px"]),
        "interlamellar_spacing_px": float(pearlite_meta["interlamellar_spacing_px"]),
        "cooling_index": float(cool_idx),
        "proeutectoid_phase": str(proeutectoid),
        "proeutectoid_fraction_target": float(proeutectoid_frac),
        "boundary_phase_bias": float(boundary_bias),
        "colony_anisotropy": "grain_oriented_lamellae",
    }
    return image_gray, phase_masks, trace


def _build_martensitic_render(
    *,
    context: SystemGenerationContext,
    stage: str,
    phase_fractions: dict[str, float],
    seed_split: dict[str, int],
    retained_austenite_used: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    size = context.size
    c_wt = float(context.composition_wt.get("C", 0.0))
    morphology_state = dict(context.transformation_state.get("morphology_state", {}))
    precipitation_state = dict(
        context.transformation_state.get("precipitation_state", {})
    )
    style = str(morphology_state.get("martensite_style", _martensite_style(c_wt)))
    cool_idx = cooling_index(getattr(context.processing, "cooling_mode", "equilibrium"))

    prior_grain_scale = clamp(
        float(
            morphology_state.get(
                "prior_austenite_grain_size_px",
                92.0 + 12.0 * max(0.0, c_wt - 0.4) - 28.0 * cool_idx,
            )
        ),
        34.0,
        120.0,
    )
    packet_size_px = clamp(
        float(
            morphology_state.get(
                "packet_size_px",
                52.0
                + (
                    10.0
                    if style == "plate_dominant"
                    else (-9.0 if style == "lath_dominant" else 0.0)
                )
                - 10.0 * cool_idx,
            )
        ),
        18.0,
        88.0,
    )
    elong = (
        1.18
        if style == "lath_dominant"
        else (1.42 if style == "mixed_lath_plate" else 1.72)
    )
    prior = _grain_map(
        size=size,
        seed=seed_split["seed_topology"],
        mean_grain_size_px=prior_grain_scale,
    )
    packets = _grain_map(
        size=size,
        seed=seed_split["seed_boundary"],
        mean_grain_size_px=packet_size_px,
        elongation=elong,
    )
    prior_bound = boundary_mask_from_labels(prior["labels"], width=2)
    packet_bound = boundary_mask_from_labels(packets["labels"], width=2)
    dist = distance_to_mask(packet_bound | prior_bound)
    boundary_pref = normalize01(
        np.exp(-dist / clamp(3.2 + 0.9 * (1.0 - cool_idx), 1.4, 5.0)).astype(np.float32)
    )

    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(seed_split["seed_lamella"])
    packet_count = int(packets["labels"].max()) + 1
    theta = rng.uniform(0.0, math.pi, size=packet_count).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=packet_count).astype(np.float32)
    spacing_base = (
        3.6
        if style == "lath_dominant"
        else (5.6 if style == "mixed_lath_plate" else 8.4)
    )
    spacing = np.clip(
        rng.normal(
            spacing_base, 0.55 if style == "lath_dominant" else 0.95, size=packet_count
        ),
        2.0,
        14.0,
    ).astype(np.float32)
    proj = xx * np.cos(theta[packets["labels"]]) + yy * np.sin(theta[packets["labels"]])
    curvature = (
        (
            multiscale_noise(
                size=size,
                seed=seed_split["seed_noise"],
                scales=((18.0, 0.6), (5.0, 0.4)),
            )
            - 0.5
        )
        * spacing_base
        * 0.7
    )
    band = np.sin(
        (2.0 * math.pi / spacing[packets["labels"]]) * (proj + curvature)
        + phase[packets["labels"]]
    )
    lath_energy = normalize01(np.abs(band))
    edge_energy = normalize01(1.0 - np.abs(band))
    packet_variation = rng.normal(0.0, 1.0, size=packet_count).astype(np.float32)
    packet_variation = packet_variation[packets["labels"]]
    packet_variation = normalize01(packet_variation) - 0.5

    recovery = float(
        precipitation_state.get(
            "recovery_level",
            {
                "martensite": 0.08,
                "martensite_tetragonal": 0.04,
                "martensite_cubic": 0.12,
                "troostite_quench": 0.32,
                "sorbite_quench": 0.46,
                "bainite": 0.42,
                "tempered_low": 0.36,
                "troostite_temper": 0.56,
                "tempered_medium": 0.62,
                "sorbite_temper": 0.78,
                "tempered_high": 0.88,
            }.get(stage, 0.28),
        )
    )
    carbide_scale_px = clamp(
        float(
            precipitation_state.get(
                "carbide_scale_px", 1.2 + 2.3 * recovery + max(0.0, c_wt - 0.3)
            )
        ),
        1.0,
        5.8,
    )
    lath_contrast = 30.0 * (1.0 - 0.55 * recovery)
    matrix = (
        134.0
        + packet_variation * 16.0
        + (
            multiscale_noise(
                size=size,
                seed=seed_split["seed_particles"],
                scales=((22.0, 0.68), (4.0, 0.32)),
            )
            - 0.5
        )
        * 12.0
    )
    image = (
        matrix
        + band * lath_contrast
        - edge_energy * (12.0 + 16.0 * min(1.0, recovery + 0.1))
    )
    image[packet_bound] -= 13.0
    image[prior_bound] -= 8.0

    recovered_grain = _grain_map(
        size=size,
        seed=seed_split["seed_particles"] + 101,
        mean_grain_size_px=clamp(packet_size_px * 1.18, 22.0, 96.0),
    )
    recovered_img = recovered_grain["image"].astype(np.float32) * 0.18 + 154.0
    recovered_img[recovered_grain["boundaries"]] -= 10.0
    image = image * (1.0 - 0.45 * recovery) + recovered_img * (0.45 * recovery)

    dominant = _dominant_structure(phase_fractions)
    ra_field = normalize01(
        boundary_pref * 0.72
        + multiscale_noise(
            size=size,
            seed=seed_split["seed_noise"] + 11,
            scales=((9.0, 0.5), (2.0, 0.5)),
        )
        * 0.28
    )
    carbide_field = normalize01(
        edge_energy * 0.52
        + boundary_pref * 0.22
        + multiscale_noise(
            size=size,
            seed=seed_split["seed_noise"] + 27,
            scales=((carbide_scale_px * 2.4, 0.35), (carbide_scale_px, 0.65)),
        )
        * 0.26
    )
    ferrite_field = normalize01(
        low_frequency_field(
            size=size, seed=seed_split["seed_particles"] + 33, sigma=24.0
        )
        * 0.74
        + (1.0 - lath_energy) * 0.26
    )
    troostite_field = normalize01(
        edge_energy * 0.45
        + multiscale_noise(
            size=size,
            seed=seed_split["seed_particles"] + 61,
            scales=((6.0, 0.55), (1.6, 0.45)),
        )
        * 0.55
    )
    sorbite_field = normalize01(
        ferrite_field * 0.62
        + multiscale_noise(
            size=size,
            seed=seed_split["seed_particles"] + 81,
            scales=((10.0, 0.4), (2.5, 0.6)),
        )
        * 0.38
    )
    bainite_field = normalize01(
        (band > 0).astype(np.float32) * 0.55
        + edge_energy * 0.20
        + multiscale_noise(
            size=size,
            seed=seed_split["seed_particles"] + 93,
            scales=((12.0, 0.55), (3.2, 0.45)),
        )
        * 0.25
    )
    mart_field = normalize01(
        lath_energy * 0.70 + (1.0 - recovery) * 0.20 + packet_variation * 0.10
    )

    ordered_fields: list[tuple[str, np.ndarray]] = []
    if (
        "AUSTENITE" in phase_fractions
        and float(phase_fractions.get("AUSTENITE", 0.0)) > 0.0
    ):
        ordered_fields.append(("AUSTENITE", ra_field))
    if (
        "CEMENTITE" in phase_fractions
        and float(phase_fractions.get("CEMENTITE", 0.0)) > 0.0
    ):
        ordered_fields.append(("CEMENTITE", carbide_field))
    if (
        "FERRITE" in phase_fractions
        and float(phase_fractions.get("FERRITE", 0.0)) > 0.0
    ):
        ordered_fields.append(("FERRITE", ferrite_field))
    if "TROOSTITE" in phase_fractions and dominant != "TROOSTITE":
        ordered_fields.append(("TROOSTITE", troostite_field))
    if "SORBITE" in phase_fractions and dominant != "SORBITE":
        ordered_fields.append(("SORBITE", sorbite_field))
    if "BAINITE" in phase_fractions and dominant != "BAINITE":
        ordered_fields.append(("BAINITE", bainite_field))
    for mart_name in ("MARTENSITE_TETRAGONAL", "MARTENSITE_CUBIC", "MARTENSITE"):
        if mart_name in phase_fractions and dominant != mart_name:
            ordered_fields.append((mart_name, mart_field))
    if dominant not in {name for name, _ in ordered_fields}:
        ordered_fields.append(
            (
                dominant,
                mart_field
                if dominant.startswith("MARTENSITE")
                else (
                    bainite_field
                    if dominant == "BAINITE"
                    else (
                        troostite_field
                        if dominant == "TROOSTITE"
                        else (sorbite_field if dominant == "SORBITE" else ferrite_field)
                    )
                ),
            )
        )

    phase_masks = allocate_phase_masks(
        size=size,
        phase_fractions=phase_fractions,
        ordered_fields=ordered_fields,
        remainder_name=dominant,
    )

    if "CEMENTITE" in phase_masks and np.any(phase_masks["CEMENTITE"] > 0):
        image[phase_masks["CEMENTITE"] > 0] -= 22.0
    if "AUSTENITE" in phase_masks and np.any(phase_masks["AUSTENITE"] > 0):
        image[phase_masks["AUSTENITE"] > 0] += 12.0
    if "FERRITE" in phase_masks and np.any(phase_masks["FERRITE"] > 0):
        image[phase_masks["FERRITE"] > 0] = (
            image[phase_masks["FERRITE"] > 0] * 0.68 + 164.0 * 0.32
        )
    if "SORBITE" in phase_masks and np.any(phase_masks["SORBITE"] > 0):
        image[phase_masks["SORBITE"] > 0] = (
            image[phase_masks["SORBITE"] > 0] * 0.72 + 150.0 * 0.28
        )
    if "TROOSTITE" in phase_masks and np.any(phase_masks["TROOSTITE"] > 0):
        image[phase_masks["TROOSTITE"] > 0] = (
            image[phase_masks["TROOSTITE"] > 0] * 0.74 + 142.0 * 0.26
        )
    if "BAINITE" in phase_masks and np.any(phase_masks["BAINITE"] > 0):
        image[phase_masks["BAINITE"] > 0] = (
            image[phase_masks["BAINITE"] > 0] * 0.76 + 136.0 * 0.24
        )

    image += (
        multiscale_noise(
            size=size,
            seed=seed_split["seed_noise"] + 3,
            scales=((10.0, 0.55), (2.6, 0.45)),
        )
        - 0.5
    ) * 6.0
    if ndimage is not None:
        image = ndimage.gaussian_filter(image, sigma=0.4 + 0.45 * recovery)
    image_gray = soft_unsharp(
        _suppress_small_inclusions(rescale_to_u8(image, lo=35.0, hi=210.0)),
        amount=max(0.16, 0.42 - 0.18 * recovery),
    )

    ra_bias = (
        float(boundary_pref[phase_masks["AUSTENITE"] > 0].mean())
        if ("AUSTENITE" in phase_masks and np.any(phase_masks["AUSTENITE"] > 0))
        else 0.0
    )
    trace = {
        "family": "martensitic_family",
        "martensite_style": str(style),
        "prior_austenite_grain_size_px": float(prior_grain_scale),
        "packet_size_px": float(packet_size_px),
        "band_spacing_px": float(spacing_base),
        "retained_austenite_distribution": (
            "boundary_films" if ra_bias > 0.5 else "mixed_films_islands"
        ),
        "retained_austenite_boundary_bias": float(ra_bias),
        "carbide_scale_px": float(carbide_scale_px),
        "temper_recovery_level": float(recovery),
        "cooling_index": float(cool_idx),
        "retained_austenite_used": float(retained_austenite_used),
    }
    return image_gray, phase_masks, trace


def _generic_render(
    *,
    context: SystemGenerationContext,
    stage: str,
    normalized_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], int]:
    labels, phase_names = _labels_from_fractions(
        size=context.size,
        seed=seed_split["seed_topology"],
        fractions=normalized_fractions,
    )
    h, w = context.size
    fragment_area = max(140, min(22000, int((h * w) // 55)))
    labels = _coarsen_phase_labels(labels, min_fragment_area=fragment_area)
    textures = fe_c_texture_map()

    phase_masks: dict[str, np.ndarray] = {}
    rendered_layers: list[str] = []
    canvas = np.zeros(context.size, dtype=np.float32)
    for idx, phase_name in enumerate(phase_names):
        mask = labels == idx
        phase_masks[str(phase_name)] = mask.astype(np.uint8)
        rendered_layers.append(str(phase_name))
        texture = _texture_for_phase(
            phase_name=phase_name,
            stage_name=stage,
            size=context.size,
            seed=seed_split["seed_particles"] + idx * 37,
            textures=textures,
        )
        canvas[mask] = texture[mask].astype(np.float32)

    boundaries = _phase_boundaries(labels)
    canvas[boundaries > 0] = np.clip(canvas[boundaries > 0] - 7.5, 0.0, 255.0)
    noise = (
        np.random.default_rng(seed_split["seed_noise"])
        .normal(0.0, 1.0, size=context.size)
        .astype(np.float32)
    )
    if ndimage is not None:
        noise = ndimage.gaussian_filter(noise, sigma=1.0)
    canvas += noise * 1.1
    image_gray = soft_unsharp(
        _suppress_small_inclusions(ensure_u8(canvas)), amount=0.44
    )
    return image_gray, phase_masks, rendered_layers, int(fragment_area)


def render_fe_c_unified(context: SystemGenerationContext) -> SystemGenerationResult:
    stage = str(context.stage or "ferrite").strip().lower()
    phase_fraction_source = str(
        getattr(context, "phase_fraction_source", "default_formula")
        or "default_formula"
    )
    phase_calibration_mode = str(
        getattr(context, "phase_calibration_mode", "default_formula")
        or "default_formula"
    )
    table_locked = bool(
        phase_fraction_source == "table_interpolated"
        or phase_calibration_mode == "table_interpolated"
    )
    input_phase_fractions = _normalize_input_fractions(dict(context.phase_fractions))
    normalized_fractions, stage_coverage_pass = _stabilize_fractions(
        stage=stage,
        input_fractions=input_phase_fractions,
        min_frac=0.02,
        table_locked=table_locked,
    )
    pure_iron_like = _is_pure_iron_like(
        stage=stage,
        phase_fractions=normalized_fractions,
        composition_wt=context.composition_wt,
    )

    qsum = dict(context.quench_summary or {})
    thermal_summary = dict(context.thermal_summary or {})
    op_summary = (
        thermal_summary.get("operation_inference", {})
        if isinstance(thermal_summary, dict)
        else {}
    )
    if not isinstance(op_summary, dict):
        op_summary = {}
    quench_effect_applied = bool(
        qsum.get("effect_applied", bool(op_summary.get("has_quench", False)))
    )
    medium_code = (
        str(qsum.get("medium_code_resolved", qsum.get("medium_code", "")))
        .strip()
        .lower()
    )
    temper_shift_applied = (
        dict(qsum.get("temper_shift_c", {}))
        if isinstance(qsum.get("temper_shift_c", {}), dict)
        else {}
    )
    as_quenched = (
        dict(qsum.get("as_quenched_prediction", {}))
        if isinstance(qsum.get("as_quenched_prediction", {}), dict)
        else {}
    )
    retained_austenite_used = float(
        as_quenched.get(
            "retained_austenite_fraction_est",
            qsum.get("retained_austenite_est_pct", 0.0) / 100.0,
        )
    )
    retained_austenite_used = float(max(0.0, min(0.7, retained_austenite_used)))

    medium_influence_applied = False
    if quench_effect_applied and stage in {
        "martensite",
        "martensite_tetragonal",
        "martensite_cubic",
        "troostite_quench",
        "sorbite_quench",
    }:
        if retained_austenite_used > 0.0 and "AUSTENITE" not in normalized_fractions:
            carrier_phase = (
                "MARTENSITE_TETRAGONAL"
                if "MARTENSITE_TETRAGONAL" in normalized_fractions
                else (
                    "MARTENSITE_CUBIC"
                    if "MARTENSITE_CUBIC" in normalized_fractions
                    else (
                        "MARTENSITE"
                        if "MARTENSITE" in normalized_fractions
                        else (
                            "TROOSTITE"
                            if "TROOSTITE" in normalized_fractions
                            else "SORBITE"
                        )
                    )
                )
            )
            ra = min(0.28, retained_austenite_used * 0.9)
            normalized_fractions["AUSTENITE"] = ra
            normalized_fractions[carrier_phase] = max(
                0.0, float(normalized_fractions.get(carrier_phase, 0.0)) - ra
            )
            normalized_fractions = normalize_phase_fractions(normalized_fractions)
            medium_influence_applied = True

    if stage in {"sorbite_temper", "tempered_high"}:
        mart_keys = [
            k for k in normalized_fractions.keys() if k.startswith("MARTENSITE")
        ]
        mart_sum = sum(float(normalized_fractions.get(k, 0.0)) for k in mart_keys)
        if mart_sum > 0.03:
            for key in mart_keys:
                normalized_fractions[key] = (
                    float(normalized_fractions.get(key, 0.0)) * 0.15
                )
            normalized_fractions["SORBITE"] = (
                float(normalized_fractions.get("SORBITE", 0.0)) + mart_sum * 0.65
            )
            normalized_fractions["FERRITE"] = (
                float(normalized_fractions.get("FERRITE", 0.0)) + mart_sum * 0.2
            )
            normalized_fractions = normalize_phase_fractions(normalized_fractions)
            medium_influence_applied = True

    seed_split = {
        "seed_topology": int(context.seed) + 1001,
        "seed_boundary": int(context.seed) + 1003,
        "seed_particles": int(context.seed) + 1007,
        "seed_lamella": int(context.seed) + 1013,
        "seed_noise": int(context.seed) + 1021,
    }

    rendered_layers: list[str] = list(normalized_fractions.keys())
    morphology_trace: dict[str, Any] = {"family": "generic"}
    fragment_area = max(1, int(context.size[0] * context.size[1] // 55))
    if pure_iron_like and stage == "ferrite":
        image_gray, phase_masks, rendered_layers, fragment_area, morphology_trace = (
            _pure_ferrite_render(
                context=context,
                seed_split=seed_split,
            )
        )
    elif stage in _SPECIALIZED_PEARLITIC_STAGES:
        image_gray, phase_masks, morphology_trace = _build_pearlitic_render(
            context=context,
            stage=stage,
            phase_fractions=normalized_fractions,
            seed_split=seed_split,
        )
        rendered_layers = sorted(list(phase_masks.keys()))
        fragment_area = int(
            max(48, morphology_trace.get("colony_size_px", 64.0) ** 2 * 0.14)
        )
    elif stage in _SPECIALIZED_MARTENSITIC_STAGES:
        image_gray, phase_masks, morphology_trace = _build_martensitic_render(
            context=context,
            stage=stage,
            phase_fractions=normalized_fractions,
            seed_split=seed_split,
            retained_austenite_used=retained_austenite_used,
        )
        rendered_layers = sorted(list(phase_masks.keys()))
        fragment_area = int(
            max(36, morphology_trace.get("packet_size_px", 40.0) ** 2 * 0.18)
        )
    else:
        image_gray, phase_masks, rendered_layers, fragment_area = _generic_render(
            context=context,
            stage=stage,
            normalized_fractions=normalized_fractions,
            seed_split=seed_split,
        )

    if (
        pure_iron_like
        and morphology_trace.get("family") != "pure_ferrite_power_voronoi"
    ):
        image_gray = _brighten_pure_ferrite_baseline(image_gray)

    visibility = build_phase_visibility_report(
        image_gray=image_gray,
        phase_masks=phase_masks,
        phase_fractions=normalized_fractions,
        tolerance_pct=float(context.phase_fraction_tolerance_pct),
    )
    composition_effect = build_composition_effect(
        system="fe-c",
        composition_wt=context.composition_wt,
        mode=str(context.composition_sensitivity_mode),
        seed=int(context.seed),
        single_phase_compensation=bool(len(phase_masks) <= 1),
    )
    liquid_fraction = float(normalized_fractions.get("LIQUID", 0.0))

    metadata: dict[str, Any] = {
        "system_generator_name": "system_fe_c",
        "resolved_stage": str(stage),
        "phase_transition_state": {
            "stage": str(stage),
            "transition_kind": (
                "none"
                if liquid_fraction <= 0.0
                else ("melting" if liquid_fraction >= 0.5 else "crystallization")
            ),
            "liquid_fraction": float(liquid_fraction),
            "solid_fraction": float(max(0.0, 1.0 - liquid_fraction)),
            "thermal_direction": "steady",
        },
        "composition_effect": composition_effect,
        "phase_visibility_report": visibility,
        **metadata_blocks_from_transformation_state(context.transformation_state),
        "engineering_trace": {
            "generation_mode": str(context.generation_mode),
            "phase_emphasis_style": str(context.phase_emphasis_style),
            "phase_fraction_tolerance_pct": float(context.phase_fraction_tolerance_pct),
            "system_generator_name": "system_fe_c",
            "blending_mode": "fractional",
            "medium_code_resolved": str(medium_code),
            "medium_influence_applied": bool(medium_influence_applied),
            "quench_effect_applied": bool(quench_effect_applied),
            "fraction_source": str(phase_fraction_source),
            "applied_realism_heuristics": {
                "stage_family": str(morphology_trace.get("family", "generic")),
                "boundary_biased_proeutectoid": bool(
                    stage in {"alpha_pearlite", "pearlite_cementite"}
                ),
                "contextual_martensite_style": bool(
                    stage in _SPECIALIZED_MARTENSITIC_STAGES
                ),
                "pure_iron_bright_baseline": bool(pure_iron_like),
            },
            "physics_guided_realism": bool(context.transformation_state),
        },
        "system_generator_extra": {
            "fe_c_unified": {
                "enabled": True,
                "stage_coverage_pass": bool(stage_coverage_pass),
                "resolved_stage": str(stage),
                "blending_mode": "fractional",
                "fallback_reason": ("" if stage_coverage_pass else "stage_not_mapped"),
            }
        },
        "fe_c_phase_render": {
            "input_phase_fractions": dict(input_phase_fractions),
            "normalized_phase_fractions": dict(normalized_fractions),
            "rendered_phase_layers": list(rendered_layers),
            "seed_split": dict(seed_split),
            "phase_masks_present": bool(phase_masks),
            "medium_influence_applied": bool(medium_influence_applied),
            "quench_effect_applied": bool(quench_effect_applied),
            "temper_shift_applied": dict(temper_shift_applied),
            "retained_austenite_used": float(retained_austenite_used),
            "homogeneity_mode": "light",
            "specialized_realism_mode": "stage_specialized",
            "fragment_filter_mode": "coarse_only",
            "specialized_fragment_filter_mode": "contextual",
            "min_fragment_area_px": int(fragment_area),
            "fraction_source": str(phase_fraction_source),
            "table_locked": bool(table_locked),
            "morphology_trace": morphology_trace,
        },
    }
    if pure_iron_like:
        metadata["system_generator_extra"]["pure_iron_baseline"] = {
            "applied": True,
            "profile": "bright_clean_ferrite_v1",
            "generator": str(
                morphology_trace.get("generator", "pure_ferrite_power_voronoi_v1")
            ),
            "expected_appearance": "almost_white_with_soft_boundaries",
        }
        metadata["engineering_trace"] = {
            **dict(metadata.get("engineering_trace", {})),
            "pure_iron_baseline_applied": True,
            "pure_iron_target": "bright_ferritic_negative_control",
        }
    # A10.3 — surface the per-grain label map at the top level of the
    # metadata dict so downstream palettes (DIC polarised) can consume it
    # without introspecting ``morphology_trace``. The key is ignored by
    # the grayscale pipeline so backward compatibility is preserved.
    grain_labels_raw = morphology_trace.get("grain_labels")
    if isinstance(grain_labels_raw, np.ndarray):
        metadata["grain_labels"] = grain_labels_raw.astype(np.int32)
    return SystemGenerationResult(
        image_gray=image_gray, phase_masks=phase_masks, metadata=metadata
    )
