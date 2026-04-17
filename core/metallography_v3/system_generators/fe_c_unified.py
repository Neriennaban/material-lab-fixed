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
# Phase 9 cleanup: все текстуры (texture_sorbite_*, texture_troostite_*,
# fe_c_texture_map) больше не нужны в fe_c_unified — они использовались
# только внутри удалённого _generic_render. Модульные renderer'ы имеют
# свои импорты.

# Phase 1 редизайна (см. docs/plans/whimsical-wandering-dawn.md):
# модульные renderer'ы семейств микроструктур зарегистрированы в
# таблице диспетчера _STAGE_TO_RENDERER. На Phase 1 они НЕ подключены
# в основной runtime-путь render_fe_c_unified — старые _build_*_render
# работают как прежде, визуальный drift = 0. Подключение по семействам
# начиная с Phase 2.
from core.metallography_v3.renderers import (  # noqa: E402
    bainite as _r_bainite,
    granular_pearlite as _r_granular_pearlite,
    high_temp_phases as _r_high_temp_phases,
    martensite as _r_martensite,
    quench_products as _r_quench_products,
    surface_layers as _r_surface_layers,
    tempered as _r_tempered,
    white_cast_iron as _r_white_cast_iron,
    widmanstatten as _r_widmanstatten,
)

_RENDERER_MODULES = (
    _r_martensite,
    _r_bainite,
    _r_tempered,
    _r_quench_products,
    _r_white_cast_iron,
    _r_high_temp_phases,
    _r_widmanstatten,
    _r_surface_layers,
    _r_granular_pearlite,
)

# Раскладка stage -> модуль. Валидируется в
# tests/renderers/test_dispatch_table.py.
_STAGE_TO_RENDERER: dict[str, Any] = {
    stage: mod
    for mod in _RENDERER_MODULES
    for stage in mod.HANDLES_STAGES
}

# Phase 2 — high_temp_phases (§1.4, §1.5, §3.1).
# Phase 3 — white_cast_iron (§1.6, §1.10).
# Остальные семейства (мартенсит/бейнит/отпуски/видманштеттен/
# поверхностные слои/зернистый перлит) пока идут по старым путям;
# подключение — Phase 4-8.
_PHASE2_ACTIVATED_STAGES: frozenset[str] = frozenset(_r_high_temp_phases.HANDLES_STAGES)
_PHASE3_ACTIVATED_STAGES: frozenset[str] = frozenset(_r_white_cast_iron.HANDLES_STAGES)
_PHASE4_ACTIVATED_STAGES: frozenset[str] = frozenset(_r_martensite.HANDLES_STAGES)
_PHASE5_ACTIVATED_STAGES: frozenset[str] = frozenset(_r_bainite.HANDLES_STAGES)
_PHASE6_ACTIVATED_STAGES: frozenset[str] = frozenset(_r_quench_products.HANDLES_STAGES)
_PHASE7_ACTIVATED_STAGES: frozenset[str] = frozenset(_r_tempered.HANDLES_STAGES)
_PHASE8_ACTIVATED_STAGES: frozenset[str] = (
    frozenset(_r_widmanstatten.HANDLES_STAGES)
    | frozenset(_r_surface_layers.HANDLES_STAGES)
    | frozenset(_r_granular_pearlite.HANDLES_STAGES)
)
_ACTIVATED_RENDERER_STAGES: frozenset[str] = (
    _PHASE2_ACTIVATED_STAGES
    | _PHASE3_ACTIVATED_STAGES
    | _PHASE4_ACTIVATED_STAGES
    | _PHASE5_ACTIVATED_STAGES
    | _PHASE6_ACTIVATED_STAGES
    | _PHASE7_ACTIVATED_STAGES
    | _PHASE8_ACTIVATED_STAGES
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
    # Phase 5 — безкарбидный бейнит §2.7 (TRIP/nanobainite, Si≥1.5%):
    # 60-85% αb + 10-30% γR + 0-15% мартенсит.
    "carbide_free_bainite": {
        "BAINITE": 0.70,
        "AUSTENITE": 0.25,
        "MARTENSITE": 0.05,
    },
    # Phase 8 — новые стадии §2.10, §3.2-3.3, §1.9.
    "widmanstatten_ferrite": {"FERRITE": 0.50, "PEARLITE": 0.50},
    "decarburized_layer": {"FERRITE": 0.70, "PEARLITE": 0.30},
    "carburized_layer": {
        "MARTENSITE": 0.35,
        "PEARLITE": 0.30,
        "FERRITE": 0.20,
        "CEMENTITE": 0.10,
        "AUSTENITE": 0.05,
    },
    "granular_pearlite": {"FERRITE": 0.85, "CEMENTITE": 0.15},
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
    "carbide_free_bainite",
    "tempered_low",
    "tempered_medium",
    "tempered_high",
    # Phase 8 additions
    "widmanstatten_ferrite",
    "decarburized_layer",
    "carburized_layer",
    "granular_pearlite",
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
# Phase 9 cleanup: дисп-ветка _SPECIALIZED_CAST_IRON_STAGES /
# _SPECIALIZED_BAINITIC_STAGES больше не вызывается в render_fe_c_unified
# (стадии теперь идут через _STAGE_TO_RENDERER), но сами константы
# сохранены как deprecated compatibility-aliases для тестов
# typesafety/taxonomy (tests/test_fe_c_new_stages_taxonomy_v3.py).
# Aliases derive from renderer HANDLES_STAGES, так что остаются
# синхронизированными.
_SPECIALIZED_CAST_IRON_STAGES: frozenset[str] = frozenset(
    {s for s in _r_white_cast_iron.HANDLES_STAGES if s.startswith("white_cast_iron_")}
)
_SPECIALIZED_BAINITIC_STAGES: frozenset[str] = frozenset(
    {"bainite_upper", "bainite_lower"}
)


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
    # Accept both "ferrite" and "alpha_pearlite" stages when carbon is
    # very low — the phase orchestrator switches from "ferrite" to
    # "alpha_pearlite" at C≈0.02%, but at these near-zero pearlite
    # fractions the micrograph should still look like pure ferrite
    # with perhaps a handful of dark spots, not a completely different
    # rendering style.
    stage_ok = stage_name in ("ferrite", "alpha_pearlite")
    return bool(
        stage_ok
        and ferritic_fraction >= 0.90
        and fe_pct >= 99.5
        and c_pct <= 0.06
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
        arr = ndimage.gaussian_filter(arr, sigma=0.3)
    lo = float(np.quantile(arr, 0.02))
    hi = float(np.quantile(arr, 0.985))
    if hi > lo + 1e-6:
        arr = (arr - lo) / (hi - lo)
        # Wider dynamic range so grain boundaries stay visible after
        # the normalisation pass.
        arr = arr * 90.0 + 148.0
    dark_floor = float(np.quantile(arr, 0.07))
    arr += max(0.0, 138.0 - dark_floor)
    arr = np.clip(arr, 132.0, 246.0)
    bright = ensure_u8(arr)
    bright = _lift_small_dark_defects(bright, max_pixels=20)
    if ndimage is not None:
        smooth = ndimage.gaussian_filter(bright.astype(np.float32), sigma=0.8)
        # Keep 40% of the high-frequency detail (boundaries) instead
        # of the previous 10% which was washing them out.
        bright = ensure_u8(smooth + (bright.astype(np.float32) - smooth) * 0.40)
    return bright


def _pure_ferrite_render(
    *,
    context: SystemGenerationContext,
    seed_split: dict[str, int],
    pearlite_fraction: float = 0.0,
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
        boundary_width_px=2.0,
        boundary_depth=0.12,
        blur_sigma_px=0.5,
    )
    image_gray = np.clip(render["image_gray"].astype(np.int16), 120, 240).astype(
        np.uint8
    )

    # When called for a near-pure-iron composition that still has a
    # small pearlite fraction (e.g. C=0.03%, pearlite≈1-3%), scatter
    # a few dark pearlite spots on grain triple-junctions so the
    # transition from pure ferrite to ferrite+pearlite is gradual
    # rather than a hard switch to a completely different renderer.
    pearlite_frac = max(0.0, min(0.15, float(pearlite_fraction)))
    labels = render.get("labels")
    if pearlite_frac > 0.005 and isinstance(labels, np.ndarray) and ndimage is not None:
        boundary = boundary_mask_from_labels(labels, width=1)
        dist = ndimage.distance_transform_edt(~boundary).astype(np.float32)
        # Triple-junction score: pixels near boundary intersections
        boundary_pref = np.exp(-((dist / 3.5) ** 2)).astype(np.float32)
        rng = np.random.default_rng(seed_split["seed_topology"] + 9999)
        noise_field = rng.normal(0.0, 1.0, size=context.size).astype(np.float32)
        if ndimage is not None:
            noise_field = ndimage.gaussian_filter(noise_field, sigma=5.0)
        combined = normalize01(boundary_pref * 0.7 + normalize01(noise_field) * 0.3)
        threshold = float(np.quantile(combined, 1.0 - pearlite_frac))
        pearlite_mask = combined >= threshold
        if ndimage is not None:
            pearlite_mask = ndimage.binary_opening(pearlite_mask, iterations=1)
        # Paint pearlite spots as dark grey (80-100)
        img_f = image_gray.astype(np.float32)
        img_f[pearlite_mask] = 80.0 + rng.uniform(0, 20, size=int(pearlite_mask.sum())).astype(np.float32)
        if ndimage is not None:
            # Slight blur on pearlite edges for natural look
            blend = ndimage.gaussian_filter(img_f, sigma=0.5)
            img_f[pearlite_mask] = blend[pearlite_mask]
        image_gray = np.clip(img_f, 0.0, 255.0).astype(np.uint8)
        phase_masks = {
            "FERRITE": (~pearlite_mask).astype(np.uint8),
            "PEARLITE": pearlite_mask.astype(np.uint8),
        }
        rendered_layers = ["FERRITE", "PEARLITE"]
    else:
        phase_masks = {"FERRITE": np.ones(context.size, dtype=np.uint8)}
        rendered_layers = ["FERRITE"]

    fragment_area = int(max(48.0, math.pi * (mean_eq_d_px * 0.5) ** 2))
    trace = {
        "family": "pure_ferrite_power_voronoi",
        **dict(render.get("metadata", {})),
    }
    raw_labels = render.get("labels")
    if isinstance(raw_labels, np.ndarray):
        trace["grain_labels"] = raw_labels.astype(np.int32)
    return image_gray, phase_masks, rendered_layers, fragment_area, trace


# Phase 9 cleanup: _build_white_cast_iron_render и
# _build_bainitic_render_split удалены — соответствующие стадии теперь
# обслуживаются модульными renderer'ами:
#   white_cast_iron  → core/metallography_v3/renderers/white_cast_iron.py
#   bainite_upper/lower/CFB → core/metallography_v3/renderers/bainite.py


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


def _compensate_edge_dimming(
    image_gray: np.ndarray, *, edge_depth: int = 30, threshold: float = 4.0
) -> np.ndarray:
    """Post-hoc compensation for systematic edge-band dimming.

    Voronoi-based generators сэмплируют seed-точки строго внутри
    [0, size), поэтому у границ изображения плотность зёрен-границ
    ≈ в 2× реже чем в центре. Это превращается в видимую тёмную
    «рамку» у всех четырёх краёв после boundary darkening + dilation.

    Алгоритм: для каждого из четырёх краёв, если mean первых
    edge_depth строк/столбцов темнее bulk (строк edge_depth..2*edge_depth)
    больше чем на threshold единиц U8, линейно подравниваем значения
    этих строк/столбцов к bulk mean. Применяется only когда разница
    systematic (> threshold), иначе no-op.
    """
    h, w = image_gray.shape
    if h <= 3 * edge_depth or w <= 3 * edge_depth:
        return image_gray
    arr = image_gray.astype(np.float32)
    # Bulk region — middle 60% of image.
    mid_y0, mid_y1 = int(h * 0.2), int(h * 0.8)
    mid_x0, mid_x1 = int(w * 0.2), int(w * 0.8)
    bulk = float(arr[mid_y0:mid_y1, mid_x0:mid_x1].mean())

    def _blend_edge(region: np.ndarray, target: float, axis: int) -> None:
        """Линейный gradient blend: на крае — полная компенсация,
        внутрь edge_depth — плавно до 0."""
        length = region.shape[axis]
        diff_mean = target - float(region.mean())
        if diff_mean <= threshold:
            return
        # weights: 1.0 на самом краю → 0.0 на inner side
        weights = np.linspace(1.0, 0.0, length, dtype=np.float32)
        if axis == 0:
            weights = weights[:, None]
        else:
            weights = weights[None, :]
        region += weights * diff_mean

    # Top edge
    top = arr[:edge_depth]
    _blend_edge(top, bulk, axis=0)
    arr[:edge_depth] = top
    # Bottom edge
    bot = arr[-edge_depth:]
    _blend_edge(bot[::-1], bulk, axis=0)
    arr[-edge_depth:] = bot
    # Left edge
    left = arr[:, :edge_depth]
    _blend_edge(left, bulk, axis=1)
    arr[:, :edge_depth] = left
    # Right edge
    right = arr[:, -edge_depth:]
    _blend_edge(right[:, ::-1], bulk, axis=1)
    arr[:, -edge_depth:] = right

    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


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
    render_ferrite_lamellae: bool = True,
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
    if render_ferrite_lamellae:
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


def _project_mask_to_grains(
    *,
    labels: np.ndarray,
    pixel_mask: np.ndarray,
    fraction_total: float,
) -> np.ndarray:
    """D3 — project a pixel-level mask onto whole Voronoi grains.

    Sort grains by their internal ``pixel_mask`` coverage in
    descending order and pick whole grains one by one until the
    cumulative pixel count reaches ``fraction_total * total_pixels``.
    Grains with zero coverage are never picked.

    The result is a clean grain-boundary-aligned mask that:

    * keeps the final ferrite / pearlite fraction close to the
      requested ``fraction_total`` (within one grain area of the
      target), and
    * preserves the boundary bias of the original pixel mask —
      grains with the highest coverage are the ones whose interior
      sits inside the high-score region of the ranking field, so the
      resulting per-pixel mean of that field stays close to what
      pixel-level allocation would have produced.
    """
    if labels.shape != pixel_mask.shape:
        raise ValueError(
            f"labels and pixel_mask shape mismatch: {labels.shape} vs {pixel_mask.shape}"
        )
    flat_labels = labels.ravel()
    if flat_labels.size == 0:
        return np.zeros(labels.shape, dtype=bool)
    n_grains = int(flat_labels.max()) + 1
    if n_grains <= 0:
        return np.zeros(labels.shape, dtype=bool)

    total_pixels = int(flat_labels.size)
    target = int(round(max(0.0, min(1.0, float(fraction_total))) * total_pixels))
    if target <= 0:
        return np.zeros(labels.shape, dtype=bool)

    hit_counts = np.bincount(
        flat_labels,
        weights=pixel_mask.ravel().astype(np.float64),
        minlength=n_grains,
    )
    total_counts = np.bincount(flat_labels, minlength=n_grains).astype(np.float64)
    safe_total = np.where(total_counts > 0, total_counts, 1.0)
    coverage = hit_counts / safe_total
    # Grains with zero size are sentinel — never pick them.
    coverage[total_counts <= 0] = -1.0

    order = np.argsort(-coverage, kind="stable")
    cumulative = 0
    picked: list[int] = []
    for gid in order:
        g = int(gid)
        if coverage[g] <= 0.0:
            break
        cnt = int(total_counts[g])
        if cnt <= 0:
            continue
        picked.append(g)
        cumulative += cnt
        if cumulative >= target:
            break

    if not picked:
        return np.zeros(labels.shape, dtype=bool)
    picked_array = np.asarray(picked, dtype=np.int64)
    return np.isin(labels, picked_array)


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
        # alpha_pearlite: ферритные ламели внутри перлита визуально
        # сливаются с матрицей чистого феррита ("двуслойный" эффект) —
        # оставляем только тёмные цементитные полосы, pearlite читается
        # как равномерно-тёмное пятно (~90) по §1.3 справочника.
        render_ferrite_lamellae=(stage != "alpha_pearlite"),
    )
    # Use the same Power Voronoi ferrite renderer as _pure_ferrite_render
    # so the ferrite grains look identical regardless of whether the
    # composition is pure iron or a low-carbon steel with a few %
    # pearlite. This eliminates the jarring visual transition at C≈0.03%.
    ferrite_render = generate_pure_ferrite_micrograph(
        size=size,
        seed=seed_split["seed_boundary"],
        mean_eq_d_px=float(grain_scale * 0.9),
        size_sigma=0.22,
        relax_iter=1,
        boundary_width_px=2.0,
        boundary_depth=0.12,
        blur_sigma_px=0.5,
    )
    ferrite_img = np.clip(
        ferrite_render["image_gray"].astype(np.float32), 150.0, 240.0
    ).astype(np.uint8)

    # Phase D.3 — proeutectoid cementite is NOT attacked by nital
    # and appears as the brightest phase in real micrographs. The
    # previous tone (~70) was a learning-mode "readable" hack; now
    # that the cementite sits in the boundary network (see below)
    # we paint it white (235-250) so it reads as the etched
    # hypereutectoid "white grain-boundary network" from §5.3.Б of
    # the TZ.
    cementite_img = 238.0 + (noise - 0.5) * 6.0 + boundary_pref * 6.0
    cementite_img = np.clip(cementite_img, 225.0, 252.0).astype(np.uint8)

    proeutectoid = (
        "FERRITE"
        if stage == "alpha_pearlite"
        else ("CEMENTITE" if stage == "pearlite_cementite" else "")
    )
    proeutectoid_frac = float(phase_fractions.get(proeutectoid, 0.0)) if proeutectoid else 0.0
    if stage == "alpha_pearlite":
        # D3 — ferrite/pearlite snap to whole Voronoi grains.
        # Pipeline: (1) pixel-level ranking with a slightly shrunk
        # target so the mask hugs the most boundary-biased pixels
        # tightly, (2) projection selects grains in order of their
        # internal coverage in that pixel mask until the cumulative
        # grain size reaches the lever-rule target fraction.
        # Grains with the highest coverage are the ones whose
        # interior sits inside the high boundary_rank region, so
        # ``boundary_phase_bias`` stays in the ≥0.60 band required
        # by ``test_proeutectoid_phases_are_boundary_biased``.
        ferr_field = normalize01(boundary_rank * 0.95 + low * 0.03 + noise * 0.02)
        ferr_pixel_mask = select_fraction_mask(
            field=ferr_field,
            available=np.ones(size, dtype=bool),
            fraction_total=float(max(0.0, proeutectoid_frac * 0.75)),
        )
        ferr_mask = _project_mask_to_grains(
            labels=labels,
            pixel_mask=ferr_pixel_mask,
            fraction_total=float(proeutectoid_frac),
        )
        phase_masks = {
            "FERRITE": ferr_mask.astype(np.uint8),
            "PEARLITE": (~ferr_mask).astype(np.uint8),
        }
    elif stage == "pearlite_cementite":
        # Phase D.3 — hypereutectoid proeutectoid cementite forms a
        # bright network along the *boundaries* of the prior
        # austenite grains, not a scattering of whole grains. The
        # grain interiors remain pearlite. We dilate the Voronoi
        # boundary mask until its coverage matches the lever-rule
        # cementite fraction; width scales with carbon content per
        # TZ §5.3.Б (1-2 px at 0.9 %C → 5-8 px at 2.0 %C), so the
        # network thickens visibly as C climbs.
        target_pixels = int(
            round(float(max(0.0, min(1.0, proeutectoid_frac))) * size[0] * size[1])
        )
        cem_mask = np.zeros(size, dtype=bool)
        if ndimage is not None and target_pixels > 0:
            base_network = boundary_mask_from_labels(labels, width=1)
            cem_mask = base_network > 0
            # Target dilation radius from the carbon content, then
            # widen one pixel at a time until we hit the target
            # coverage — this gives continuous control rather than
            # quantised jumps and never overshoots the lever-rule
            # value dramatically.
            c_clamped = float(max(0.77, min(2.14, c_wt)))
            carbon_lerp = (c_clamped - 0.77) / (2.14 - 0.77)
            target_width = 1 + int(round(carbon_lerp * 6.0))  # 1..7 px
            max_iter = max(target_width + 2, 12)
            for _ in range(max_iter):
                if int(cem_mask.sum()) >= target_pixels:
                    break
                cem_mask = ndimage.binary_dilation(cem_mask, iterations=1)
            # If we overshot, prune: keep the pixels with the
            # largest ``boundary_pref`` score so the thinnest part
            # of the network is sacrificed first.
            cur = int(cem_mask.sum())
            if cur > target_pixels:
                scores = boundary_pref.astype(np.float32)
                scores_in_net = np.where(cem_mask, scores, -1.0)
                flat_scores = scores_in_net.ravel()
                keep = np.argpartition(-flat_scores, target_pixels)[:target_pixels]
                kept = np.zeros(flat_scores.size, dtype=bool)
                kept[keep] = True
                cem_mask = kept.reshape(size)
        else:
            # ndimage not available — fall back to the old grain-level
            # allocation rather than crashing.
            cem_field = normalize01(boundary_rank * 0.97 + low * 0.02 + noise * 0.01)
            cem_pixel_mask = select_fraction_mask(
                field=cem_field,
                available=np.ones(size, dtype=bool),
                fraction_total=float(max(0.0, proeutectoid_frac * 0.75)),
            )
            cem_mask = _project_mask_to_grains(
                labels=labels,
                pixel_mask=cem_pixel_mask,
                fraction_total=float(proeutectoid_frac),
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

    # Grain-boundary darkening is skipped on cementite pixels so the
    # bright proeutectoid network in hypereutectoid presets is not
    # eroded — cementite is nital-inert and must stay light.
    cementite_pixel_mask = (
        phase_masks["CEMENTITE"] > 0 if "CEMENTITE" in phase_masks else None
    )
    boundary_darken_mask = boundaries.copy()
    if cementite_pixel_mask is not None:
        boundary_darken_mask &= ~cementite_pixel_mask
    canvas[boundary_darken_mask] -= 6.0
    canvas += (
        multiscale_noise(
            size=size,
            seed=seed_split["seed_noise"] + 5,
            scales=((14.0, 0.65), (3.4, 0.35)),
        )
        - 0.5
    ) * 4.0
    # D2 — raise the lower rescale bound so the final normalisation
    # does not stretch any stray dark pixels toward pure black. The
    # pearlite tone remains clearly darker than the ferrite matrix
    # (it sits ~70-95 pre-rescale) but loses the gritty black spots
    # that the previous lo=25 created on ferrite-dominated frames.
    image_gray = soft_unsharp(
        _suppress_small_inclusions(rescale_to_u8(canvas, lo=40.0, hi=245.0)),
        amount=0.38,
    )
    # Phase D.3 — repaint the proeutectoid cementite network on top
    # of the post-processed frame. The generic rescale + suppression
    # + unsharp chain was flattening the bright cementite band to
    # ~170 because its area fraction (6-11 %) is too small to pull
    # the histogram stretch. We stamp the saved ``cementite_img``
    # tone back onto the mask so the grain-boundary network always
    # reads as nearly white (nital-inert Fe₃C) in the final image.
    if cementite_pixel_mask is not None and cementite_pixel_mask.any():
        image_gray = image_gray.copy()
        image_gray[cementite_pixel_mask] = cementite_img[cementite_pixel_mask]

    # Edge-dimming compensation: Voronoi seed'ы сэмплируются внутри
    # [0, size), поэтому у верхнего/нижнего/левого/правого края плотность
    # γ-границ ≈ в 2× реже чем в центре. После boundary darkening,
    # cementite dilation и rescale это превращается в видимую тёмную
    # «рамку» (~8 единиц U8 разницы для pearlite_cementite). Косметический
    # фикс: если первые/последние N строк (или столбцов) статистически
    # темнее bulk, подравниваем mean строки к bulk. Применяется только
    # при заметной разнице (>4 U8).
    image_gray = _compensate_edge_dimming(image_gray)

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


# Phase 9 cleanup: _build_martensitic_render и _generic_render удалены.
# См. core/metallography_v3/renderers/martensite.py (§2.1-2.4) +
# renderers/tempered.py + renderers/quench_products.py.


_FALLBACK_PHASE_TONE: dict[str, float] = {
    "FERRITE": 218.0,
    "PEARLITE": 105.0,
    "CEMENTITE": 235.0,
    "AUSTENITE": 225.0,
    "DELTA_FERRITE": 150.0,
    "LIQUID": 225.0,
    "MARTENSITE": 80.0,
    "MARTENSITE_TETRAGONAL": 75.0,
    "MARTENSITE_CUBIC": 88.0,
    "TROOSTITE": 60.0,
    "SORBITE": 115.0,
    "BAINITE": 130.0,
    "LEDEBURITE": 165.0,
    "CEMENTITE_PRIMARY": 245.0,
}


def _fallback_render(
    *,
    context: SystemGenerationContext,
    stage: str,
    normalized_fractions: dict[str, float],
    seed_split: dict[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], int, dict[str, Any]]:
    """Safety-net для unknown/custom stages.

    До Phase 9 эту роль играл ``_generic_render``. Новый минимализм:
    аллоцируем phase_masks через ``_labels_from_fractions`` и заливаем
    каждую фазу её tabular-тоном. Низкокачественный, но корректный
    drop-in fallback — предотвращает silent downgrade в
    ``generate_fe_c → run_phase_map_system`` при попадании
    пользовательских/опечатанных stage names.
    """
    size = context.size
    seed = int(seed_split.get("seed_noise", context.seed))

    dominant = _dominant_structure(normalized_fractions)
    # _labels_from_fractions возвращает tuple (labels_array, names_list).
    label_arr, phase_names = _labels_from_fractions(
        size=size,
        seed=seed,
        fractions=normalized_fractions,
    )

    canvas = np.full(size, 128.0, dtype=np.float32)
    phase_masks: dict[str, np.ndarray] = {}
    for idx, name in enumerate(phase_names):
        mask = label_arr == idx
        if not mask.any():
            continue
        phase_masks[name] = mask.astype(np.uint8)
        tone = _FALLBACK_PHASE_TONE.get(_canon_phase_name(name), 128.0)
        canvas[mask] = tone
    # Лёгкая текстура + сглаживание краёв — избегаем сплошных блоков.
    canvas += (
        multiscale_noise(size=size, seed=seed + 11, scales=((12.0, 0.6), (4.0, 0.4)))
        - 0.5
    ) * 10.0
    image_gray = soft_unsharp(
        rescale_to_u8(np.clip(canvas, 0.0, 255.0), lo=20.0, hi=235.0),
        amount=0.22,
    )

    rendered_layers = list(phase_masks.keys()) or [dominant]
    fragment_area = max(1, size[0] * size[1] // max(1, len(rendered_layers)))
    morphology_trace: dict[str, Any] = {
        "family": "fallback_generic",
        "stage": stage,
        "phase_count": len(rendered_layers),
        "dominant_phase": dominant,
    }
    return image_gray, phase_masks, rendered_layers, fragment_area, morphology_trace


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
    if pure_iron_like or stage == "ferrite":
        # Stage "ferrite" всегда — pure-ferrite Power Voronoi рендер.
        # Pass the pearlite fraction so near-pure compositions (C≈0.02-0.05%)
        # get a few dark pearlite spots instead of a hard visual switch.
        pearlite_frac_for_render = float(normalized_fractions.get("PEARLITE", 0.0))
        image_gray, phase_masks, rendered_layers, fragment_area, morphology_trace = (
            _pure_ferrite_render(
                context=context,
                seed_split=seed_split,
                pearlite_fraction=pearlite_frac_for_render,
            )
        )
    elif stage in _ACTIVATED_RENDERER_STAGES and stage in _STAGE_TO_RENDERER:
        # Новые семейственные renderer'ы:
        # Phase 2: high_temp_phases (austenite/δ/γ-cementite/liquid/…)
        # Phase 3: white_cast_iron (ledeburite + 3 чугуна, §1.6/§1.10)
        # Phase 4-8 добавят martensite/bainite/tempered/quench_products/
        # widmanstatten/surface_layers/granular_pearlite.
        _r_out = _STAGE_TO_RENDERER[stage].render(
            context=context,
            stage=stage,
            phase_fractions=normalized_fractions,
            seed_split=seed_split,
        )
        image_gray = _r_out.image_gray
        phase_masks = {k: np.asarray(v, dtype=np.uint8) for k, v in _r_out.phase_masks.items()}
        morphology_trace = dict(_r_out.morphology_trace)
        rendered_layers = list(_r_out.rendered_layers) or sorted(list(phase_masks.keys()))
        fragment_area = int(_r_out.fragment_area or 0)
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
    else:
        # Phase 9 cleanup: safe-net для custom/opaque stage names
        # (resolve_fe_c_stage пропускает требование unknown stage
        # verbatim, и generate_fe_c не должен уходить в legacy
        # run_phase_map_system silently). Минимальный рендер по
        # нормализованным долям: aggregated tone field через
        # allocate_phase_masks + per-phase заливка.
        image_gray, phase_masks, rendered_layers, fragment_area, morphology_trace = (
            _fallback_render(
                context=context,
                stage=stage,
                normalized_fractions=normalized_fractions,
                seed_split=seed_split,
            )
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
    # A10.3 — surface the per-grain label map through a private
    # ``_grain_labels`` key inside ``metadata``. ``morphology_engine``
    # pops this key before any JSON serialisation happens so the
    # ndarray never reaches ``metadata_json_safe``. The label map is
    # *also* stripped from ``morphology_trace`` before it is attached
    # to the metadata (see below) so the trace can still be rendered
    # to JSON without errors.
    grain_labels_raw = morphology_trace.pop("grain_labels", None)
    if isinstance(grain_labels_raw, np.ndarray):
        metadata["_grain_labels"] = grain_labels_raw.astype(np.int32)
    return SystemGenerationResult(
        image_gray=image_gray, phase_masks=phase_masks, metadata=metadata
    )
