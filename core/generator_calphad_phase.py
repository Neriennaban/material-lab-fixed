from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from .calphad.phase_mapper import map_phase_to_texture
from .generator_eutectic import generate_aged_aluminum_structure, generate_eutectic_al_si
from .generator_grains import generate_grain_structure
from .generator_pearlite import generate_martensite_structure, generate_pearlite_structure


_SENSITIVITY_GAIN = {
    "realistic": 1.25,
    "educational": 2.15,
    "high_contrast": 3.10,
}

_EDU_STYLES = {"contrast_texture", "max_contrast", "morphology_only"}

_BASE_ELEMENT_BY_SYSTEM = {
    "fe-c": "Fe",
    "fe-si": "Fe",
    "al-si": "Al",
    "cu-zn": "Cu",
    "al-cu-mg": "Al",
}

_PHASE_TONE_HINTS = {
    "FERRITE": 184.0,
    "BCC": 176.0,
    "PEARLITE": 112.0,
    "CEMENTITE": 82.0,
    "CARBIDE": 88.0,
    "AUSTENITE": 150.0,
    "FCC": 148.0,
    "GRAPHITE": 70.0,
    "LIQUID": 190.0,
    "INTERMETALLIC": 118.0,
    "SIGMA": 106.0,
    "SI": 120.0,
    "PRECIP": 112.0,
}

_RULEBOOK_DIR = Path(__file__).resolve().parent / "rulebook"
_TEXTBOOK_RULES_PATH = _RULEBOOK_DIR / "textbook_visual_rules_v3.json"


def _load_textbook_rules() -> dict[str, Any]:
    if not _TEXTBOOK_RULES_PATH.exists():
        return {}
    try:
        return json.loads(_TEXTBOOK_RULES_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


_TEXTBOOK_RULES = _load_textbook_rules()
_TEXTURE_CACHE: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
_TEXTURE_CACHE_MAX = 96


def _rules_for_system(system: str) -> dict[str, Any]:
    systems = _TEXTBOOK_RULES.get("systems", {})
    if not isinstance(systems, dict):
        return {}
    payload = systems.get(str(system).strip().lower(), {})
    return payload if isinstance(payload, dict) else {}


def _rules_defaults() -> dict[str, Any]:
    payload = _TEXTBOOK_RULES.get("defaults", {})
    return payload if isinstance(payload, dict) else {}


def _cache_get_texture(key: tuple[Any, ...]) -> np.ndarray | None:
    img = _TEXTURE_CACHE.get(key)
    if img is None:
        return None
    _TEXTURE_CACHE.move_to_end(key)
    return img


def _cache_put_texture(key: tuple[Any, ...], image: np.ndarray) -> np.ndarray:
    _TEXTURE_CACHE[key] = image
    _TEXTURE_CACHE.move_to_end(key)
    while len(_TEXTURE_CACHE) > _TEXTURE_CACHE_MAX:
        _TEXTURE_CACHE.popitem(last=False)
    return image


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

    base = _BASE_ELEMENT_BY_SYSTEM.get(sys_name, "")
    if base and base in comp:
        total = float(sum(comp.values()))
        alloying = max(0.0, total - float(comp.get(base, 0.0)))
        return _clip01(alloying / 30.0)
    return _clip01(float(sum(comp.values())) / 100.0)


def _liquid_texture(
    size: tuple[int, int],
    seed: int,
    brightness: int = 145,
    swirl_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    noise = rng.normal(0.0, 1.0, size=size).astype(np.float32)
    if ndimage is not None:
        noise = ndimage.gaussian_filter(noise, sigma=10.0)
    swirl = np.sin(xx / max(1, w) * 7.0 * swirl_scale) + np.cos(yy / max(1, h) * 5.0 * swirl_scale)
    image = brightness + noise * (22.0 * noise_scale) + swirl * 7.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _texture_by_family(
    family: str,
    size: tuple[int, int],
    seed: int,
    *,
    system: str,
    stage: str,
    phase_name: str,
    visual_profile_id: str,
    solute_index: float,
    sensitivity_gain: float,
) -> np.ndarray:
    key = str(family).strip().lower()
    scale = float(max(0.0, solute_index) * max(0.1, sensitivity_gain))
    sys_name = str(system).strip().lower()
    stage_name = str(stage).strip().lower()
    phase_name_u = str(phase_name).upper()
    phase_token = "".join(ch for ch in phase_name_u if ch.isalnum())[:18]
    seed_bucket = int(seed) // 13
    cache_key = (
        sys_name,
        stage_name,
        phase_token,
        key,
        int(size[0]),
        int(size[1]),
        int(seed_bucket),
        str(visual_profile_id),
        round(float(solute_index), 4),
        round(float(sensitivity_gain), 4),
    )
    cached = _cache_get_texture(cache_key)
    if cached is not None:
        return cached

    if key == "liquid":
        brightness = int(round(148.0 + 9.0 * scale))
        swirl_scale = 1.0 + 0.15 * scale
        noise_scale = 1.0 + 0.18 * scale
        return _cache_put_texture(cache_key, _liquid_texture(
            size=size,
            seed=seed,
            brightness=brightness,
            swirl_scale=swirl_scale,
            noise_scale=noise_scale,
        ))

    if key in {"martensite", "bainite"}:
        needle_count = int(round(3800.0 + 1700.0 * scale))
        sys_rules = _rules_for_system(sys_name)
        tparams = sys_rules.get("texture_family_params", {}) if isinstance(sys_rules, dict) else {}
        mart = tparams.get("martensite", {}) if isinstance(tparams, dict) else {}
        if isinstance(mart, dict):
            needle_count = int(round(float(mart.get("needle_count_base", needle_count)) + 980.0 * scale))
        return _cache_put_texture(
            cache_key,
            generate_martensite_structure(size=size, seed=seed, needle_count=needle_count)["image"],
        )

    if key in {"pearlite", "cementite", "carbide"}:
        lamella_period_px = float(np.clip(5.2 - 1.8 * scale, 2.6, 8.2))
        pearlite_fraction = float(np.clip(0.76 + 0.17 * scale, 0.3, 0.985))
        sys_rules = _rules_for_system(sys_name)
        tparams = sys_rules.get("texture_family_params", {}) if isinstance(sys_rules, dict) else {}
        pearlite_rules = tparams.get("pearlite", {}) if isinstance(tparams, dict) else {}
        if isinstance(pearlite_rules, dict):
            lamella_base = float(pearlite_rules.get("lamella_period_px_base", lamella_period_px))
            lamella_min = float(pearlite_rules.get("lamella_period_px_min", 2.8))
            lamella_max = float(pearlite_rules.get("lamella_period_px_max", 8.8))
            lamella_period_px = float(np.clip(lamella_base - 1.2 * scale, lamella_min, lamella_max))
        return _cache_put_texture(cache_key, generate_pearlite_structure(
            size=size,
            seed=seed,
            pearlite_fraction=pearlite_fraction,
            lamella_period_px=lamella_period_px,
        )["image"])

    if key in {"eutectic", "si", "intermetallic"}:
        si_phase_fraction = float(np.clip(0.23 + 0.22 * scale, 0.08, 0.75))
        morphology = "branched"
        if sys_name == "al-si" and ("SI" in phase_name_u or "EUT" in phase_name_u):
            morphology = "network"
        return _cache_put_texture(cache_key, generate_eutectic_al_si(
            size=size,
            seed=seed,
            si_phase_fraction=si_phase_fraction,
            morphology=morphology,
        )["image"])

    if key in {"precipitate", "aging"}:
        precipitate_fraction = float(np.clip(0.08 + 0.11 * scale, 0.03, 0.30))
        precipitate_scale_px = float(np.clip(1.5 + 1.6 * scale, 0.8, 4.5))
        return _cache_put_texture(cache_key, generate_aged_aluminum_structure(
            size=size,
            seed=seed,
            precipitate_fraction=precipitate_fraction,
            precipitate_scale_px=precipitate_scale_px,
        )["image"])

    mean_grain_size_px = float(np.clip(58.0 * (1.0 - 0.18 * scale), 36.0, 72.0))
    boundary_contrast = float(np.clip(0.52 + 0.14 * scale, 0.25, 0.95))
    return _cache_put_texture(cache_key, generate_grain_structure(
        size=size,
        seed=seed,
        mean_grain_size_px=mean_grain_size_px,
        boundary_contrast=boundary_contrast,
    )["image"])


def _normalize_fractions(phase_fractions: dict[str, float]) -> dict[str, float]:
    cleaned = {str(k): max(0.0, float(v)) for k, v in phase_fractions.items() if float(v) > 1e-8}
    total = float(sum(cleaned.values()))
    if total <= 1e-12:
        return {"UNKNOWN": 1.0}
    return {k: float(v / total) for k, v in cleaned.items()}


def _single_phase_compensation(
    image: np.ndarray,
    *,
    seed: int,
    solute_index: float,
    sensitivity_gain: float,
) -> tuple[np.ndarray, bool]:
    if solute_index <= 1e-6:
        return image, False
    if ndimage is None:
        return image, False

    rng = np.random.default_rng(seed + 104729)
    low = rng.normal(0.0, 1.0, size=image.shape).astype(np.float32)
    sigma = max(8.0, float(min(image.shape[0], image.shape[1])) * 0.08)
    low = ndimage.gaussian_filter(low, sigma=sigma)
    lo = float(low.min())
    hi = float(low.max())
    if hi <= lo + 1e-9:
        return image, False
    low = 2.0 * (low - lo) / (hi - lo) - 1.0

    amp = float(np.clip((7.0 + 11.0 * solute_index) * sensitivity_gain, 5.0, 28.0))
    out = image.astype(np.float32) + low * amp
    return np.clip(out, 0, 255).astype(np.uint8), True


def _build_labels_random(
    size: tuple[int, int],
    rng: np.random.Generator,
    probs: np.ndarray,
) -> np.ndarray:
    return rng.choice(len(probs), size=size, p=probs).astype(np.int32)


def _build_labels_readable(
    size: tuple[int, int],
    rng: np.random.Generator,
    probs: np.ndarray,
) -> np.ndarray:
    if len(probs) <= 1:
        return np.zeros(size, dtype=np.int32)

    h, w = size
    field = rng.normal(0.0, 1.0, size=size).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    field += 0.30 * np.sin(xx / max(1, w) * np.pi * 2.0)
    field += 0.24 * np.cos(yy / max(1, h) * np.pi * 1.6)
    if ndimage is not None:
        sigma = max(3.0, float(min(size)) / 24.0)
        field = ndimage.gaussian_filter(field, sigma=sigma)

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
    return labels.reshape(size)


def _smooth_field(rng: np.random.Generator, size: tuple[int, int], sigma: float, bias: float = 0.0) -> np.ndarray:
    field = rng.normal(0.0, 1.0, size=size).astype(np.float32)
    if ndimage is not None:
        field = ndimage.gaussian_filter(field, sigma=max(0.1, sigma))
    if abs(bias) > 1e-9:
        h, w = size
        yy, xx = np.mgrid[0:h, 0:w]
        field += float(bias) * (
            0.45 * np.sin(xx / max(1, w) * np.pi * 2.0) + 0.30 * np.cos(yy / max(1, h) * np.pi * 1.6)
        )
    return field


def _phase_idx(names: list[str], token: str) -> int:
    needle = str(token).upper()
    for idx, name in enumerate(names):
        if needle in str(name).upper():
            return idx
    return -1


def _build_labels_textbook(
    size: tuple[int, int],
    rng: np.random.Generator,
    probs: np.ndarray,
    names: list[str],
    *,
    system: str,
    stage: str,
) -> np.ndarray:
    labels = _build_labels_readable(size=size, rng=rng, probs=probs)
    sys_name = str(system).strip().lower()
    stage_name = str(stage).strip().lower()
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]

    if sys_name == "fe-c":
        ferrite_idx = _phase_idx(names, "FERRITE")
        pearlite_idx = _phase_idx(names, "PEARLITE")
        cementite_idx = _phase_idx(names, "CEMENTITE")
        carbide_idx = _phase_idx(names, "CARBIDE")
        cementite_main = cementite_idx if cementite_idx >= 0 else carbide_idx
        mart_idx = _phase_idx(names, "MARTENSITE")

        if pearlite_idx >= 0:
            p_target = float(probs[pearlite_idx])
            c_target = float(probs[cementite_main]) if cementite_main >= 0 else 0.0
            total_pc = min(0.98, p_target + c_target)
            field = _smooth_field(rng, size, sigma=max(2.5, min(size) / 20.0), bias=0.4)
            thr = np.quantile(field, 1.0 - max(0.02, total_pc))
            colony_mask = field >= thr
            if ferrite_idx >= 0:
                labels[~colony_mask] = ferrite_idx
            labels[colony_mask] = pearlite_idx
            if cementite_main >= 0 and total_pc > 1e-9:
                frac_c = np.clip(c_target / total_pc, 0.03, 0.85)
                net = np.abs(np.sin((xx * 0.10) + (yy * 0.08))) + 0.35 * _smooth_field(rng, size, sigma=2.2)
                v = net[colony_mask]
                if v.size > 8:
                    c_thr = np.quantile(v, 1.0 - frac_c)
                    c_mask = colony_mask & (net >= c_thr)
                    labels[c_mask] = cementite_main

        if mart_idx >= 0 and ("martensite" in stage_name or "tempered" in stage_name or "bainite" in stage_name):
            m_target = float(probs[mart_idx])
            packet_orient = rng.uniform(0.0, np.pi, size=6)
            packet = np.zeros(size, dtype=np.float32)
            for angle in packet_orient:
                packet += np.abs(np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * 0.12 + angle * 2.3))
            packet += 0.35 * _smooth_field(rng, size, sigma=1.9)
            m_thr = np.quantile(packet, 1.0 - np.clip(m_target, 0.02, 0.95))
            labels[packet >= m_thr] = mart_idx

    elif sys_name == "al-si":
        eut_idx = _phase_idx(names, "EUTECTIC")
        si_idx = _phase_idx(names, "SI")
        alpha_idx = _phase_idx(names, "FCC_A1")
        if alpha_idx < 0:
            alpha_idx = _phase_idx(names, "ALPHA")
        if eut_idx >= 0 or si_idx >= 0:
            e_target = float(probs[eut_idx]) if eut_idx >= 0 else 0.0
            s_target = float(probs[si_idx]) if si_idx >= 0 else 0.0
            total = min(0.98, max(0.02, e_target + s_target))
            field = 0.65 * np.abs(np.sin(xx * 0.085) + np.cos(yy * 0.095)) + 0.35 * _smooth_field(
                rng, size, sigma=max(1.6, min(size) / 30.0)
            )
            thr = np.quantile(field, 1.0 - total)
            eut_zone = field >= thr
            if alpha_idx >= 0:
                labels[~eut_zone] = alpha_idx
            if eut_idx >= 0:
                labels[eut_zone] = eut_idx
            if si_idx >= 0 and total > 1e-9:
                frac_si = np.clip(s_target / total, 0.02, 0.92)
                si_field = np.abs(np.sin(xx * 0.22 + yy * 0.17)) + 0.25 * _smooth_field(rng, size, sigma=1.3)
                vv = si_field[eut_zone]
                if vv.size > 8:
                    si_thr = np.quantile(vv, 1.0 - frac_si)
                    si_mask = eut_zone & (si_field >= si_thr)
                    labels[si_mask] = si_idx

    elif sys_name == "cu-zn":
        alpha_idx = _phase_idx(names, "ALPHA")
        beta_idx = _phase_idx(names, "BETA")
        deform_idx = _phase_idx(names, "DEFORMATION")
        if alpha_idx >= 0 and beta_idx >= 0:
            b_target = float(probs[beta_idx])
            angle = rng.uniform(0.2, 1.2)
            stripes = np.abs(np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * 0.12 + 0.7))
            stripes += 0.30 * _smooth_field(rng, size, sigma=max(1.8, min(size) / 40.0))
            thr = np.quantile(stripes, 1.0 - np.clip(b_target, 0.02, 0.8))
            labels[:] = alpha_idx
            labels[stripes >= thr] = beta_idx
            if deform_idx >= 0:
                d_target = float(probs[deform_idx])
                dfield = np.abs(np.sin((xx * 0.21) - (yy * 0.16) + 0.9))
                d_thr = np.quantile(dfield, 1.0 - np.clip(d_target, 0.01, 0.28))
                labels[dfield >= d_thr] = deform_idx

    elif sys_name == "fe-si":
        matrix_idx = _phase_idx(names, "BCC")
        inter_idx = _phase_idx(names, "INTERMETALLIC")
        if matrix_idx >= 0 and inter_idx >= 0:
            i_target = float(probs[inter_idx])
            field = _smooth_field(rng, size, sigma=max(1.2, min(size) / 45.0))
            thr = np.quantile(field, 1.0 - np.clip(i_target, 0.01, 0.45))
            labels[:] = matrix_idx
            labels[field >= thr] = inter_idx

    return labels


def _fraction_vector(labels: np.ndarray, n: int) -> np.ndarray:
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=n).astype(np.float64)
    return counts / max(1.0, float(flat.size))


def _fraction_error_pct(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    denom = np.maximum(target, 1e-6)
    return np.abs(actual - target) / denom * 100.0


def _enforce_fraction_tolerance(
    labels: np.ndarray,
    *,
    target: np.ndarray,
    tolerance_pct: float,
    rng: np.random.Generator,
    max_iter: int = 16,
) -> np.ndarray:
    n = int(target.size)
    tol = float(max(0.0, tolerance_pct))
    flat = labels.ravel()
    total = max(1, flat.size)

    for _ in range(max_iter):
        actual = _fraction_vector(flat.reshape(labels.shape), n)
        err = _fraction_error_pct(target, actual)
        if float(err.max(initial=0.0)) <= tol:
            break

        delta = actual - target
        over_idx = int(np.argmax(delta))
        under_idx = int(np.argmin(delta))
        if delta[over_idx] <= 0.0 or delta[under_idx] >= 0.0:
            break

        move_frac = min(delta[over_idx], -delta[under_idx])
        move_count = int(max(1.0, round(move_frac * total * 0.85)))
        source_idx = np.flatnonzero(flat == over_idx)
        if source_idx.size == 0:
            break
        picked = rng.choice(source_idx, size=min(move_count, source_idx.size), replace=False)
        flat[picked] = under_idx

    return flat.reshape(labels.shape)


def _phase_tone_target(
    phase_name: str,
    family: str,
    emphasis_style: str,
    *,
    system: str,
    stage: str,
) -> float:
    name_u = str(phase_name).upper()
    fam_u = str(family).upper()
    tone = 144.0
    defaults = _rules_defaults()
    default_tones = defaults.get("phase_tone_targets", {}) if isinstance(defaults, dict) else {}
    sys_rules = _rules_for_system(system)
    system_tones = sys_rules.get("phase_tone_targets", {}) if isinstance(sys_rules, dict) else {}

    tone_found = False
    if isinstance(system_tones, dict):
        for token, value in system_tones.items():
            if str(token).upper() in name_u:
                tone = float(value)
                tone_found = True
                break
    if not tone_found and isinstance(default_tones, dict):
        for token, value in default_tones.items():
            if str(token).upper() in name_u:
                tone = float(value)
                tone_found = True
                break
    for token, value in _PHASE_TONE_HINTS.items():
        if tone_found:
            break
        if token in name_u or token in fam_u:
            tone = float(value)
            break

    if emphasis_style == "max_contrast":
        if tone >= 145.0:
            tone = min(220.0, tone + 18.0)
        else:
            tone = max(52.0, tone - 18.0)
    elif emphasis_style == "morphology_only":
        tone = 0.5 * tone + 72.0
    return float(np.clip(tone, 24.0, 235.0))


def _apply_phase_emphasis(
    image: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    families: list[str],
    *,
    emphasis_style: str,
    system: str,
    stage: str,
) -> np.ndarray:
    if emphasis_style not in _EDU_STYLES:
        emphasis_style = "contrast_texture"

    alpha_by_style = {
        "contrast_texture": 0.50,
        "max_contrast": 0.72,
        "morphology_only": 0.28,
    }
    defaults = _rules_defaults()
    sys_rules = _rules_for_system(system)
    default_edges = defaults.get("boundary_emphasis", {}) if isinstance(defaults, dict) else {}
    system_edges = sys_rules.get("boundary_emphasis", {}) if isinstance(sys_rules, dict) else {}
    edge_by_style = {
        "contrast_texture": 20.0,
        "max_contrast": 32.0,
        "morphology_only": 12.0,
    }
    if isinstance(default_edges, dict):
        for k in tuple(edge_by_style.keys()):
            if k in default_edges:
                edge_by_style[k] = float(default_edges[k])
    if isinstance(system_edges, dict):
        for k in tuple(edge_by_style.keys()):
            if k in system_edges:
                edge_by_style[k] = float(system_edges[k])

    out = image.astype(np.float32)
    alpha = float(alpha_by_style.get(emphasis_style, 0.38))
    for idx, name in enumerate(names):
        mask = labels == idx
        if not np.any(mask):
            continue
        tone = _phase_tone_target(
            name,
            families[idx],
            emphasis_style,
            system=system,
            stage=stage,
        )
        out[mask] = out[mask] * (1.0 - alpha) + tone * alpha

    if ndimage is not None:
        edges = np.zeros(labels.shape, dtype=np.float32)
        edges[:-1, :] = np.maximum(edges[:-1, :], (labels[:-1, :] != labels[1:, :]).astype(np.float32))
        edges[:, :-1] = np.maximum(edges[:, :-1], (labels[:, :-1] != labels[:, 1:]).astype(np.float32))
        edges = ndimage.binary_dilation(edges > 0.0, iterations=1).astype(np.float32)
        out += edges * float(edge_by_style.get(emphasis_style, 14.0))

    return np.clip(out, 0, 255).astype(np.uint8)


def _phase_separability_score(
    image: np.ndarray,
    phase_masks: dict[str, np.ndarray],
    names: list[str],
) -> float:
    means: list[float] = []
    stds: list[float] = []
    for name in names:
        mask = phase_masks.get(name)
        if not isinstance(mask, np.ndarray):
            continue
        pix = image[mask > 0]
        if pix.size < 32:
            continue
        means.append(float(np.mean(pix)))
        stds.append(float(np.std(pix)) + 1e-6)

    if len(means) < 2:
        return 0.0

    ratios: list[float] = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            d = abs(means[i] - means[j]) / (stds[i] + stds[j])
            ratios.append(float(d))
    if not ratios:
        return 0.0
    raw = float(np.mean(ratios))
    score = float(1.0 - np.exp(-0.9 * raw))
    return float(np.clip(score, 0.0, 1.0))


def generate_calphad_phase_structure(
    *,
    size: tuple[int, int],
    seed: int,
    system: str,
    phase_fractions: dict[str, float],
    transition_state: dict[str, Any] | None = None,
    kinetics_result: dict[str, Any] | None = None,
    top_n_phases: int = 6,
    composition_wt: dict[str, float] | None = None,
    equilibrium_result: dict[str, Any] | None = None,
    composition_sensitivity_mode: str = "realistic",
    generation_mode: str = "realistic_visual",
    phase_emphasis_style: str = "contrast_texture",
    phase_fraction_tolerance_pct: float = 20.0,
    visual_profile_id: str = "",
) -> dict[str, Any]:
    mode = str(composition_sensitivity_mode or "realistic").strip().lower()
    if mode not in _SENSITIVITY_GAIN:
        mode = "realistic"
    gain = float(_SENSITIVITY_GAIN[mode])

    gen_mode = str(generation_mode or "realistic_visual").strip().lower()
    if gen_mode not in {"realistic_visual", "edu_engineering"}:
        gen_mode = "realistic_visual"
    emphasis = str(phase_emphasis_style or "contrast_texture").strip().lower()
    if emphasis not in _EDU_STYLES:
        emphasis = "contrast_texture"
    tolerance_pct = float(np.clip(float(phase_fraction_tolerance_pct), 0.0, 100.0))

    comp_hash, seed_offset = _composition_signature(composition_wt)
    solute = _solute_index(system=system, composition_wt=composition_wt)
    stage_hint = ""
    if isinstance(transition_state, dict):
        stage_hint = str(transition_state.get("stage", "")).strip().lower()

    rng = np.random.default_rng(int(seed) + int(seed_offset))
    fractions = _normalize_fractions(phase_fractions)
    ranked = sorted(fractions.items(), key=lambda x: x[1], reverse=True)[: max(1, int(top_n_phases))]
    names = [name for name, _ in ranked]
    probs = np.asarray([frac for _, frac in ranked], dtype=float)
    probs = probs / probs.sum()

    textures: dict[str, np.ndarray] = {}
    families: list[str] = []
    for idx, phase_name in enumerate(names):
        family = map_phase_to_texture(system=system, phase_name=phase_name)
        families.append(family)
        tex_seed = int(seed) + int(seed_offset) + 200 + idx * 17
        textures[phase_name] = _texture_by_family(
            family=family,
            size=size,
            seed=tex_seed,
            system=system,
            stage=stage_hint,
            phase_name=phase_name,
            visual_profile_id=visual_profile_id,
            solute_index=solute,
            sensitivity_gain=gain,
        )

    if gen_mode == "edu_engineering":
        labels = _build_labels_textbook(
            size=size,
            rng=rng,
            probs=probs,
            names=names,
            system=system,
            stage=stage_hint,
        )
        labels = _enforce_fraction_tolerance(
            labels,
            target=probs,
            tolerance_pct=tolerance_pct,
            rng=rng,
            max_iter=20,
        )
    else:
        labels = _build_labels_random(size=size, rng=rng, probs=probs)

    image = np.zeros(size, dtype=np.float32)
    phase_masks: dict[str, np.ndarray] = {}
    for idx, phase_name in enumerate(names):
        mask = labels == idx
        phase_masks[phase_name] = mask.astype(np.uint8)
        tex = textures[phase_name]
        image[mask] = tex[mask]

    if ndimage is not None:
        image = ndimage.gaussian_filter(image, sigma=0.45 if gen_mode == "edu_engineering" else 0.55)
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)

    if gen_mode == "edu_engineering":
        image_u8 = _apply_phase_emphasis(
            image=image_u8,
            labels=labels,
            names=names,
            families=families,
            emphasis_style=emphasis,
            system=system,
            stage=stage_hint,
        )

    single_phase_compensation = False
    if len(names) == 1:
        single_gain = gain if gen_mode != "edu_engineering" else max(0.65, gain * 0.72)
        image_u8, single_phase_compensation = _single_phase_compensation(
            image_u8,
            seed=int(seed) + int(seed_offset),
            solute_index=solute,
            sensitivity_gain=single_gain,
        )

    liquid_mask = np.zeros(size, dtype=np.uint8)
    for phase_name, mask in phase_masks.items():
        if "LIQUID" in phase_name.upper():
            liquid_mask = np.maximum(liquid_mask, mask.astype(np.uint8))
    if liquid_mask.any():
        phase_masks["L"] = liquid_mask
        phase_masks["solid"] = (1 - (liquid_mask > 0).astype(np.uint8)).astype(np.uint8)

    actual = _fraction_vector(labels, len(names))
    err_pct = _fraction_error_pct(probs, actual)
    target_map = {str(name): float(probs[idx]) for idx, name in enumerate(names)}
    actual_map = {str(name): float(actual[idx]) for idx, name in enumerate(names)}
    err_map = {str(name): float(err_pct[idx]) for idx, name in enumerate(names)}
    within_tolerance = bool(float(np.max(err_pct, initial=0.0)) <= tolerance_pct)
    separability_score = _phase_separability_score(image=image_u8, phase_masks=phase_masks, names=names)

    phase_visibility_report = {
        "target_phase_fractions": target_map,
        "achieved_phase_fractions": actual_map,
        "fraction_error_pct": err_map,
        "within_tolerance": within_tolerance,
        "separability_score": float(round(separability_score, 6)),
    }
    engineering_trace = {
        "generation_mode": gen_mode,
        "phase_emphasis_style": emphasis,
        "phase_fraction_tolerance_pct": float(tolerance_pct),
    }

    metadata = {
        "system": str(system),
        "phase_fractions": {str(k): float(v) for k, v in fractions.items()},
        "phase_order": names,
        "transition_state": transition_state or {},
        "kinetics_result": kinetics_result or {},
        "equilibrium_result": equilibrium_result or {},
        "generator_name": "calphad_phase",
        "visual_profile_id": str(visual_profile_id or ""),
        "composition_effect": {
            "mode": mode,
            "solute_index": float(round(solute, 6)),
            "composition_hash": comp_hash,
            "seed_offset": int(seed_offset),
            "single_phase_compensation": bool(single_phase_compensation),
        },
        "phase_visibility_report": phase_visibility_report,
        "engineering_trace": engineering_trace,
    }
    return {"image": image_u8, "phase_masks": phase_masks, "metadata": metadata}
