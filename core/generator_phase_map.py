from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from .generator_eutectic import generate_aged_aluminum_structure, generate_eutectic_al_si
from .generator_grains import generate_grain_structure
from .generator_pearlite import (
    generate_martensite_structure,
    generate_pearlite_structure,
    generate_sorbite_structure,
    generate_tempered_steel_structure,
    generate_troostite_structure,
)

SYSTEM_STAGE_ORDER: dict[str, list[str]] = {
    "fe-c": [
        "liquid",
        "liquid_gamma",
        "delta_ferrite",
        "austenite",
        "ferrite",
        "alpha_gamma",
        "gamma_cementite",
        "alpha_pearlite",
        "pearlite",
        "pearlite_cementite",
        "ledeburite",
        "white_cast_iron_hypoeutectic",
        "white_cast_iron_eutectic",
        "white_cast_iron_hypereutectic",
        "martensite",
        "martensite_tetragonal",
        "martensite_cubic",
        "troostite_quench",
        "troostite_temper",
        "sorbite_quench",
        "sorbite_temper",
        "bainite",
        "bainite_upper",
        "bainite_lower",
        "tempered_low",
        "tempered_medium",
        "tempered_high",
    ],
    "al-si": [
        "liquid",
        "liquid_alpha",
        "liquid_si",
        "alpha_eutectic",
        "eutectic",
        "primary_si_eutectic",
        "supersaturated",
        "aged",
    ],
    "cu-zn": [
        "liquid",
        "liquid_alpha",
        "liquid_beta",
        "alpha",
        "alpha_beta",
        "beta",
        "beta_prime",
        "cold_worked",
    ],
    "al-cu-mg": [
        "solutionized",
        "quenched",
        "natural_aged",
        "artificial_aged",
        "overaged",
    ],
    "fe-si": [
        "liquid",
        "liquid_ferrite",
        "hot_ferrite",
        "recrystallized_ferrite",
        "cold_worked_ferrite",
    ],
}

_SYSTEM_ALIASES = {
    "fe-c": "fe-c",
    "fec": "fe-c",
    "fe_c": "fe-c",
    "steel": "fe-c",
    "carbon_steel": "fe-c",
    "al-si": "al-si",
    "alsi": "al-si",
    "al_si": "al-si",
    "cast_al_si": "al-si",
    "cu-zn": "cu-zn",
    "cuzn": "cu-zn",
    "cu_zn": "cu-zn",
    "brass": "cu-zn",
    "al-cu-mg": "al-cu-mg",
    "al_cu_mg": "al-cu-mg",
    "duralumin": "al-cu-mg",
    "fe-si": "fe-si",
    "fesi": "fe-si",
    "fe_si": "fe-si",
}

_STAGE_ALIASES = {
    "alpha+gamma": "alpha_gamma",
    "gamma+cementite": "gamma_cementite",
    "alpha+pearlite": "alpha_pearlite",
    "pearlite+cementite": "pearlite_cementite",
    "liquid+gamma": "liquid_gamma",
    "liquid+alpha": "liquid_alpha",
    "alpha+beta": "alpha_beta",
    "primary_si+eutectic": "primary_si_eutectic",
    "medium_temper": "tempered_medium",
    "mid_temper": "tempered_medium",
    "high_temper": "tempered_high",
    "low_temper": "tempered_low",
    "bainitic": "bainite",
    "martensite_t": "martensite_tetragonal",
    "martensite_c": "martensite_cubic",
    "troostite_q": "troostite_quench",
    "troostite_t": "troostite_temper",
    "sorbite_q": "sorbite_quench",
    "sorbite_t": "sorbite_temper",
    "бейнит": "bainite",
    "upper_bainite": "bainite_upper",
    "lower_bainite": "bainite_lower",
    "верхний_бейнит": "bainite_upper",
    "нижний_бейнит": "bainite_lower",
    "bainite_u": "bainite_upper",
    "bainite_l": "bainite_lower",
    "white_cast_iron_hypo": "white_cast_iron_hypoeutectic",
    "white_cast_iron_eut": "white_cast_iron_eutectic",
    "white_cast_iron_hyper": "white_cast_iron_hypereutectic",
    "белый_чугун_доэвт": "white_cast_iron_hypoeutectic",
    "белый_чугун_эвт": "white_cast_iron_eutectic",
    "белый_чугун_заэвт": "white_cast_iron_hypereutectic",
    "hypoeutectic_white_cast_iron": "white_cast_iron_hypoeutectic",
    "eutectic_white_cast_iron": "white_cast_iron_eutectic",
    "hypereutectic_white_cast_iron": "white_cast_iron_hypereutectic",
}


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _linear_fraction(value: float, low: float, high: float) -> float:
    if high <= low + 1e-9:
        return 0.0
    return _clamp((float(value) - float(low)) / (float(high) - float(low)), 0.0, 1.0)


def _thermal_direction(thermal_slope: float | None) -> str:
    if thermal_slope is None:
        return "steady"
    slope = float(thermal_slope)
    if slope > 1e-9:
        return "heating"
    if slope < -1e-9:
        return "cooling"
    return "steady"


def normalize_system(system: str) -> str:
    key = system.strip().lower().replace(" ", "_")
    return _SYSTEM_ALIASES.get(key, system.strip().lower())


def normalize_stage(stage: str) -> str:
    key = stage.strip().lower().replace(" ", "_")
    return _STAGE_ALIASES.get(key, key)


def supported_stages(system: str) -> list[str]:
    return list(SYSTEM_STAGE_ORDER.get(normalize_system(system), []))


def _smooth_noise(noise: np.ndarray, sigma: float) -> np.ndarray:
    if ndimage is not None:
        return ndimage.gaussian_filter(noise, sigma=max(0.1, sigma))

    radius = max(1, int(round(sigma * 2)))
    out = noise.copy()
    for _ in range(radius):
        up = np.pad(out[:-1, :], ((1, 0), (0, 0)), mode="edge")
        down = np.pad(out[1:, :], ((0, 1), (0, 0)), mode="edge")
        left = np.pad(out[:, :-1], ((0, 0), (1, 0)), mode="edge")
        right = np.pad(out[:, 1:], ((0, 0), (0, 1)), mode="edge")
        out = (out + up + down + left + right) / 5.0
    return out


def _blend_images(images: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if not images:
        raise ValueError("images list must not be empty")
    if len(images) != len(weights):
        raise ValueError("images and weights length mismatch")

    total = sum(max(0.0, float(w)) for w in weights)
    norm = [max(0.0, float(w)) / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)
    canvas = np.zeros_like(images[0], dtype=np.float32)
    for image, w in zip(images, norm, strict=True):
        canvas += image.astype(np.float32) * float(w)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _liquid_texture(size: tuple[int, int], seed: int, brightness: int = 145) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = size
    yy, xx = np.mgrid[0:height, 0:width]
    low = rng.normal(0.0, 1.0, size=size).astype(np.float32)
    low = _smooth_noise(low, sigma=12.0)
    swirl = np.sin(xx / max(width, 1) * 6.0) + np.cos(yy / max(height, 1) * 4.5)
    image = brightness + low * 22.0 + swirl * 7.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _phase_mask(size: tuple[int, int], seed: int, fraction: float, sigma: float = 8.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.random(size, dtype=np.float32)
    field = _smooth_noise(noise, sigma=sigma)
    threshold = np.quantile(field, 1.0 - _clamp(fraction, 0.0, 1.0))
    return field >= threshold


def _dark_particles(
    image: np.ndarray,
    seed: int,
    count: int,
    radius_range: tuple[float, float],
    tone_range: tuple[int, int] = (20, 90),
    angular: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = image.shape
    pil = Image.fromarray(image, mode="L")
    draw = ImageDraw.Draw(pil)
    for _ in range(max(1, int(count))):
        cx = float(rng.uniform(0, width - 1))
        cy = float(rng.uniform(0, height - 1))
        r = float(rng.uniform(radius_range[0], radius_range[1]))
        tone = int(rng.integers(tone_range[0], tone_range[1] + 1))
        if angular:
            points = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0.0, 2.0 * np.pi, num=5, endpoint=False)]
            draw.polygon(points, fill=tone)
        else:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=tone)
    return np.asarray(pil, dtype=np.uint8)


def resolve_fe_c_stage(c_wt: float, temperature_c: float, cooling_mode: str, requested_stage: str) -> str:
    stage = normalize_stage(requested_stage)
    if stage and stage != "auto":
        return stage

    c = max(0.0, float(c_wt))
    temp = float(temperature_c)
    mode = cooling_mode.strip().lower()

    if mode in {"quench", "quenched", "water_quench", "oil_quench"} and temp <= 260.0:
        return "martensite"
    if mode in {"troostite_quench", "troostite"}:
        return "troostite_quench"
    if mode in {"sorbite_quench", "sorbite"}:
        return "sorbite_quench"
    if mode.startswith("bain"):
        # Explicit upper/lower bainite only if the cooling mode string
        # mentions it ("bainite_upper", "bainite_lower", "upper_bainite",
        # "lower_bainite"). Otherwise keep the legacy generic "bainite"
        # stage so existing presets render unchanged.
        if "upper" in mode:
            return "bainite_upper"
        if "lower" in mode or "нижний" in mode:
            return "bainite_lower"
        return "bainite"
    if mode.startswith("temper"):
        if temp <= 250.0:
            return "tempered_low"
        if temp <= 450.0:
            return "tempered_medium"
        return "tempered_high"

    liquidus = 1538.0 - 83.0 * min(c, 4.3)
    solidus = 1493.0 - 58.0 * c if c <= 2.14 else 1147.0 + (4.3 - min(c, 4.3)) * 38.0

    if temp >= liquidus:
        return "liquid"
    if temp >= solidus:
        return "liquid_gamma"
    if c < 0.10 and temp >= 1394.0:
        return "delta_ferrite"

    if c <= 0.77:
        a3 = 912.0 - 203.0 * np.sqrt(max(c, 0.0))
        if temp >= max(727.0, a3):
            return "austenite"
    else:
        acm = 727.0 + 160.0 * min(c - 0.77, 1.5)
        if temp >= max(727.0, acm):
            return "austenite" if c <= 0.77 else "gamma_cementite"

    if temp >= 727.0:
        if c < 0.77:
            return "alpha_gamma"
        if c <= 2.14:
            return "gamma_cementite"
        return "ledeburite"

    if c <= 0.022:
        return "ferrite"
    if c < 0.77:
        return "alpha_pearlite"
    if c <= 0.77:
        return "pearlite"
    if c <= 2.14:
        return "pearlite_cementite"
    # Cast iron (C > 2.14 %). We keep the legacy "ledeburite" stage as the
    # auto-resolved default so that existing presets (e.g. grey cast iron)
    # render unchanged. The new split — ``white_cast_iron_hypoeutectic`` /
    # ``white_cast_iron_eutectic`` / ``white_cast_iron_hypereutectic`` — is
    # **opt-in**: a preset must request it explicitly via ``requested_stage``.
    # The specialised render functions in ``fe_c_unified.py`` pick up the
    # new stage names through the aliases in ``_STAGE_ALIASES``.
    return "ledeburite"


def resolve_al_si_stage(si_wt: float, temperature_c: float, cooling_mode: str, requested_stage: str) -> str:
    stage = normalize_stage(requested_stage)
    if stage and stage != "auto":
        return stage

    si = max(0.0, float(si_wt))
    temp = float(temperature_c)
    mode = cooling_mode.strip().lower()
    if mode in {"quench", "quenched"} and temp <= 120.0:
        return "supersaturated"
    if mode.startswith("aged"):
        return "aged"
    if temp >= 660.0:
        return "liquid"
    if temp >= 577.0:
        return "liquid_alpha" if si < 12.6 else "liquid_si"
    if si < 7.0:
        return "alpha_eutectic"
    if si <= 13.0:
        return "eutectic"
    return "primary_si_eutectic"


def resolve_cu_zn_stage(
    zn_wt: float,
    temperature_c: float,
    cooling_mode: str,
    requested_stage: str,
    deformation_pct: float,
) -> str:
    stage = normalize_stage(requested_stage)
    if stage and stage != "auto":
        return stage

    zn = max(0.0, float(zn_wt))
    temp = float(temperature_c)
    mode = cooling_mode.strip().lower()
    if (mode.startswith("cold") or mode.startswith("deform")) and deformation_pct > 1.0:
        return "cold_worked"
    liquidus = _clamp(1085.0 - 3.0 * min(max(0.0, zn), 60.0), 900.0, 1085.0)
    solidus = _clamp(liquidus - 120.0, 760.0, 1000.0)

    if temp >= liquidus:
        return "liquid"
    if temp >= solidus:
        return "liquid_alpha" if zn < 46.0 else "liquid_beta"
    if zn < 35.0:
        return "alpha"
    if zn < 46.0:
        return "alpha_beta"
    if zn < 50.0:
        return "beta"
    return "beta_prime"


def resolve_al_cu_mg_stage(
    temperature_c: float,
    cooling_mode: str,
    requested_stage: str,
    aging_temperature_c: float,
    aging_hours: float,
) -> str:
    stage = normalize_stage(requested_stage)
    if stage and stage != "auto":
        return stage

    temp = float(temperature_c)
    mode = cooling_mode.strip().lower()
    age_temp = float(aging_temperature_c)
    age_hours = float(aging_hours)

    if temp >= 500.0 and mode in {"equilibrium", "solutionized"}:
        return "solutionized"
    if mode in {"quench", "quenched", "water_quench"}:
        return "quenched"
    if mode.startswith("natural"):
        return "natural_aged"
    if mode.startswith("overaged") or age_hours >= 20.0 or age_temp >= 220.0:
        return "overaged"
    if mode.startswith("aged") or age_hours >= 2.0:
        return "artificial_aged"
    return "quenched"


def resolve_fe_si_stage(
    temperature_c: float,
    cooling_mode: str,
    requested_stage: str,
    deformation_pct: float,
    si_wt: float = 0.0,
) -> str:
    stage = normalize_stage(requested_stage)
    if stage and stage != "auto":
        return stage

    temp = float(temperature_c)
    mode = cooling_mode.strip().lower()
    si = max(0.0, float(si_wt))
    liquidus = _clamp(1538.0 - 10.0 * si, 1470.0, 1538.0)
    solidus = _clamp(1493.0 - 15.0 * si, 1400.0, 1493.0)
    if temp >= liquidus:
        return "liquid"
    if temp >= solidus:
        return "liquid_ferrite"
    if temp >= 900.0:
        return "hot_ferrite"
    if mode.startswith("cold") or deformation_pct > 1.0:
        return "cold_worked_ferrite"
    return "recrystallized_ferrite"


def _generate_fe_c(
    size: tuple[int, int],
    seed: int,
    stage: str,
    composition: dict[str, float],
    liquid_fraction: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    c = float(composition.get("C", 0.0))

    if stage == "liquid":
        return _liquid_texture(size=size, seed=seed, brightness=140), {"L": 1.0}

    if stage == "liquid_gamma":
        liquid = _liquid_texture(size=size, seed=seed, brightness=142)
        gamma = generate_grain_structure(
            size=size,
            seed=seed + 11,
            mean_grain_size_px=72,
            grain_size_jitter=0.14,
            boundary_width_px=1,
            boundary_contrast=0.25,
        )["image"]
        lf = _clamp(0.62 if liquid_fraction is None else liquid_fraction, 0.05, 0.95)
        return _blend_images([liquid, gamma], [lf, 1.0 - lf]), {"L": lf, "gamma": 1.0 - lf}

    if stage == "delta_ferrite":
        ferrite = generate_grain_structure(
            size=size,
            seed=seed + 17,
            mean_grain_size_px=85,
            grain_size_jitter=0.12,
            boundary_width_px=2,
            boundary_contrast=0.35,
        )["image"]
        return ferrite, {"delta": 1.0}

    if stage == "austenite":
        gamma = generate_grain_structure(
            size=size,
            seed=seed + 23,
            mean_grain_size_px=56,
            grain_size_jitter=0.18,
            boundary_width_px=2,
            boundary_contrast=0.38,
        )["image"]
        return gamma, {"gamma": 1.0}

    if stage == "ferrite":
        ferrite = generate_grain_structure(
            size=size,
            seed=seed + 29,
            mean_grain_size_px=64,
            grain_size_jitter=0.2,
            boundary_width_px=2,
            boundary_contrast=0.42,
        )["image"]
        image = np.clip(ferrite.astype(np.float32) + 16.0, 0, 255).astype(np.uint8)
        return image, {"alpha": 1.0}

    if stage == "alpha_gamma":
        grains = generate_grain_structure(
            size=size,
            seed=seed + 31,
            mean_grain_size_px=52,
            grain_size_jitter=0.2,
            boundary_width_px=2,
            boundary_contrast=0.44,
        )["image"]
        alpha_fraction = _clamp((0.77 - c) / 0.77, 0.12, 0.88)
        mask = _phase_mask(size=size, seed=seed + 32, fraction=alpha_fraction, sigma=10.0)
        image = grains.astype(np.float32)
        image[mask] += 18.0
        image[~mask] -= 10.0
        return np.clip(image, 0, 255).astype(np.uint8), {"alpha": float(mask.mean()), "gamma": float((~mask).mean())}

    if stage == "gamma_cementite":
        gamma = generate_grain_structure(
            size=size,
            seed=seed + 41,
            mean_grain_size_px=46,
            grain_size_jitter=0.17,
            boundary_width_px=2,
            boundary_contrast=0.42,
        )["image"]
        cement = generate_pearlite_structure(
            size=size,
            seed=seed + 42,
            pearlite_fraction=0.36 + 0.2 * _clamp(c, 0.0, 1.5),
            lamella_period_px=5.2,
            colony_size_px=70.0,
            ferrite_brightness=170,
            cementite_brightness=60,
        )["image"]
        image = _blend_images([gamma, cement], [0.65, 0.35])
        return image, {"gamma": 0.68, "Fe3C": 0.32}

    if stage in {"alpha_pearlite", "pearlite", "pearlite_cementite"}:
        if stage == "alpha_pearlite":
            # Lever rule: P = (C - 0.02) / (0.77 - 0.02).  The old
            # Lever rule: P = (C - C_α) / (C_eut - C_α).
            # C_α = 0.022 wt% (max solubility of C in α-Fe at 727°C).
            # C_eut = 0.76 wt% (eutectoid composition).
            # At C=0.03%: ~1.1% pearlite; at C=0.10%: ~10.6%.
            pearlite_fraction = _clamp((c - 0.022) / 0.738, 0.0, 0.95)
        elif stage == "pearlite":
            pearlite_fraction = 0.96
        else:
            pearlite_fraction = 0.98

        base = generate_pearlite_structure(
            size=size,
            seed=seed + 51,
            pearlite_fraction=pearlite_fraction,
            lamella_period_px=_clamp(8.4 - 4.3 * c, 3.8, 8.4),
            colony_size_px=98.0,
            ferrite_brightness=180,
            cementite_brightness=58,
        )["image"]

        if stage == "pearlite_cementite":
            cementite_extra = generate_grain_structure(
                size=size,
                seed=seed + 52,
                mean_grain_size_px=84,
                grain_size_jitter=0.14,
                boundary_width_px=3,
                boundary_contrast=0.65,
            )["boundaries"]
            image = base.copy()
            image[cementite_extra] = np.clip(image[cementite_extra].astype(np.int16) - 35, 0, 255).astype(np.uint8)
            image = _dark_particles(
                image=image,
                seed=seed + 53,
                count=max(20, (size[0] * size[1]) // 48_000),
                radius_range=(2.0, 5.0),
                tone_range=(20, 55),
            )
            return image, {"pearlite": 0.84, "Fe3C": 0.16}

        if stage == "pearlite":
            return base, {"pearlite": 0.96, "alpha": 0.04}
        return base, {"alpha": float(1.0 - pearlite_fraction), "pearlite": float(pearlite_fraction)}

    if stage == "ledeburite":
        eut = generate_eutectic_al_si(
            size=size,
            seed=seed + 61,
            si_phase_fraction=0.56,
            eutectic_scale_px=5.8,
            morphology="network",
        )["image"]
        pearlite = generate_pearlite_structure(
            size=size,
            seed=seed + 62,
            pearlite_fraction=0.84,
            lamella_period_px=4.8,
            colony_size_px=86.0,
            ferrite_brightness=170,
            cementite_brightness=56,
        )["image"]
        image = _blend_images([eut, pearlite], [0.56, 0.44])
        return image, {"ledeburite": 0.62, "pearlite": 0.38}

    if stage == "martensite":
        mart = generate_martensite_structure(
            size=size,
            seed=seed + 71,
            needle_count=4200,
            needle_length_px=(8, 95),
            packet_spread_deg=15.0,
        )["image"]
        return mart, {"martensite": 1.0}

    if stage in {"martensite_tetragonal", "martensite_cubic"}:
        is_t = stage == "martensite_tetragonal"
        mart = generate_martensite_structure(
            size=size,
            seed=seed + (73 if is_t else 74),
            needle_count=(4600 if is_t else 3600),
            needle_length_px=((10, 110) if is_t else (8, 78)),
            packet_spread_deg=(13.0 if is_t else 18.0),
        )["image"]
        if is_t:
            mart = np.clip(mart.astype(np.float32) * 0.92 + 4.0, 0, 255).astype(np.uint8)
            return mart, {"martensite_t": 0.9, "cementite": 0.1}
        mart = np.clip(mart.astype(np.float32) * 1.05 + 8.0, 0, 255).astype(np.uint8)
        return mart, {"martensite_c": 0.94, "cementite": 0.06}

    if stage in {"troostite_quench", "troostite_temper"}:
        tr = generate_troostite_structure(
            size=size,
            seed=seed + (84 if stage == "troostite_quench" else 85),
            mode=("quench" if stage == "troostite_quench" else "temper"),
        )["image"]
        return tr, {"troostite": 0.88, "cementite": 0.12}

    if stage in {"sorbite_quench", "sorbite_temper"}:
        sb = generate_sorbite_structure(
            size=size,
            seed=seed + (86 if stage == "sorbite_quench" else 87),
            mode=("quench" if stage == "sorbite_quench" else "temper"),
        )["image"]
        return sb, {"sorbite": 0.84, "cementite": 0.16}

    if stage == "bainite":
        lower = generate_martensite_structure(
            size=size,
            seed=seed + 75,
            needle_count=2600,
            needle_length_px=(6, 62),
            packet_spread_deg=22.0,
        )["image"]
        carbides = generate_pearlite_structure(
            size=size,
            seed=seed + 76,
            pearlite_fraction=0.35,
            lamella_period_px=5.8,
            colony_size_px=68.0,
            ferrite_brightness=172,
            cementite_brightness=58,
        )["image"]
        image = _blend_images([lower, carbides], [0.7, 0.3])
        image = _dark_particles(
            image=image,
            seed=seed + 77,
            count=max(18, (size[0] * size[1]) // 55_000),
            radius_range=(1.8, 4.0),
            tone_range=(28, 70),
        )
        return image, {"bainite": 1.0}

    if stage in {"tempered_low", "tempered_medium", "tempered_high"}:
        temper_map = {"tempered_low": 200, "tempered_medium": 400, "tempered_high": 600}
        tempered = generate_tempered_steel_structure(
            size=size,
            seed=seed + 80,
            temper_temperature_c=temper_map[stage],
        )["image"]
        return tempered, {"tempered_matrix": 1.0}

    fallback = generate_pearlite_structure(size=size, seed=seed + 90)["image"]
    return fallback, {"unknown": 1.0}


def _generate_al_si(
    size: tuple[int, int],
    seed: int,
    stage: str,
    composition: dict[str, float],
    liquid_fraction: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    si = float(composition.get("Si", 7.0))

    if stage == "liquid":
        return _liquid_texture(size=size, seed=seed, brightness=155), {"L": 1.0}

    if stage == "liquid_alpha":
        liquid = _liquid_texture(size=size, seed=seed, brightness=158)
        alpha = generate_grain_structure(
            size=size,
            seed=seed + 101,
            mean_grain_size_px=72,
            grain_size_jitter=0.15,
            boundary_width_px=1,
            boundary_contrast=0.25,
        )["image"]
        lf = _clamp(0.65 if liquid_fraction is None else liquid_fraction, 0.05, 0.95)
        return _blend_images([liquid, alpha], [lf, 1.0 - lf]), {"L": lf, "alpha_Al": 1.0 - lf}

    if stage == "liquid_si":
        liquid = _liquid_texture(size=size, seed=seed, brightness=154)
        solid_si = _dark_particles(
            image=np.full(size, 138, dtype=np.uint8),
            seed=seed + 103,
            count=max(20, (size[0] * size[1]) // 35_000),
            radius_range=(2.0, 7.0),
            tone_range=(35, 90),
            angular=True,
        )
        lf = _clamp(0.62 if liquid_fraction is None else liquid_fraction, 0.05, 0.95)
        image = _blend_images([liquid, solid_si], [lf, 1.0 - lf])
        return image, {"L": lf, "Si": 1.0 - lf}

    if stage == "supersaturated":
        matrix = generate_grain_structure(
            size=size,
            seed=seed + 107,
            mean_grain_size_px=55,
            grain_size_jitter=0.2,
            boundary_width_px=2,
            boundary_contrast=0.34,
        )["image"]
        image = np.clip(matrix.astype(np.float32) + 10.0, 0, 255).astype(np.uint8)
        return image, {"alpha_sss": 1.0}

    if stage == "aged":
        cu = float(composition.get("Cu", 4.0))
        mg = float(composition.get("Mg", 1.2))
        precip = _clamp(0.04 + 0.014 * cu + 0.008 * mg, 0.06, 0.18)
        aged = generate_aged_aluminum_structure(
            size=size,
            seed=seed + 109,
            precipitate_fraction=precip,
            precipitate_scale_px=2.0,
        )["image"]
        return aged, {"alpha_Al": 0.88, "precipitates": 0.12}

    if stage in {"alpha_eutectic", "eutectic", "primary_si_eutectic"}:
        si_phase = _clamp(0.06 + 0.022 * si, 0.1, 0.72)
        morphology = "network" if stage == "alpha_eutectic" else ("needle" if stage == "primary_si_eutectic" else "branched")
        eut = generate_eutectic_al_si(
            size=size,
            seed=seed + 111,
            si_phase_fraction=si_phase,
            eutectic_scale_px=_clamp(9.0 - 0.26 * si, 4.0, 9.4),
            morphology=morphology,
        )["image"]
        if stage == "alpha_eutectic":
            alpha = generate_grain_structure(
                size=size,
                seed=seed + 112,
                mean_grain_size_px=76,
                grain_size_jitter=0.16,
                boundary_width_px=2,
                boundary_contrast=0.32,
            )["image"]
            image = _blend_images([alpha, eut], [0.58, 0.42])
            return image, {"alpha_Al": 0.58, "eutectic": 0.42}
        if stage == "primary_si_eutectic":
            image = _dark_particles(
                image=eut,
                seed=seed + 113,
                count=max(24, (size[0] * size[1]) // 34_000),
                radius_range=(3.0, 9.0),
                tone_range=(20, 55),
                angular=True,
            )
            return image, {"primary_Si": 0.25, "eutectic": 0.75}
        return eut, {"eutectic": 1.0}

    fallback = generate_eutectic_al_si(size=size, seed=seed + 118)["image"]
    return fallback, {"unknown": 1.0}


def _generate_cu_zn(
    size: tuple[int, int],
    seed: int,
    stage: str,
    composition: dict[str, float],
    deformation_pct: float,
    liquid_fraction: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    zn = float(composition.get("Zn", 32.0))

    if stage == "liquid":
        return _liquid_texture(size=size, seed=seed, brightness=150), {"L": 1.0}

    if stage in {"liquid_alpha", "liquid_beta"}:
        liquid = _liquid_texture(size=size, seed=seed, brightness=151)
        solid = generate_grain_structure(
            size=size,
            seed=seed + 200,
            mean_grain_size_px=48 if stage == "liquid_beta" else 58,
            grain_size_jitter=0.18,
            boundary_width_px=2,
            boundary_contrast=0.52,
        )["image"]
        if stage == "liquid_beta":
            solid = np.clip(solid.astype(np.float32) - 12.0, 0, 255).astype(np.uint8)
        lf = _clamp(0.58 if liquid_fraction is None else liquid_fraction, 0.05, 0.95)
        image = _blend_images([liquid, solid], [lf, 1.0 - lf])
        frac_name = "beta" if stage == "liquid_beta" else "alpha"
        return image, {"L": lf, frac_name: 1.0 - lf}

    if stage == "cold_worked":
        elong = _clamp(1.0 + deformation_pct / 35.0, 1.05, 3.0)
        grains = generate_grain_structure(
            size=size,
            seed=seed + 201,
            mean_grain_size_px=60,
            grain_size_jitter=0.16,
            equiaxed=0.8,
            elongation=elong,
            orientation_deg=8.0,
            boundary_width_px=2,
            boundary_contrast=0.56,
        )["image"]
        return grains, {"deformed_alpha": 1.0}

    if stage == "alpha":
        alpha = generate_grain_structure(
            size=size,
            seed=seed + 203,
            mean_grain_size_px=_clamp(56 + 0.22 * (zn - 32.0), 34, 85),
            grain_size_jitter=0.18,
            equiaxed=1.05,
            boundary_width_px=2,
            boundary_contrast=0.54,
        )["image"]
        return alpha, {"alpha": 1.0}

    if stage == "alpha_beta":
        base = generate_grain_structure(
            size=size,
            seed=seed + 206,
            mean_grain_size_px=52,
            grain_size_jitter=0.2,
            equiaxed=1.0,
            boundary_width_px=2,
            boundary_contrast=0.56,
        )["image"]
        beta_frac = _clamp((zn - 35.0) / 11.0, 0.12, 0.7)
        mask = _phase_mask(size=size, seed=seed + 207, fraction=beta_frac, sigma=8.5)
        image = base.astype(np.float32)
        image[mask] -= 26.0
        image[~mask] += 8.0
        return np.clip(image, 0, 255).astype(np.uint8), {"alpha": float((~mask).mean()), "beta": float(mask.mean())}

    if stage in {"beta", "beta_prime"}:
        beta = generate_grain_structure(
            size=size,
            seed=seed + 210,
            mean_grain_size_px=44,
            grain_size_jitter=0.15,
            equiaxed=0.95,
            boundary_width_px=2,
            boundary_contrast=0.62,
        )["image"]
        image = np.clip(beta.astype(np.float32) - 14.0, 0, 255).astype(np.uint8)
        if stage == "beta_prime":
            network = generate_eutectic_al_si(
                size=size,
                seed=seed + 211,
                si_phase_fraction=0.25,
                eutectic_scale_px=6.0,
                morphology="network",
            )["image"]
            image = _blend_images([image, network], [0.74, 0.26])
        return image, {"beta": 1.0}

    fallback = generate_grain_structure(size=size, seed=seed + 214)["image"]
    return fallback, {"unknown": 1.0}


def _generate_al_cu_mg(
    size: tuple[int, int],
    seed: int,
    stage: str,
    composition: dict[str, float],
    aging_hours: float,
) -> tuple[np.ndarray, dict[str, float]]:
    cu = float(composition.get("Cu", 4.4))
    mg = float(composition.get("Mg", 1.5))
    base_fraction = _clamp(0.03 + 0.012 * cu + 0.009 * mg, 0.05, 0.2)

    if stage == "solutionized":
        grains = generate_grain_structure(
            size=size,
            seed=seed + 301,
            mean_grain_size_px=58,
            grain_size_jitter=0.16,
            equiaxed=1.04,
            boundary_width_px=2,
            boundary_contrast=0.36,
        )["image"]
        return np.clip(grains.astype(np.float32) + 12.0, 0, 255).astype(np.uint8), {"alpha_sss": 1.0}

    if stage == "quenched":
        grains = generate_grain_structure(
            size=size,
            seed=seed + 303,
            mean_grain_size_px=54,
            grain_size_jitter=0.2,
            equiaxed=1.02,
            boundary_width_px=2,
            boundary_contrast=0.4,
        )["image"]
        noise = np.random.default_rng(seed + 304).normal(0.0, 4.0, size=size).astype(np.float32)
        return np.clip(grains.astype(np.float32) + noise, 0, 255).astype(np.uint8), {"alpha_sss": 1.0}

    if stage == "natural_aged":
        image = generate_aged_aluminum_structure(
            size=size,
            seed=seed + 305,
            precipitate_fraction=_clamp(base_fraction * 0.45, 0.03, 0.08),
            precipitate_scale_px=1.5,
        )["image"]
        return image, {"matrix": 0.95, "precipitates": 0.05}

    if stage == "artificial_aged":
        growth = 1.0 + _clamp(aging_hours / 12.0, 0.0, 1.2)
        image = generate_aged_aluminum_structure(
            size=size,
            seed=seed + 307,
            precipitate_fraction=_clamp(base_fraction * 0.9 * growth, 0.07, 0.16),
            precipitate_scale_px=2.0,
        )["image"]
        return image, {"matrix": 0.88, "precipitates": 0.12}

    if stage == "overaged":
        image = generate_aged_aluminum_structure(
            size=size,
            seed=seed + 309,
            precipitate_fraction=_clamp(base_fraction * 1.2, 0.11, 0.22),
            precipitate_scale_px=3.1,
        )["image"]
        if ndimage is not None:
            image = ndimage.gaussian_filter(image.astype(np.float32), sigma=0.9).clip(0, 255).astype(np.uint8)
        return image, {"matrix": 0.82, "coarse_precipitates": 0.18}

    fallback = generate_aged_aluminum_structure(size=size, seed=seed + 311)["image"]
    return fallback, {"unknown": 1.0}


def _generate_fe_si(
    size: tuple[int, int],
    seed: int,
    stage: str,
    composition: dict[str, float],
    deformation_pct: float,
    liquid_fraction: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    si = float(composition.get("Si", 1.4))

    if stage == "liquid":
        return _liquid_texture(size=size, seed=seed, brightness=146), {"L": 1.0}

    if stage == "liquid_ferrite":
        liquid = _liquid_texture(size=size, seed=seed, brightness=147)
        ferrite = generate_grain_structure(
            size=size,
            seed=seed + 400,
            mean_grain_size_px=80,
            grain_size_jitter=0.14,
            boundary_width_px=2,
            boundary_contrast=0.32,
        )["image"]
        lf = _clamp(0.55 if liquid_fraction is None else liquid_fraction, 0.05, 0.95)
        image = _blend_images([liquid, ferrite], [lf, 1.0 - lf])
        return image, {"L": lf, "ferrite": 1.0 - lf}

    if stage == "hot_ferrite":
        hot = generate_grain_structure(
            size=size,
            seed=seed + 401,
            mean_grain_size_px=_clamp(88 + 1.8 * si, 70, 110),
            grain_size_jitter=0.15,
            boundary_width_px=2,
            boundary_contrast=0.33,
        )["image"]
        return hot, {"ferrite": 1.0}

    if stage == "cold_worked_ferrite":
        elong = _clamp(1.0 + deformation_pct / 40.0, 1.05, 2.8)
        cold = generate_grain_structure(
            size=size,
            seed=seed + 402,
            mean_grain_size_px=62,
            grain_size_jitter=0.16,
            equiaxed=0.82,
            elongation=elong,
            orientation_deg=5.0,
            boundary_width_px=2,
            boundary_contrast=0.56,
        )["image"]
        return cold, {"deformed_ferrite": 1.0}

    recr = generate_grain_structure(
        size=size,
        seed=seed + 403,
        mean_grain_size_px=_clamp(52 + 2.0 * si, 36, 84),
        grain_size_jitter=0.14,
        boundary_width_px=2,
        boundary_contrast=0.46,
    )["image"]
    return recr, {"recrystallized_ferrite": 1.0}


def resolve_phase_transition_state(
    system: str,
    composition: dict[str, float] | None,
    processing: Any,
    thermal_slope: float | None = None,
    requested_stage: str = "auto",
    liquid_fraction_hint: float | None = None,
) -> dict[str, Any]:
    composition_map = {str(k): float(v) for k, v in (composition or {}).items()}
    system_name = normalize_system(system)
    if isinstance(processing, dict):
        temperature_c = float(processing.get("temperature_c", 20.0))
        cooling_mode = str(processing.get("cooling_mode", "equilibrium"))
        deformation_pct = float(processing.get("deformation_pct", 0.0))
        aging_temperature_c = float(processing.get("aging_temperature_c", 180.0))
        aging_hours = float(processing.get("aging_hours", 8.0))
    else:
        temperature_c = float(getattr(processing, "temperature_c", 20.0))
        cooling_mode = str(getattr(processing, "cooling_mode", "equilibrium"))
        deformation_pct = float(getattr(processing, "deformation_pct", 0.0))
        aging_temperature_c = float(getattr(processing, "aging_temperature_c", 180.0))
        aging_hours = float(getattr(processing, "aging_hours", 8.0))

    stage_input = requested_stage or "auto"
    stage_normalized = normalize_stage(stage_input)

    if system_name == "fe-c":
        c = float(composition_map.get("C", 0.0))
        resolved_stage = resolve_fe_c_stage(
            c_wt=c,
            temperature_c=temperature_c,
            cooling_mode=cooling_mode,
            requested_stage=stage_input,
        )
        liquidus = 1538.0 - 83.0 * min(c, 4.3)
        solidus = 1493.0 - 58.0 * c if c <= 2.1 else 1147.0 + (4.3 - min(c, 4.3)) * 38.0
        default_liquid = 1.0 if resolved_stage == "liquid" else (_linear_fraction(temperature_c, solidus, liquidus) if resolved_stage == "liquid_gamma" else 0.0)
    elif system_name == "al-si":
        si = float(composition_map.get("Si", 0.0))
        resolved_stage = resolve_al_si_stage(
            si_wt=si,
            temperature_c=temperature_c,
            cooling_mode=cooling_mode,
            requested_stage=stage_input,
        )
        if si <= 12.6:
            liquidus = 660.0 - (660.0 - 577.0) * _clamp(si / 12.6, 0.0, 1.0)
        else:
            liquidus = 577.0 + (700.0 - 577.0) * _clamp((si - 12.6) / (25.0 - 12.6), 0.0, 1.0)
        solidus = 577.0
        default_liquid = 1.0 if resolved_stage == "liquid" else (_linear_fraction(temperature_c, solidus, liquidus) if resolved_stage in {"liquid_alpha", "liquid_si"} else 0.0)
    elif system_name == "cu-zn":
        zn = float(composition_map.get("Zn", 0.0))
        resolved_stage = resolve_cu_zn_stage(
            zn_wt=zn,
            temperature_c=temperature_c,
            cooling_mode=cooling_mode,
            requested_stage=stage_input,
            deformation_pct=deformation_pct,
        )
        liquidus = _clamp(1085.0 - 3.0 * min(max(0.0, zn), 60.0), 900.0, 1085.0)
        solidus = _clamp(liquidus - 120.0, 760.0, 1000.0)
        default_liquid = 1.0 if resolved_stage == "liquid" else (_linear_fraction(temperature_c, solidus, liquidus) if resolved_stage in {"liquid_alpha", "liquid_beta"} else 0.0)
    elif system_name == "fe-si":
        si = float(composition_map.get("Si", 0.0))
        resolved_stage = resolve_fe_si_stage(
            temperature_c=temperature_c,
            cooling_mode=cooling_mode,
            requested_stage=stage_input,
            deformation_pct=deformation_pct,
            si_wt=si,
        )
        liquidus = _clamp(1538.0 - 10.0 * min(max(0.0, si), 6.0), 1470.0, 1538.0)
        solidus = _clamp(1493.0 - 15.0 * min(max(0.0, si), 6.0), 1400.0, 1493.0)
        default_liquid = 1.0 if resolved_stage == "liquid" else (_linear_fraction(temperature_c, solidus, liquidus) if resolved_stage == "liquid_ferrite" else 0.0)
    elif system_name == "al-cu-mg":
        resolved_stage = resolve_al_cu_mg_stage(
            temperature_c=temperature_c,
            cooling_mode=cooling_mode,
            requested_stage=stage_input,
            aging_temperature_c=aging_temperature_c,
            aging_hours=aging_hours,
        )
        default_liquid = 0.0
    else:
        resolved_stage = stage_normalized if stage_normalized and stage_normalized != "auto" else "unknown"
        default_liquid = 0.0

    liquid_fraction = float(default_liquid)
    if liquid_fraction_hint is not None:
        liquid_fraction = _clamp(float(liquid_fraction_hint), 0.0, 1.0)
    solid_fraction = _clamp(1.0 - liquid_fraction, 0.0, 1.0)
    direction = _thermal_direction(thermal_slope)
    transition_kind = "none"
    if 0.0 < liquid_fraction < 1.0:
        if direction == "heating":
            transition_kind = "melting"
        elif direction == "cooling":
            transition_kind = "crystallization"

    return {
        "stage": resolved_stage,
        "transition_kind": transition_kind,
        "liquid_fraction": float(liquid_fraction),
        "solid_fraction": float(solid_fraction),
        "thermal_direction": direction,
    }


def generate_phase_stage_structure(
    size: tuple[int, int],
    seed: int,
    system: str = "fe-c",
    composition: dict[str, float] | None = None,
    stage: str = "auto",
    temperature_c: float = 20.0,
    cooling_mode: str = "equilibrium",
    deformation_pct: float = 0.0,
    aging_temperature_c: float = 180.0,
    aging_hours: float = 8.0,
    thermal_slope: float | None = None,
    liquid_fraction: float | None = None,
) -> dict[str, Any]:
    """
    Generate educational phase-stage microstructures for multiple alloy systems.
    """

    composition_map = {str(k): float(v) for k, v in (composition or {}).items()}
    system_name = normalize_system(system)
    requested_stage = stage or "auto"
    transition_state = resolve_phase_transition_state(
        system=system_name,
        composition=composition_map,
        processing={
            "temperature_c": float(temperature_c),
            "cooling_mode": str(cooling_mode),
            "deformation_pct": float(deformation_pct),
            "aging_temperature_c": float(aging_temperature_c),
            "aging_hours": float(aging_hours),
        },
        thermal_slope=thermal_slope,
        requested_stage=requested_stage,
        liquid_fraction_hint=liquid_fraction,
    )
    resolved = str(transition_state["stage"])
    liquid_fraction_value = float(transition_state["liquid_fraction"])

    if system_name == "fe-c":
        image, fractions = _generate_fe_c(
            size=size,
            seed=seed,
            stage=resolved,
            composition=composition_map,
            liquid_fraction=liquid_fraction_value if resolved == "liquid_gamma" else None,
        )
    elif system_name == "al-si":
        image, fractions = _generate_al_si(
            size=size,
            seed=seed,
            stage=resolved,
            composition=composition_map,
            liquid_fraction=liquid_fraction_value if resolved in {"liquid_alpha", "liquid_si"} else None,
        )
    elif system_name == "cu-zn":
        image, fractions = _generate_cu_zn(
            size=size,
            seed=seed,
            stage=resolved,
            composition=composition_map,
            deformation_pct=float(deformation_pct),
            liquid_fraction=liquid_fraction_value if resolved in {"liquid_alpha", "liquid_beta"} else None,
        )
    elif system_name == "al-cu-mg":
        image, fractions = _generate_al_cu_mg(
            size=size,
            seed=seed,
            stage=resolved,
            composition=composition_map,
            aging_hours=aging_hours,
        )
    elif system_name == "fe-si":
        image, fractions = _generate_fe_si(
            size=size,
            seed=seed,
            stage=resolved,
            composition=composition_map,
            deformation_pct=float(deformation_pct),
            liquid_fraction=liquid_fraction_value if resolved == "liquid_ferrite" else None,
        )
    else:
        raise ValueError(f"Unsupported phase system: {system}")

    fractions_out = {str(k): float(v) for k, v in fractions.items()}
    if "L" not in fractions_out:
        fractions_out["L"] = float(liquid_fraction_value)
    if "solid" not in fractions_out:
        fractions_out["solid"] = float(_clamp(1.0 - liquid_fraction_value, 0.0, 1.0))

    phase_masks: dict[str, np.ndarray] | None = None
    if liquid_fraction_value >= 0.999:
        phase_masks = {
            "L": np.ones(size, dtype=np.uint8),
            "solid": np.zeros(size, dtype=np.uint8),
        }
    elif 0.001 < liquid_fraction_value < 0.999:
        mask_liquid = _phase_mask(size=size, seed=seed + 9_901, fraction=liquid_fraction_value, sigma=8.6)
        phase_masks = {
            "L": mask_liquid.astype(np.uint8),
            "solid": (~mask_liquid).astype(np.uint8),
        }

    return {
        "image": image.astype(np.uint8, copy=False),
        "phase_masks": phase_masks,
        "metadata": {
            "system": system_name,
            "requested_stage": normalize_stage(requested_stage),
            "resolved_stage": resolved,
            "temperature_c": float(temperature_c),
            "cooling_mode": cooling_mode,
            "deformation_pct": float(deformation_pct),
            "aging_temperature_c": float(aging_temperature_c),
            "aging_hours": float(aging_hours),
            "phase_fractions": fractions_out,
            "phase_transition_state": transition_state,
            "supported_stages": supported_stages(system_name),
        },
    }
