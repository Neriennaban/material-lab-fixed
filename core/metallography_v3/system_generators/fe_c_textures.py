from __future__ import annotations

from typing import Callable

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from core.generator_eutectic import generate_eutectic_al_si
from core.generator_grains import generate_grain_structure
from core.generator_pearlite import (
    generate_martensite_structure,
    generate_pearlite_structure,
    generate_sorbite_structure,
    generate_troostite_structure,
)


def _ensure_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image.astype(np.float32), 0.0, 255.0).astype(np.uint8)


def _smooth_noise(seed: int, size: tuple[int, int], sigma: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, 1.0, size=size).astype(np.float32)
    if ndimage is not None:
        return ndimage.gaussian_filter(noise, sigma=max(0.2, float(sigma)))
    return noise


def _blend(images: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if not images:
        raise ValueError("images list is empty")
    if len(images) != len(weights):
        raise ValueError("images and weights size mismatch")
    total = float(sum(max(0.0, float(w)) for w in weights))
    if total <= 1e-9:
        norm = [1.0 / float(len(images))] * len(images)
    else:
        norm = [max(0.0, float(w)) / total for w in weights]
    canvas = np.zeros_like(images[0], dtype=np.float32)
    for image, weight in zip(images, norm, strict=True):
        canvas += image.astype(np.float32) * float(weight)
    return _ensure_u8(canvas)


def texture_liquid(size: tuple[int, int], seed: int) -> np.ndarray:
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    low = _smooth_noise(seed + 3, size=size, sigma=11.0)
    mid = _smooth_noise(seed + 7, size=size, sigma=4.0)
    swirl = np.sin(xx / max(1, w) * 6.2) + np.cos(yy / max(1, h) * 4.4)
    image = 146.0 + low * 19.0 + mid * 7.0 + swirl * 6.0
    return _ensure_u8(image)


def texture_ferrite(size: tuple[int, int], seed: int) -> np.ndarray:
    grain_data = generate_grain_structure(
        size=size,
        seed=int(seed) + 11,
        mean_grain_size_px=74.0,
        grain_size_jitter=0.16,
        boundary_width_px=1,
        boundary_contrast=0.16,
    )
    labels = grain_data["labels"]
    # Nital etching is orientation-dependent: each grain gets a random
    # base tone in the 200-230 range (±15 units variation between grains).
    rng = np.random.default_rng(int(seed) + 12)
    n_grains = int(labels.max()) + 1
    grain_tones = rng.uniform(200.0, 230.0, size=n_grains).astype(np.float32)
    ferrite = grain_tones[labels]
    # Add subtle intra-grain noise for surface roughness
    ferrite += _smooth_noise(seed + 19, size=size, sigma=6.0) * 3.5
    # Darken grain boundaries
    boundaries = grain_data.get("boundaries")
    if boundaries is not None:
        ferrite[boundaries] -= 45.0
    if ndimage is not None:
        ferrite = ndimage.gaussian_filter(ferrite, sigma=0.35)
    return _ensure_u8(ferrite)


def texture_delta_ferrite(size: tuple[int, int], seed: int) -> np.ndarray:
    ferrite = generate_grain_structure(
        size=size,
        seed=int(seed) + 13,
        mean_grain_size_px=92.0,
        grain_size_jitter=0.12,
        boundary_width_px=1,
        boundary_contrast=0.14,
    )["image"].astype(np.float32)
    ferrite = ferrite * 0.8 + 50.0
    ferrite += _smooth_noise(seed + 21, size=size, sigma=7.0) * 2.2
    if ndimage is not None:
        ferrite = ndimage.gaussian_filter(ferrite, sigma=0.4)
    return _ensure_u8(ferrite)


def texture_austenite(size: tuple[int, int], seed: int) -> np.ndarray:
    gamma = generate_grain_structure(
        size=size,
        seed=int(seed) + 17,
        mean_grain_size_px=62.0,
        grain_size_jitter=0.18,
        boundary_width_px=1,
        boundary_contrast=0.18,
    )["image"].astype(np.float32)
    gamma = gamma * 0.8 + 34.0
    gamma += _smooth_noise(seed + 23, size=size, sigma=5.5) * 3.0
    if ndimage is not None:
        gamma = ndimage.gaussian_filter(gamma, sigma=0.35)
    return _ensure_u8(gamma)


def texture_pearlite_colonies(size: tuple[int, int], seed: int) -> np.ndarray:
    image = generate_pearlite_structure(
        size=size,
        seed=int(seed) + 23,
        pearlite_fraction=0.92,
        lamella_period_px=5.6,
        colony_size_px=92.0,
        ferrite_brightness=95,
        cementite_brightness=175,
    )["image"].astype(np.float32)
    if ndimage is not None:
        image = ndimage.gaussian_filter(image, sigma=0.55)
    # Average pearlite brightness should be ~85-105 (dark gray, unresolved look)
    image = image * 0.72 + 8.0
    return _ensure_u8(image)


def texture_cementite_network(size: tuple[int, int], seed: int) -> np.ndarray:
    # Cementite is NOT attacked by nital → brightest phase (~235-250).
    # The network forms at prior austenite grain boundaries in hypereutectoid steel.
    base = np.full(size, 235.0, dtype=np.float32)
    base += _smooth_noise(seed + 31, size=size, sigma=3.2) * 5.0
    grains = generate_grain_structure(
        size=size,
        seed=int(seed) + 37,
        mean_grain_size_px=84.0,
        grain_size_jitter=0.15,
        boundary_width_px=2,
        boundary_contrast=0.62,
    )
    boundaries = grains["boundaries"]
    net = base.copy()
    # Cementite at grain boundaries is slightly brighter (thicker, flatter)
    net[boundaries] = np.clip(net[boundaries] + 12.0, 0.0, 255.0)
    blob = _smooth_noise(seed + 41, size=size, sigma=2.6)
    bright_mask = blob > np.quantile(blob, 0.97)
    net[bright_mask] = np.clip(net[bright_mask] + 8.0, 0.0, 255.0)
    if ndimage is not None:
        net = ndimage.gaussian_filter(net, sigma=0.55)
    return _ensure_u8(net)


def texture_martensite_tetragonal(size: tuple[int, int], seed: int) -> np.ndarray:
    mart = generate_martensite_structure(
        size=size,
        seed=int(seed) + 43,
        needle_count=4600,
        needle_length_px=(10, 108),
        packet_spread_deg=12.0,
    )["image"].astype(np.float32)
    return _ensure_u8(mart * 0.9 + 8.0)


def texture_martensite_cubic(size: tuple[int, int], seed: int) -> np.ndarray:
    mart = generate_martensite_structure(
        size=size,
        seed=int(seed) + 47,
        needle_count=3600,
        needle_length_px=(8, 76),
        packet_spread_deg=18.0,
    )["image"].astype(np.float32)
    return _ensure_u8(mart * 1.03 + 7.0)


def texture_troostite_quench(size: tuple[int, int], seed: int) -> np.ndarray:
    return _ensure_u8(generate_troostite_structure(size=size, seed=int(seed) + 53, mode="quench")["image"])


def texture_troostite_temper(size: tuple[int, int], seed: int) -> np.ndarray:
    return _ensure_u8(generate_troostite_structure(size=size, seed=int(seed) + 59, mode="temper")["image"])


def texture_sorbite_quench(size: tuple[int, int], seed: int) -> np.ndarray:
    return _ensure_u8(
        generate_sorbite_structure(
            size=size,
            seed=int(seed) + 61,
            mode="quench",
            include_particles=False,
        )["image"]
    )


def texture_sorbite_temper(size: tuple[int, int], seed: int) -> np.ndarray:
    return _ensure_u8(
        generate_sorbite_structure(
            size=size,
            seed=int(seed) + 67,
            mode="temper",
            include_particles=False,
        )["image"]
    )


def texture_bainite(size: tuple[int, int], seed: int) -> np.ndarray:
    troostite = texture_troostite_quench(size=size, seed=seed + 71)
    mart = texture_martensite_cubic(size=size, seed=seed + 73)
    return _blend([troostite, mart], [0.72, 0.28])


def texture_ledeburite(size: tuple[int, int], seed: int) -> np.ndarray:
    eut = generate_eutectic_al_si(
        size=size,
        seed=int(seed) + 79,
        si_phase_fraction=0.55,
        eutectic_scale_px=5.6,
        morphology="network",
    )["image"]
    pearlite = texture_pearlite_colonies(size=size, seed=seed + 83)
    return _blend([_ensure_u8(eut), pearlite], [0.58, 0.42])


def texture_tempered_high(size: tuple[int, int], seed: int) -> np.ndarray:
    sorbite = texture_sorbite_temper(size=size, seed=seed + 89)
    ferrite = texture_ferrite(size=size, seed=seed + 97)
    return _blend([sorbite, ferrite], [0.76, 0.24])


def fe_c_texture_map() -> dict[str, Callable[[tuple[int, int], int], np.ndarray]]:
    return {
        "LIQUID": texture_liquid,
        "FERRITE": texture_ferrite,
        "DELTA_FERRITE": texture_delta_ferrite,
        "AUSTENITE": texture_austenite,
        "PEARLITE": texture_pearlite_colonies,
        "CEMENTITE": texture_cementite_network,
        "MARTENSITE": texture_martensite_cubic,
        "MARTENSITE_TETRAGONAL": texture_martensite_tetragonal,
        "MARTENSITE_CUBIC": texture_martensite_cubic,
        "TROOSTITE": texture_troostite_temper,
        "SORBITE": texture_sorbite_temper,
        "BAINITE": texture_bainite,
        "LEDEBURITE": texture_ledeburite,
        "TEMPERED_MATRIX": texture_tempered_high,
    }
