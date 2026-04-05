from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


def _smooth_noise(noise: np.ndarray, sigma: float) -> np.ndarray:
    if ndimage is not None:
        return ndimage.gaussian_filter(noise, sigma=max(0.1, sigma))

    # Simple fallback blur without scipy.
    radius = max(1, int(round(sigma * 2)))
    out = noise.copy()
    for _ in range(radius):
        up = np.pad(out[:-1, :], ((1, 0), (0, 0)), mode="edge")
        down = np.pad(out[1:, :], ((0, 1), (0, 0)), mode="edge")
        left = np.pad(out[:, :-1], ((0, 0), (1, 0)), mode="edge")
        right = np.pad(out[:, 1:], ((0, 0), (0, 1)), mode="edge")
        out = (out + up + down + left + right) / 5.0
    return out


def generate_eutectic_al_si(
    size: tuple[int, int],
    seed: int,
    si_phase_fraction: float = 0.32,
    eutectic_scale_px: float = 7.5,
    morphology: str = "branched",
) -> dict[str, Any]:
    """Generate educational Al-Si eutectic-like microstructure."""

    rng = np.random.default_rng(seed)
    height, width = size
    yy, xx = np.mgrid[0:height, 0:width]

    base_noise = rng.random((height, width), dtype=np.float32)
    smooth = _smooth_noise(base_noise, sigma=max(1.0, eutectic_scale_px * 0.35))

    pitch = max(2.0, float(eutectic_scale_px))
    orientations = [0.0, math.pi / 3.0, 2.0 * math.pi / 3.0]
    directional = np.zeros_like(smooth)
    for angle in orientations:
        wave = np.sin((xx * math.cos(angle) + yy * math.sin(angle)) * (2.0 * math.pi / pitch))
        directional += wave
    directional /= len(orientations)

    if morphology == "needle":
        field = 0.3 * smooth + 0.7 * np.abs(directional)
    elif morphology == "fibrous":
        branch = 0.5 * smooth + 0.5 * (directional > 0).astype(np.float32)
        field = 0.72 * _smooth_noise(branch, sigma=max(0.8, eutectic_scale_px * 0.28)) + 0.28 * smooth
    elif morphology == "network":
        field = 0.55 * smooth + 0.45 * (directional > 0).astype(np.float32)
    else:  # branched default
        field = 0.45 * smooth + 0.55 * np.abs(directional)

    frac = float(np.clip(si_phase_fraction, 0.02, 0.9))
    threshold = np.quantile(field, 1.0 - frac)
    si_mask = field >= threshold

    # Build the grayscale image: matrix Al is brighter, Si is darker.
    image = np.full((height, width), 178, dtype=np.uint8)
    image[si_mask] = 72

    if ndimage is not None:
        edges = ndimage.binary_dilation(si_mask, iterations=1) ^ si_mask
        image[edges] = 96
        image = ndimage.gaussian_filter(image.astype(np.float32), sigma=0.65).clip(0, 255).astype(np.uint8)

    return {
        "image": image,
        "phase_mask": si_mask.astype(np.uint8),
        "metadata": {
            "si_phase_fraction": float(si_mask.mean()),
            "morphology": morphology,
            "eutectic_scale_px": float(eutectic_scale_px),
        },
    }


def generate_aged_aluminum_structure(
    size: tuple[int, int],
    seed: int,
    precipitate_fraction: float = 0.08,
    precipitate_scale_px: float = 2.2,
) -> dict[str, Any]:
    """Generate educational quenched/aged Al alloy texture."""

    rng = np.random.default_rng(seed)
    height, width = size
    matrix = np.full((height, width), 175, dtype=np.float32)
    noise = rng.normal(0.0, 8.0, size=(height, width)).astype(np.float32)
    field = _smooth_noise(noise, sigma=5.0)
    matrix += field

    spots = rng.random((height, width), dtype=np.float32)
    smooth_spots = _smooth_noise(spots, sigma=max(0.8, precipitate_scale_px))
    threshold = np.quantile(smooth_spots, 1.0 - np.clip(precipitate_fraction, 0.01, 0.4))
    precipitates = smooth_spots > threshold
    matrix[precipitates] = 85

    if ndimage is not None:
        matrix = ndimage.gaussian_filter(matrix, sigma=0.7)

    image = np.clip(matrix, 0, 255).astype(np.uint8)
    return {
        "image": image,
        "phase_mask": precipitates.astype(np.uint8),
        "metadata": {
            "precipitate_fraction": float(precipitates.mean()),
            "precipitate_scale_px": float(precipitate_scale_px),
        },
    }
