from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from .voronoi import generate_voronoi_labels


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _smooth(field: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(0.05, float(sigma))
    if ndimage is not None:
        return ndimage.gaussian_filter(field, sigma=sigma)

    radius = max(1, int(round(2.0 * sigma)))
    out = field.copy()
    for _ in range(radius):
        up = np.pad(out[:-1, :], ((1, 0), (0, 0)), mode="edge")
        down = np.pad(out[1:, :], ((0, 1), (0, 0)), mode="edge")
        left = np.pad(out[:, :-1], ((0, 0), (1, 0)), mode="edge")
        right = np.pad(out[:, 1:], ((0, 0), (0, 1)), mode="edge")
        out = (out + up + down + left + right) / 5.0
    return out


def _metric_from_orientation(orientation_deg: float, elongation: float) -> np.ndarray:
    theta = math.radians(float(orientation_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    stretch = np.array([[1.0 / max(0.25, elongation), 0.0], [0.0, max(0.25, elongation)]], dtype=np.float32)
    return rot @ stretch @ rot.T


def _build_seed_points(size: tuple[int, int], spacing: float, rng: np.random.Generator) -> np.ndarray:
    height, width = size
    area = float(height * width)
    expected = max(12, int(area / max(60.0, spacing * spacing * 0.72)))
    pts = np.column_stack(
        (
            rng.uniform(0, height - 1, size=expected),
            rng.uniform(0, width - 1, size=expected),
        )
    ).astype(np.float32)
    return pts


def _boundaries(labels: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(labels, dtype=bool)
    mask[:-1, :] |= labels[:-1, :] != labels[1:, :]
    mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return mask


def _fraction_mask(field: np.ndarray, fraction: float) -> np.ndarray:
    frac = _clamp(fraction, 0.0, 1.0)
    if frac <= 0.0:
        return np.zeros_like(field, dtype=bool)
    if frac >= 1.0:
        return np.ones_like(field, dtype=bool)
    threshold = np.quantile(field, frac)
    return field <= threshold


def generate_dendritic_cast(
    size: tuple[int, int],
    seed: int,
    cooling_rate: float = 45.0,
    thermal_gradient: float = 0.65,
    undercooling: float = 35.0,
    primary_arm_spacing: float = 34.0,
    secondary_arm_factor: float = 0.42,
    interdendritic_fraction: float = 0.32,
    porosity_fraction: float = 0.006,
    gradient_angle_deg: float = 20.0,
) -> dict[str, Any]:
    """
    Deterministic pseudo-physical dendritic casting texture generator.

    The model is educational: it combines anisotropic Voronoi domains (grains),
    oriented arm growth fields, and inter-dendritic segregation.
    """

    rng = np.random.default_rng(int(seed))
    height, width = size

    spacing = _clamp(float(primary_arm_spacing), 8.0, 180.0)
    thermal = _clamp(float(thermal_gradient), 0.0, 1.8)
    cooling = _clamp(float(cooling_rate), 0.5, 300.0)
    under = _clamp(float(undercooling), 0.0, 220.0)
    secondary_factor = _clamp(float(secondary_arm_factor), 0.1, 1.2)
    inter_frac = _clamp(float(interdendritic_fraction), 0.02, 0.85)
    por_frac = _clamp(float(porosity_fraction), 0.0, 0.25)

    anisotropy = _clamp(1.0 + thermal * 1.2 + 0.0035 * cooling, 1.0, 4.0)
    metric = _metric_from_orientation(float(gradient_angle_deg), anisotropy)
    seeds = _build_seed_points(size=size, spacing=spacing, rng=rng)
    labels = generate_voronoi_labels(size=size, points=seeds, metric_matrix=metric)
    grain_count = int(labels.max()) + 1

    # Grain-level dendrite orientation tracks thermal gradient with stochastic spread.
    base_angle = math.radians(float(gradient_angle_deg))
    spread = math.radians(_clamp(20.0 - thermal * 6.0, 4.0, 25.0))
    grain_theta = rng.normal(loc=base_angle, scale=spread, size=grain_count).astype(np.float32)
    grain_phase = rng.uniform(0.0, 2.0 * math.pi, size=grain_count).astype(np.float32)
    seed_y = seeds[:, 0]
    seed_x = seeds[:, 1]

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    theta_map = grain_theta[labels]
    phase_map = grain_phase[labels]
    sx = seed_x[labels]
    sy = seed_y[labels]
    dx = xx - sx
    dy = yy - sy

    local_s = dx * np.cos(theta_map) + dy * np.sin(theta_map)
    local_t = -dx * np.sin(theta_map) + dy * np.cos(theta_map)

    primary_period = spacing * _clamp(1.1 - 0.0025 * under + 0.0018 * cooling, 0.55, 1.4)
    sec_period = max(2.5, primary_period * secondary_factor)
    complexity = _clamp(0.55 + 0.006 * under + 0.002 * cooling, 0.55, 2.0)

    trunk = np.exp(-((np.abs(local_t) / max(1.0, 0.24 * spacing)) ** 1.35))
    primary_wave = 0.5 + 0.5 * np.sin((2.0 * math.pi / max(primary_period, 2.0)) * local_s + phase_map)
    secondary_wave = 0.5 + 0.5 * np.cos((2.0 * math.pi / sec_period) * (local_t + 0.4 * local_s))
    secondary_env = np.exp(-((np.abs(local_t) / max(1.0, 0.65 * spacing)) ** 2.0))

    dendrite_field = trunk * primary_wave * (0.72 + 0.28 * secondary_wave * secondary_env * complexity)
    dendrite_field = dendrite_field.astype(np.float32)

    low_noise = _smooth(rng.normal(0.0, 1.0, size=size).astype(np.float32), sigma=max(2.0, spacing * 0.24))
    low_noise = (low_noise - low_noise.min()) / max(1e-6, low_noise.max() - low_noise.min())
    dendrite_field = 0.82 * dendrite_field + 0.18 * low_noise
    dendrite_field = np.clip(dendrite_field, 0.0, 1.0)

    inter_mask = _fraction_mask(dendrite_field, inter_frac)
    boundary_mask = _boundaries(labels)

    seg_noise = _smooth(rng.normal(0.0, 1.0, size=size).astype(np.float32), sigma=max(1.2, spacing * 0.18))
    seg_noise = (seg_noise - seg_noise.min()) / max(1e-6, seg_noise.max() - seg_noise.min())

    gray = (132.0 + 62.0 * dendrite_field).astype(np.float32)
    gray[inter_mask] = (98.0 + 45.0 * seg_noise[inter_mask]).astype(np.float32)
    gray[boundary_mask] -= 16.0

    porosity_mask = np.zeros(size, dtype=bool)
    if por_frac > 0.0 and np.any(inter_mask):
        pore_field = _smooth(rng.random(size, dtype=np.float32), sigma=max(0.8, spacing * 0.11))
        pore_field = (pore_field - pore_field.min()) / max(1e-6, pore_field.max() - pore_field.min())
        target_in_inter = _clamp(por_frac / max(float(inter_mask.mean()), 1e-9), 0.0, 1.0)
        threshold = np.quantile(pore_field[inter_mask], 1.0 - target_in_inter)
        porosity_mask = inter_mask & (pore_field >= threshold)
        gray[porosity_mask] = rng.uniform(20.0, 58.0, size=int(porosity_mask.sum())).astype(np.float32)

    if ndimage is not None:
        gray = ndimage.gaussian_filter(gray, sigma=0.65)

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=2).astype(np.float32)
    # Slight warm tint for dendrite cores and cooler tint for inter-dendritic channels.
    rgb[..., 0] *= 1.03
    rgb[..., 2] *= 0.96
    rgb[inter_mask, 0] *= 0.92
    rgb[inter_mask, 1] *= 0.98
    rgb[inter_mask, 2] *= 1.06
    rgb[porosity_mask] *= 0.72
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    phase_masks = {
        "dendrite_core": (~inter_mask).astype(np.uint8),
        "interdendritic": inter_mask.astype(np.uint8),
        "porosity": porosity_mask.astype(np.uint8),
    }

    return {
        "image": gray_u8,
        "image_gray": gray_u8,
        "image_rgb": rgb_u8,
        "labels": labels,
        "phase_masks": phase_masks,
        "metadata": {
            "generator": "dendritic_cast",
            "grain_count": grain_count,
            "cooling_rate": cooling,
            "thermal_gradient": thermal,
            "undercooling": under,
            "primary_arm_spacing": spacing,
            "secondary_arm_factor": secondary_factor,
            "interdendritic_fraction": float(inter_mask.mean()),
            "porosity_fraction": float(porosity_mask.mean()),
            "gradient_angle_deg": float(gradient_angle_deg),
        },
    }
