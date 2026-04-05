from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover - fallback path
    ndimage = None

from .voronoi import generate_voronoi_labels


def _metric_from_elongation(elongation: float, orientation_deg: float) -> np.ndarray:
    elong = max(0.25, float(elongation))
    theta = math.radians(float(orientation_deg))
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float32,
    )
    scale = np.array([[1.0 / elong, 0.0], [0.0, elong]], dtype=np.float32)
    return rot @ scale @ rot.T


def _draw_circular_defects(
    image: np.ndarray,
    rng: np.random.Generator,
    fraction: float,
    radius_px: tuple[float, float],
    value_range: tuple[int, int],
) -> int:
    fraction = max(0.0, float(fraction))
    if fraction <= 0.0:
        return 0

    height, width = image.shape
    area = height * width
    mean_r = max(1.0, (radius_px[0] + radius_px[1]) * 0.5)
    approx_count = int(fraction * area / (math.pi * mean_r * mean_r))
    count = max(1, approx_count)

    pil = Image.fromarray(image, mode="L")
    draw = ImageDraw.Draw(pil)
    for _ in range(count):
        cx = float(rng.uniform(0, width - 1))
        cy = float(rng.uniform(0, height - 1))
        radius = float(rng.uniform(radius_px[0], radius_px[1]))
        tone = int(rng.integers(value_range[0], value_range[1] + 1))
        bounds = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(bounds, fill=tone)
    image[:, :] = np.asarray(pil, dtype=np.uint8)
    return count


def _build_boundaries(labels: np.ndarray, width: int) -> np.ndarray:
    borders = np.zeros_like(labels, dtype=bool)
    borders[:-1, :] |= labels[:-1, :] != labels[1:, :]
    borders[:, :-1] |= labels[:, :-1] != labels[:, 1:]

    if width <= 1:
        return borders

    if ndimage is not None:
        return ndimage.binary_dilation(borders, iterations=width - 1)

    expanded = borders.copy()
    for _ in range(width - 1):
        up = np.pad(expanded[:-1, :], ((1, 0), (0, 0)), mode="constant")
        down = np.pad(expanded[1:, :], ((0, 1), (0, 0)), mode="constant")
        left = np.pad(expanded[:, :-1], ((0, 0), (1, 0)), mode="constant")
        right = np.pad(expanded[:, 1:], ((0, 0), (0, 1)), mode="constant")
        expanded = expanded | up | down | left | right
    return expanded


def generate_grain_structure(
    size: tuple[int, int],
    seed: int,
    mean_grain_size_px: float = 42.0,
    grain_size_jitter: float = 0.25,
    equiaxed: float = 1.0,
    elongation: float = 1.0,
    orientation_deg: float = 0.0,
    boundary_width_px: int = 2,
    boundary_contrast: float = 0.5,
    pore_fraction: float = 0.0,
    inclusion_fraction: float = 0.0,
) -> dict[str, Any]:
    """
    Generate a grain microstructure map and grayscale image.

    Output dictionary:
    - image: uint8 grayscale image
    - labels: int32 grain labels
    - boundaries: bool boundary mask
    - metadata: lightweight generation metadata
    """

    rng = np.random.default_rng(seed)
    height, width = size
    area = height * width
    mean_size = max(8.0, float(mean_grain_size_px))

    # Grain count is estimated from average grain area.
    grain_area = max(64.0, (mean_size * mean_size) * max(0.35, float(equiaxed)))
    grain_count = int(max(16, area / grain_area))
    points = np.column_stack(
        (
            rng.uniform(0, height - 1, size=grain_count),
            rng.uniform(0, width - 1, size=grain_count),
        )
    ).astype(np.float32)

    jitter = float(np.clip(grain_size_jitter, 0.0, 0.95))
    if jitter > 0:
        noise = rng.normal(0.0, mean_size * 0.2 * jitter, size=points.shape)
        points += noise.astype(np.float32)
        points[:, 0] = np.clip(points[:, 0], 0, height - 1)
        points[:, 1] = np.clip(points[:, 1], 0, width - 1)

    metric = _metric_from_elongation(elongation=elongation, orientation_deg=orientation_deg)
    labels = generate_voronoi_labels(size=size, points=points, metric_matrix=metric)
    boundaries = _build_boundaries(labels, width=max(1, int(boundary_width_px)))

    tones = rng.normal(loc=150.0, scale=22.0, size=grain_count).clip(70, 230).astype(np.uint8)
    image = tones[labels]

    contrast = float(np.clip(boundary_contrast, 0.0, 1.0))
    if contrast > 0:
        image[boundaries] = np.clip(image[boundaries] * (1.0 - contrast), 0, 255).astype(np.uint8)

    # Pores are dark, inclusions slightly bright or dark.
    pore_count = _draw_circular_defects(
        image=image,
        rng=rng,
        fraction=pore_fraction,
        radius_px=(2.0, mean_size * 0.18),
        value_range=(10, 55),
    )
    inclusion_count = _draw_circular_defects(
        image=image,
        rng=rng,
        fraction=inclusion_fraction,
        radius_px=(1.5, mean_size * 0.14),
        value_range=(65, 200),
    )

    return {
        "image": image,
        "labels": labels,
        "boundaries": boundaries,
        "metadata": {
            "grain_count": grain_count,
            "pore_count": pore_count,
            "inclusion_count": inclusion_count,
            "mean_grain_size_px": mean_size,
            "elongation": float(elongation),
            "orientation_deg": float(orientation_deg),
        },
    }

