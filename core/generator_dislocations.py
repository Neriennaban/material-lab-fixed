from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


def scale_um_per_px_for_magnification(magnification: int) -> float:
    """
    Approximate scale conversion for educational use.

    At 100x, we assume ~1.2 um/px. This scales inversely with magnification.
    """

    mag = max(25, int(magnification))
    return 1.2 * (100.0 / mag)


def _sample_positions(
    rng: np.random.Generator,
    count: int,
    size: tuple[int, int],
    distribution: str,
) -> np.ndarray:
    height, width = size
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    if distribution == "cluster":
        cluster_count = max(2, count // 120)
        centers = np.column_stack(
            (
                rng.uniform(0, height - 1, size=cluster_count),
                rng.uniform(0, width - 1, size=cluster_count),
            )
        )
        picked = rng.integers(0, cluster_count, size=count)
        jitter = rng.normal(0.0, min(height, width) * 0.04, size=(count, 2))
        pts = centers[picked] + jitter
    elif distribution == "gradient":
        pts = np.zeros((count, 2), dtype=np.float32)
        for idx in range(count):
            y = rng.beta(1.3, 3.2) * (height - 1)
            x = rng.uniform(0, width - 1)
            pts[idx] = (y, x)
    else:
        pts = np.column_stack(
            (rng.uniform(0, height - 1, size=count), rng.uniform(0, width - 1, size=count))
        )

    pts[:, 0] = np.clip(pts[:, 0], 0, height - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, width - 1)
    return pts.astype(np.float32, copy=False)


def _pit_polygon(cx: float, cy: float, radius: float, shape: str, rotation_rad: float) -> list[tuple[float, float]]:
    if shape == "square":
        angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    elif shape == "diamond":
        angles = [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]
    else:  # triangle default
        angles = [math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6]

    return [
        (
            cx + radius * math.cos(a + rotation_rad),
            cy + radius * math.sin(a + rotation_rad),
        )
        for a in angles
    ]


def generate_dislocation_pits(
    size: tuple[int, int],
    seed: int,
    density_per_mm2: float = 200_000.0,
    magnification: int = 200,
    pit_shape: str = "triangle",
    distribution: str = "uniform",
    pit_radius_px: tuple[float, float] = (1.8, 4.5),
) -> dict[str, Any]:
    """
    Generate etch-pit style texture for monocrystalline silicon.

    Returns:
    - image: uint8
    - pit_positions: list[[y, x], ...]
    - metadata: pit count + calculated area
    """

    rng = np.random.default_rng(seed)
    height, width = size

    background = np.full((height, width), 170, dtype=np.float32)
    gradient = np.linspace(0, 15, width, dtype=np.float32)[None, :]
    background = background + gradient
    background += rng.normal(0.0, 2.0, size=(height, width)).astype(np.float32)
    if ndimage is not None:
        background = ndimage.gaussian_filter(background, sigma=1.2)

    scale_um_px = scale_um_per_px_for_magnification(magnification)
    area_mm2 = (height * scale_um_px * width * scale_um_px) / 1_000_000.0
    expected = max(1, int(float(density_per_mm2) * area_mm2))
    count = int(min(expected, 80_000))

    positions = _sample_positions(
        rng=rng,
        count=count,
        size=size,
        distribution=distribution.lower().strip(),
    )

    canvas = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(canvas)
    for pos in positions:
        cy, cx = float(pos[0]), float(pos[1])
        radius = float(rng.uniform(pit_radius_px[0], pit_radius_px[1]))
        polygon = _pit_polygon(
            cx=cx,
            cy=cy,
            radius=radius,
            shape=pit_shape.lower().strip(),
            rotation_rad=float(rng.uniform(0.0, math.pi)),
        )
        fill = int(rng.integers(35, 75))
        draw.polygon(polygon, fill=fill)

    image = np.asarray(canvas, dtype=np.uint8)
    if ndimage is not None:
        image = ndimage.gaussian_filter(image.astype(np.float32), sigma=0.4).clip(0, 255).astype(np.uint8)

    return {
        "image": image,
        "pit_positions": positions.astype(np.float32),
        "metadata": {
            "pit_count": int(count),
            "pit_shape": pit_shape,
            "distribution": distribution,
            "area_mm2": float(area_mm2),
            "density_per_mm2_target": float(density_per_mm2),
        },
    }

