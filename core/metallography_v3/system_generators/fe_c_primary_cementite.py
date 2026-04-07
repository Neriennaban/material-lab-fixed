"""A1 — primary cementite needle/plate renderer for hypereutectic
white cast iron (4.3 < %C ≤ 6.67).

Reference: §5.3.В of the TZ. The phase appears as long bright Fe₃C
plates that traverse the field of view in 2-3 dominant directions
(crystallographic of Fe₃C). Width grows with %C and decreases with
cooling rate. Edges are mildly serrated.

The renderer is intentionally lightweight: it draws bright line
segments on top of a baseline image using PIL, then adds a small
amount of multiscale smoothing for blending. The output is a
``(H, W) uint8`` image — colour palettes (e.g. NITAL warm) are still
applied later by ``apply_color_palette``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


def _ensure_u8(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def render_primary_cementite_needles(
    *,
    size: tuple[int, int],
    seed: int,
    c_wt: float,
    base_image: np.ndarray | None = None,
    needle_count: int | None = None,
    cooling_rate_c_per_s: float = 5.0,
) -> dict[str, Any]:
    """Render primary cementite needles on a hypereutectic cast-iron base.

    Parameters
    ----------
    size : (H, W)
        Output image dimensions.
    seed : int
        RNG seed for reproducibility.
    c_wt : float
        Carbon content in weight percent. Drives needle width and
        coverage — higher %C → wider, more numerous needles.
    base_image : np.ndarray, optional
        If provided, the renderer paints needles on top of this
        background (typically a leopard ledeburite texture). Otherwise
        a flat mid-gray background is used.
    needle_count : int, optional
        Override the auto-computed needle count.
    cooling_rate_c_per_s : float
        Cooling rate during solidification — higher rate → narrower
        needles (less time to grow).
    """
    h, w = size
    rng = np.random.default_rng(int(seed) + 211)

    if base_image is None:
        base = np.full(size, 168.0, dtype=np.float32)
        base += 4.0 * rng.normal(0.0, 1.0, size=size).astype(np.float32)
    else:
        base = base_image.astype(np.float32).copy()

    # Pick three dominant orientations to mimic the crystallographic
    # variants of Fe₃C precipitation.
    primary_dirs = [
        float(rng.uniform(0.0, math.pi)) for _ in range(3)
    ]

    if needle_count is None:
        # Coverage grows linearly with C content above 4.3 %.
        excess_c = max(0.0, float(c_wt) - 4.3)
        coverage_factor = 0.6 + 0.55 * excess_c  # 0.6 .. 1.9
        # Cooling rate inverse: slow cooling = fewer, larger needles.
        rate_factor = 1.0 / max(0.5, math.sqrt(float(cooling_rate_c_per_s)))
        needle_count = int(60.0 * coverage_factor * rate_factor)
    needle_count = max(20, int(needle_count))

    # Width band scales with %C and inverse with cooling rate.
    base_width = 4.0 + 6.0 * max(0.0, float(c_wt) - 4.3) / 2.5
    width_low = max(2.0, base_width * 0.6)
    width_high = max(width_low + 1.0, base_width * 1.5)

    canvas = Image.fromarray(_ensure_u8(base), mode="L")
    draw = ImageDraw.Draw(canvas)
    diag = math.hypot(h, w)
    bright_low = 240
    bright_high = 254
    for _ in range(needle_count):
        # Pick orientation from one of the dominant directions plus
        # small jitter.
        dir_idx = int(rng.integers(0, len(primary_dirs)))
        angle = primary_dirs[dir_idx] + float(rng.normal(0.0, math.pi / 36.0))
        length = float(rng.uniform(0.30, 0.85)) * diag
        cx = float(rng.uniform(0.0, w))
        cy = float(rng.uniform(0.0, h))
        dx = math.cos(angle) * length * 0.5
        dy = math.sin(angle) * length * 0.5
        p0 = (cx - dx, cy - dy)
        p1 = (cx + dx, cy + dy)
        width_px = int(rng.uniform(width_low, width_high))
        tone = int(rng.integers(bright_low, bright_high + 1))
        draw.line((p0, p1), fill=tone, width=max(1, width_px))

    out = np.asarray(canvas, dtype=np.uint8).astype(np.float32)
    if ndimage is not None:
        out = ndimage.gaussian_filter(out, sigma=0.6)
    image_u8 = _ensure_u8(out)

    needle_mask = (image_u8 > (bright_low - 4)).astype(np.uint8)
    return {
        "image": image_u8,
        "needle_mask": needle_mask,
        "metadata": {
            "needle_count": int(needle_count),
            "primary_directions_rad": [float(d) for d in primary_dirs],
            "width_band_px": [float(width_low), float(width_high)],
            "c_wt": float(c_wt),
            "cooling_rate_c_per_s": float(cooling_rate_c_per_s),
        },
    }
