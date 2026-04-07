"""A3 — austenite dendrites for hypoeutectic white cast iron
(2.14 < %C < 4.3).

Reference: §5.5 of the TZ. In hypoeutectic white cast iron the first
solid that nucleates from the melt is primary austenite, which grows
as branching dendrites; after eutectoid decomposition at 727 °C the
dendrites become pearlite, leaving a darker tree-like network in a
brighter ledeburite matrix.

The renderer is intentionally lightweight: it walks an L-system that
emits straight stems with secondary side arms at ±60° to the main
trunk and paints them onto the supplied base image (typically the
``texture_ledeburite_leopard`` output). Two to four orders of
branching are produced; the trunk count and arm length scale with
the carbon excess above the eutectic point ``(4.3 - C)``.
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


def _draw_branch(
    draw: ImageDraw.ImageDraw,
    *,
    cx: float,
    cy: float,
    angle: float,
    length: float,
    width: int,
    tone: int,
    rng: np.random.Generator,
    order: int,
    max_order: int,
) -> None:
    """Recursively draw a branching dendrite arm."""
    if order > max_order or length < 4.0 or width < 1:
        return
    end_x = cx + math.cos(angle) * length
    end_y = cy + math.sin(angle) * length
    draw.line(
        ((cx, cy), (end_x, end_y)),
        fill=tone,
        width=max(1, int(width)),
    )

    # Place a few side branches along the trunk.
    branch_count = max(1, int(round(length / 14.0)))
    for k in range(1, branch_count + 1):
        t = float(k) / float(branch_count + 1)
        bx = cx + (end_x - cx) * t
        by = cy + (end_y - cy) * t
        for sign in (-1.0, 1.0):
            side_angle = angle + sign * float(rng.uniform(math.pi / 4.0, math.pi / 2.5))
            side_length = length * float(rng.uniform(0.40, 0.65))
            side_width = max(1, width - 1)
            _draw_branch(
                draw,
                cx=bx,
                cy=by,
                angle=side_angle,
                length=side_length,
                width=side_width,
                tone=tone,
                rng=rng,
                order=order + 1,
                max_order=max_order,
            )


def render_fe_c_austenite_dendrites(
    *,
    size: tuple[int, int],
    seed: int,
    c_wt: float,
    base_image: np.ndarray | None = None,
    cooling_rate_c_per_s: float = 5.0,
    trunk_count: int | None = None,
) -> dict[str, Any]:
    """Render hypoeutectic austenite dendrites on top of a base image.

    Parameters
    ----------
    size : (H, W)
        Output image dimensions.
    seed : int
        RNG seed for reproducibility.
    c_wt : float
        Carbon content in weight percent. Must be in the
        2.14-4.3 % range; values outside the range produce a degenerate
        output (no dendrites for very low C, almost solid coverage at
        the eutectic point).
    base_image : np.ndarray, optional
        Background to paint dendrites onto. If ``None`` a flat
        bright field is used.
    cooling_rate_c_per_s : float
        Cooling rate during solidification — higher rate → finer,
        more numerous dendrites with shorter primary arms.
    trunk_count : int, optional
        Override the auto-computed primary trunk count.
    """
    h, w = size
    rng = np.random.default_rng(int(seed) + 4099)

    if base_image is None:
        base = np.full(size, 200.0, dtype=np.float32)
        base += 5.0 * rng.normal(0.0, 1.0, size=size).astype(np.float32)
    else:
        base = base_image.astype(np.float32).copy()

    # Carbon excess below the eutectic point. The closer to 2.14 % C
    # the more austenite, hence more dendrites.
    excess = max(0.0, 4.3 - float(c_wt))
    excess = min(excess, 4.3 - 2.14)  # cap at 2.16

    if trunk_count is None:
        # Slow cooling → fewer but bigger trunks; fast cooling →
        # many small trunks.
        rate = max(0.5, float(cooling_rate_c_per_s))
        rate_factor = math.sqrt(1.0 / rate)  # ∈ (0, 1.4]
        trunk_count = int(round(4.0 + 6.0 * (excess / 2.16) / rate_factor))
    trunk_count = max(2, int(trunk_count))

    # Trunk length: half the diagonal at minimum, scales with excess.
    diag = math.hypot(h, w)
    trunk_length = diag * (0.30 + 0.15 * (excess / 2.16))
    base_width = int(round(3.0 + 2.0 * (excess / 2.16)))
    max_order = 3 if excess < 1.0 else 4
    dendrite_tone = 70  # Dark, after eutectoid decomposition → pearlite

    canvas = Image.fromarray(_ensure_u8(base), mode="L")
    draw = ImageDraw.Draw(canvas)
    for _ in range(trunk_count):
        cx = float(rng.uniform(0.0, w))
        cy = float(rng.uniform(0.0, h))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        length = trunk_length * float(rng.uniform(0.85, 1.15))
        _draw_branch(
            draw,
            cx=cx,
            cy=cy,
            angle=angle,
            length=length,
            width=base_width,
            tone=dendrite_tone,
            rng=rng,
            order=1,
            max_order=max_order,
        )

    out = np.asarray(canvas, dtype=np.uint8).astype(np.float32)
    if ndimage is not None:
        out = ndimage.gaussian_filter(out, sigma=0.7)
    image_u8 = _ensure_u8(out)

    # The dendritic mask is the set of pixels noticeably darker than
    # the surrounding base — useful for downstream metrics.
    delta = base.astype(np.float32) - image_u8.astype(np.float32)
    dendrite_mask = (delta > 20.0).astype(np.uint8)

    return {
        "image": image_u8,
        "dendrite_mask": dendrite_mask,
        "metadata": {
            "trunk_count": int(trunk_count),
            "trunk_length_px": float(trunk_length),
            "base_width_px": int(base_width),
            "max_branch_order": int(max_order),
            "c_wt": float(c_wt),
            "cooling_rate_c_per_s": float(cooling_rate_c_per_s),
        },
    }
