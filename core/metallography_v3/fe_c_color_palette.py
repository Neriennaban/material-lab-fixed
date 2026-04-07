"""Post-process colour palette for Fe-C microstructures (A10.0).

The rendering pipeline produces a grayscale frame and per-phase masks.
When a preset selects a non-trivial ``color_mode`` (nital_warm, DIC,
tint etch...), this module is the single place that turns the gray +
masks into a proper 3-channel RGB frame. The rest of the pipeline
stays grayscale — prep, etch, optics and policy passes are untouched.

This keeps the scope of A10.0 small: one new function called at the very
end of ``MetallographyPipelineV3.generate`` replaces the old
``image_rgb = np.broadcast_to(gray[..., None], (h, w, 3))`` placeholder.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from core.metallography_v3.fe_c_palettes import (
    DIC_POLARIZED_MODE,
    GRAYSCALE_MODE,
    NITAL_WARM_MODE,
    TINT_ETCH_BLUE_YELLOW_MODE,
    get_palette,
    hsv_to_rgb,
    lerp_rgb,
)


def _to_rgb_grayscale(image_gray: np.ndarray) -> np.ndarray:
    """Replicate the legacy ``np.broadcast_to`` path as a contiguous array."""
    arr = np.ascontiguousarray(image_gray.astype(np.uint8))
    return np.repeat(arr[:, :, None], 3, axis=2)


def _resolve_mask(
    phase_masks: dict[str, np.ndarray] | None,
    phase_key: str,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if not isinstance(phase_masks, dict):
        return None
    key = str(phase_key).strip().upper()
    for name, mask in phase_masks.items():
        if str(name).strip().upper() == key and isinstance(mask, np.ndarray):
            arr = mask
            if arr.shape != shape:
                continue
            return arr > 0
    return None


def _boundary_mask(labels: np.ndarray | None, thickness: int = 1) -> np.ndarray | None:
    if labels is None or ndimage is None:
        return None
    if labels.ndim != 2:
        return None
    grad_y = np.abs(np.diff(labels.astype(np.int32), axis=0, prepend=labels[:1]))
    grad_x = np.abs(np.diff(labels.astype(np.int32), axis=1, prepend=labels[:, :1]))
    mask = (grad_y > 0) | (grad_x > 0)
    if thickness > 1:
        mask = ndimage.binary_dilation(mask, iterations=max(0, thickness - 1))
    return mask


def _apply_phase_tint(
    *,
    image_gray: np.ndarray,
    phase_masks: dict[str, np.ndarray] | None,
    palette: dict[str, Any],
    seed: int,
) -> np.ndarray:
    """Colour a grayscale frame by mapping phase masks to RGB gradients.

    For every phase listed in ``palette['phase_rgb']`` the corresponding
    mask gets a per-pixel interpolation between ``low`` and ``high``
    driven by the original grayscale intensity. Pixels that are not
    covered by any known phase mask fall back to the ``fallback_low_rgb``
    / ``fallback_high_rgb`` gradient so the output stays consistent.
    """
    h, w = image_gray.shape[:2]
    gray = image_gray.astype(np.float32)
    gmin = float(gray.min())
    gmax = float(gray.max())
    span = max(1.0, gmax - gmin)
    norm = (gray - gmin) / span  # 0..1 per pixel

    phase_rgb: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = dict(
        palette.get("phase_rgb", {})
    )
    fallback_low = tuple(int(c) for c in palette.get("fallback_low_rgb", (30, 30, 30)))
    fallback_high = tuple(int(c) for c in palette.get("fallback_high_rgb", (220, 220, 220)))

    out = np.empty((h, w, 3), dtype=np.uint8)
    # Start with the fallback gradient so every pixel gets a colour
    # even when phase masks are missing or partial.
    for channel in range(3):
        out[..., channel] = np.clip(
            fallback_low[channel] + (fallback_high[channel] - fallback_low[channel]) * norm,
            0.0,
            255.0,
        ).astype(np.uint8)

    if not phase_rgb:
        return out

    for phase_key, (low_rgb, high_rgb) in phase_rgb.items():
        mask = _resolve_mask(phase_masks, phase_key, (h, w))
        if mask is None or not mask.any():
            continue
        phase_norm = norm[mask]
        low = np.asarray(low_rgb, dtype=np.float32)
        high = np.asarray(high_rgb, dtype=np.float32)
        colours = low[None, :] + (high - low)[None, :] * phase_norm[:, None]
        out[mask] = np.clip(colours, 0.0, 255.0).astype(np.uint8)

    boundary_rgb = palette.get("boundary_rgb")
    boundary_blend = float(palette.get("boundary_blend", 0.0))
    if boundary_rgb is not None and boundary_blend > 0.0:
        # Approximate boundaries from the grayscale frame itself: very
        # dark thin bands are grain/phase edges. Using the gray channel
        # avoids needing a label map here.
        if ndimage is not None:
            blurred = ndimage.gaussian_filter(gray, sigma=1.0)
            edge_mask = (gray - blurred) < -12.0
        else:
            edge_mask = gray < (gmin + 0.08 * span)
        if edge_mask.any():
            bnd = np.asarray(boundary_rgb, dtype=np.float32)
            mix = float(max(0.0, min(1.0, boundary_blend)))
            out[edge_mask] = np.clip(
                (1.0 - mix) * out[edge_mask].astype(np.float32)
                + mix * bnd[None, :],
                0.0,
                255.0,
            ).astype(np.uint8)

    return out


def _apply_dic_polarized(
    *,
    image_gray: np.ndarray,
    phase_masks: dict[str, np.ndarray] | None,
    labels: np.ndarray | None,
    palette: dict[str, Any],
    seed: int,
) -> np.ndarray:
    """Colour each grain with a random HSV hue for a DIC look."""
    h, w = image_gray.shape[:2]
    out = _to_rgb_grayscale(image_gray).copy()

    # Without a label map we cannot tint per grain — fall back to gray.
    if labels is None or labels.ndim != 2:
        return out
    if labels.shape != (h, w):
        return out

    rng = np.random.default_rng(int(seed) + 9137)
    hue_low, hue_high = palette.get("hue_range", (0.0, 1.0))
    sat_low, sat_high = palette.get("saturation_range", (0.1, 0.4))
    val_low, val_high = palette.get("value_range", (0.5, 0.9))
    intragrain_jitter = float(palette.get("intragrain_value_jitter", 0.1))

    unique_labels = np.unique(labels)
    gray_norm = image_gray.astype(np.float32) / 255.0

    for label in unique_labels:
        mask = labels == label
        if not mask.any():
            continue
        hue = float(rng.uniform(hue_low, hue_high))
        sat = float(rng.uniform(sat_low, sat_high))
        base_val = float(rng.uniform(val_low, val_high))
        # Combine the random base value with the actual grayscale
        # intensity so intra-grain structure (noise, twins) remains
        # visible instead of being flattened.
        local_gray = gray_norm[mask]
        effective_val = np.clip(
            base_val * (0.85 + 0.30 * local_gray) - intragrain_jitter * 0.5,
            0.0,
            1.0,
        )
        colours = np.zeros((local_gray.size, 3), dtype=np.uint8)
        for idx, v in enumerate(effective_val):
            colours[idx] = hsv_to_rgb(hue, sat, float(v))
        out[mask] = colours

    boundary_rgb = palette.get("boundary_rgb")
    if boundary_rgb is not None:
        bnd_mask = _boundary_mask(labels, thickness=2)
        if bnd_mask is not None and bnd_mask.any():
            bnd = np.asarray(boundary_rgb, dtype=np.uint8)
            out[bnd_mask] = bnd
    return out


def apply_color_palette(
    *,
    image_gray: np.ndarray,
    phase_masks: dict[str, np.ndarray] | None,
    color_mode: str,
    seed: int,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Return a ``(H, W, 3)`` uint8 RGB frame for the requested palette.

    * ``grayscale_nital`` — simply stacks the grayscale channel in three
      copies (matches the legacy ``_to_rgb`` behaviour so existing
      presets remain byte-identical).
    * ``nital_warm`` / ``tint_etch_blue_yellow`` — phase-tint palette
      based on per-phase low/high RGB ramps.
    * ``dic_polarized`` — HSV grain colouring mimicking DIC/Nomarski
      reflected light. Requires a label map; silently falls back to
      grayscale when none is available.
    """
    if image_gray is None:
        raise ValueError("image_gray is required")
    if image_gray.ndim != 2:
        raise ValueError("image_gray must be a 2D (H, W) array")

    gray_u8 = image_gray.astype(np.uint8)
    mode = str(color_mode or GRAYSCALE_MODE).strip().lower()

    if mode == GRAYSCALE_MODE:
        return _to_rgb_grayscale(gray_u8)

    palette = get_palette(mode)
    kind = str(palette.get("kind", "grayscale"))

    if kind == "grayscale":
        return _to_rgb_grayscale(gray_u8)

    if mode == DIC_POLARIZED_MODE or kind == "dic_polarized":
        return _apply_dic_polarized(
            image_gray=gray_u8,
            phase_masks=phase_masks,
            labels=labels,
            palette=palette,
            seed=int(seed),
        )

    if kind == "phase_tint" or mode in (
        NITAL_WARM_MODE,
        TINT_ETCH_BLUE_YELLOW_MODE,
    ):
        return _apply_phase_tint(
            image_gray=gray_u8,
            phase_masks=phase_masks,
            palette=palette,
            seed=int(seed),
        )

    # Unknown kind → safe fallback.
    return _to_rgb_grayscale(gray_u8)
