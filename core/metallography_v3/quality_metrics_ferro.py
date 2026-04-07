"""Quality metrics for Fe-C microstructure renderings (Phase C1).

The plan calls for objective ways to compare a generated frame against
either a reference photograph or the lever-rule prediction. This module
implements a small set of CPU-only metrics that work without any deep
learning dependencies; the optional LPIPS / FID metrics live behind
``try``-import guards so the module degrades gracefully when those
libraries are not available.

Provided metrics:

* ``phase_fraction_error`` — relative error between rendered phase
  masks and the lever-rule expectation. Returns a per-phase dict
  plus a maximum error.
* ``histogram_intersection`` — RGB or grayscale histogram intersection
  in [0, 1]; 1.0 = identical distributions.
* ``ssim_vs_reference`` — sanity-check SSIM via
  ``skimage.metrics.structural_similarity`` if available, otherwise
  a numpy fallback (Pearson correlation on the same-size frames).
* ``fft_lamellae_period_px`` — peak frequency of a 1D radial FFT
  spectrum, useful for measuring perlite interlamellar spacing.
* ``hough_orientation_histogram`` — coarse 8-bin orientation histogram
  derived from a Sobel gradient field; lets a test verify that an
  ostensibly-uniform pearlite area has no dominant direction.
* ``grain_size_astm`` — wraps ``core.measurements`` to expose the ASTM
  E112 grain size from a label map.

All functions are pure: they read numpy arrays and return Python
primitives (or small dicts). They never touch the filesystem.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from skimage.metrics import structural_similarity as _skimage_ssim  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _skimage_ssim = None

try:
    from scipy import ndimage as _ndimage  # type: ignore
except Exception:  # pragma: no cover
    _ndimage = None


# ---------------------------------------------------------------------------
# Phase fraction error
# ---------------------------------------------------------------------------


def _resolve_mask(masks: dict[str, np.ndarray], phase: str) -> np.ndarray | None:
    if not isinstance(masks, dict):
        return None
    target = str(phase).strip().upper()
    for name, mask in masks.items():
        if str(name).strip().upper() == target and isinstance(mask, np.ndarray):
            return mask > 0
    return None


def phase_fraction_error(
    *,
    phase_masks: dict[str, np.ndarray],
    expected_fractions: dict[str, float],
    total_pixels: int | None = None,
) -> dict[str, Any]:
    """Per-phase relative error between actual and expected fractions.

    The output dict contains:

    * ``actual`` — measured fraction per phase (0..1).
    * ``expected`` — expected fraction per phase (0..1).
    * ``relative_error_pct`` — ``|actual - expected| / max(expected, 1e-6) * 100``.
    * ``max_relative_error_pct`` — the worst-offending phase.

    A missing mask is treated as zero coverage.
    """
    if total_pixels is None:
        total = 0
        for mask in phase_masks.values():
            if isinstance(mask, np.ndarray):
                total = mask.size
                break
        total_pixels = max(1, total)

    actual: dict[str, float] = {}
    relative: dict[str, float] = {}
    for phase, expected in expected_fractions.items():
        mask = _resolve_mask(phase_masks, phase)
        if mask is None:
            measured = 0.0
        else:
            measured = float(mask.sum()) / float(total_pixels)
        actual[str(phase).upper()] = measured
        denom = max(1e-6, float(expected))
        relative[str(phase).upper()] = abs(measured - float(expected)) / denom * 100.0

    return {
        "actual": actual,
        "expected": {str(k).upper(): float(v) for k, v in expected_fractions.items()},
        "relative_error_pct": relative,
        "max_relative_error_pct": float(max(relative.values()) if relative else 0.0),
    }


# ---------------------------------------------------------------------------
# Histogram intersection
# ---------------------------------------------------------------------------


def histogram_intersection(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    bins: int = 64,
) -> float:
    """Compute the normalised histogram intersection between two
    images. Works on grayscale (2D) or RGB (3D) inputs of the same
    shape; for RGB the channels are concatenated and the metric is
    averaged across channels."""
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"shape mismatch: {image_a.shape} vs {image_b.shape}"
        )
    if image_a.ndim == 2:
        return _hist_inter_channel(image_a, image_b, bins=bins)
    if image_a.ndim == 3 and image_a.shape[2] == 3:
        scores = [
            _hist_inter_channel(image_a[..., c], image_b[..., c], bins=bins)
            for c in range(3)
        ]
        return float(np.mean(scores))
    raise ValueError(f"unsupported image shape: {image_a.shape}")


def _hist_inter_channel(
    a: np.ndarray, b: np.ndarray, *, bins: int
) -> float:
    hist_a, _ = np.histogram(a, bins=bins, range=(0, 256), density=True)
    hist_b, _ = np.histogram(b, bins=bins, range=(0, 256), density=True)
    hist_a = hist_a / max(1e-12, float(hist_a.sum()))
    hist_b = hist_b / max(1e-12, float(hist_b.sum()))
    return float(np.minimum(hist_a, hist_b).sum())


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def ssim_vs_reference(
    image: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Sanity-check structural similarity. Uses ``skimage.metrics`` if
    available, otherwise falls back to a Pearson correlation on the
    centred images. Both inputs must have the same shape."""
    if image.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: {image.shape} vs {reference.shape}"
        )
    if _skimage_ssim is not None:
        try:
            data_range = float(image.max() - image.min()) or 1.0
            return float(
                _skimage_ssim(image, reference, data_range=data_range)
            )
        except Exception:
            pass
    # Pearson correlation fallback.
    a = image.astype(np.float32).ravel()
    b = reference.astype(np.float32).ravel()
    a_centred = a - a.mean()
    b_centred = b - b.mean()
    denom = float(np.linalg.norm(a_centred) * np.linalg.norm(b_centred))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(a_centred, b_centred) / denom)


# ---------------------------------------------------------------------------
# FFT lamella period
# ---------------------------------------------------------------------------


def fft_lamellae_period_px(image: np.ndarray) -> float:
    """Estimate the dominant lamella period (in pixels) of a 2D
    grayscale image via the radial average of the 2D FFT spectrum.

    Returns 0.0 when the image has no detectable directional
    structure (uniform pearlite at low magnification).
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D grayscale")
    arr = image.astype(np.float32) - float(image.mean())
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    max_r = min(cy, cx) - 1
    if max_r <= 4:
        return 0.0
    radial = np.bincount(
        radius.ravel(), weights=spectrum.ravel(), minlength=max_r + 1
    )
    counts = np.bincount(radius.ravel(), minlength=max_r + 1)
    radial = radial[: max_r + 1]
    counts = counts[: max_r + 1]
    counts[counts == 0] = 1
    radial = radial / counts
    # Skip the DC bin (index 0). Find the strongest non-DC peak.
    radial[0] = 0.0
    radial[1] = 0.0  # very low frequency, dominated by background tone
    peak_radius = int(np.argmax(radial))
    if peak_radius < 2 or radial[peak_radius] < radial.mean() * 1.5:
        return 0.0
    return float(min(h, w) / float(peak_radius))


# ---------------------------------------------------------------------------
# Hough orientation histogram (coarse)
# ---------------------------------------------------------------------------


def hough_orientation_histogram(
    image: np.ndarray,
    *,
    bins: int = 8,
) -> np.ndarray:
    """Return a coarse orientation histogram (radians ∈ [0, π)) of
    the gradient field of ``image``. Used to verify whether an
    ostensibly-uniform pearlite area has any dominant direction
    (the histogram should be near-uniform when it does not).
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D grayscale")
    if _ndimage is None:
        # Manual Sobel.
        gx = np.zeros_like(image, dtype=np.float32)
        gy = np.zeros_like(image, dtype=np.float32)
        gx[:, 1:-1] = image[:, 2:].astype(np.float32) - image[:, :-2].astype(np.float32)
        gy[1:-1, :] = image[2:, :].astype(np.float32) - image[:-2, :].astype(np.float32)
    else:
        gx = _ndimage.sobel(image.astype(np.float32), axis=1)
        gy = _ndimage.sobel(image.astype(np.float32), axis=0)
    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx) % np.pi
    weights = magnitude.ravel()
    angles = angle.ravel()
    threshold = float(weights.mean()) * 1.5
    keep = weights > threshold
    if keep.sum() < 32:
        return np.full(bins, 1.0 / float(bins), dtype=np.float32)
    edges = np.linspace(0.0, np.pi, bins + 1)
    hist, _ = np.histogram(angles[keep], bins=edges, weights=weights[keep])
    if hist.sum() <= 0.0:
        return np.full(bins, 1.0 / float(bins), dtype=np.float32)
    return (hist / hist.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Grain size (ASTM E112) wrapper
# ---------------------------------------------------------------------------


def grain_size_astm(
    labels: np.ndarray,
    *,
    um_per_px: float,
) -> float:
    """ASTM E112 grain size number from a label map and pixel pitch."""
    from core.measurements import (
        astm_grain_size_from_intercept_length_um,
        mean_lineal_intercept,
    )

    if labels.ndim != 2:
        raise ValueError("labels must be a 2D label map")
    h, w = labels.shape
    # Count label changes along horizontal and vertical scan lines.
    horizontal_changes = int((labels[:, 1:] != labels[:, :-1]).sum())
    vertical_changes = int((labels[1:, :] != labels[:-1, :]).sum())
    total_intersections = horizontal_changes + vertical_changes
    if total_intersections <= 0:
        return float("nan")
    total_length_px = float(h * (w - 1) + (h - 1) * w)
    total_length_um = total_length_px * float(um_per_px)
    li_um = mean_lineal_intercept(
        total_test_line_length_um=total_length_um,
        intersections=total_intersections,
    )
    return astm_grain_size_from_intercept_length_um(li_um)
