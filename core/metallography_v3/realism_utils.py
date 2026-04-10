
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def smooth(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing with a downsample/upsample fast path.

    Phase D.2 — ``scipy.ndimage.gaussian_filter`` costs O(N · sigma)
    per axis; when ``sigma`` is a meaningful fraction of the image
    size (as in ``multiscale_noise`` with ``sigma=18..30`` on a
    2K/4K/16K canvas) it becomes the top bottleneck of the pipeline.
    For ``sigma >= 6`` we downsample the field to a buffer where
    the effective sigma is ~3, run a small-sigma gaussian there,
    and zoom-interpolate the result back to the original shape. The
    output is visually indistinguishable from the full-resolution
    pass (gaussian smoothing on a down-sampled grid produces the
    same low-pass band) but scales linearly with output area.
    """
    sigma = max(0.05, float(sigma))
    if ndimage is None:
        return field
    # Small sigma — the classic path is already cheap enough and
    # sub-pixel features matter, so no approximation.
    h, w = field.shape[:2]
    if sigma < 5.0 or max(h, w) < 512:
        return ndimage.gaussian_filter(field, sigma=sigma)
    factor = max(2, int(round(sigma / 2.5)))
    small_h = max(8, h // factor)
    small_w = max(8, w // factor)
    zoom_down_y = float(small_h) / float(h)
    zoom_down_x = float(small_w) / float(w)
    small = ndimage.zoom(
        field.astype(np.float32, copy=False),
        (zoom_down_y, zoom_down_x),
        order=1,
        prefilter=False,
    )
    small = ndimage.gaussian_filter(small, sigma=max(0.5, sigma / factor))
    zoom_up_y = float(h) / float(small.shape[0])
    zoom_up_x = float(w) / float(small.shape[1])
    # Upsample with cubic spline interpolation. Use ndimage.zoom with
    # order=3 and then crop/pad to the exact target shape. The cubic
    # spline eliminates the visible grid artifacts that bilinear
    # produced at integer-pixel transitions of the downsampled buffer.
    zoom_up_y = float(h) / float(small.shape[0])
    zoom_up_x = float(w) / float(small.shape[1])
    upsampled = ndimage.zoom(
        small, (zoom_up_y, zoom_up_x), order=3, prefilter=True,
    ).astype(np.float32)
    # Crop or pad by at most 1 pixel to match the exact target shape.
    if upsampled.shape != field.shape:
        out = np.empty(field.shape, dtype=np.float32)
        th = min(h, upsampled.shape[0])
        tw = min(w, upsampled.shape[1])
        out[:th, :tw] = upsampled[:th, :tw]
        if th < h:
            out[th:, :] = out[th - 1 : th, :]
        if tw < w:
            out[:, tw:] = out[:, tw - 1 : tw]
        upsampled = out
    return upsampled


def normalize01(field: np.ndarray) -> np.ndarray:
    arr = field.astype(np.float32, copy=False)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo + 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def boundary_mask_from_labels(labels: np.ndarray, width: int = 1) -> np.ndarray:
    borders = np.zeros_like(labels, dtype=bool)
    borders[:-1, :] |= labels[:-1, :] != labels[1:, :]
    borders[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    if int(width) > 1 and ndimage is not None:
        borders = ndimage.binary_dilation(borders, iterations=max(1, int(width) - 1))
    return borders


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    mask_b = mask.astype(bool, copy=False)
    if ndimage is not None:
        return ndimage.distance_transform_edt(~mask_b).astype(np.float32)
    return (~mask_b).astype(np.float32)


def multiscale_noise(
    *,
    size: tuple[int, int],
    seed: int,
    scales: tuple[tuple[float, float], ...] = ((18.0, 0.55), (7.0, 0.30), (2.5, 0.15)),
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.zeros(size, dtype=np.float32)
    for sigma, weight in scales:
        layer = rng.normal(0.0, 1.0, size=size).astype(np.float32)
        layer = smooth(layer, sigma=max(0.3, float(sigma)))
        out += layer * float(weight)
    return normalize01(out)


def select_fraction_mask(*, field: np.ndarray, available: np.ndarray | None, fraction_total: float) -> np.ndarray:
    target = int(round(clamp(float(fraction_total), 0.0, 1.0) * field.size))
    out = np.zeros(field.shape, dtype=bool)
    if target <= 0:
        return out
    available_b = np.ones(field.shape, dtype=bool) if available is None else available.astype(bool, copy=False)
    flat_idx = np.flatnonzero(available_b.ravel())
    if flat_idx.size == 0:
        return out
    if target >= flat_idx.size:
        out.ravel()[flat_idx] = True
        return out
    values = field.ravel()[flat_idx]
    pick = np.argpartition(values, -target)[-target:]
    out.ravel()[flat_idx[pick]] = True
    return out


def allocate_phase_masks(
    *,
    size: tuple[int, int],
    phase_fractions: dict[str, float],
    ordered_fields: list[tuple[str, np.ndarray]],
    remainder_name: str | None = None,
) -> dict[str, np.ndarray]:
    available = np.ones(size, dtype=bool)
    out: dict[str, np.ndarray] = {}
    normalized = {str(k): float(max(0.0, v)) for k, v in dict(phase_fractions or {}).items() if float(v) > 1e-9}
    norm_sum = float(sum(normalized.values()))
    if norm_sum <= 1e-9:
        return {"solid": np.ones(size, dtype=np.uint8)}
    normalized = {k: float(v) / norm_sum for k, v in normalized.items()}

    handled: set[str] = set()
    for name, field in ordered_fields:
        frac = float(normalized.get(name, 0.0))
        handled.add(str(name))
        if frac <= 0.0:
            continue
        mask = select_fraction_mask(field=field, available=available, fraction_total=frac)
        out[str(name)] = mask.astype(np.uint8)
        available &= ~mask

    residual_names = [name for name in normalized.keys() if name not in handled]
    if remainder_name is not None and remainder_name in normalized and remainder_name not in handled:
        residual_names = [x for x in residual_names if x != remainder_name] + [remainder_name]

    for idx, name in enumerate(residual_names):
        frac = float(normalized.get(name, 0.0))
        if frac <= 0.0:
            continue
        if idx == len(residual_names) - 1:
            mask = available.copy()
        else:
            mask = select_fraction_mask(
                field=multiscale_noise(size=size, seed=7919 + idx * 97),
                available=available,
                fraction_total=frac,
            )
        out[str(name)] = mask.astype(np.uint8)
        available &= ~mask

    if not out:
        out["solid"] = np.ones(size, dtype=np.uint8)
    elif int(available.sum()) > 0:
        dominant = max(out.items(), key=lambda item: int(np.asarray(item[1]).sum()))[0]
        out[dominant] = np.clip(out[dominant].astype(np.uint8) + available.astype(np.uint8), 0, 1).astype(np.uint8)
    return out


def draw_particle_mask(
    *,
    size: tuple[int, int],
    seed: int,
    fraction_total: float,
    radius_range: tuple[float, float],
    angular: bool = False,
    elongation_range: tuple[float, float] = (1.0, 1.0),
    angle_spread_deg: float = 180.0,
    restrict_to: np.ndarray | None = None,
) -> np.ndarray:
    h, w = size
    restrict = np.ones(size, dtype=bool) if restrict_to is None else restrict_to.astype(bool, copy=False)
    if not np.any(restrict) or fraction_total <= 0.0:
        return np.zeros(size, dtype=bool)

    rng = np.random.default_rng(int(seed))
    mean_r = max(0.6, float(radius_range[0] + radius_range[1]) * 0.5)
    target_px = int(round(clamp(float(fraction_total), 0.0, 1.0) * h * w))
    if target_px <= 0:
        return np.zeros(size, dtype=bool)
    approx_count = max(1, int(target_px / max(1.0, math.pi * mean_r * mean_r)))
    pil = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(pil)
    ys, xs = np.nonzero(restrict)
    if len(xs) == 0:
        return np.zeros(size, dtype=bool)
    for _ in range(approx_count):
        ridx = int(rng.integers(0, len(xs)))
        cx = float(xs[ridx])
        cy = float(ys[ridx])
        r = float(rng.uniform(radius_range[0], radius_range[1]))
        elong = float(rng.uniform(elongation_range[0], elongation_range[1]))
        angle = math.radians(float(rng.uniform(-angle_spread_deg, angle_spread_deg)))
        if angular:
            pts = []
            n = int(rng.integers(4, 7))
            for k in range(n):
                ang = angle + 2.0 * math.pi * float(k) / float(n) + float(rng.normal(0.0, 0.18))
                rad = r * float(rng.uniform(0.72, 1.28))
                pts.append((cx + math.cos(ang) * rad * elong, cy + math.sin(ang) * rad))
            draw.polygon(pts, fill=255)
        else:
            dx = r * elong
            dy = r
            draw.ellipse((cx - dx, cy - dy, cx + dx, cy + dy), fill=255)
    mask = np.asarray(pil, dtype=np.uint8) > 0
    mask &= restrict
    if ndimage is not None:
        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_closing(mask, iterations=1)
    return mask.astype(bool)


def blur_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    return normalize01(smooth(mask.astype(np.float32), sigma=max(0.2, float(sigma))))


def rescale_to_u8(image: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    arr = image.astype(np.float32)
    if lo is None:
        lo = float(np.quantile(arr, 0.01))
    if hi is None:
        hi = float(np.quantile(arr, 0.99))
    if hi <= lo + 1e-9:
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)
    arr = (arr - float(lo)) / (float(hi) - float(lo))
    return np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)


def low_frequency_field(size: tuple[int, int], seed: int, sigma: float = 24.0) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    field = rng.normal(0.0, 1.0, size=size).astype(np.float32)
    return normalize01(smooth(field, sigma=max(0.3, sigma)))


def cooling_index(cooling_mode: str) -> float:
    mode = str(cooling_mode or "").strip().lower()
    mapping = {
        "furnace": 0.10,
        "annealed": 0.12,
        "equilibrium": 0.18,
        "slow": 0.22,
        "air": 0.40,
        "normalized": 0.55,
        "fast_air": 0.62,
        "oil": 0.70,
        "oil_quench": 0.76,
        "water": 0.90,
        "water_quench": 0.95,
        "brine": 1.00,
        "quench": 0.92,
        "aged": 0.20,
    }
    for key, value in mapping.items():
        if key in mode:
            return float(value)
    return 0.35


def build_twins_from_labels(
    *,
    labels: np.ndarray,
    seed: int,
    fraction_of_grains: float = 0.45,
    spacing_px: float = 12.0,
    width_px: float = 1.6,
) -> np.ndarray:
    h, w = labels.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(int(seed))
    grain_count = int(labels.max()) + 1
    active = rng.random(grain_count) < clamp(float(fraction_of_grains), 0.0, 1.0)
    theta = rng.uniform(-math.pi / 2.0, math.pi / 2.0, size=grain_count).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=grain_count).astype(np.float32)
    proj = xx * np.cos(theta[labels]) + yy * np.sin(theta[labels])
    wave = np.sin((2.0 * math.pi / float(max(3.0, spacing_px))) * proj + phase[labels])
    twin = active[labels] & (np.abs(wave) > max(0.1, 1.0 - 0.25 * float(width_px)))
    if ndimage is not None:
        twin = ndimage.binary_opening(twin, iterations=1)
    return twin.astype(bool)
