from __future__ import annotations

import math
from collections import OrderedDict
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from core.optical_mode_transfer import apply_optical_mode_transfer
from core.psf_engineering import apply_live_psf_profile
from core.ui_v2_utils import estimate_um_per_px
from core.virtual_slide import extract_field_of_view_from_array

try:
    from scipy import ndimage, signal, special  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None
    signal = None
    special = None


_VIEW_CACHE_MAX = 12
_VIEW_CACHE: OrderedDict[tuple[Any, ...], tuple[np.ndarray, dict[str, Any]]] = (
    OrderedDict()
)
_PSF_CACHE_MAX = 64
_NOISE_CACHE_MAX = 24
_MASK_CACHE_MAX = 24


def _cache_key_float(value: float | None, *, precision: int = 4) -> int:
    if value is None:
        return 0
    return int(round(float(value) * (10**precision)))


def _stable_signature_items(payload: dict[str, Any] | None) -> tuple[Any, ...]:
    if not isinstance(payload, dict):
        return ()
    items: list[tuple[str, Any]] = []
    for key, value in sorted(payload.items(), key=lambda item: str(item[0])):
        if isinstance(value, (str, int, float, bool)) or value is None:
            items.append((str(key), value))
        else:
            items.append((str(key), str(value)))
    return tuple(items)


def _cache_touch(
    cache: OrderedDict[tuple[Any, ...], tuple[np.ndarray, dict[str, Any]]],
    key: tuple[Any, ...],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    item = cache.get(key)
    if item is None:
        return None
    cache.move_to_end(key)
    return item


def _cache_set(
    cache: OrderedDict[tuple[Any, ...], tuple[np.ndarray, dict[str, Any]]],
    key: tuple[Any, ...],
    value: tuple[np.ndarray, dict[str, Any]],
    *,
    max_size: int,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _sample_view_cache_key(
    sample: np.ndarray,
    optical_ferromagnetic_mask_sample: np.ndarray | None,
    magnification: int,
    pan_x: float,
    pan_y: float,
    output_size: tuple[int, int],
    focus: float,
    focus_distance_mm: float | None,
    focus_target_mm: float | None,
    focus_quality: float | None,
    um_per_px_100x: float | None,
    brightness: float,
    contrast: float,
    vignette_strength: float,
    uneven_strength: float,
    noise_sigma: float,
    add_dust: bool,
    add_scratches: bool,
    etch_uneven: float,
    optical_mode: str,
    psf_profile: str,
    psf_strength: float,
    sectioning_shear_deg: float,
    hybrid_balance: float,
    optical_mode_parameters: tuple[Any, ...],
    optical_context_signature: tuple[Any, ...],
    seed: int,
) -> tuple[Any, ...]:
    return (
        sample.__array_interface__["data"][0],
        sample.shape[0],
        sample.shape[1],
        0
        if optical_ferromagnetic_mask_sample is None
        else optical_ferromagnetic_mask_sample.__array_interface__["data"][0],
        0
        if optical_ferromagnetic_mask_sample is None
        else optical_ferromagnetic_mask_sample.shape[0],
        0
        if optical_ferromagnetic_mask_sample is None
        else optical_ferromagnetic_mask_sample.shape[1],
        int(magnification),
        _cache_key_float(pan_x),
        _cache_key_float(pan_y),
        int(output_size[0]),
        int(output_size[1]),
        _cache_key_float(focus),
        _cache_key_float(focus_distance_mm),
        _cache_key_float(focus_target_mm),
        _cache_key_float(focus_quality),
        _cache_key_float(um_per_px_100x),
        _cache_key_float(brightness, precision=3),
        _cache_key_float(contrast, precision=3),
        _cache_key_float(vignette_strength),
        _cache_key_float(uneven_strength),
        _cache_key_float(noise_sigma),
        1 if bool(add_dust) else 0,
        1 if bool(add_scratches) else 0,
        _cache_key_float(etch_uneven),
        str(optical_mode or "brightfield"),
        str(psf_profile or "standard"),
        _cache_key_float(psf_strength),
        _cache_key_float(sectioning_shear_deg),
        _cache_key_float(hybrid_balance),
        tuple(optical_mode_parameters),
        tuple(optical_context_signature),
        int(seed),
    )


def _resize_u8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image, mode="L")
    resized = pil.resize((size[1], size[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def extract_field_of_view(
    sample: np.ndarray,
    magnification: int,
    pan_x: float = 0.5,
    pan_y: float = 0.5,
    output_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Crop sample according to virtual magnification and pan.

    `sample` is treated as reference at 100x.

    The implementation uses a multiresolution virtual slide cache so repeated
    changes of magnification or stage position are rendered without resampling
    the full-resolution source on every frame. The returned metadata preserves
    the base-level crop coordinates for downstream scale audit logic.
    """
    crop, meta = extract_field_of_view_from_array(
        sample,
        magnification=max(100, int(magnification)),
        pan_x=pan_x,
        pan_y=pan_y,
        output_size=output_size,
        reference_magnification=100.0,
    )
    return crop.astype(np.uint8, copy=False), dict(meta)


def apply_focus(image: np.ndarray, focus: float) -> np.ndarray:
    focus_clamped = float(np.clip(focus, 0.0, 1.0))
    sigma = (1.0 - focus_clamped) * 2.5
    if sigma <= 0.02:
        return image.copy()

    if ndimage is not None:
        blur = ndimage.gaussian_filter(image.astype(np.float32), sigma=sigma)
        return np.clip(blur, 0, 255).astype(np.uint8)

    pil = Image.fromarray(image, mode="L")
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=sigma))  # type: ignore[name-defined]
    return np.asarray(blurred, dtype=np.uint8)


def _effective_objective_magnification(magnification: int) -> float:
    # UI magnification is total microscope magnification; optical objective is approximated as /10.
    return max(10.0, float(magnification) / 10.0)


def _estimate_objective_na(magnification: int) -> float:
    anchors_mag = np.asarray(
        [100.0, 200.0, 300.0, 400.0, 500.0, 600.0], dtype=np.float64
    )
    anchors_na = np.asarray([0.25, 0.40, 0.52, 0.65, 0.75, 0.85], dtype=np.float64)
    return float(np.interp(float(magnification), anchors_mag, anchors_na))


def _optical_limit_summary(magnification: int) -> dict[str, float | bool]:
    wavelength_um = 0.55
    na = _estimate_objective_na(magnification)
    resolution_limit_um = 0.61 * wavelength_um / max(na, 1e-6)
    depth_of_field_um = wavelength_um / max(na * na, 1e-6)
    max_useful_magnification = 1000.0 * na
    return {
        "objective_numerical_aperture": float(na),
        "optical_resolution_limit_um": float(resolution_limit_um),
        "approx_depth_of_field_um": float(depth_of_field_um),
        "max_useful_magnification": float(max_useful_magnification),
        "empty_magnification_risk": bool(
            float(magnification) > max_useful_magnification
        ),
    }


def _thin_lens_image_distance_mm(
    object_distance_mm: float, focal_length_mm: float
) -> float:
    safe_object_distance = max(float(object_distance_mm), float(focal_length_mm) + 1e-6)
    inv = (1.0 / float(focal_length_mm)) - (1.0 / safe_object_distance)
    return float(1.0 / max(inv, 1e-9))


@lru_cache(maxsize=_PSF_CACHE_MAX)
def _disk_kernel_cached(radius_px_key: int) -> np.ndarray:
    radius_px = float(radius_px_key) / 10000.0
    if radius_px <= 0.15:
        return np.asarray([[1.0]], dtype=np.float32)
    radius_i = max(1, int(math.ceil(radius_px)))
    yy, xx = np.mgrid[-radius_i : radius_i + 1, -radius_i : radius_i + 1]
    rr = np.sqrt(xx * xx + yy * yy)
    kernel = (rr <= float(radius_px)).astype(np.float32)
    total = float(kernel.sum())
    if total <= 0.0:
        return np.asarray([[1.0]], dtype=np.float32)
    return kernel / total


@lru_cache(maxsize=_PSF_CACHE_MAX)
def _airy_kernel_cached(first_zero_radius_px_key: int) -> np.ndarray:
    first_zero_radius_px = float(first_zero_radius_px_key) / 10000.0
    if first_zero_radius_px <= 0.15 or special is None:
        return np.asarray([[1.0]], dtype=np.float32)
    radius_i = max(2, int(math.ceil(first_zero_radius_px * 4.0)))
    yy, xx = np.mgrid[-radius_i : radius_i + 1, -radius_i : radius_i + 1]
    rr = np.sqrt(xx * xx + yy * yy).astype(np.float64)
    x = 3.8317059702075125 * rr / max(float(first_zero_radius_px), 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = (2.0 * special.j1(x) / x) ** 2
    kernel[rr == 0.0] = 1.0
    kernel = np.clip(kernel.astype(np.float32), 0.0, None)
    total = float(kernel.sum())
    if total <= 0.0:
        return np.asarray([[1.0]], dtype=np.float32)
    return kernel / total


@lru_cache(maxsize=_PSF_CACHE_MAX)
def _combined_psf_cached(
    airy_radius_px_key: int, defocus_radius_px_key: int
) -> np.ndarray:
    airy = _airy_kernel_cached(airy_radius_px_key)
    disk = _disk_kernel_cached(defocus_radius_px_key)
    if airy.shape == (1, 1):
        return disk
    if disk.shape == (1, 1):
        return airy
    if signal is None:
        return airy
    combined = signal.fftconvolve(airy, disk, mode="full").astype(np.float32)
    total = float(combined.sum())
    if total <= 0.0:
        return np.asarray([[1.0]], dtype=np.float32)
    return combined / total


def _disk_kernel(radius_px: float) -> np.ndarray:
    return _disk_kernel_cached(_cache_key_float(radius_px))


def _airy_kernel(first_zero_radius_px: float) -> np.ndarray:
    return _airy_kernel_cached(_cache_key_float(first_zero_radius_px))


def _combine_psf_kernels(
    airy_kernel: np.ndarray, disk_kernel: np.ndarray
) -> np.ndarray:
    if airy_kernel.shape == (1, 1):
        return disk_kernel
    if disk_kernel.shape == (1, 1):
        return airy_kernel
    if signal is None:
        return airy_kernel
    combined = signal.fftconvolve(airy_kernel, disk_kernel, mode="full").astype(
        np.float32
    )
    total = float(combined.sum())
    if total <= 0.0:
        return np.asarray([[1.0]], dtype=np.float32)
    return combined / total


def _physical_focus_kernel(
    *,
    magnification: int,
    focus_distance_mm: float,
    focus_target_mm: float,
    um_per_px: float,
) -> tuple[np.ndarray, dict[str, float]]:
    wavelength_um = 0.55
    tube_length_mm = 160.0
    optical_mag = _effective_objective_magnification(magnification)
    focal_length_mm = tube_length_mm / optical_mag
    na = _estimate_objective_na(magnification)

    image_distance_mm = _thin_lens_image_distance_mm(focus_distance_mm, focal_length_mm)
    image_distance_target_mm = _thin_lens_image_distance_mm(
        focus_target_mm, focal_length_mm
    )
    aperture_diameter_mm = max(1e-6, 2.0 * focal_length_mm * na)
    coc_sensor_mm = (
        aperture_diameter_mm
        * abs(image_distance_mm - image_distance_target_mm)
        / max(image_distance_mm, 1e-6)
    )
    defocus_diameter_um = (coc_sensor_mm * 1000.0) / optical_mag
    defocus_radius_px_geom = max(
        0.0, 0.5 * defocus_diameter_um / max(float(um_per_px), 1e-6)
    )

    airy_radius_um = 0.61 * wavelength_um / max(na, 1e-6)
    airy_radius_px = max(0.0, airy_radius_um / max(float(um_per_px), 1e-6))
    depth_of_field_um = wavelength_um / max(na * na, 1e-6)
    focus_error_um = abs(float(focus_distance_mm) - float(focus_target_mm)) * 1000.0
    defocus_radius_px_dof = airy_radius_px * (
        0.02 * focus_error_um / max(depth_of_field_um, 1e-6)
    )
    defocus_radius_px = max(defocus_radius_px_geom, defocus_radius_px_dof)

    airy = _airy_kernel(airy_radius_px)
    disk = _disk_kernel(min(defocus_radius_px, 64.0))
    if airy.shape == (1, 1):
        kernel = disk
    elif disk.shape == (1, 1):
        kernel = airy
    else:
        kernel = _combined_psf_cached(
            _cache_key_float(airy_radius_px),
            _cache_key_float(min(defocus_radius_px, 64.0)),
        )
    return kernel, {
        "optical_magnification": float(optical_mag),
        "focal_length_mm": float(focal_length_mm),
        "numerical_aperture": float(na),
        "image_distance_mm": float(image_distance_mm),
        "image_distance_target_mm": float(image_distance_target_mm),
        "defocus_diameter_um": float(defocus_diameter_um),
        "depth_of_field_um": float(depth_of_field_um),
        "focus_error_um": float(focus_error_um),
        "defocus_radius_px_geom": float(defocus_radius_px_geom),
        "defocus_radius_px_dof": float(defocus_radius_px_dof),
        "defocus_radius_px": float(defocus_radius_px),
        "airy_radius_um": float(airy_radius_um),
        "airy_radius_px": float(airy_radius_px),
        "psf_support_px": float(max(kernel.shape)),
    }


def apply_physical_focus(
    image: np.ndarray,
    *,
    magnification: int,
    focus_distance_mm: float,
    focus_target_mm: float,
    um_per_px: float,
) -> tuple[np.ndarray, dict[str, float]]:
    kernel, metrics = _physical_focus_kernel(
        magnification=magnification,
        focus_distance_mm=focus_distance_mm,
        focus_target_mm=focus_target_mm,
        um_per_px=um_per_px,
    )
    if kernel.shape == (1, 1):
        return image.copy(), metrics

    image_f = image.astype(np.float32, copy=False)
    kernel_f = kernel.astype(np.float32, copy=False)

    # Small PSF kernels are faster with direct convolution than with a full-image FFT.
    if ndimage is not None and max(kernel_f.shape) <= 31:
        blurred = ndimage.convolve(image_f, kernel_f, mode="constant", cval=0.0)
        return np.clip(blurred, 0.0, 255.0).astype(np.uint8), metrics

    if signal is not None:
        oa_convolve = getattr(signal, "oaconvolve", None)
        if oa_convolve is not None and max(kernel_f.shape) >= 48:
            blurred = oa_convolve(image_f, kernel_f, mode="same")
        else:
            blurred = signal.fftconvolve(image_f, kernel_f, mode="same")
        return np.clip(blurred, 0.0, 255.0).astype(np.uint8), metrics

    # Fallback path if scipy.signal is unavailable.
    radius = max(
        metrics.get("airy_radius_px", 0.0), metrics.get("defocus_radius_px", 0.0)
    )
    focus_equivalent = float(np.clip(1.0 - radius / 8.0, 0.0, 1.0))
    return apply_focus(image, focus_equivalent), metrics


def _smooth_float(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.01:
        return image.astype(np.float32, copy=False)
    if ndimage is not None:
        return ndimage.gaussian_filter(image.astype(np.float32), sigma=float(sigma))
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(
        pil.filter(ImageFilter.GaussianBlur(radius=float(sigma))), dtype=np.float32
    )


def _low_frequency_noise(shape: tuple[int, int], seed: int, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    low = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    low = _smooth_float(low, sigma=max(1.0, float(sigma)))
    low -= float(low.mean())
    std = float(low.std())
    if std > 1e-6:
        low /= std
    return low


@lru_cache(maxsize=_NOISE_CACHE_MAX)
def _low_frequency_noise_cached(
    shape0: int, shape1: int, seed: int, sigma_key: int
) -> np.ndarray:
    sigma = float(sigma_key) / 10000.0
    rng = np.random.default_rng(int(seed))
    low = rng.normal(0.0, 1.0, size=(shape0, shape1)).astype(np.float32)
    low = _smooth_float(low, sigma=max(1.0, sigma))
    low -= float(low.mean())
    std = float(low.std())
    if std > 1e-6:
        low /= std
    return low


def _illumination_mask(
    shape: tuple[int, int],
    vignette_strength: float,
    uneven_strength: float,
    *,
    magnification: int = 200,
    seed: int = 0,
) -> np.ndarray:
    return _illumination_mask_cached(
        shape0=int(shape[0]),
        shape1=int(shape[1]),
        magnification=int(magnification),
        seed=int(seed),
        vignette_strength_key=_cache_key_float(vignette_strength),
        uneven_strength_key=_cache_key_float(uneven_strength),
    )


@lru_cache(maxsize=_MASK_CACHE_MAX)
def _illumination_mask_cached(
    shape0: int,
    shape1: int,
    magnification: int,
    seed: int,
    vignette_strength_key: int,
    uneven_strength_key: int,
) -> np.ndarray:
    height, width = int(shape0), int(shape1)
    rng = np.random.default_rng(int(seed) + 17)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    cx = (width - 1) * (0.50 + rng.uniform(-0.035, 0.035))
    cy = (height - 1) * (0.50 + rng.uniform(-0.035, 0.035))
    sx = max(width * (0.86 + rng.uniform(-0.05, 0.05)), 1.0)
    sy = max(height * (0.86 + rng.uniform(-0.05, 0.05)), 1.0)
    rr = np.sqrt(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)

    vignette_level = float(np.clip(float(vignette_strength_key) / 10000.0, 0.0, 1.0))
    vignette = 1.0 - vignette_level * np.clip(
        rr ** (1.55 + 0.0006 * max(100, int(magnification))), 0.0, 1.0
    )

    hotspot_rx = max(
        width * (0.20 + 45.0 / max(120.0, float(magnification) + 60.0)), 1.0
    )
    hotspot_ry = max(
        height * (0.20 + 45.0 / max(120.0, float(magnification) + 60.0)), 1.0
    )
    hotspot = np.exp(-(((xx - cx) / hotspot_rx) ** 2 + ((yy - cy) / hotspot_ry) ** 2))

    tilt_x = rng.uniform(-0.08, 0.08)
    tilt_y = rng.uniform(-0.08, 0.08)
    plane = (
        1.0
        + tilt_x * ((xx / max(width - 1, 1)) - 0.5)
        + tilt_y * ((yy / max(height - 1, 1)) - 0.5)
    )

    sigma = max(height, width) * (0.16 + 40.0 / max(float(magnification), 120.0))
    low = _low_frequency_noise_cached(
        height, width, int(seed) + 29, _cache_key_float(sigma)
    )
    uneven = 1.0 + float(np.clip(float(uneven_strength_key) / 10000.0, 0.0, 1.0)) * (
        0.11 * low + 0.07 * hotspot
    )

    mask = vignette * plane * uneven
    return np.clip(mask, 0.42, 1.28).astype(np.float32)


def apply_lighting(
    image: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0,
    vignette_strength: float = 0.2,
    uneven_strength: float = 0.1,
    *,
    magnification: int = 200,
    seed: int = 0,
) -> np.ndarray:
    img = image.astype(np.float32)
    img = (img - 127.5) * float(contrast) + 127.5
    img = img + (float(brightness) - 1.0) * 50.0

    mask = _illumination_mask(
        shape=image.shape,
        vignette_strength=vignette_strength,
        uneven_strength=uneven_strength,
        magnification=magnification,
        seed=seed,
    )
    img *= mask
    return np.clip(img, 0, 255).astype(np.uint8)


def add_artifacts(
    image: np.ndarray,
    seed: int,
    noise_sigma: float = 4.0,
    add_dust: bool = False,
    add_scratches: bool = False,
    etch_uneven: float = 0.0,
    *,
    magnification: int = 200,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    img = image.astype(np.float32)

    sigma = max(0.0, float(noise_sigma))
    if sigma > 0:
        shot = rng.normal(0.0, 1.0, size=img.shape).astype(np.float32)
        shot *= np.sqrt(np.clip(img, 8.0, 255.0) / 255.0) * (0.55 * sigma)
        read = rng.normal(0.0, 0.42 * sigma, size=img.shape).astype(np.float32)
        fixed_pattern = _low_frequency_noise_cached(
            int(img.shape[0]),
            int(img.shape[1]),
            int(seed) + 71,
            _cache_key_float(max(img.shape) * 0.22),
        ) * (0.08 * sigma)
        img += shot + read + fixed_pattern

    if etch_uneven > 0:
        low = _low_frequency_noise_cached(
            int(img.shape[0]),
            int(img.shape[1]),
            int(seed) + 113,
            _cache_key_float(max(img.shape) * 0.10),
        )
        ripple = _low_frequency_noise_cached(
            int(img.shape[0]),
            int(img.shape[1]),
            int(seed) + 127,
            _cache_key_float(max(img.shape) * 0.035),
        )
        etch_term = (0.72 * low + 0.28 * ripple) * float(np.clip(etch_uneven, 0.0, 1.0))
        img += etch_term * (12.0 + 0.014 * max(100, int(magnification)))

    if add_dust:
        halo = np.zeros_like(img, dtype=np.float32)
        dust = Image.new("L", (img.shape[1], img.shape[0]), 0)
        draw = ImageDraw.Draw(dust)
        dust_count = max(3, int((img.shape[0] * img.shape[1]) / 95_000))
        for _ in range(dust_count):
            cx = float(rng.uniform(0, img.shape[1] - 1))
            cy = float(rng.uniform(0, img.shape[0] - 1))
            radius = float(rng.uniform(1.4, 4.6))
            tone = int(rng.integers(16, 80))
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius), fill=tone
            )
            yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
            rr2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / max((radius * 2.7) ** 2, 1e-6)
            halo += np.exp(-rr2).astype(np.float32) * float(rng.uniform(2.0, 6.0))
        dust_arr = np.asarray(dust, dtype=np.float32)
        dust_arr = _smooth_float(dust_arr, sigma=0.9)
        img -= dust_arr * 0.45
        img += halo * 0.22

    if add_scratches:
        scratch_mask = Image.new("L", (img.shape[1], img.shape[0]), 0)
        draw = ImageDraw.Draw(scratch_mask)
        scratch_count = max(2, int(img.shape[1] / 360))
        for _ in range(scratch_count):
            angle = float(
                rng.uniform(-35.0, 35.0) + rng.choice([0.0, 90.0], p=[0.7, 0.3])
            )
            theta = math.radians(angle)
            length = float(rng.uniform(img.shape[1] * 0.18, img.shape[1] * 0.72))
            cx = float(rng.uniform(0, img.shape[1] - 1))
            cy = float(rng.uniform(0, img.shape[0] - 1))
            dx = math.cos(theta) * length * 0.5
            dy = math.sin(theta) * length * 0.5
            width = max(1, int(rng.integers(1, 3)))
            tone = int(rng.integers(80, 170))
            draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=tone, width=width)
        scratch_arr = _smooth_float(
            np.asarray(scratch_mask, dtype=np.float32), sigma=0.55
        )
        img -= scratch_arr * 0.18
        img += _smooth_float(scratch_arr, sigma=1.2) * 0.05

    return np.clip(img, 0, 255).astype(np.uint8)


def simulate_microscope_view(
    sample: np.ndarray,
    optical_ferromagnetic_mask_sample: np.ndarray | None = None,
    magnification: int = 200,
    pan_x: float = 0.5,
    pan_y: float = 0.5,
    output_size: tuple[int, int] = (1024, 1024),
    focus: float = 1.0,
    focus_distance_mm: float | None = None,
    focus_target_mm: float | None = None,
    focus_quality: float | None = None,
    um_per_px_100x: float | None = None,
    brightness: float = 1.0,
    contrast: float = 1.0,
    vignette_strength: float = 0.15,
    uneven_strength: float = 0.08,
    noise_sigma: float = 3.5,
    add_dust: bool = False,
    add_scratches: bool = False,
    etch_uneven: float = 0.0,
    optical_mode: str = "brightfield",
    optical_mode_parameters: dict[str, Any] | None = None,
    optical_context: dict[str, Any] | None = None,
    psf_profile: str = "standard",
    psf_strength: float = 0.0,
    sectioning_shear_deg: float = 35.0,
    hybrid_balance: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_key = _sample_view_cache_key(
        sample=sample,
        optical_ferromagnetic_mask_sample=optical_ferromagnetic_mask_sample,
        magnification=magnification,
        pan_x=pan_x,
        pan_y=pan_y,
        output_size=output_size,
        focus=focus,
        focus_distance_mm=focus_distance_mm,
        focus_target_mm=focus_target_mm,
        focus_quality=focus_quality,
        um_per_px_100x=um_per_px_100x,
        brightness=brightness,
        contrast=contrast,
        vignette_strength=vignette_strength,
        uneven_strength=uneven_strength,
        noise_sigma=noise_sigma,
        add_dust=add_dust,
        add_scratches=add_scratches,
        etch_uneven=etch_uneven,
        optical_mode=optical_mode,
        psf_profile=psf_profile,
        psf_strength=psf_strength,
        sectioning_shear_deg=sectioning_shear_deg,
        hybrid_balance=hybrid_balance,
        optical_mode_parameters=_stable_signature_items(optical_mode_parameters),
        optical_context_signature=_stable_signature_items(optical_context),
        seed=seed,
    )
    cached = _cache_touch(_VIEW_CACHE, cache_key)
    if cached is not None:
        return cached[0], dict(cached[1])

    view, fov_meta = extract_field_of_view(
        sample=sample,
        magnification=magnification,
        pan_x=pan_x,
        pan_y=pan_y,
        output_size=output_size,
    )
    ferromagnetic_mask_view = None
    if (
        isinstance(optical_ferromagnetic_mask_sample, np.ndarray)
        and optical_ferromagnetic_mask_sample.shape == sample.shape
    ):
        ferromagnetic_mask_view, _ = extract_field_of_view(
            sample=optical_ferromagnetic_mask_sample.astype(np.uint8, copy=False),
            magnification=magnification,
            pan_x=pan_x,
            pan_y=pan_y,
            output_size=output_size,
        )
    focus_distance_value = (
        None if focus_distance_mm is None else float(focus_distance_mm)
    )
    focus_target_value = None if focus_target_mm is None else float(focus_target_mm)
    um_per_px = estimate_um_per_px(
        um_per_px_100x=um_per_px_100x,
        crop_size_px=fov_meta.get("crop_size_px"),
        output_size_px=output_size,
    )

    physical_metrics: dict[str, float] = {}
    if focus_distance_value is not None and focus_target_value is not None:
        focused, physical_metrics = apply_physical_focus(
            view,
            magnification=magnification,
            focus_distance_mm=focus_distance_value,
            focus_target_mm=focus_target_value,
            um_per_px=um_per_px,
        )
        focus_error = abs(focus_distance_value - focus_target_value)
        tolerance_mm = max(0.18, 0.75 - 0.0009 * float(magnification))
        effective_focus = float(np.clip(1.0 - focus_error / tolerance_mm, 0.0, 1.0))
    elif focus_quality is not None:
        effective_focus = float(np.clip(focus_quality, 0.0, 1.0))
        focused = apply_focus(view, focus=effective_focus)
    else:
        effective_focus = float(focus)
        focused = apply_focus(view, focus=effective_focus)
    focused, psf_meta = apply_live_psf_profile(
        original_view=view,
        focused_view=focused,
        microscope_profile={
            "psf_profile": psf_profile,
            "psf_strength": psf_strength,
            "sectioning_shear_deg": sectioning_shear_deg,
            "hybrid_balance": hybrid_balance,
        },
        focus_distance_mm=focus_distance_value,
        focus_target_mm=focus_target_value,
        focus_quality=effective_focus,
    )
    pure_iron_like = bool(
        dict(optical_context or {}).get("pure_iron_baseline_applied", False)
    )
    lighting_brightness = float(brightness)
    lighting_vignette = float(vignette_strength)
    lighting_uneven = float(uneven_strength)
    artifact_noise = float(noise_sigma)
    if pure_iron_like:
        lighting_brightness = max(lighting_brightness, 1.04)
        lighting_vignette *= 0.35
        lighting_uneven *= 0.45
        artifact_noise *= 0.35
    optical_mode_meta: dict[str, Any] = {}
    focused_mode, optical_mode_meta = apply_optical_mode_transfer(
        image_gray=focused,
        optical_mode=optical_mode,
        optical_mode_parameters=optical_mode_parameters,
        ferromagnetic_mask=(
            None
            if ferromagnetic_mask_view is None
            else np.clip(ferromagnetic_mask_view.astype(np.float32) / 255.0, 0.0, 1.0)
        ),
        pure_iron_like=pure_iron_like,
    )
    lit = apply_lighting(
        focused_mode,
        brightness=lighting_brightness,
        contrast=contrast,
        vignette_strength=lighting_vignette,
        uneven_strength=lighting_uneven,
        magnification=magnification,
        seed=seed,
    )
    artifacted = add_artifacts(
        lit,
        seed=seed,
        noise_sigma=artifact_noise,
        add_dust=add_dust,
        add_scratches=add_scratches,
        etch_uneven=etch_uneven,
        magnification=magnification,
    )
    if (
        pure_iron_like
        and str(optical_mode or "brightfield").strip().lower() == "brightfield"
    ):
        arr = artifacted.astype(np.float32)
        dark_floor = float(np.quantile(arr, 0.05))
        arr += max(0.0, 128.0 - dark_floor)
        artifacted = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    fov_meta["focus_distance_mm"] = focus_distance_value
    fov_meta["focus_target_mm"] = focus_target_value
    fov_meta["focus_error_mm"] = (
        None
        if focus_distance_value is None or focus_target_value is None
        else float(abs(focus_distance_value - focus_target_value))
    )
    fov_meta["focus_quality"] = float(effective_focus)
    fov_meta["um_per_px"] = float(um_per_px)
    fov_meta["optical_mode"] = str(optical_mode or "brightfield")
    fov_meta["optical_mode_parameters"] = dict(
        optical_mode_meta.get("mode_parameters", {})
    )
    fov_meta["psf_profile"] = str(psf_meta.get("psf_profile", "standard"))
    fov_meta["psf_strength"] = float(psf_meta.get("psf_strength", 0.0))
    fov_meta["sectioning_shear_deg"] = float(psf_meta.get("sectioning_shear_deg", 35.0))
    fov_meta["hybrid_balance"] = float(psf_meta.get("hybrid_balance", 0.5))
    fov_meta["focus_profile_mode"] = str(psf_meta.get("focus_profile_mode", "standard"))
    fov_meta["axial_profile_mode"] = str(psf_meta.get("axial_profile_mode", "standard"))
    fov_meta["psf_engineering_applied"] = bool(
        psf_meta.get("psf_engineering_applied", False)
    )
    fov_meta["effective_dof_factor"] = float(psf_meta.get("effective_dof_factor", 1.0))
    fov_meta["sectioning_active"] = bool(psf_meta.get("sectioning_active", False))
    fov_meta["sectioning_suppression_score"] = float(
        psf_meta.get("sectioning_suppression_score", 0.0)
    )
    fov_meta["axial_shift_signature"] = float(
        psf_meta.get("lateral_shift_signature", 0.0)
    )
    fov_meta["extended_dof_retention_score"] = float(
        psf_meta.get("extended_dof_retention_score", 0.0)
    )
    fov_meta["sectioning_directionality_score"] = float(
        psf_meta.get("sectioning_directionality_score", 0.0)
    )
    fov_meta["optical_limit_summary"] = _optical_limit_summary(magnification)
    fov_meta.update(dict(optical_mode_meta.get("mode_parameters", {})))
    if pure_iron_like:
        fov_meta["pure_iron_baseline_applied"] = True
    if psf_meta.get("rotation_signature_deg") is not None:
        fov_meta["rotation_signature_deg"] = float(psf_meta["rotation_signature_deg"])
    if physical_metrics:
        fov_meta["focus_optics"] = dict(physical_metrics)

    _cache_set(
        _VIEW_CACHE, cache_key, (artifacted, dict(fov_meta)), max_size=_VIEW_CACHE_MAX
    )
    return artifacted, fov_meta
