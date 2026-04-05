from __future__ import annotations

import math
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:  # pragma: no cover - optional acceleration
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


_ARRAY_SLIDE_CACHE_MAX = 16
_PATH_SLIDE_CACHE_MAX = 32
_CACHE_LOCK = threading.RLock()
_ARRAY_SLIDE_CACHE: OrderedDict[tuple[Any, ...], tuple[weakref.ReferenceType[np.ndarray], "VirtualSlide"]] = OrderedDict()
_PATH_SLIDE_CACHE: OrderedDict[tuple[Any, ...], "VirtualSlide"] = OrderedDict()


@dataclass(frozen=True, slots=True)
class PyramidLevel:
    image: np.ndarray
    scale: float

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.image.shape)


@dataclass(slots=True)
class VirtualSlide:
    levels: tuple[PyramidLevel, ...]
    base_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("VirtualSlide requires at least one pyramid level")

    def _score_level(self, *, level: PyramidLevel, crop_size_px: tuple[int, int], output_size: tuple[int, int]) -> float:
        crop_h = max(1.0, float(crop_size_px[0]) * float(level.scale))
        crop_w = max(1.0, float(crop_size_px[1]) * float(level.scale))
        out_h = max(1.0, float(output_size[0]))
        out_w = max(1.0, float(output_size[1]))
        ratio_h = out_h / crop_h
        ratio_w = out_w / crop_w
        worst_ratio = max(ratio_h, ratio_w, 1.0 / max(ratio_h, 1e-6), 1.0 / max(ratio_w, 1e-6))
        # Prefer near-1 resampling, but slightly tolerate moderate upscaling for sharper results.
        bias = 0.08 if ratio_h > 1.0 or ratio_w > 1.0 else 0.0
        return abs(math.log2(max(worst_ratio, 1e-6))) + bias

    def choose_level(self, *, crop_size_px: tuple[int, int], output_size: tuple[int, int] | None) -> tuple[int, PyramidLevel]:
        if output_size is None:
            return 0, self.levels[0]
        best_idx = 0
        best_score = float("inf")
        for idx, level in enumerate(self.levels):
            score = self._score_level(level=level, crop_size_px=crop_size_px, output_size=output_size)
            if score < best_score:
                best_idx = idx
                best_score = score
        return best_idx, self.levels[best_idx]

    def extract_pixels(
        self,
        *,
        origin_px: tuple[int, int],
        crop_size_px: tuple[int, int],
        output_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        base_h, base_w = self.base_shape
        y0_base = int(np.clip(int(origin_px[0]), 0, max(base_h - 1, 0)))
        x0_base = int(np.clip(int(origin_px[1]), 0, max(base_w - 1, 0)))
        crop_h_base = int(np.clip(int(crop_size_px[0]), 1, max(base_h - y0_base, 1)))
        crop_w_base = int(np.clip(int(crop_size_px[1]), 1, max(base_w - x0_base, 1)))

        level_idx, level = self.choose_level(crop_size_px=(crop_h_base, crop_w_base), output_size=output_size)
        scale = float(level.scale)
        image = level.image
        lvl_h, lvl_w = int(image.shape[0]), int(image.shape[1])

        y0_lvl = int(round(float(y0_base) * scale))
        x0_lvl = int(round(float(x0_base) * scale))
        crop_h_lvl = max(1, int(round(float(crop_h_base) * scale)))
        crop_w_lvl = max(1, int(round(float(crop_w_base) * scale)))

        y0_lvl = int(np.clip(y0_lvl, 0, max(lvl_h - 1, 0)))
        x0_lvl = int(np.clip(x0_lvl, 0, max(lvl_w - 1, 0)))
        crop_h_lvl = int(np.clip(crop_h_lvl, 1, max(lvl_h - y0_lvl, 1)))
        crop_w_lvl = int(np.clip(crop_w_lvl, 1, max(lvl_w - x0_lvl, 1)))

        crop = image[y0_lvl : y0_lvl + crop_h_lvl, x0_lvl : x0_lvl + crop_w_lvl]
        if output_size is not None and (crop.shape[0] != int(output_size[0]) or crop.shape[1] != int(output_size[1])):
            crop = _resize_u8(crop, output_size, downscale_preference=False)

        return crop.astype(np.uint8, copy=False), {
            "crop_origin_px": [int(y0_base), int(x0_base)],
            "crop_size_px": [int(crop_h_base), int(crop_w_base)],
            "crop_origin_px_level": [int(y0_lvl), int(x0_lvl)],
            "crop_size_px_level": [int(crop_h_lvl), int(crop_w_lvl)],
            "pyramid_level": int(level_idx),
            "pyramid_scale": float(scale),
            "level_shape_px": [int(lvl_h), int(lvl_w)],
        }

    def extract_normalized(
        self,
        *,
        magnification: int,
        pan_x: float = 0.5,
        pan_y: float = 0.5,
        output_size: tuple[int, int] | None = None,
        reference_magnification: float = 100.0,
        minimum_crop_px: int = 32,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        base_h, base_w = self.base_shape
        mag = max(int(reference_magnification), int(magnification))
        ratio = float(reference_magnification) / float(mag)
        crop_h = max(minimum_crop_px, int(round(base_h * ratio)))
        crop_w = max(minimum_crop_px, int(round(base_w * ratio)))
        crop_h = min(crop_h, base_h)
        crop_w = min(crop_w, base_w)

        pan_x_safe = float(np.clip(pan_x, 0.0, 1.0))
        pan_y_safe = float(np.clip(pan_y, 0.0, 1.0))
        x0 = int(round((base_w - crop_w) * pan_x_safe))
        y0 = int(round((base_h - crop_h) * pan_y_safe))
        x0 = max(0, min(x0, base_w - crop_w))
        y0 = max(0, min(y0, base_h - crop_h))

        crop, meta = self.extract_pixels(
            origin_px=(y0, x0),
            crop_size_px=(crop_h, crop_w),
            output_size=output_size,
        )
        meta["area_fraction_of_sample"] = float((crop_h * crop_w) / max(1.0, float(base_h * base_w)))
        meta["reference_magnification"] = float(reference_magnification)
        meta["requested_magnification"] = int(magnification)
        return crop, meta


def _cache_put(cache: OrderedDict, key: tuple[Any, ...], value: Any, *, max_size: int) -> Any:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)
    return value


def _pil_resize(image: np.ndarray, size: tuple[int, int], *, resample: int) -> np.ndarray:
    if image.ndim == 2:
        pil = Image.fromarray(image.astype(np.uint8, copy=False), mode="L")
    elif image.ndim == 3 and image.shape[2] == 3:
        pil = Image.fromarray(image.astype(np.uint8, copy=False), mode="RGB")
    elif image.ndim == 3 and image.shape[2] == 4:
        pil = Image.fromarray(image.astype(np.uint8, copy=False), mode="RGBA")
    else:
        raise ValueError(f"Unsupported image shape for resize: {image.shape}")
    resized = pil.resize((int(size[1]), int(size[0])), resample=resample)
    return np.asarray(resized, dtype=np.uint8)


def _resize_u8(image: np.ndarray, size: tuple[int, int], *, downscale_preference: bool = True) -> np.ndarray:
    target_h = max(1, int(size[0]))
    target_w = max(1, int(size[1]))
    if int(image.shape[0]) == target_h and int(image.shape[1]) == target_w:
        return image.astype(np.uint8, copy=False)

    if cv2 is not None:
        if downscale_preference and target_h <= int(image.shape[0]) and target_w <= int(image.shape[1]):
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        return np.asarray(resized, dtype=np.uint8)

    resample = Image.Resampling.BILINEAR
    if downscale_preference and target_h <= int(image.shape[0]) and target_w <= int(image.shape[1]):
        resample = Image.Resampling.BOX
    return _pil_resize(image, (target_h, target_w), resample=resample)


def build_slide_pyramid(
    image: np.ndarray,
    *,
    min_side: int = 96,
    downscale: float = 2.0,
    max_levels: int = 8,
) -> VirtualSlide:
    arr = np.asarray(image)
    if arr.ndim not in {2, 3}:
        raise ValueError(f"Unsupported image rank for pyramid: {arr.ndim}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    levels: list[PyramidLevel] = [PyramidLevel(image=arr, scale=1.0)]
    current = arr
    current_scale = 1.0
    for _ in range(max_levels - 1):
        h, w = int(current.shape[0]), int(current.shape[1])
        if min(h, w) <= int(min_side):
            break
        next_h = max(1, int(round(h / float(downscale))))
        next_w = max(1, int(round(w / float(downscale))))
        if next_h == h and next_w == w:
            break
        current = _resize_u8(current, (next_h, next_w), downscale_preference=True)
        current_scale = float(current.shape[0]) / float(arr.shape[0])
        levels.append(PyramidLevel(image=current, scale=current_scale))
        if min(next_h, next_w) <= int(min_side):
            break
    return VirtualSlide(levels=tuple(levels), base_shape=(int(arr.shape[0]), int(arr.shape[1])))


def _array_cache_key(array: np.ndarray) -> tuple[Any, ...]:
    return (
        int(id(array)),
        int(array.__array_interface__["data"][0]),
        tuple(int(v) for v in array.shape),
        str(array.dtype),
        int(array.strides[0]) if array.ndim >= 1 else 0,
        int(array.strides[1]) if array.ndim >= 2 else 0,
    )


def get_array_slide(array: np.ndarray) -> VirtualSlide:
    arr = np.asarray(array)
    key = _array_cache_key(arr)
    with _CACHE_LOCK:
        cached = _ARRAY_SLIDE_CACHE.get(key)
        if cached is not None:
            ref, slide = cached
            if ref() is arr:
                _ARRAY_SLIDE_CACHE.move_to_end(key)
                return slide
            _ARRAY_SLIDE_CACHE.pop(key, None)
        slide = build_slide_pyramid(arr)
        ref = weakref.ref(arr)
        _cache_put(_ARRAY_SLIDE_CACHE, key, (ref, slide), max_size=_ARRAY_SLIDE_CACHE_MAX)
        return slide


def get_path_slide(path: str | Path) -> VirtualSlide:
    p = Path(path).resolve()
    stat = p.stat()
    key = (str(p), int(stat.st_mtime_ns), int(stat.st_size))
    with _CACHE_LOCK:
        cached = _PATH_SLIDE_CACHE.get(key)
        if cached is not None:
            _PATH_SLIDE_CACHE.move_to_end(key)
            return cached
    with Image.open(p) as img:
        if img.mode not in {"L", "RGB", "RGBA"}:
            img = img.convert("L")
        arr = np.asarray(img, dtype=np.uint8)
    slide = build_slide_pyramid(arr)
    with _CACHE_LOCK:
        return _cache_put(_PATH_SLIDE_CACHE, key, slide, max_size=_PATH_SLIDE_CACHE_MAX)


def extract_field_of_view_from_array(
    sample: np.ndarray,
    *,
    magnification: int,
    pan_x: float = 0.5,
    pan_y: float = 0.5,
    output_size: tuple[int, int] | None = None,
    reference_magnification: float = 100.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    slide = get_array_slide(sample)
    return slide.extract_normalized(
        magnification=magnification,
        pan_x=pan_x,
        pan_y=pan_y,
        output_size=output_size,
        reference_magnification=reference_magnification,
    )


__all__ = [
    "PyramidLevel",
    "VirtualSlide",
    "build_slide_pyramid",
    "extract_field_of_view_from_array",
    "get_array_slide",
    "get_path_slide",
]
