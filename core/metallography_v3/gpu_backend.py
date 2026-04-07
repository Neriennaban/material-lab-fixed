"""C4 — optional GPU acceleration helpers (CuPy fallback).

The renderer is CPU-bound by design so the project does not require
any GPU dependency. When ``cupy`` is installed and a GPU is available
this module exposes a tiny helper that mirrors a few hot numpy ops
on the GPU; on systems without ``cupy`` it transparently falls back
to numpy so the rest of the codebase keeps working unchanged.

The intent is *additive*: nothing in the existing pipeline imports
this module yet — it is opt-in for callers that explicitly want the
acceleration. Tests cover the public surface and verify that the
fallback path is byte-equivalent to plain numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    import cupy as _cupy  # type: ignore

    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - intentional broad guard
    _cupy = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False


def is_gpu_available() -> bool:
    """Return ``True`` when ``cupy`` is importable and the runtime
    reports at least one CUDA device. The check is purely diagnostic;
    callers should still treat any CuPy operation as optional and
    catch ``ImportError`` / ``RuntimeError`` themselves."""
    if not _CUPY_AVAILABLE or _cupy is None:
        return False
    try:  # pragma: no cover - GPU runtime side effect
        return bool(int(_cupy.cuda.runtime.getDeviceCount()) > 0)
    except Exception:
        return False


def to_device(array: np.ndarray, *, use_gpu: bool = False) -> Any:
    """Move ``array`` to the GPU when ``use_gpu`` is true and CuPy is
    available; otherwise return the original numpy array unchanged.
    The return type is intentionally ``Any`` because callers must
    treat the result as opaque and use ``to_host`` to come back."""
    if use_gpu and _CUPY_AVAILABLE and _cupy is not None:
        try:  # pragma: no cover - GPU side effect
            return _cupy.asarray(array)
        except Exception:
            return array
    return array


def to_host(array: Any) -> np.ndarray:
    """Bring an array back to the host as a numpy array. Works on
    both numpy and CuPy inputs."""
    if _CUPY_AVAILABLE and _cupy is not None and isinstance(array, _cupy.ndarray):
        try:  # pragma: no cover - GPU side effect
            return _cupy.asnumpy(array)
        except Exception:
            return np.asarray(array)
    return np.asarray(array)


def gaussian_blur(
    array: np.ndarray,
    *,
    sigma: float,
    use_gpu: bool = False,
) -> np.ndarray:
    """Gaussian blur with optional CuPy acceleration. Falls back to
    ``scipy.ndimage`` (or a manual box-blur approximation) when GPU
    is not available."""
    if use_gpu and _CUPY_AVAILABLE and _cupy is not None:
        try:  # pragma: no cover - GPU side effect
            from cupyx.scipy import ndimage as _cupy_ndimage  # type: ignore

            gpu_array = _cupy.asarray(array.astype(np.float32))
            blurred = _cupy_ndimage.gaussian_filter(gpu_array, sigma=float(sigma))
            return _cupy.asnumpy(blurred)
        except Exception:
            pass
    try:
        from scipy import ndimage  # type: ignore

        return ndimage.gaussian_filter(
            array.astype(np.float32), sigma=float(max(0.05, sigma))
        )
    except Exception:
        # Fallback box-blur with the same shape and dtype.
        radius = int(round(max(0.0, sigma)))
        if radius == 0:
            return array.astype(np.float32)
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
        kernel /= float(kernel.sum())
        from numpy.lib.stride_tricks import sliding_window_view

        padded = np.pad(array.astype(np.float32), radius, mode="reflect")
        windows = sliding_window_view(padded, kernel.shape)
        return np.einsum("ijkl,kl->ij", windows, kernel)


def describe_backend() -> dict[str, Any]:
    """Diagnostic dict consumed by tests and CLI ``--info`` output."""
    return {
        "cupy_installed": _CUPY_AVAILABLE,
        "gpu_available": is_gpu_available(),
        "fallback": "scipy.ndimage" if not is_gpu_available() else "cupy",
    }
