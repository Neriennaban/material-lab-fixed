"""Tests for the C4 optional GPU backend (CuPy fallback).

CuPy is *not* a hard dependency, so the tests run on the CPU
fallback path. They check the public surface and the byte-equivalence
of the fallback Gaussian blur against ``scipy.ndimage``.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.metallography_v3 import gpu_backend


class GpuBackendDescribeTest(unittest.TestCase):
    def test_describe_backend_returns_expected_keys(self) -> None:
        info = gpu_backend.describe_backend()
        self.assertIn("cupy_installed", info)
        self.assertIn("gpu_available", info)
        self.assertIn("fallback", info)
        # CuPy is not a project dependency — must be False on CI.
        self.assertFalse(info["gpu_available"])

    def test_is_gpu_available_returns_bool(self) -> None:
        self.assertIsInstance(gpu_backend.is_gpu_available(), bool)


class GpuBackendArrayHelpersTest(unittest.TestCase):
    def test_to_device_passthrough_without_gpu(self) -> None:
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = gpu_backend.to_device(arr, use_gpu=False)
        self.assertIs(result, arr)

    def test_to_device_passthrough_when_no_cupy(self) -> None:
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        # When CuPy is not available the function returns the original
        # array even when ``use_gpu=True``.
        result = gpu_backend.to_device(arr, use_gpu=True)
        self.assertIsInstance(result, np.ndarray)

    def test_to_host_returns_numpy_array(self) -> None:
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = gpu_backend.to_host(arr)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, arr))


class GpuBackendBlurTest(unittest.TestCase):
    def test_fallback_matches_scipy_gaussian(self) -> None:
        from scipy import ndimage

        arr = np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32)
        sigma = 1.5
        a = gpu_backend.gaussian_blur(arr, sigma=sigma, use_gpu=False)
        b = ndimage.gaussian_filter(arr, sigma=sigma)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_blur_returns_same_shape(self) -> None:
        arr = np.zeros((32, 48), dtype=np.float32)
        result = gpu_backend.gaussian_blur(arr, sigma=2.0)
        self.assertEqual(result.shape, arr.shape)

    def test_zero_sigma_returns_finite_image(self) -> None:
        arr = np.ones((16, 16), dtype=np.float32)
        result = gpu_backend.gaussian_blur(arr, sigma=0.0)
        self.assertEqual(result.shape, arr.shape)
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == "__main__":
    unittest.main()
