from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from core.virtual_slide import extract_field_of_view_from_array, get_array_slide, get_path_slide


class VirtualSlideRuntimeTests(unittest.TestCase):
    def test_array_slide_is_cached_by_identity(self) -> None:
        arr = np.arange(512 * 512, dtype=np.uint32).reshape(512, 512) % 251
        arr = arr.astype(np.uint8)
        slide_a = get_array_slide(arr)
        slide_b = get_array_slide(arr)
        self.assertIs(slide_a, slide_b)
        self.assertGreaterEqual(len(slide_a.levels), 2)

    def test_extract_field_of_view_uses_multiresolution_metadata(self) -> None:
        yy, xx = np.mgrid[0:4096, 0:4096]
        arr = ((xx * 3 + yy * 5) % 256).astype(np.uint8)
        crop, meta = extract_field_of_view_from_array(
            arr,
            magnification=400,
            pan_x=0.65,
            pan_y=0.25,
            output_size=(512, 512),
        )
        self.assertEqual(crop.shape, (512, 512))
        self.assertIn("pyramid_level", meta)
        self.assertIn("pyramid_scale", meta)
        self.assertGreaterEqual(int(meta["pyramid_level"]), 1)
        self.assertEqual(meta["requested_magnification"], 400)
        self.assertEqual(meta["crop_size_px"], [1024, 1024])

    def test_path_slide_extract_matches_manual_crop_semantics(self) -> None:
        arr = np.zeros((600, 900), dtype=np.uint8)
        arr[120:360, 200:500] = 255
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mask.png"
            Image.fromarray(arr, mode="L").save(path)
            slide = get_path_slide(path)
            crop, meta = slide.extract_pixels(
                origin_px=(120, 200),
                crop_size_px=(240, 300),
                output_size=(240, 300),
            )
        self.assertEqual(crop.shape, (240, 300))
        self.assertGreater(float(crop.mean()), 250.0)
        self.assertEqual(meta["crop_origin_px"], [120, 200])
        self.assertEqual(meta["crop_size_px"], [240, 300])


if __name__ == "__main__":
    unittest.main()
