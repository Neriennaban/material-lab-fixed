"""Image export utilities."""

from pathlib import Path
from typing import Iterable, Union

import numpy as np
from PIL import Image


def _to_pil_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}")

    arr = image
    if arr.dtype != np.uint8:
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


def save_image(image: Union[np.ndarray, Image.Image], path: Union[str, Path]) -> Path:
    """
    Save an image to disk.

    Args:
        image: RGB image as numpy array or PIL Image
        path: Output file path

    Returns:
        Path object of saved file
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    _to_pil_image(image).save(output)

    return output


def save_image_bundle(
    image: Union[np.ndarray, Image.Image],
    *,
    output_dir: Union[str, Path],
    base_name: str,
    formats: Iterable[str],
) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        suffix = str(fmt).strip().lstrip(".").lower()
        if not suffix:
            continue
        path = output_root / f"{base_name}.{suffix}"
        save_image(image, path)
        saved.append(path)
    return saved
