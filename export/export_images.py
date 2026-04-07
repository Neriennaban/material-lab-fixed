"""Image export utilities."""

import json
from pathlib import Path
from typing import Iterable, Mapping, Union

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


def save_phase_masks(
    phase_masks: Mapping[str, np.ndarray],
    *,
    output_dir: Union[str, Path],
    base_name: str = "phase_mask",
    write_legend: bool = True,
) -> dict[str, Union[Path, dict]]:
    """C2 — export per-phase binary masks for ML datasets.

    Each mask is written as ``{base_name}_{INDEX}_{PHASE_LOWER}.png``
    (8-bit PNG, 0/255). When ``write_legend`` is true an additional
    ``{base_name}_legend.json`` file maps the integer index back to
    the phase name and stores per-phase pixel coverage statistics
    (fraction of image area).

    Returns a dict with the saved paths under ``"masks"``, the
    legend dict under ``"legend"`` and the legend file path under
    ``"legend_path"`` (when ``write_legend`` is enabled).
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    legend_entries: list[dict] = []
    total_pixels: int | None = None

    for index, (phase_name, mask) in enumerate(phase_masks.items()):
        if not isinstance(mask, np.ndarray):
            continue
        if mask.ndim != 2:
            raise ValueError(
                f"phase mask '{phase_name}' must be 2D, got shape {mask.shape}"
            )
        binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
        if total_pixels is None:
            total_pixels = int(binary.size)
        slug = str(phase_name).strip().lower().replace(" ", "_")
        out_path = output_root / f"{base_name}_{index:02d}_{slug}.png"
        Image.fromarray(binary, mode="L").save(out_path)
        saved[str(phase_name).upper()] = out_path

        if write_legend:
            coverage = float(int((binary > 0).sum())) / float(max(1, binary.size))
            legend_entries.append(
                {
                    "index": index,
                    "phase": str(phase_name).upper(),
                    "file": out_path.name,
                    "coverage_fraction": coverage,
                }
            )

    legend: dict = {
        "base_name": base_name,
        "total_pixels": total_pixels,
        "phase_count": len(saved),
        "phases": legend_entries,
    }
    legend_path: Path | None = None
    if write_legend:
        legend_path = output_root / f"{base_name}_legend.json"
        legend_path.write_text(
            json.dumps(legend, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    result: dict[str, Union[Path, dict]] = {
        "masks": saved,  # type: ignore[dict-item]
        "legend": legend,
    }
    if legend_path is not None:
        result["legend_path"] = legend_path
    return result
