from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


@dataclass(slots=True)
class JeffriesResult:
    p_inside: float
    q_intercepted: float
    equivalent_grains_n: float
    area_mm2: float
    grains_per_mm2: float
    astm_grain_size_number: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def jeffries_method(
    p_inside: float, q_intercepted: float, area_mm2: float
) -> JeffriesResult:
    """
    Classic educational Jeffries count:
    N = p + q/2
    N_A = N / area
    """

    area = max(float(area_mm2), 1e-12)
    n_equivalent = float(p_inside) + float(q_intercepted) * 0.5
    n_a = n_equivalent / area

    # ASTM grain size estimate (approximate educational conversion).
    n_100 = n_a * 645.16
    astm = 1.0 + math.log2(max(n_100, 1e-12))
    return JeffriesResult(
        p_inside=float(p_inside),
        q_intercepted=float(q_intercepted),
        equivalent_grains_n=n_equivalent,
        area_mm2=area,
        grains_per_mm2=n_a,
        astm_grain_size_number=astm,
    )


@dataclass(slots=True)
class DislocationDensityResult:
    counts: list[int]
    areas_mm2: list[float]
    densities_per_mm2: list[float]
    average_density_per_mm2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dislocation_density(count: int, area_mm2: float) -> float:
    area = max(float(area_mm2), 1e-12)
    return float(count) / area


def average_dislocation_density(
    counts: list[int], areas_mm2: list[float]
) -> DislocationDensityResult:
    if len(counts) != len(areas_mm2):
        raise ValueError("counts and areas_mm2 must have same length")
    if not counts:
        raise ValueError("counts must not be empty")
    densities = [
        dislocation_density(c, a) for c, a in zip(counts, areas_mm2, strict=True)
    ]
    return DislocationDensityResult(
        counts=[int(v) for v in counts],
        areas_mm2=[float(v) for v in areas_mm2],
        densities_per_mm2=densities,
        average_density_per_mm2=float(np.mean(densities)),
    )


@dataclass(slots=True)
class PhaseFractionResult:
    thresholds: list[int]
    class_area_percent: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def phase_fraction_by_thresholds(
    image: np.ndarray,
    thresholds: list[int] | tuple[int, ...] = (90, 150),
) -> PhaseFractionResult:
    arr = image.astype(np.uint8)
    th = sorted(int(t) for t in thresholds)
    if not th:
        raise ValueError("At least one threshold is required")

    classes: dict[str, float] = {}
    prev = 0
    total = arr.size
    for value in th:
        mask = (arr >= prev) & (arr < value)
        key = f"{prev}-{value - 1}"
        classes[key] = float(mask.sum() * 100.0 / total)
        prev = value
    tail = arr >= prev
    classes[f"{prev}-255"] = float(tail.sum() * 100.0 / total)

    return PhaseFractionResult(thresholds=th, class_area_percent=classes)


def auto_count_dark_pits(
    image: np.ndarray, threshold: int = 80, min_pixels: int = 4
) -> int:
    """
    Semi-automatic pit counting using connected dark components.
    """

    mask = image.astype(np.uint8) < int(threshold)
    if ndimage is not None:
        labels, count = ndimage.label(mask)
        if count == 0:
            return 0
        sizes = np.bincount(labels.ravel())
        # labels[0] is background
        return int(np.sum(sizes[1:] >= max(1, int(min_pixels))))

    # Lightweight flood-fill fallback.
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area >= max(1, int(min_pixels)):
                components += 1
    return components


def estimate_jeffries_from_labels(
    labels: np.ndarray,
    roi: tuple[int, int, int, int],
    area_mm2: float,
) -> JeffriesResult:
    """
    Semi-automatic Jeffries from a labeled image in ROI.
    ROI: x, y, w, h.
    """

    x, y, w, h = roi
    subset = labels[y : y + h, x : x + w]
    all_ids = np.unique(subset)
    border_ids = np.unique(
        np.concatenate(
            (subset[0, :], subset[-1, :], subset[:, 0], subset[:, -1]), axis=0
        )
    )
    inside_ids = np.setdiff1d(all_ids, border_ids, assume_unique=False)
    return jeffries_method(
        p_inside=float(len(inside_ids)),
        q_intercepted=float(len(border_ids)),
        area_mm2=area_mm2,
    )


def mean_lineal_intercept(
    total_test_line_length_um: float, intersections: int
) -> float:
    total = float(max(0.0, total_test_line_length_um))
    count = int(intersections)
    if total <= 1e-12 or count <= 0:
        return 0.0
    return float(total / count)


def astm_grain_size_from_intercept_length_cm(l_bar_cm: float) -> float:
    length_cm = float(max(l_bar_cm, 1e-12))
    return float(-10.0 - 6.64 * math.log10(length_cm))


def astm_grain_size_from_intercept_length_um(l_bar_um: float) -> float:
    return astm_grain_size_from_intercept_length_cm(float(l_bar_um) / 10_000.0)


def surface_density_from_line_intersections(pl_mm_inv: float) -> float:
    return float(max(0.0, 2.0 * float(pl_mm_inv)))


def line_density_from_plane_points(pa_mm2_inv: float) -> float:
    return float(max(0.0, 2.0 * float(pa_mm2_inv)))


def lamellar_spacing_metrics(
    total_test_line_length_um: float,
    intersections: int,
) -> dict[str, float]:
    random_spacing_um = mean_lineal_intercept(total_test_line_length_um, intersections)
    if random_spacing_um <= 1e-12:
        return {
            "interlamellar_random_spacing_um": 0.0,
            "interlamellar_true_spacing_um": 0.0,
            "lamella_intersection_density_mm_inv": 0.0,
            "lamella_surface_density_mm_inv": 0.0,
        }
    nl_mm_inv = float(int(intersections) / (float(total_test_line_length_um) / 1000.0))
    true_spacing_um = float(random_spacing_um / 2.0)
    lamella_surface_density_mm_inv = float(4000.0 / random_spacing_um)
    return {
        "interlamellar_random_spacing_um": float(random_spacing_um),
        "interlamellar_true_spacing_um": true_spacing_um,
        "lamella_intersection_density_mm_inv": nl_mm_inv,
        "lamella_surface_density_mm_inv": lamella_surface_density_mm_inv,
    }


def mean_free_path_from_volume_fraction(
    volume_fraction: float, nl_mm_inv: float
) -> float:
    vv = float(min(max(volume_fraction, 0.0), 1.0))
    nl = float(max(nl_mm_inv, 0.0))
    if nl <= 1e-12:
        return 0.0
    return float((1.0 - vv) / nl)


def export_dislocation_table(
    result: DislocationDensityResult, path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field_index", "count", "area_mm2", "density_per_mm2"])
        for idx, (c, a, d) in enumerate(
            zip(result.counts, result.areas_mm2, result.densities_per_mm2, strict=True),
            start=1,
        ):
            writer.writerow([idx, c, a, d])
        writer.writerow(["avg", "", "", result.average_density_per_mm2])
