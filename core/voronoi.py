from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover - fallback path
    cKDTree = None


def generate_voronoi_labels(
    size: tuple[int, int],
    points: np.ndarray,
    metric_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Return nearest-site labels for a 2D regular grid."""

    height, width = size
    yy, xx = np.mgrid[0:height, 0:width]
    coords = np.column_stack((yy.ravel(), xx.ravel())).astype(np.float32, copy=False)
    seeds = points.astype(np.float32, copy=False)

    if metric_matrix is not None:
        metric = metric_matrix.astype(np.float32, copy=False)
        coords = coords @ metric.T
        seeds = seeds @ metric.T

    if cKDTree is not None:
        tree = cKDTree(seeds)
        _, idx = tree.query(coords, k=1, workers=-1)
        return idx.reshape((height, width)).astype(np.int32, copy=False)

    # Fallback when scipy is unavailable. Chunking avoids very large arrays.
    labels = np.zeros(coords.shape[0], dtype=np.int32)
    best = np.full(coords.shape[0], np.inf, dtype=np.float32)
    chunk = 32
    for start in range(0, seeds.shape[0], chunk):
        part = seeds[start : start + chunk]
        delta = coords[:, None, :] - part[None, :, :]
        dist2 = np.sum(delta * delta, axis=2)
        local_ix = np.argmin(dist2, axis=1)
        local_best = dist2[np.arange(dist2.shape[0]), local_ix]
        mask = local_best < best
        best[mask] = local_best[mask]
        labels[mask] = start + local_ix[mask]
    return labels.reshape((height, width))

