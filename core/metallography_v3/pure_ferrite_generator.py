from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy import ndimage as ndi  # type: ignore
except Exception:  # pragma: no cover
    ndi = None

try:
    from scipy.spatial.transform import Rotation as SciPyRotation  # type: ignore
except Exception:  # pragma: no cover
    SciPyRotation = None


def _init_jittered_points(
    h: int,
    w: int,
    n_points: int,
    rng: np.random.Generator,
    jitter: float = 0.42,
) -> np.ndarray:
    nx = max(1, int(round(math.sqrt(n_points * w / max(h, 1)))))
    ny = max(1, int(math.ceil(n_points / max(nx, 1))))
    xs = np.linspace(w / (2 * nx), w - w / (2 * nx), nx)
    ys = np.linspace(h / (2 * ny), h - h / (2 * ny), ny)
    grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    if len(grid) > n_points:
        idx = rng.choice(len(grid), size=n_points, replace=False)
        grid = grid[idx]

    cell_w = w / max(nx, 1)
    cell_h = h / max(ny, 1)
    grid[:, 0] += rng.uniform(-jitter * cell_w, jitter * cell_w, size=len(grid))
    grid[:, 1] += rng.uniform(-jitter * cell_h, jitter * cell_h, size=len(grid))
    grid[:, 0] = np.clip(grid[:, 0], 0, w - 1)
    grid[:, 1] = np.clip(grid[:, 1], 0, h - 1)
    return grid.astype(np.float32)


def _power_voronoi_labels(
    h: int,
    w: int,
    points: np.ndarray,
    weights: np.ndarray,
    tile: int = 192,
) -> np.ndarray:
    labels = np.empty((h, w), dtype=np.int32)
    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)
    ws = weights.astype(np.float32)
    for y0 in range(0, h, tile):
        y1 = min(y0 + tile, h)
        yy = np.arange(y0, y1, dtype=np.float32)[:, None, None]
        for x0 in range(0, w, tile):
            x1 = min(x0 + tile, w)
            xx = np.arange(x0, x1, dtype=np.float32)[None, :, None]
            score = (
                (xx - xs[None, None, :]) ** 2
                + (yy - ys[None, None, :]) ** 2
                - ws[None, None, :]
            )
            labels[y0:y1, x0:x1] = np.argmin(score, axis=2)
    return labels


def _lloyd_relax_power(
    h: int,
    w: int,
    points: np.ndarray,
    weights: np.ndarray,
    *,
    n_iter: int,
    rng: np.random.Generator,
    tile: int = 192,
) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    n = len(points)
    for _ in range(max(0, int(n_iter))):
        labels = _power_voronoi_labels(h, w, points, weights, tile=tile)
        flat = labels.ravel()
        counts = np.bincount(flat, minlength=n).astype(np.float32)
        sumx = np.bincount(flat, weights=xx.ravel(), minlength=n)
        sumy = np.bincount(flat, weights=yy.ravel(), minlength=n)
        ok = counts > 0
        points[ok, 0] = sumx[ok] / counts[ok]
        points[ok, 1] = sumy[ok] / counts[ok]
        if np.any(~ok):
            points[~ok] = np.column_stack(
                [
                    rng.uniform(0, w - 1, size=(~ok).sum()),
                    rng.uniform(0, h - 1, size=(~ok).sum()),
                ]
            )
    labels = _power_voronoi_labels(h, w, points, weights, tile=tile)
    return labels, points.astype(np.float32)


def _random_bcc_orientation_response(
    n_grains: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if SciPyRotation is not None:
        mats = (
            SciPyRotation.random(n_grains, random_state=rng)
            .as_matrix()
            .astype(np.float32)
        )
        n_sample = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        n_cryst = np.einsum("nij,j->ni", np.transpose(mats, (0, 2, 1)), n_sample)
        c100 = np.max(np.abs(n_cryst), axis=1).astype(np.float32)
        orientation = np.arctan2(mats[:, 1, 0], mats[:, 0, 0]).astype(np.float32)
        return c100, orientation

    vec = rng.normal(size=(n_grains, 3)).astype(np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms <= 1e-6] = 1.0
    vec /= norms
    c100 = np.max(np.abs(vec), axis=1).astype(np.float32)
    orientation = np.arctan2(vec[:, 1], vec[:, 0]).astype(np.float32)
    return c100, orientation


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return boundary


def generate_pure_ferrite_micrograph(
    *,
    size: tuple[int, int],
    seed: int,
    mean_eq_d_px: float = 72.0,
    size_sigma: float = 0.22,
    relax_iter: int = 1,
    boundary_width_px: float = 2.2,
    boundary_depth: float = 0.13,
    blur_sigma_px: float = 0.6,
    tile: int = 192,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    h, w = int(size[0]), int(size[1])
    area_px2 = float(h * w)
    mean_eq_d_px = float(max(8.0, mean_eq_d_px))
    mean_grain_area_px2 = math.pi * (mean_eq_d_px**2) / 4.0
    n_grains = max(30, int(round(area_px2 / max(mean_grain_area_px2, 1.0))))

    points = _init_jittered_points(h, w, n_grains, rng)
    eq_d_px = mean_eq_d_px * np.exp(float(size_sigma) * rng.standard_normal(n_grains))
    r_px = 0.5 * eq_d_px
    weights = 0.35 * (r_px**2 - np.mean(r_px**2))
    labels, _points = _lloyd_relax_power(
        h,
        w,
        points,
        weights.astype(np.float32),
        n_iter=int(relax_iter),
        rng=rng,
        tile=tile,
    )
    n = len(points)
    boundary = _boundary_mask(labels)
    if ndi is not None:
        dist_to_gb = ndi.distance_transform_edt(~boundary).astype(np.float32)
    else:
        dist_to_gb = np.where(boundary, 0.0, 1.0).astype(np.float32)

    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=n).astype(np.float32)
    cx = np.bincount(flat, weights=xx.ravel(), minlength=n) / np.maximum(counts, 1.0)
    cy = np.bincount(flat, weights=yy.ravel(), minlength=n) / np.maximum(counts, 1.0)

    c100, orientation_rad_per_grain = _random_bcc_orientation_response(n, rng)
    grain_brightness = 0.84 + 0.10 * (c100**3.0)
    base = grain_brightness[labels]

    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    ux = np.cos(theta)
    uy = np.sin(theta)
    grad_amp = rng.uniform(0.005, 0.015, size=n)
    local_grad = grad_amp[labels] * (
        ((xx - cx[labels]) * ux[labels] + (yy - cy[labels]) * uy[labels])
        / max(mean_eq_d_px, 1e-6)
    )

    lf = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    if ndi is not None:
        lf = ndi.gaussian_filter(lf, sigma=0.10 * min(h, w))
    lf = (lf - lf.mean()) / (lf.std() + 1e-6)
    lf *= 0.010

    gb_sigma_px = max(0.8, float(boundary_width_px))
    gb_dark = float(boundary_depth) * np.exp(-((dist_to_gb / gb_sigma_px) ** 2))
    img = base + local_grad + lf - gb_dark
    img += 0.006 * rng.normal(size=(h, w)).astype(np.float32)
    if ndi is not None:
        img = ndi.gaussian_filter(img, sigma=float(max(0.0, blur_sigma_px)))
    q01 = float(np.quantile(img, 0.01))
    if q01 < 0.43:
        img += (0.43 - q01) * 0.9
    img = np.clip(img, 0.0, 1.0)

    orientation_map = orientation_rad_per_grain[labels].astype(np.float32)
    return {
        "image_gray": np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8),
        "labels": labels.astype(np.int32),
        "boundary": boundary.astype(np.uint8),
        "orientation_rad": orientation_map,
        "metadata": {
            "generator": "pure_ferrite_power_voronoi_v1",
            "grain_count": int(n_grains),
            "mean_eq_d_px": float(mean_eq_d_px),
            "size_sigma": float(size_sigma),
            "relax_iter": int(relax_iter),
        },
    }
