from __future__ import annotations

import math
from typing import Callable

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from core.generator_eutectic import generate_eutectic_al_si
from core.generator_grains import generate_grain_structure
from core.generator_pearlite import (
    generate_martensite_structure,
    generate_pearlite_structure,
    generate_sorbite_structure,
    generate_troostite_structure,
)


def _ensure_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image.astype(np.float32), 0.0, 255.0).astype(np.uint8)


def _smooth_noise(seed: int, size: tuple[int, int], sigma: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, 1.0, size=size).astype(np.float32)
    if ndimage is not None:
        return ndimage.gaussian_filter(noise, sigma=max(0.2, float(sigma)))
    return noise


def _blend(images: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if not images:
        raise ValueError("images list is empty")
    if len(images) != len(weights):
        raise ValueError("images and weights size mismatch")
    total = float(sum(max(0.0, float(w)) for w in weights))
    if total <= 1e-9:
        norm = [1.0 / float(len(images))] * len(images)
    else:
        norm = [max(0.0, float(w)) / total for w in weights]
    canvas = np.zeros_like(images[0], dtype=np.float32)
    for image, weight in zip(images, norm, strict=True):
        canvas += image.astype(np.float32) * float(weight)
    return _ensure_u8(canvas)


def texture_liquid(size: tuple[int, int], seed: int) -> np.ndarray:
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    low = _smooth_noise(seed + 3, size=size, sigma=11.0)
    mid = _smooth_noise(seed + 7, size=size, sigma=4.0)
    swirl = np.sin(xx / max(1, w) * 6.2) + np.cos(yy / max(1, h) * 4.4)
    image = 146.0 + low * 19.0 + mid * 7.0 + swirl * 6.0
    return _ensure_u8(image)


def texture_ferrite(
    size: tuple[int, int],
    seed: int,
    *,
    add_twins: bool = False,
    anisotropic_etching: bool = False,
) -> np.ndarray:
    """Render polygonal ferrite grains with nital etching.

    Optional refinements introduced in A7/A9:

    * ``add_twins`` (A7) — with probability ~7 % per grain, draw a
      single straight annealing twin band across the grain. The twin
      tone is offset ±3 % from the host grain so it remains a subtle
      feature rather than a full-contrast lamella.
    * ``anisotropic_etching`` (A9) — modulate the per-grain base tone
      with a low-frequency multiscale-noise field driven by the same
      RNG used for grain orientation, so each grain has a slight
      orientation-dependent contrast variation similar to real nital
      etching of α-Fe.

    Both flags default to ``False`` so the legacy ``texture_ferrite``
    call remains byte-identical for backward compatibility (the
    snapshot baseline test guards this).
    """

    grain_data = generate_grain_structure(
        size=size,
        seed=int(seed) + 11,
        mean_grain_size_px=74.0,
        grain_size_jitter=0.16,
        boundary_width_px=1,
        boundary_contrast=0.16,
    )
    labels = grain_data["labels"]
    # Nital etching is orientation-dependent: each grain gets a random
    # base tone in the 200-230 range (±15 units variation between grains).
    rng = np.random.default_rng(int(seed) + 12)
    n_grains = int(labels.max()) + 1
    grain_tones = rng.uniform(200.0, 230.0, size=n_grains).astype(np.float32)

    if anisotropic_etching:
        # Per-grain orientation surrogate (random angle in [0, π]) is
        # mapped to a small tone offset via a smooth periodic function
        # so adjacent grains with similar angles do not get identical
        # contrast — this approximates the visible facet-by-facet
        # variation of real nital-etched α-Fe.
        orientation = rng.uniform(0.0, math.pi, size=n_grains).astype(np.float32)
        anisotropy_offset = (np.sin(2.0 * orientation) * 6.0).astype(np.float32)
        grain_tones = grain_tones + anisotropy_offset
        np.clip(grain_tones, 188.0, 240.0, out=grain_tones)

    ferrite = grain_tones[labels]

    if add_twins:
        # Annealing twins — straight bands inside ~7 % of grains. The
        # twin tone is offset by ±3 % from the host grain so it stays
        # subtle (real twins are weak contrast features, not lamellae).
        h, w = size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        twin_field = np.zeros(size, dtype=np.float32)
        twin_rng = np.random.default_rng(int(seed) + 14)
        for grain_id in range(n_grains):
            if twin_rng.random() > 0.07:
                continue
            mask = labels == grain_id
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            cy = float(ys.mean())
            cx = float(xs.mean())
            angle = float(twin_rng.uniform(0.0, math.pi))
            # Project pixels onto the perpendicular of the twin axis;
            # a thin band of width ~2-4 px around the centre is the
            # twin lamella.
            proj = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
            band_width = float(twin_rng.uniform(1.5, 3.0))
            band = mask & (np.abs(proj) < band_width)
            if not band.any():
                continue
            tone_shift = float(twin_rng.choice([-1.0, 1.0])) * float(
                twin_rng.uniform(5.0, 9.0)
            )
            twin_field[band] += tone_shift
        ferrite = ferrite + twin_field

    # Add subtle intra-grain noise for surface roughness
    ferrite += _smooth_noise(seed + 19, size=size, sigma=6.0) * 3.5
    # Darken grain boundaries
    boundaries = grain_data.get("boundaries")
    if boundaries is not None:
        ferrite[boundaries] -= 45.0
    if ndimage is not None:
        ferrite = ndimage.gaussian_filter(ferrite, sigma=0.35)
    return _ensure_u8(ferrite)


def texture_delta_ferrite(size: tuple[int, int], seed: int) -> np.ndarray:
    ferrite = generate_grain_structure(
        size=size,
        seed=int(seed) + 13,
        mean_grain_size_px=92.0,
        grain_size_jitter=0.12,
        boundary_width_px=1,
        boundary_contrast=0.14,
    )["image"].astype(np.float32)
    ferrite = ferrite * 0.8 + 50.0
    ferrite += _smooth_noise(seed + 21, size=size, sigma=7.0) * 2.2
    if ndimage is not None:
        ferrite = ndimage.gaussian_filter(ferrite, sigma=0.4)
    return _ensure_u8(ferrite)


def texture_austenite(size: tuple[int, int], seed: int) -> np.ndarray:
    gamma = generate_grain_structure(
        size=size,
        seed=int(seed) + 17,
        mean_grain_size_px=62.0,
        grain_size_jitter=0.18,
        boundary_width_px=1,
        boundary_contrast=0.18,
    )["image"].astype(np.float32)
    gamma = gamma * 0.8 + 34.0
    gamma += _smooth_noise(seed + 23, size=size, sigma=5.5) * 3.0
    if ndimage is not None:
        gamma = ndimage.gaussian_filter(gamma, sigma=0.35)
    return _ensure_u8(gamma)


def texture_pearlite_colonies(size: tuple[int, int], seed: int) -> np.ndarray:
    image = generate_pearlite_structure(
        size=size,
        seed=int(seed) + 23,
        pearlite_fraction=0.92,
        lamella_period_px=5.6,
        colony_size_px=92.0,
        ferrite_brightness=95,
        cementite_brightness=175,
    )["image"].astype(np.float32)
    if ndimage is not None:
        image = ndimage.gaussian_filter(image, sigma=0.55)
    # Average pearlite brightness should be ~85-105 (dark gray, unresolved look)
    image = image * 0.72 + 8.0
    return _ensure_u8(image)


def texture_cementite_network(
    size: tuple[int, int],
    seed: int,
    *,
    c_wt: float | None = None,
    cooling_rate_c_per_s: float | None = None,
) -> np.ndarray:
    """Render the secondary cementite grain-boundary network.

    Cementite is NOT attacked by nital → brightest phase (~235-250).
    The network forms at prior austenite grain boundaries in
    hypereutectoid steel.

    A5 — when ``c_wt`` is supplied the boundary width scales linearly
    from ~1.5 px at the eutectoid composition (0.77 % C) to ~7 px at
    the steel-cast iron limit (2.14 % C). ``cooling_rate_c_per_s``
    further widens the network for slow furnace cooling. Both
    parameters default to ``None`` so existing presets and tests stay
    byte-identical with the legacy fixed ``boundary_width_px=2``.
    """
    if c_wt is None:
        boundary_width_px = 2
    else:
        # Linear interpolation between (0.77 %C → 1.5 px) and
        # (2.14 %C → 7.0 px). Clamp so wildly out-of-range carbon
        # contents (pure ferrite, cast iron) cannot generate runaway
        # thicknesses.
        c_normalised = max(0.77, min(2.14, float(c_wt)))
        carbon_lerp = (c_normalised - 0.77) / (2.14 - 0.77)
        target_width = 1.5 + carbon_lerp * (7.0 - 1.5)
        if cooling_rate_c_per_s is not None:
            # Slow cooling (furnace) → +1 px per 10× rate decrease
            # below 1 °C/s. Fast cooling shrinks the network slightly.
            rate_clamped = max(0.05, float(cooling_rate_c_per_s))
            rate_factor = -math.log10(rate_clamped)  # >0 for slow, <0 for fast
            target_width += max(-1.5, min(2.0, rate_factor * 0.8))
        boundary_width_px = int(max(1, round(target_width)))

    base = np.full(size, 235.0, dtype=np.float32)
    base += _smooth_noise(seed + 31, size=size, sigma=3.2) * 5.0
    grains = generate_grain_structure(
        size=size,
        seed=int(seed) + 37,
        mean_grain_size_px=84.0,
        grain_size_jitter=0.15,
        boundary_width_px=boundary_width_px,
        boundary_contrast=0.62,
    )
    boundaries = grains["boundaries"]
    net = base.copy()
    # Cementite at grain boundaries is slightly brighter (thicker, flatter)
    net[boundaries] = np.clip(net[boundaries] + 12.0, 0.0, 255.0)
    blob = _smooth_noise(seed + 41, size=size, sigma=2.6)
    bright_mask = blob > np.quantile(blob, 0.97)
    net[bright_mask] = np.clip(net[bright_mask] + 8.0, 0.0, 255.0)
    if ndimage is not None:
        net = ndimage.gaussian_filter(net, sigma=0.55)
    return _ensure_u8(net)


def texture_martensite_tetragonal(size: tuple[int, int], seed: int) -> np.ndarray:
    mart = generate_martensite_structure(
        size=size,
        seed=int(seed) + 43,
        needle_count=4600,
        needle_length_px=(10, 108),
        packet_spread_deg=12.0,
    )["image"].astype(np.float32)
    return _ensure_u8(mart * 0.9 + 8.0)


def texture_martensite_cubic(size: tuple[int, int], seed: int) -> np.ndarray:
    mart = generate_martensite_structure(
        size=size,
        seed=int(seed) + 47,
        needle_count=3600,
        needle_length_px=(8, 76),
        packet_spread_deg=18.0,
    )["image"].astype(np.float32)
    return _ensure_u8(mart * 1.03 + 7.0)


def texture_troostite_quench(size: tuple[int, int], seed: int) -> np.ndarray:
    return _ensure_u8(generate_troostite_structure(size=size, seed=int(seed) + 53, mode="quench")["image"])


def texture_troostite_temper(size: tuple[int, int], seed: int) -> np.ndarray:
    return _ensure_u8(generate_troostite_structure(size=size, seed=int(seed) + 59, mode="temper")["image"])


def texture_sorbite_quench(size: tuple[int, int], seed: int) -> np.ndarray:
    return _ensure_u8(
        generate_sorbite_structure(
            size=size,
            seed=int(seed) + 61,
            mode="quench",
            include_particles=False,
        )["image"]
    )


def texture_sorbite_temper(size: tuple[int, int], seed: int) -> np.ndarray:
    return _ensure_u8(
        generate_sorbite_structure(
            size=size,
            seed=int(seed) + 67,
            mode="temper",
            include_particles=False,
        )["image"]
    )


def texture_bainite(size: tuple[int, int], seed: int) -> np.ndarray:
    troostite = texture_troostite_quench(size=size, seed=seed + 71)
    mart = texture_martensite_cubic(size=size, seed=seed + 73)
    return _blend([troostite, mart], [0.72, 0.28])


def texture_bainite_upper(size: tuple[int, int], seed: int) -> np.ndarray:
    """Upper bainite (350-550 °C): feathery packets of parallel ferrite
    plates separated by Fe3C layers. Coarser than lower bainite.

    Implementation: a Voronoi packet field with low elongation provides
    the packet topology, then a sinusoidal lamellar pattern with a
    relatively long period (~9 px) is overlaid for the feathery look.
    """

    h, w = size
    # Packets: a few large grains with random orientation give the
    # feathery direction.
    packets = generate_grain_structure(
        size=size,
        seed=int(seed) + 91,
        mean_grain_size_px=120.0,
        grain_size_jitter=0.25,
        equiaxed=0.55,
        elongation=1.6,
        orientation_deg=0.0,
        boundary_width_px=1,
        boundary_contrast=0.0,
    )
    labels = packets["labels"]
    n_packets = int(labels.max()) + 1
    rng = np.random.default_rng(int(seed) + 95)
    theta = rng.uniform(0.0, math.pi, size=n_packets).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    theta_field = theta[labels]
    projection = xx * np.cos(theta_field) + yy * np.sin(theta_field)
    period = 9.0  # coarse upper-bainite spacing in pixels
    lamella = np.sin((2.0 * math.pi / period) * projection)
    base = np.full(size, 132.0, dtype=np.float32)
    base += lamella * 18.0
    base += _smooth_noise(int(seed) + 97, size=size, sigma=4.0) * 6.0
    boundaries = np.zeros_like(labels, dtype=bool)
    boundaries[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundaries[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    base[boundaries] -= 22.0
    if ndimage is not None:
        base = ndimage.gaussian_filter(base, sigma=0.55)
    return _ensure_u8(base)


def texture_bainite_lower(size: tuple[int, int], seed: int) -> np.ndarray:
    """Lower bainite (200-350 °C): needle-like ferrite laths with Fe3C
    precipitates inside the laths at ~55-60° to the lath axis.

    Implementation: blend a martensite-like needle field with a
    higher-frequency sinusoidal "intra-lath carbide" overlay aligned
    near the lath axis to suggest the inner-lath cementite particles.
    """

    h, w = size
    needle_base = generate_martensite_structure(
        size=size,
        seed=int(seed) + 101,
        needle_count=3000,
        needle_length_px=(8, 60),
        packet_spread_deg=10.0,
    )["image"].astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(int(seed) + 103)
    angle = float(rng.uniform(0.0, math.pi))
    cementite_overlay = np.sin(
        (2.0 * math.pi / 3.5)
        * (xx * math.cos(angle + math.pi / 3.2) + yy * math.sin(angle + math.pi / 3.2))
    )
    base = needle_base * 0.94 + cementite_overlay * 12.0 + 6.0
    base += _smooth_noise(int(seed) + 107, size=size, sigma=2.0) * 3.0
    if ndimage is not None:
        base = ndimage.gaussian_filter(base, sigma=0.45)
    return _ensure_u8(base)


def texture_ledeburite(size: tuple[int, int], seed: int) -> np.ndarray:
    eut = generate_eutectic_al_si(
        size=size,
        seed=int(seed) + 79,
        si_phase_fraction=0.55,
        eutectic_scale_px=5.6,
        morphology="network",
    )["image"]
    pearlite = texture_pearlite_colonies(size=size, seed=seed + 83)
    return _blend([_ensure_u8(eut), pearlite], [0.58, 0.42])


def texture_ledeburite_leopard(size: tuple[int, int], seed: int) -> np.ndarray:
    """A2 — leopard texture for white cast iron ledeburite.

    A bright cementite matrix is sprinkled with quasi-periodic dark
    pearlite blobs. The blobs come from thresholding a multiscale
    smooth noise field, which gives them a controlled mean size and a
    near-uniform spacing.
    """

    h, w = size
    rng = np.random.default_rng(int(seed) + 79)
    matrix = np.full(size, 218.0, dtype=np.float32)  # bright cementite
    matrix += _smooth_noise(int(seed) + 81, size=size, sigma=2.5) * 6.0

    # Two-scale noise field for the dark pearlite blobs.
    blob_field = _smooth_noise(int(seed) + 83, size=size, sigma=4.5)
    blob_field += 0.4 * _smooth_noise(int(seed) + 85, size=size, sigma=9.0)
    threshold = float(np.quantile(blob_field, 0.55))
    blob_mask = blob_field > threshold

    pearlite_tone = 60.0 + rng.uniform(-6.0, 6.0, size=size).astype(np.float32)
    pearlite_tone += _smooth_noise(int(seed) + 87, size=size, sigma=1.6) * 6.0
    matrix[blob_mask] = pearlite_tone[blob_mask]

    if ndimage is not None:
        matrix = ndimage.gaussian_filter(matrix, sigma=0.5)
    return _ensure_u8(matrix)


def texture_tempered_high(size: tuple[int, int], seed: int) -> np.ndarray:
    sorbite = texture_sorbite_temper(size=size, seed=seed + 89)
    ferrite = texture_ferrite(size=size, seed=seed + 97)
    return _blend([sorbite, ferrite], [0.76, 0.24])


def fe_c_texture_map() -> dict[str, Callable[[tuple[int, int], int], np.ndarray]]:
    return {
        "LIQUID": texture_liquid,
        "FERRITE": texture_ferrite,
        "DELTA_FERRITE": texture_delta_ferrite,
        "AUSTENITE": texture_austenite,
        "PEARLITE": texture_pearlite_colonies,
        "CEMENTITE": texture_cementite_network,
        # A1 — primary cementite needles in hypereutectic white cast
        # iron. Until the dedicated needle renderer (A1) lands the key
        # is mapped to the existing cementite-network texture so the
        # phase template at least has *some* renderer to fall back on.
        "CEMENTITE_PRIMARY": texture_cementite_network,
        "MARTENSITE": texture_martensite_cubic,
        "MARTENSITE_TETRAGONAL": texture_martensite_tetragonal,
        "MARTENSITE_CUBIC": texture_martensite_cubic,
        "TROOSTITE": texture_troostite_temper,
        "SORBITE": texture_sorbite_temper,
        "BAINITE": texture_bainite,
        "BAINITE_UPPER": texture_bainite_upper,
        "BAINITE_LOWER": texture_bainite_lower,
        "LEDEBURITE": texture_ledeburite,
        # A2 — leopard-style hierarchical cementite-with-pearlite-blobs
        # texture for white cast iron. Existing presets continue to use
        # the legacy "LEDEBURITE" key (which delegates to
        # ``texture_ledeburite``) so the new identifier is opt-in.
        "LEDEBURITE_LEOPARD": texture_ledeburite_leopard,
        "TEMPERED_MATRIX": texture_tempered_high,
    }
