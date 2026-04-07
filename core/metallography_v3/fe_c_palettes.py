"""Colour palettes for Fe-C phase renderings (A10.0).

Each palette is a small JSON-like dict that describes how to map a
grayscale frame + phase masks to an RGB output that mimics a real
optical-microscopy aesthetic (``nital_warm`` yellow-cream ferrite, DIC
polarised grain colouring, Klemm/LePerra tint etching, etc.).

The palettes are consumed by
``core.metallography_v3.fe_c_color_palette.apply_color_palette`` which is
the single post-process entry point the pipeline calls at the very end of
``MetallographyPipelineV3.generate``. Keeping palette constants separate
from the transformation code keeps them easy to tweak without touching
rendering logic.
"""

from __future__ import annotations

from typing import Any

# Palette identifiers accepted by ``SynthesisProfileV3.color_mode``.
GRAYSCALE_MODE: str = "grayscale_nital"
NITAL_WARM_MODE: str = "nital_warm"
DIC_POLARIZED_MODE: str = "dic_polarized"
TINT_ETCH_BLUE_YELLOW_MODE: str = "tint_etch_blue_yellow"

SUPPORTED_COLOR_MODES: tuple[str, ...] = (
    GRAYSCALE_MODE,
    NITAL_WARM_MODE,
    DIC_POLARIZED_MODE,
    TINT_ETCH_BLUE_YELLOW_MODE,
)

# ---------------------------------------------------------------------------
# RGB range helpers
# ---------------------------------------------------------------------------

RGB = tuple[int, int, int]
RGB_RANGE = tuple[RGB, RGB]


def _clip_rgb(rgb: tuple[float, float, float]) -> RGB:
    return (
        int(max(0.0, min(255.0, rgb[0]))),
        int(max(0.0, min(255.0, rgb[1]))),
        int(max(0.0, min(255.0, rgb[2]))),
    )


def lerp_rgb(low: RGB, high: RGB, t: float) -> RGB:
    """Linear interpolation in RGB space. ``t`` is clamped to [0, 1]."""
    tt = max(0.0, min(1.0, float(t)))
    return _clip_rgb(
        (
            low[0] + (high[0] - low[0]) * tt,
            low[1] + (high[1] - low[1]) * tt,
            low[2] + (high[2] - low[2]) * tt,
        )
    )


def hsv_to_rgb(h: float, s: float, v: float) -> RGB:
    """HSV (h ∈ [0, 1], s, v ∈ [0, 1]) → 8-bit RGB."""
    h = float(h) % 1.0
    s = max(0.0, min(1.0, float(s)))
    v = max(0.0, min(1.0, float(v)))
    if s <= 0.0:
        val = int(round(v * 255.0))
        return (val, val, val)
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return _clip_rgb((r * 255.0, g * 255.0, b * 255.0))


# ---------------------------------------------------------------------------
# Palette tables
# ---------------------------------------------------------------------------

# Grayscale baseline — used when ``color_mode`` is absent or set to the
# default. The numbers mirror the mean brightness of each phase observed in
# the current texture functions so the legacy grayscale output can still be
# described in the same structure.
GRAYSCALE_NITAL: dict[str, Any] = {
    "kind": "grayscale",
    "boundary_darken": 0.0,
}

# Palette derived from the AISI 1020-1050 reference series at 400× with
# nital etching on a warm broadband illumination: ferrite appears as a
# yellow-cream polygon, pearlite as a dark brown "blob", cementite (when
# present) is the brightest phase. Ranges (low, high) bracket intra-grain
# brightness variation.
NITAL_WARM: dict[str, Any] = {
    "kind": "phase_tint",
    "phase_rgb": {
        "FERRITE": ((244, 214, 107), (255, 233, 154)),
        "DELTA_FERRITE": ((236, 204, 98), (250, 222, 140)),
        "AUSTENITE": ((229, 196, 100), (248, 220, 142)),
        "PEARLITE": ((61, 54, 40), (91, 75, 50)),
        "CEMENTITE": ((240, 232, 210), (255, 250, 230)),
        "CEMENTITE_PRIMARY": ((240, 232, 210), (255, 250, 230)),
        "LEDEBURITE": ((162, 142, 92), (196, 176, 118)),
        "MARTENSITE": ((90, 70, 50), (130, 100, 75)),
        "MARTENSITE_TETRAGONAL": ((88, 68, 48), (128, 98, 72)),
        "MARTENSITE_CUBIC": ((96, 78, 54), (138, 110, 82)),
        "TROOSTITE": ((74, 60, 42), (110, 90, 60)),
        "SORBITE": ((102, 84, 58), (140, 114, 78)),
        "BAINITE": ((86, 70, 50), (126, 102, 74)),
    },
    # Base brightness remap for any phase that is not listed above:
    # dark grayscale values get mapped to the "pearlite" palette and
    # bright ones to "ferrite" so the output stays warm-tinted even for
    # transitional stages.
    "fallback_low_rgb": (55, 50, 38),
    "fallback_high_rgb": (252, 230, 152),
    # Boundary rings are kept dark brown so grain edges stay visible.
    "boundary_rgb": (20, 15, 8),
    "boundary_blend": 0.55,
}

# DIC / polarised reflected-light palette for ferritic microstructures.
# Each grain gets a random hue within a narrow saturation range and a
# brightness lift near the grain boundary to fake the rim-lighting effect
# typical of Nomarski imaging.
DIC_POLARIZED: dict[str, Any] = {
    "kind": "dic_polarized",
    "hue_range": (0.0, 1.0),
    "saturation_range": (0.10, 0.40),
    "value_range": (0.50, 0.90),
    "boundary_rgb": (235, 235, 235),
    "boundary_value_boost": 0.35,
    "intragrain_value_jitter": 0.10,
}

# Klemm / Le Perra / Beraha-style tint etching. Bainitic packets tint to
# yellow, the matrix tints to blue — carbides stay pale. This produces the
# distinctive blue/yellow bainite look seen in the reference image.
TINT_ETCH_BLUE_YELLOW: dict[str, Any] = {
    "kind": "phase_tint",
    "phase_rgb": {
        "FERRITE": ((245, 225, 115), (255, 240, 155)),
        "AUSTENITE": ((60, 95, 165), (100, 140, 205)),
        "PEARLITE": ((55, 85, 140), (95, 130, 185)),
        "BAINITE": ((60, 95, 165), (100, 140, 205)),
        "BAINITE_UPPER": ((60, 95, 165), (100, 140, 205)),
        "BAINITE_LOWER": ((48, 80, 150), (90, 128, 190)),
        "CEMENTITE": ((220, 220, 200), (245, 245, 220)),
        "MARTENSITE": ((52, 82, 150), (92, 128, 190)),
    },
    "fallback_low_rgb": (48, 80, 150),
    "fallback_high_rgb": (252, 236, 150),
    "boundary_rgb": (15, 18, 32),
    "boundary_blend": 0.45,
}


PALETTES: dict[str, dict[str, Any]] = {
    GRAYSCALE_MODE: GRAYSCALE_NITAL,
    NITAL_WARM_MODE: NITAL_WARM,
    DIC_POLARIZED_MODE: DIC_POLARIZED,
    TINT_ETCH_BLUE_YELLOW_MODE: TINT_ETCH_BLUE_YELLOW,
}


def get_palette(color_mode: str) -> dict[str, Any]:
    """Return the palette dict for ``color_mode``.

    Unknown identifiers silently fall back to the grayscale palette so a
    malformed preset never aborts generation; callers that care about the
    distinction should consult ``SUPPORTED_COLOR_MODES`` instead.
    """
    key = str(color_mode or GRAYSCALE_MODE).strip().lower()
    return PALETTES.get(key, GRAYSCALE_NITAL)
