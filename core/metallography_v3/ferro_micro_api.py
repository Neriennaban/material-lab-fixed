"""ferro-micro public API facade (TZ §10).

A thin convenience wrapper around ``MetallographyPipelineV3`` that
exposes the parameter names from §10 of the TZ — ``carbon``,
``cooling_rate``, ``magnification``, ``etchant``, etc. — and returns a
small ``GeneratedSample`` namedtuple instead of the full
``GenerationOutputV3``. The goal is to make the simplest "render one
image" use case a one-liner without forcing the caller to construct
the full v3 request payload.

Example
-------

>>> from core.metallography_v3 import ferro_micro_api as fm
>>> sample = fm.generate(carbon=0.45, magnification=400, seed=42)
>>> sample.image.shape
(1024, 1024, 3)

When ``return_info=True`` the function returns a ``SampleInfo`` dict
with the measured phase fractions, ASTM grain size and Brinell
hardness estimate from the pipeline metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts_v3 import (
    EtchProfileV3,
    GenerationOutputV3,
    MetallographyRequestV3,
    PhaseModelConfigV3,
    SynthesisProfileV3,
    ThermalProgramV3,
)
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRESETS_DIR = REPO_ROOT / "presets_v3"
DEFAULT_PROFILES_DIR = REPO_ROOT / "profiles_v3"


@dataclass(slots=True)
class GeneratedSample:
    """Public output of ``ferro_micro_api.generate``.

    The ``image`` field always holds an RGB ``(H, W, 3) uint8`` array
    so consumers can pass it straight to PIL/Matplotlib without
    branching on grayscale vs colour.
    """

    image: np.ndarray
    image_gray: np.ndarray
    phase_masks: dict[str, np.ndarray]
    metadata: dict[str, Any]
    info: dict[str, Any] | None = None


_ETCHANT_TO_REAGENT = {
    "nital": "nital_2",
    "nital_2": "nital_2",
    "nital_5": "nital_5",
    "picral": "picral",
    "kalling": "kalling",
    "klemm": "klemm_1",
    "klemm_1": "klemm_1",
    "le_perra": "le_perra",
    "beraha": "beraha_iii",
    "beraha_iii": "beraha_iii",
}


def _resolve_etch_reagent(etchant: str | None) -> str:
    if not etchant:
        return "nital_2"
    key = str(etchant).strip().lower().replace(" ", "_").replace("-", "_")
    return _ETCHANT_TO_REAGENT.get(key, key)


def _build_thermal_program(
    *,
    austenitization_temp: float | None,
    holding_time_min: float,
    cooling_rate: float,
    quench_medium_code: str | None = None,
) -> ThermalProgramV3:
    """Build a simple 4-point thermal program from §10 parameters."""
    austenitize_t = float(austenitization_temp) if austenitization_temp else 870.0
    hold_s = max(60.0, float(holding_time_min) * 60.0)
    rate_c_per_s = max(0.001, float(cooling_rate))
    cool_span = max(1.0, (austenitize_t - 20.0) / rate_c_per_s)
    points = [
        {"time_s": 0.0, "temperature_c": 20.0, "label": "Start", "locked": True},
        {
            "time_s": 600.0,
            "temperature_c": austenitize_t,
            "label": "Austenitize",
            "locked": False,
        },
        {
            "time_s": 600.0 + hold_s,
            "temperature_c": austenitize_t,
            "label": "Hold",
            "locked": False,
        },
        {
            "time_s": 600.0 + hold_s + cool_span,
            "temperature_c": 20.0,
            "label": "Cool to RT",
            "locked": False,
        },
    ]
    medium = quench_medium_code or (
        "water_20" if rate_c_per_s >= 30.0 else "air"
    )
    payload = {
        "points": points,
        "quench": {
            "medium_code": medium,
            "quench_time_s": 0.0,
            "bath_temperature_c": 20.0,
            "sample_temperature_c": austenitize_t,
            "custom_medium_name": medium,
            "custom_severity_factor": 1.0,
        },
        "sampling_mode": "per_degree",
        "degree_step_c": 1.0,
        "max_frames": 320,
    }
    return ThermalProgramV3.from_dict(payload)


def _build_request(
    *,
    carbon: float,
    width: int,
    height: int,
    cooling_rate: float,
    austenitization_temp: float | None,
    holding_time: float,
    magnification: int,
    etchant: str | None,
    color_mode: str,
    seed: int,
    thermal_program: list[dict[str, Any]] | None,
) -> MetallographyRequestV3:
    composition = {"Fe": round(100.0 - float(carbon), 3), "C": round(float(carbon), 3)}

    synthesis = SynthesisProfileV3(
        profile_id="ferro_micro_api",
        color_mode=str(color_mode),
        contrast_target=1.05,
        boundary_sharpness=1.10,
        artifact_level=0.15,
    )

    etch_reagent = _resolve_etch_reagent(etchant)
    etch = EtchProfileV3.from_dict(
        {"reagent": etch_reagent, "time_s": 9.0, "temperature_c": 22.0}
    )

    if thermal_program:
        thermal = ThermalProgramV3.from_dict({"points": thermal_program})
    else:
        thermal = _build_thermal_program(
            austenitization_temp=austenitization_temp,
            holding_time_min=float(holding_time),
            cooling_rate=float(cooling_rate),
        )

    return MetallographyRequestV3(
        sample_id=f"ferro_micro_C{int(round(carbon * 100))}",
        composition_wt=composition,
        system_hint="fe-c",
        thermal_program=thermal,
        etch_profile=etch,
        synthesis_profile=synthesis,
        phase_model=PhaseModelConfigV3(),
        microscope_profile={
            "magnification": int(magnification),
            "focus": 0.95,
            "brightness": 1.0,
            "contrast": 1.1,
            "optical_mode": "brightfield",
            "simulate_preview": False,
        },
        seed=int(seed),
        resolution=(int(height), int(width)),
        strict_validation=False,
    )


def _summarise_info(output: GenerationOutputV3) -> dict[str, Any]:
    masks = output.phase_masks or {}
    total = 0
    for mask in masks.values():
        if isinstance(mask, np.ndarray):
            total = mask.size
            break
    total = max(1, total)
    fractions = {
        str(name).upper(): float(np.asarray(mask > 0).sum()) / float(total)
        for name, mask in masks.items()
        if isinstance(mask, np.ndarray)
    }

    metadata = dict(output.metadata or {})
    measurements = dict(metadata.get("measurements", {}))
    grain = measurements.get("grain_size_astm")
    hardness_block = metadata.get("hardness", metadata.get("hbw_estimate"))
    hardness_hv: float | None = None
    if isinstance(hardness_block, dict):
        hv = hardness_block.get("hv") or hardness_block.get("vickers")
        if hv is not None:
            try:
                hardness_hv = float(hv)
            except Exception:
                hardness_hv = None
    return {
        "phases": fractions,
        "grain_size_astm": grain,
        "hardness_hv": hardness_hv,
        "metadata": metadata,
    }


def generate(
    *,
    carbon: float,
    width: int = 1024,
    height: int = 1024,
    cooling_rate: float = 1.0,
    austenitization_temp: float | None = None,
    holding_time: float = 60.0,
    magnification: int = 200,
    etchant: str | None = "nital",
    color_mode: str = "grayscale_nital",
    seed: int = 42,
    thermal_program: list[dict[str, Any]] | None = None,
    presets_dir: str | Path | None = None,
    profiles_dir: str | Path | None = None,
    return_info: bool = False,
) -> GeneratedSample:
    """Render a single Fe-C microstructure from the ``ferro-micro``
    API surface (TZ §10).

    Parameters
    ----------
    carbon : float
        Weight-percent carbon content (0..6.67).
    width, height : int
        Output image dimensions in pixels.
    cooling_rate : float
        Cooling rate in °C/s. Drives the auto-generated thermal
        program when ``thermal_program`` is ``None``.
    austenitization_temp : float, optional
        Austenitisation temperature (°C). Defaults to 870 °C.
    holding_time : float
        Hold time at austenitisation temperature, in minutes.
    magnification : int
        Microscope magnification (100 / 200 / 500 / 1000…).
    etchant : str, optional
        Friendly etchant name (``"nital"``, ``"picral"``,
        ``"klemm_1"``, …). ``None`` keeps the default ``nital_2``.
    color_mode : str
        ``"grayscale_nital"`` (default), ``"nital_warm"``,
        ``"dic_polarized"`` or ``"tint_etch_blue_yellow"``.
    seed : int
        RNG seed for reproducibility.
    thermal_program : list of dicts, optional
        Custom thermal program in the format used by
        ``ThermalProgramV3.from_dict({"points": ...})``. Overrides
        ``cooling_rate``/``austenitization_temp``/``holding_time``.
    return_info : bool
        When ``True`` populates the ``info`` field of the returned
        :class:`GeneratedSample` with phase fractions, ASTM grain
        size and the Brinell hardness estimate (if available).
    """
    pipeline = MetallographyPipelineV3(
        presets_dir=Path(presets_dir) if presets_dir else DEFAULT_PRESETS_DIR,
        profiles_dir=Path(profiles_dir) if profiles_dir else DEFAULT_PROFILES_DIR,
    )
    request = _build_request(
        carbon=carbon,
        width=width,
        height=height,
        cooling_rate=cooling_rate,
        austenitization_temp=austenitization_temp,
        holding_time=holding_time,
        magnification=magnification,
        etchant=etchant,
        color_mode=color_mode,
        seed=seed,
        thermal_program=thermal_program,
    )
    output = pipeline.generate(request)
    return GeneratedSample(
        image=output.image_rgb,
        image_gray=output.image_gray,
        phase_masks=output.phase_masks or {},
        metadata=dict(output.metadata),
        info=_summarise_info(output) if return_info else None,
    )


# ---------------------------------------------------------------------------
# Convenience preset wrappers (TZ §10 ``presets``)
# ---------------------------------------------------------------------------


def _generate_from_preset_name(
    name: str,
    *,
    presets_dir: str | Path | None = None,
    profiles_dir: str | Path | None = None,
    return_info: bool = False,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    color_mode: str | None = None,
) -> GeneratedSample:
    pipeline = MetallographyPipelineV3(
        presets_dir=Path(presets_dir) if presets_dir else DEFAULT_PRESETS_DIR,
        profiles_dir=Path(profiles_dir) if profiles_dir else DEFAULT_PROFILES_DIR,
    )
    payload = pipeline.load_preset(name)
    if width is not None or height is not None:
        existing = payload.get("resolution") or [1024, 1024]
        h_existing = int(existing[0])
        w_existing = int(existing[1])
        payload["resolution"] = [
            int(height) if height is not None else h_existing,
            int(width) if width is not None else w_existing,
        ]
    if seed is not None:
        payload["seed"] = int(seed)
    if color_mode is not None:
        synth = dict(payload.get("synthesis_profile") or {})
        synth["color_mode"] = str(color_mode)
        payload["synthesis_profile"] = synth
    request = pipeline.request_from_preset(payload)
    output = pipeline.generate(request)
    return GeneratedSample(
        image=output.image_rgb,
        image_gray=output.image_gray,
        phase_masks=output.phase_masks or {},
        metadata=dict(output.metadata),
        info=_summarise_info(output) if return_info else None,
    )


_PRESET_ALIAS_MAP = {
    "armco": "fe_armco_annealed_v3",
    "steel_08": "steel_08_annealed_v3",
    "steel_10": "steel_10_annealed_v3",
    "steel_20": "steel_20_annealed_v3",
    "steel_45": "fe_c_hypoeutectoid_textbook",
    "steel_45_quenched": "steel_45_quenched_water_v3",
    "steel_u8": "steel_u8_tool_textbook",
    "steel_u10": "steel_u10_annealed_v3",
    "steel_u12": "steel_u12_annealed_v3",
    "steel_u13": "steel_u13_annealed_v3",
    "cast_iron_grey": "cast_iron_grey_textbook",
    "cast_iron_white_hypoeutectic": "cast_iron_white_hypoeutectic_v3",
    "cast_iron_white_eutectic": "cast_iron_white_eutectic_v3",
    "cast_iron_white_hypereutectic": "cast_iron_white_hypereutectic_v3",
}


class _PresetGroup:
    """Tiny dispatcher that mirrors the §10 ``presets.steel_20()`` style."""

    def __getattr__(self, name: str):
        preset_name = _PRESET_ALIAS_MAP.get(name.lower())
        if preset_name is None:
            raise AttributeError(f"unknown ferro-micro preset alias: {name}")
        return lambda **kwargs: _generate_from_preset_name(
            preset_name, **kwargs
        )

    def list_aliases(self) -> list[str]:
        """Return the sorted list of supported preset aliases."""
        return sorted(_PRESET_ALIAS_MAP.keys())


presets = _PresetGroup()
