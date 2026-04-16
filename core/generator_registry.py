from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from .contracts_v2 import ProcessingState
from .crm_fe_c_generator import generate_crm_fe_c_rgb
from .generator_dendritic import generate_dendritic_cast
from .generator_dislocations import generate_dislocation_pits
from .generator_eutectic import generate_aged_aluminum_structure, generate_eutectic_al_si
from .generator_grains import generate_grain_structure
from .generator_calphad_phase import generate_calphad_phase_structure
from .generator_pearlite import (
    generate_martensite_structure,
    generate_pearlite_structure,
    generate_tempered_steel_structure,
)
from .generator_phase_map import generate_phase_stage_structure, supported_stages

GeneratorHandler = Callable[
    [tuple[int, int], int, dict[str, float], ProcessingState, dict[str, Any], str, dict[str, Any] | None],
    dict[str, Any],
]


def _call_with_allowed(func: Callable[..., dict[str, Any]], kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = {k: v for k, v in kwargs.items() if k in _allowed_parameters(func)}
    return func(**allowed)


@lru_cache(maxsize=None)
def _allowed_parameters(func: Callable[..., dict[str, Any]]) -> frozenset[str]:
    return frozenset(inspect.signature(func).parameters)


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3 and gray.shape[2] == 3:
        return gray.astype(np.uint8, copy=False)
    if gray.ndim == 3 and gray.shape[2] == 4:
        return gray[:, :, :3].astype(np.uint8, copy=False)
    return np.stack([gray.astype(np.uint8, copy=False)] * 3, axis=2)


def _normalize_output(raw: dict[str, Any], default_name: str) -> dict[str, Any]:
    image_gray = raw.get("image_gray", raw.get("image"))
    image_rgb = raw.get("image_rgb")

    if image_gray is None and image_rgb is None:
        raise ValueError(f"Generator '{default_name}' did not return image data.")

    if image_rgb is None:
        if isinstance(image_gray, np.ndarray) and image_gray.ndim == 3 and image_gray.shape[2] == 3:
            image_rgb = image_gray.astype(np.uint8, copy=False)
            image_gray = image_rgb.mean(axis=2).astype(np.uint8)
        else:
            assert isinstance(image_gray, np.ndarray)
            image_gray = image_gray.astype(np.uint8, copy=False)
            image_rgb = _to_rgb(image_gray)
    else:
        assert isinstance(image_rgb, np.ndarray)
        image_rgb = image_rgb.astype(np.uint8, copy=False)
        if image_gray is None:
            image_gray = image_rgb.mean(axis=2).astype(np.uint8)
        else:
            assert isinstance(image_gray, np.ndarray)
            image_gray = image_gray.astype(np.uint8, copy=False)

    phase_masks = raw.get("phase_masks")
    if phase_masks is None and raw.get("phase_mask") is not None:
        phase_masks = {"phase_1": raw["phase_mask"]}
    if isinstance(phase_masks, dict):
        normalized_masks: dict[str, np.ndarray] = {}
        for key, mask in phase_masks.items():
            if isinstance(mask, np.ndarray):
                normalized_masks[str(key)] = (mask > 0).astype(np.uint8)
        phase_masks = normalized_masks
    else:
        phase_masks = None

    metadata = dict(raw.get("metadata", {}))
    return {
        "image_gray": image_gray,
        "image_rgb": image_rgb,
        "phase_masks": phase_masks,
        "metadata": metadata,
    }


class GeneratorRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, GeneratorHandler] = {}
        self._aliases: dict[str, str] = {}
        self._register_defaults()

    def register(self, name: str, handler: GeneratorHandler, aliases: list[str] | None = None) -> None:
        canonical = name.strip().lower()
        self._handlers[canonical] = handler
        self._aliases[canonical] = canonical
        for alias in aliases or []:
            self._aliases[alias.strip().lower()] = canonical

    def canonical_name(self, name: str) -> str:
        key = name.strip().lower()
        return self._aliases.get(key, key)

    def available_generators(self) -> list[str]:
        return sorted(self._handlers.keys())

    def generate(
        self,
        name: str,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any] | None = None,
        inferred_system: str = "custom-multicomponent",
        route_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        canonical = self.canonical_name(name)
        handler = self._handlers.get(canonical)
        if handler is None:
            available = ", ".join(self.available_generators())
            raise ValueError(f"Unknown generator '{name}'. Available: {available}")
        params = dict(generator_params or {})
        raw = handler(size, int(seed), composition, processing, params, inferred_system, route_context)
        out = _normalize_output(raw=raw, default_name=canonical)
        out["metadata"]["generator_name"] = canonical
        return out

    def _register_defaults(self) -> None:
        self.register("auto", self._run_auto_placeholder, aliases=["universal_auto"])
        self.register("grains", self._run_grains, aliases=["grain"])
        self.register("pearlite", self._run_pearlite)
        self.register("eutectic", self._run_eutectic)
        self.register("dislocations", self._run_dislocations)
        self.register("martensite", self._run_martensite)
        self.register("tempered", self._run_tempered)
        self.register("aged_al", self._run_aged_al)
        self.register("phase_map", self._run_phase_map, aliases=["phase", "alloy_phase", "legacy_phase_map"])
        self.register("calphad_phase", self._run_calphad_phase, aliases=["calphad"])
        self.register("crm_fe_c", self._run_crm_fe_c, aliases=["crm", "crm_style"])
        self.register("dendritic_cast", self._run_dendritic_cast, aliases=["dendritic"])

    def _run_auto_placeholder(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raise ValueError(
            "Generator 'auto' is resolved only in GenerationPipelineV2. "
            "Use pipeline.generate(...) instead of direct registry call."
        )

    def _run_grains(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params = {
            "size": size,
            "seed": seed,
            **generator_params,
        }
        if "elongation" not in params and processing.deformation_pct > 0:
            params["elongation"] = min(3.2, 1.0 + processing.deformation_pct / 32.0)
        return _call_with_allowed(generate_grain_structure, params)

    def _run_pearlite(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params = {"size": size, "seed": seed, **generator_params}
        if "pearlite_fraction" not in params:
            c = float(composition.get("C", 0.4))
            params["pearlite_fraction"] = float(np.clip(c / 0.82, 0.08, 0.98))
        return _call_with_allowed(generate_pearlite_structure, params)

    def _run_eutectic(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params = {"size": size, "seed": seed, **generator_params}
        if "si_phase_fraction" not in params:
            si = float(composition.get("Si", 12.0))
            params["si_phase_fraction"] = float(np.clip(0.06 + 0.022 * si, 0.1, 0.75))
        return _call_with_allowed(generate_eutectic_al_si, params)

    def _run_dislocations(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params = {"size": size, "seed": seed, **generator_params}
        if "magnification" not in params:
            params["magnification"] = 200
        return _call_with_allowed(generate_dislocation_pits, params)

    def _run_martensite(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return _call_with_allowed(generate_martensite_structure, {"size": size, "seed": seed, **generator_params})

    def _run_tempered(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params = dict(generator_params)
        temperature = int(params.pop("temper_temperature_c", max(120, min(650, int(processing.temperature_c)))))
        return generate_tempered_steel_structure(size=size, seed=seed, temper_temperature_c=temperature)

    def _run_aged_al(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params = {"size": size, "seed": seed, **generator_params}
        if "precipitate_fraction" not in params:
            cu = float(composition.get("Cu", 4.0))
            mg = float(composition.get("Mg", 1.4))
            params["precipitate_fraction"] = float(np.clip(0.03 + 0.012 * cu + 0.009 * mg, 0.04, 0.22))
        return _call_with_allowed(generate_aged_aluminum_structure, params)

    def _run_phase_map(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        requested_system = str(generator_params.get("system", inferred_system)).strip().lower()
        candidates = {"fe-c", "al-si", "cu-zn", "fe-si", "al-cu-mg"}
        if requested_system not in candidates:
            requested_system = "fe-c"

        requested_stage = str(generator_params.get("stage", "auto")).strip().lower()
        if route_context and isinstance(route_context, dict):
            requested_stage = str(route_context.get("resolved_stage", requested_stage))
            requested_system = str(route_context.get("inferred_system", requested_system))

        result = generate_phase_stage_structure(
            size=size,
            seed=seed,
            system=requested_system,
            composition=composition,
            stage=requested_stage,
            temperature_c=float(generator_params.get("temperature_c", processing.temperature_c)),
            cooling_mode=str(generator_params.get("cooling_mode", processing.cooling_mode)),
            deformation_pct=float(generator_params.get("deformation_pct", processing.deformation_pct)),
            aging_temperature_c=float(generator_params.get("aging_temperature_c", processing.aging_temperature_c)),
            aging_hours=float(generator_params.get("aging_hours", processing.aging_hours)),
            thermal_slope=(
                None
                if generator_params.get("thermal_slope") is None
                else float(generator_params.get("thermal_slope"))
            ),
            liquid_fraction=(
                None
                if generator_params.get("liquid_fraction") is None
                else float(generator_params.get("liquid_fraction"))
            ),
        )
        out = dict(result)
        out["metadata"] = dict(result.get("metadata", {}))
        out["metadata"]["supported_stages"] = supported_stages(requested_system)
        return out

    def _run_crm_fe_c(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        carbon_pct = float(generator_params.get("carbon_pct", composition.get("C", 0.8)))
        grains_count = int(generator_params.get("grains_count", max(30, (size[0] * size[1]) // 15_000)))
        iron_type = str(generator_params.get("iron_type", "auto"))
        distortion = float(generator_params.get("distortion_level", 0.6))
        rgb, fractions = generate_crm_fe_c_rgb(
            width=size[1],
            height=size[0],
            carbon_pct=carbon_pct,
            grains_count=grains_count,
            seed=seed,
            iron_type=iron_type,
            distortion_level=distortion,
        )
        return {
            "image_rgb": rgb,
            "metadata": {
                "fractions": fractions,
                "carbon_pct": carbon_pct,
                "grains_count": grains_count,
                "iron_type": iron_type,
                "distortion_level": distortion,
            },
        }

    def _run_calphad_phase(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        calphad_result = generator_params.get("calphad_result")
        if not isinstance(calphad_result, dict):
            raise ValueError("CALPHAD_CONTEXT_MISSING: generator 'calphad_phase' requires 'calphad_result' in params")
        stable = calphad_result.get("stable_phases", {})
        if not isinstance(stable, dict) or not stable:
            raise ValueError("CALPHAD_RESULT_EMPTY: no stable phases returned by CALPHAD solver")
        system_name = str(generator_params.get("system", inferred_system)).strip().lower()
        transition_state = generator_params.get("phase_transition_state")
        kinetics_result = generator_params.get("kinetics_result")
        top_n = int(generator_params.get("top_n_phases", 6))
        return generate_calphad_phase_structure(
            size=size,
            seed=seed,
            system=system_name,
            phase_fractions={str(k): float(v) for k, v in stable.items()},
            transition_state=(transition_state if isinstance(transition_state, dict) else None),
            kinetics_result=(kinetics_result if isinstance(kinetics_result, dict) else None),
            top_n_phases=top_n,
            composition_wt={str(k): float(v) for k, v in composition.items()},
            equilibrium_result=calphad_result,
            composition_sensitivity_mode=str(generator_params.get("composition_sensitivity_mode", "realistic")),
        )

    def _run_dendritic_cast(
        self,
        size: tuple[int, int],
        seed: int,
        composition: dict[str, float],
        processing: ProcessingState,
        generator_params: dict[str, Any],
        inferred_system: str,
        route_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params = {"size": size, "seed": seed, **generator_params}
        if "cooling_rate" not in params:
            mode = processing.cooling_mode.lower().strip()
            default_rate = 18.0 if mode in {"equilibrium", "slow_cool"} else 80.0
            params["cooling_rate"] = default_rate
        if "undercooling" not in params:
            params["undercooling"] = max(8.0, min(120.0, params["cooling_rate"] * 0.55))
        return _call_with_allowed(generate_dendritic_cast, params)
