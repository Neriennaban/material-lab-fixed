from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .composition import apply_composition_rules, merge_composition
from .generator_dislocations import generate_dislocation_pits
from .generator_eutectic import generate_aged_aluminum_structure, generate_eutectic_al_si
from .generator_grains import generate_grain_structure
from .generator_phase_map import generate_phase_stage_structure
from .generator_pearlite import (
    generate_martensite_structure,
    generate_pearlite_structure,
    generate_tempered_steel_structure,
)
from .imaging import simulate_microscope_view
from .materials import MaterialPreset, list_presets, load_preset


@dataclass(slots=True)
class GenerationResult:
    preset: MaterialPreset
    sample_image: np.ndarray
    sample_metadata: dict[str, Any]
    aux: dict[str, Any]
    view_image: np.ndarray | None = None
    view_metadata: dict[str, Any] | None = None

    def metadata_for_export(self) -> dict[str, Any]:
        payload = {
            "preset": self.preset.to_dict(),
            "sample_metadata": self.sample_metadata,
        }
        if self.view_metadata is not None:
            payload["view_metadata"] = self.view_metadata
        return payload

    def metadata_json(self) -> str:
        return json.dumps(self.metadata_for_export(), ensure_ascii=False, indent=2)


class GenerationEngine:
    """Orchestrates generation + virtual microscope rendering."""

    def __init__(self, presets_dir: str | Path | None = None) -> None:
        if presets_dir is None:
            self.presets_dir = Path(__file__).resolve().parents[1] / "presets"
        else:
            self.presets_dir = Path(presets_dir)
        self._view_cache: OrderedDict[tuple[Any, ...], tuple[np.ndarray, dict[str, Any]]] = OrderedDict()
        self._view_cache_size = 16

    def list_preset_paths(self) -> list[Path]:
        return list_presets(self.presets_dir)

    def list_preset_names(self) -> list[str]:
        return [p.stem for p in self.list_preset_paths()]

    def load_preset(self, name_or_path: str | Path) -> MaterialPreset:
        candidate = Path(name_or_path)
        if candidate.exists():
            return load_preset(candidate)

        by_name = self.presets_dir / f"{name_or_path}.json"
        if by_name.exists():
            return load_preset(by_name)
        raise FileNotFoundError(f"Preset not found: {name_or_path}")

    @staticmethod
    def _q(v: float, precision: int = 4) -> int:
        return int(round(float(v) * (10**precision)))

    def _view_cache_key(self, result: GenerationResult, params: dict[str, Any], output_size: tuple[int, int]) -> tuple[Any, ...]:
        sample = result.sample_image
        return (
            sample.__array_interface__["data"][0],
            int(sample.shape[0]),
            int(sample.shape[1]),
            int(params.get("magnification", 200)),
            self._q(params.get("pan_x", 0.5)),
            self._q(params.get("pan_y", 0.5)),
            self._q(params.get("focus", 1.0)),
            self._q(params.get("brightness", 1.0), precision=3),
            self._q(params.get("contrast", 1.0), precision=3),
            self._q(params.get("vignette_strength", 0.15)),
            self._q(params.get("uneven_strength", 0.08)),
            self._q(params.get("noise_sigma", 3.5)),
            1 if bool(params.get("add_dust", False)) else 0,
            1 if bool(params.get("add_scratches", False)) else 0,
            self._q(params.get("etch_uneven", 0.0)),
            int(params.get("seed", result.sample_metadata["seed"] + 1000)),
            int(output_size[0]),
            int(output_size[1]),
        )

    def _cache_view(self, key: tuple[Any, ...], view: np.ndarray, meta: dict[str, Any]) -> None:
        self._view_cache[key] = (view, dict(meta))
        self._view_cache.move_to_end(key)
        while len(self._view_cache) > self._view_cache_size:
            self._view_cache.popitem(last=False)

    def generate_sample(
        self,
        preset: MaterialPreset,
        seed_override: int | None = None,
        image_size_override: tuple[int, int] | None = None,
        composition_override: dict[str, Any] | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ) -> GenerationResult:
        seed = int(seed_override if seed_override is not None else preset.seed)
        size = image_size_override or preset.image_size
        composition = merge_composition(preset.composition, composition_override)
        generator_name = preset.generator.lower().strip()
        params, composition_notes = apply_composition_rules(
            preset=preset,
            generation_params=dict(preset.generation),
            composition=composition,
        )
        if generation_overrides:
            if generator_name in {"phase_map", "phase", "alloy_phase"}:
                params.update(generation_overrides)
            else:
                # For non-phase generators, only override existing numeric controls.
                for key, value in generation_overrides.items():
                    if key in params:
                        params[key] = value

        if generator_name == "grains":
            generated = generate_grain_structure(size=size, seed=seed, **params)
        elif generator_name == "pearlite":
            generated = generate_pearlite_structure(size=size, seed=seed, **params)
        elif generator_name == "eutectic":
            generated = generate_eutectic_al_si(size=size, seed=seed, **params)
        elif generator_name == "dislocations":
            generated = generate_dislocation_pits(size=size, seed=seed, **params)
        elif generator_name == "martensite":
            generated = generate_martensite_structure(size=size, seed=seed, **params)
        elif generator_name == "tempered":
            temperature = int(params.pop("temper_temperature_c", 300))
            generated = generate_tempered_steel_structure(
                size=size,
                seed=seed,
                temper_temperature_c=temperature,
            )
            generated["metadata"]["temper_temperature_c"] = temperature
        elif generator_name == "aged_al":
            generated = generate_aged_aluminum_structure(size=size, seed=seed, **params)
        elif generator_name in {"phase_map", "phase", "alloy_phase"}:
            phase_allowed = {
                "system",
                "stage",
                "temperature_c",
                "cooling_mode",
                "deformation_pct",
                "aging_temperature_c",
                "aging_hours",
            }
            phase_params = {k: v for k, v in params.items() if k in phase_allowed}
            generated = generate_phase_stage_structure(
                size=size,
                seed=seed,
                composition=composition,
                **phase_params,
            )
        else:
            raise ValueError(f"Unsupported generator type: {preset.generator}")

        image = generated["image"].astype(np.uint8, copy=False)
        sample_meta = {
            "generator": preset.generator,
            "seed": seed,
            "size_px": [int(size[0]), int(size[1])],
            "composition_wt": composition,
            "composition_notes": composition_notes,
            "generation_overrides": generation_overrides or {},
            **dict(generated.get("metadata", {})),
        }
        aux = {k: v for k, v in generated.items() if k not in {"image", "metadata"}}
        return GenerationResult(
            preset=preset,
            sample_image=image,
            sample_metadata=sample_meta,
            aux=aux,
        )

    def render_view(
        self,
        result: GenerationResult,
        microscope_overrides: dict[str, Any] | None = None,
        output_size: tuple[int, int] = (1024, 1024),
    ) -> GenerationResult:
        params = dict(result.preset.microscope)
        if microscope_overrides:
            params.update(microscope_overrides)

        magnification = int(params.get("magnification", 200))
        pan_x = float(params.get("pan_x", 0.5))
        pan_y = float(params.get("pan_y", 0.5))
        focus = float(params.get("focus", 1.0))
        brightness = float(params.get("brightness", 1.0))
        contrast = float(params.get("contrast", 1.0))
        vignette = float(params.get("vignette_strength", 0.15))
        uneven = float(params.get("uneven_strength", 0.08))
        noise_sigma = float(params.get("noise_sigma", 3.5))
        add_dust = bool(params.get("add_dust", False))
        add_scratches = bool(params.get("add_scratches", False))
        etch_uneven = float(params.get("etch_uneven", 0.0))
        view_seed = int(params.get("seed", result.sample_metadata["seed"] + 1000))
        cache_key = self._view_cache_key(result, params, output_size)
        cached = self._view_cache.get(cache_key)
        if cached is not None:
            self._view_cache.move_to_end(cache_key)
            result.view_image = cached[0]
            result.view_metadata = dict(cached[1])
            return result

        view, fov_meta = simulate_microscope_view(
            sample=result.sample_image,
            magnification=magnification,
            pan_x=pan_x,
            pan_y=pan_y,
            output_size=output_size,
            focus=focus,
            brightness=brightness,
            contrast=contrast,
            vignette_strength=vignette,
            uneven_strength=uneven,
            noise_sigma=noise_sigma,
            add_dust=add_dust,
            add_scratches=add_scratches,
            etch_uneven=etch_uneven,
            seed=view_seed,
        )

        result.view_image = view
        result.view_metadata = {
            "magnification": magnification,
            "focus": focus,
            "brightness": brightness,
            "contrast": contrast,
            "vignette_strength": vignette,
            "uneven_strength": uneven,
            "noise_sigma": noise_sigma,
            "add_dust": add_dust,
            "add_scratches": add_scratches,
            "etch_uneven": etch_uneven,
            "view_seed": view_seed,
            **fov_meta,
        }
        self._cache_view(cache_key, result.view_image, result.view_metadata)
        return result

    def generate_from_preset(
        self,
        name_or_path: str | Path,
        seed_override: int | None = None,
        image_size_override: tuple[int, int] | None = None,
        composition_override: dict[str, Any] | None = None,
        generation_overrides: dict[str, Any] | None = None,
        microscope_overrides: dict[str, Any] | None = None,
        output_size: tuple[int, int] = (1024, 1024),
    ) -> GenerationResult:
        preset = self.load_preset(name_or_path)
        result = self.generate_sample(
            preset=preset,
            seed_override=seed_override,
            image_size_override=image_size_override,
            composition_override=composition_override,
            generation_overrides=generation_overrides,
        )
        return self.render_view(
            result=result,
            microscope_overrides=microscope_overrides,
            output_size=output_size,
        )
