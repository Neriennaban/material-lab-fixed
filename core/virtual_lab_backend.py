from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts_v3 import GenerationOutputV3, MetallographyRequestV3
from core.imaging import simulate_microscope_view
from core.optical_mode_transfer import build_ferromagnetic_mask
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3
from core.microscope_measurements import derive_um_per_px_100x
from core.virtual_slide import get_array_slide


@dataclass(slots=True)
class MicroscopeState:
    objective: int = 200
    stage_x: float = 0.5
    stage_y: float = 0.5
    focus_distance_mm: float | None = None
    brightness: float = 1.0
    contrast: float = 1.0
    vignette_strength: float = 0.25
    uneven_strength: float = 0.08
    noise_sigma: float = 4.0
    add_dust: bool = False
    add_scratches: bool = False
    etch_uneven: float = 0.0
    optical_mode: str = "brightfield"
    optical_mode_parameters: dict[str, Any] = field(default_factory=dict)
    psf_profile: str = "standard"
    psf_strength: float = 0.0
    sectioning_shear_deg: float = 35.0
    hybrid_balance: float = 0.5
    output_size: tuple[int, int] = (1024, 1024)
    seed: int = 1234

    def normalized(self) -> "MicroscopeState":
        return MicroscopeState(
            objective=max(100, int(self.objective)),
            stage_x=float(np.clip(self.stage_x, 0.0, 1.0)),
            stage_y=float(np.clip(self.stage_y, 0.0, 1.0)),
            focus_distance_mm=(None if self.focus_distance_mm is None else float(self.focus_distance_mm)),
            brightness=float(max(0.1, self.brightness)),
            contrast=float(max(0.1, self.contrast)),
            vignette_strength=float(np.clip(self.vignette_strength, 0.0, 1.0)),
            uneven_strength=float(np.clip(self.uneven_strength, 0.0, 1.0)),
            noise_sigma=float(max(0.0, self.noise_sigma)),
            add_dust=bool(self.add_dust),
            add_scratches=bool(self.add_scratches),
            etch_uneven=float(np.clip(self.etch_uneven, 0.0, 1.0)),
            optical_mode=str(self.optical_mode or "brightfield"),
            optical_mode_parameters=dict(self.optical_mode_parameters or {}),
            psf_profile=str(self.psf_profile or "standard"),
            psf_strength=float(np.clip(self.psf_strength, 0.0, 1.0)),
            sectioning_shear_deg=float(self.sectioning_shear_deg),
            hybrid_balance=float(np.clip(self.hybrid_balance, 0.0, 1.0)),
            output_size=(max(64, int(self.output_size[0])), max(64, int(self.output_size[1]))),
            seed=int(self.seed),
        )


@dataclass(slots=True)
class PreparedMicroscopeSlide:
    output: GenerationOutputV3
    sample_id: str
    image_gray: np.ndarray
    image_rgb: np.ndarray
    metadata: dict[str, Any]
    source: str
    phase_masks: dict[str, np.ndarray] = field(default_factory=dict)
    feature_masks: dict[str, np.ndarray] = field(default_factory=dict)
    prep_maps: dict[str, np.ndarray] = field(default_factory=dict)

    def warm_runtime_cache(self) -> None:
        get_array_slide(self.image_gray)
        for payload in (self.phase_masks, self.feature_masks, self.prep_maps):
            for arr in payload.values():
                if isinstance(arr, np.ndarray):
                    get_array_slide(arr)

    @property
    def inferred_system(self) -> str:
        return str(self.metadata.get("inferred_system", ""))

    @property
    def final_stage(self) -> str:
        return str(self.metadata.get("final_stage", ""))


class VirtualLabBackend:
    """
    Backend facade for the laboratory complex.

    Responsibilities:
    1. Generate a metallographic slide from a V3 preset or request.
    2. Warm multiresolution runtime caches for fast microscope work.
    3. Render microscope frames with objective, XY stage and physical-focus logic.

    This class is intentionally UI-agnostic so it can be used from PySide6,
    a web service, or automated laboratory scenarios.
    """

    def __init__(
        self,
        *,
        presets_dir: str | Path | None = None,
        profiles_dir: str | Path | None = None,
        generator_version: str = "v3.0.0+realtime",
    ) -> None:
        self.pipeline = MetallographyPipelineV3(
            presets_dir=presets_dir,
            profiles_dir=profiles_dir,
            generator_version=generator_version,
        )

    def generate_slide(self, request: MetallographyRequestV3 | dict[str, Any]) -> PreparedMicroscopeSlide:
        req = request if isinstance(request, MetallographyRequestV3) else MetallographyRequestV3.from_dict(dict(request))
        output = self.pipeline.generate(req)
        slide = PreparedMicroscopeSlide(
            output=output,
            sample_id=str(req.sample_id),
            image_gray=output.image_gray.astype(np.uint8, copy=False),
            image_rgb=output.image_rgb.astype(np.uint8, copy=False),
            metadata=dict(output.metadata),
            source=str(req.sample_id),
            phase_masks={str(k): np.asarray(v, dtype=np.uint8) for k, v in (output.phase_masks or {}).items()},
            feature_masks={str(k): np.asarray(v, dtype=np.uint8) for k, v in (output.feature_masks or {}).items()},
            prep_maps={str(k): np.asarray(v, dtype=np.uint8) for k, v in (output.prep_maps or {}).items()},
        )
        slide.warm_runtime_cache()
        return slide

    def generate_from_preset(self, name_or_path: str | Path) -> PreparedMicroscopeSlide:
        payload = self.pipeline.load_preset(name_or_path)
        request = self.pipeline.request_from_preset(payload)
        return self.generate_slide(request)

    @staticmethod
    def objective_nominal_distance_mm(objective: int) -> float:
        return float(np.clip(36.0 - 0.045 * float(objective), 9.0, 32.0))

    @classmethod
    def focus_target_mm(cls, objective: int, stage_x: float, stage_y: float) -> float:
        return float(
            cls.objective_nominal_distance_mm(objective)
            + (float(stage_x) - 0.5) * 1.4
            + (float(stage_y) - 0.5) * -1.0
        )

    @staticmethod
    def focus_quality(objective: int, focus_distance_mm: float, focus_target_mm: float) -> float:
        tolerance_mm = max(0.18, 0.75 - 0.0009 * float(objective))
        error_mm = abs(float(focus_distance_mm) - float(focus_target_mm))
        return float(np.clip(1.0 - error_mm / tolerance_mm, 0.0, 1.0))

    @staticmethod
    def _resolve_um_per_px_100x(metadata: dict[str, Any]) -> tuple[float, str]:
        return derive_um_per_px_100x(metadata, default=1.0)

    def render_microscope_frame(
        self,
        slide: PreparedMicroscopeSlide,
        state: MicroscopeState | dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        current = state if isinstance(state, MicroscopeState) else MicroscopeState(**(state or {}))
        current = current.normalized()
        focus_target_mm = self.focus_target_mm(current.objective, current.stage_x, current.stage_y)
        focus_distance_mm = float(current.focus_distance_mm) if current.focus_distance_mm is not None else float(focus_target_mm)
        um_per_px_100x, scale_source = self._resolve_um_per_px_100x(slide.metadata)
        ferromagnetic_mask, _ = build_ferromagnetic_mask(
            slide.phase_masks,
            pure_iron_like=bool(dict(slide.metadata.get("pure_iron_baseline", {})).get("applied", False)),
            size=slide.image_gray.shape,
        )

        view, view_meta = simulate_microscope_view(
            sample=slide.image_gray,
            optical_ferromagnetic_mask_sample=np.clip(ferromagnetic_mask * 255.0, 0.0, 255.0).astype(np.uint8),
            magnification=current.objective,
            pan_x=current.stage_x,
            pan_y=current.stage_y,
            output_size=current.output_size,
            focus_distance_mm=focus_distance_mm,
            focus_target_mm=focus_target_mm,
            um_per_px_100x=um_per_px_100x,
            brightness=current.brightness,
            contrast=current.contrast,
            vignette_strength=current.vignette_strength,
            uneven_strength=current.uneven_strength,
            noise_sigma=current.noise_sigma,
            add_dust=current.add_dust,
            add_scratches=current.add_scratches,
            etch_uneven=current.etch_uneven,
            optical_mode=str(current.optical_mode or slide.metadata.get("request_v3", {}).get("microscope_profile", {}).get("optical_mode", "brightfield")),
            optical_mode_parameters=dict(current.optical_mode_parameters or slide.metadata.get("request_v3", {}).get("microscope_profile", {}).get("optical_mode_parameters", {})),
            optical_context={
                "inferred_system": str(slide.metadata.get("inferred_system", "")),
                "final_stage": str(slide.metadata.get("final_stage", "")),
                "pure_iron_baseline_applied": bool(dict(slide.metadata.get("pure_iron_baseline", {})).get("applied", False)),
            },
            psf_profile=current.psf_profile,
            psf_strength=current.psf_strength,
            sectioning_shear_deg=current.sectioning_shear_deg,
            hybrid_balance=current.hybrid_balance,
            seed=current.seed,
        )
        quality = self.focus_quality(current.objective, focus_distance_mm, focus_target_mm)
        frame_meta = {
            **dict(view_meta),
            "sample_id": str(slide.sample_id),
            "source": str(slide.source),
            "objective": int(current.objective),
            "stage_x": float(current.stage_x),
            "stage_y": float(current.stage_y),
            "focus_distance_mm": float(focus_distance_mm),
            "focus_target_mm": float(focus_target_mm),
            "focus_quality": float(quality),
            "focus_in_range": bool(quality >= 0.45),
            "scale_source": str(scale_source),
            "psf_profile": str(current.psf_profile),
            "optical_mode": str(current.optical_mode),
            "psf_strength": float(current.psf_strength),
            "sectioning_shear_deg": float(current.sectioning_shear_deg),
            "hybrid_balance": float(current.hybrid_balance),
            "inferred_system": str(slide.metadata.get("inferred_system", "")),
            "final_stage": str(slide.metadata.get("final_stage", "")),
            "etch_summary": dict(slide.metadata.get("etch_summary", {})) if isinstance(slide.metadata.get("etch_summary"), dict) else {},
            "prep_summary": dict(slide.metadata.get("prep_summary", {})) if isinstance(slide.metadata.get("prep_summary"), dict) else {},
            "pure_iron_baseline": dict(slide.metadata.get("pure_iron_baseline", {})) if isinstance(slide.metadata.get("pure_iron_baseline"), dict) else {},
            "pure_iron_optical_recommendation": (
                dict(slide.metadata.get("pure_iron_optical_recommendation", {}))
                if isinstance(slide.metadata.get("pure_iron_optical_recommendation"), dict)
                else {}
            ),
            "optical_recommendation": (
                dict(slide.metadata.get("optical_recommendation", {}))
                if isinstance(slide.metadata.get("optical_recommendation"), dict)
                else {}
            ),
            "electron_microscopy_guidance": (
                dict(slide.metadata.get("electron_microscopy_guidance", {}))
                if isinstance(slide.metadata.get("electron_microscopy_guidance"), dict)
                else {}
            ),
            "pure_iron_electropolish_profile": str(slide.metadata.get("pure_iron_electropolish_profile", "")),
            "pure_iron_polarized_extinction_score": float(slide.metadata.get("pure_iron_polarized_extinction_score", 0.0) or 0.0),
            "single_phase_negative_control": bool(slide.metadata.get("single_phase_negative_control", False)),
            "multiphase_separability_applicable": bool(slide.metadata.get("multiphase_separability_applicable", True)),
        }
        pure_iron = frame_meta["pure_iron_baseline"]
        if pure_iron:
            frame_meta["pure_iron_baseline_applied"] = bool(pure_iron.get("applied", False))
            frame_meta["pure_iron_cleanliness_score"] = float(pure_iron.get("cleanliness_score", 0.0) or 0.0)
            frame_meta["pure_iron_dark_defect_suppression"] = float(pure_iron.get("dark_defect_suppression", 0.0) or 0.0)
            frame_meta["pure_iron_boundary_visibility_score"] = float(pure_iron.get("boundary_visibility_score", 0.0) or 0.0)
            frame_meta["pure_iron_dark_defect_warning"] = bool(
                frame_meta["pure_iron_baseline_applied"]
                and frame_meta["pure_iron_cleanliness_score"] < 0.55
            )
        return view.astype(np.uint8, copy=False), frame_meta


__all__ = [
    "MicroscopeState",
    "PreparedMicroscopeSlide",
    "VirtualLabBackend",
]
