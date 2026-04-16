from __future__ import annotations

import json
import sys
from copy import deepcopy
from functools import lru_cache
from importlib import import_module
from pathlib import Path

from core.app_paths import get_app_base_dir


def apply_runtime_patches() -> None:
    _install_cache_manager_alias()
    _patch_metallography_runtime()


def apply_ui_runtime_patches() -> None:
    _patch_sample_factory_window_v3()
    _patch_microscope_window()


def _install_cache_manager_alias() -> None:
    try:
        performance_module = import_module("core.performance")
    except Exception:
        return
    sys.modules.setdefault("core.cache_manager", performance_module)


def _patch_metallography_runtime() -> None:
    try:
        fm = import_module("core.metallography_v3.ferro_micro_api")
        pipeline_mod = import_module("core.metallography_v3.pipeline_v3")
    except Exception:
        return

    if getattr(fm, "_runtime_patched", False):
        return

    def _file_signature(path: Path) -> tuple[str, int, int]:
        resolved = path.resolve()
        stat = resolved.stat()
        return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)

    @lru_cache(maxsize=128)
    def _load_cached_preset(
        path_str: str,
        mtime_ns: int,
        file_size: int,
    ) -> dict[str, object]:
        del mtime_ns, file_size
        payload = json.loads(Path(path_str).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError(f"Preset payload must be object: {path_str}")
        return payload

    def _load_preset(self, name_or_path: str | Path) -> dict[str, object]:
        candidate = Path(name_or_path)
        if not candidate.exists():
            candidate = self.presets_dir / f"{name_or_path}.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Preset not found: {name_or_path}")
        return deepcopy(_load_cached_preset(*_file_signature(candidate)))

    pipeline_mod.MetallographyPipelineV3.load_preset = _load_preset

    def _runtime_dirs(
        *,
        presets_dir: str | Path | None,
        profiles_dir: str | Path | None,
    ) -> tuple[Path, Path]:
        return (
            (Path(presets_dir) if presets_dir else fm.DEFAULT_PRESETS_DIR).resolve(),
            (Path(profiles_dir) if profiles_dir else fm.DEFAULT_PROFILES_DIR).resolve(),
        )

    @lru_cache(maxsize=8)
    def _get_pipeline(
        presets_dir: str,
        profiles_dir: str,
    ):
        return pipeline_mod.MetallographyPipelineV3(
            presets_dir=Path(presets_dir),
            profiles_dir=Path(profiles_dir),
        )

    def _generate(
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
        thermal_program: list[dict[str, object]] | None = None,
        presets_dir: str | Path | None = None,
        profiles_dir: str | Path | None = None,
        return_info: bool = False,
    ):
        runtime_presets_dir, runtime_profiles_dir = _runtime_dirs(
            presets_dir=presets_dir,
            profiles_dir=profiles_dir,
        )
        pipeline = _get_pipeline(
            str(runtime_presets_dir),
            str(runtime_profiles_dir),
        )
        request = fm._build_request(
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
        return fm.GeneratedSample(
            image=output.image_rgb,
            image_gray=output.image_gray,
            phase_masks=output.phase_masks or {},
            metadata=dict(output.metadata),
            info=fm._summarise_info(output) if return_info else None,
        )

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
    ):
        runtime_presets_dir, runtime_profiles_dir = _runtime_dirs(
            presets_dir=presets_dir,
            profiles_dir=profiles_dir,
        )
        pipeline = _get_pipeline(
            str(runtime_presets_dir),
            str(runtime_profiles_dir),
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
        return fm.GeneratedSample(
            image=output.image_rgb,
            image_gray=output.image_gray,
            phase_masks=output.phase_masks or {},
            metadata=dict(output.metadata),
            info=fm._summarise_info(output) if return_info else None,
        )

    fm._runtime_dirs = _runtime_dirs
    fm._get_pipeline = _get_pipeline
    fm.generate = _generate
    fm._generate_from_preset_name = _generate_from_preset_name
    fm._runtime_patched = True


def _path_signature(path: Path) -> tuple[str, int, int] | None:
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return None
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=256)
def _load_json_dict_cached(
    path_str: str,
    mtime_ns: int,
    file_size: int,
    encoding: str,
) -> dict[str, object] | None:
    del mtime_ns, file_size
    try:
        payload = json.loads(Path(path_str).read_text(encoding=encoding))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_json_dict(
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> dict[str, object] | None:
    cache_key = _path_signature(Path(path))
    if cache_key is None:
        return None
    payload = _load_json_dict_cached(*cache_key, encoding)
    return None if payload is None else dict(payload)


def _patch_sample_factory_window_v3() -> None:
    try:
        module = import_module("ui_qt.sample_factory_window_v3")
    except Exception:
        return
    if getattr(module, "_ui_runtime_patched", False):
        return

    def _json_load(path: Path) -> dict[str, object]:
        payload = _load_json_dict(path, encoding="utf-8-sig")
        return {} if payload is None else payload

    original_init = module.SampleFactoryWindowV3.__init__

    def _patched_init(self, presets_dir=None, profiles_dir=None):
        base_dir = get_app_base_dir()
        resolved_presets_dir = (
            Path(presets_dir) if presets_dir is not None else base_dir / "presets_v3"
        )
        resolved_profiles_dir = (
            Path(profiles_dir) if profiles_dir is not None else base_dir / "profiles_v3"
        )
        if getattr(sys, "frozen", False):
            if not resolved_presets_dir.exists():
                fallback = base_dir / "presets_v3"
                if fallback.exists():
                    resolved_presets_dir = fallback
            if not resolved_profiles_dir.exists():
                fallback = base_dir / "profiles_v3"
                if fallback.exists():
                    resolved_profiles_dir = fallback
        return original_init(
            self,
            presets_dir=resolved_presets_dir,
            profiles_dir=resolved_profiles_dir,
        )

    module._json_load = _json_load
    module.SampleFactoryWindowV3.__init__ = _patched_init
    module._ui_runtime_patched = True


def _patch_microscope_window() -> None:
    try:
        module = import_module("ui_qt.microscope_window")
    except Exception:
        return
    if getattr(module, "_ui_runtime_patched", False):
        return

    module._load_json_dict = _load_json_dict

    def _find_manifest_for_image(self, image_path: Path):
        direct = image_path.with_name(f"{image_path.stem}_manifest.json")
        payload = _load_json_dict(direct)
        if isinstance(payload, dict):
            return direct, payload

        for candidate in sorted(image_path.parent.glob("*_manifest.json")):
            payload = _load_json_dict(candidate)
            if not isinstance(payload, dict):
                continue
            manifest_image = payload.get("image")
            if not isinstance(manifest_image, str):
                continue
            try:
                manifest_image_path = Path(manifest_image).resolve()
                if manifest_image_path == image_path.resolve():
                    return candidate, payload
            except Exception:
                if Path(manifest_image).name == image_path.name:
                    return candidate, payload
        return None, None

    def _load_profile_on_start(self) -> None:
        path = self._profile_path()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            module.save_json(self._profile_payload_from_controls(), path)
            return
        payload = _load_json_dict(path)
        if isinstance(payload, dict):
            self._apply_profile_payload(payload)

    def _load_profile_from_ui(self) -> None:
        path, _ = module.QFileDialog.getOpenFileName(
            self,
            "Загрузить профиль микроскопа",
            str(self._profile_path()),
            "JSON (*.json)",
        )
        if not path:
            return
        self.profile_path_edit.setText(path)
        payload = _load_json_dict(Path(path))
        if not isinstance(payload, dict):
            module.QMessageBox.critical(
                self,
                "Профиль",
                "Ошибка загрузки профиля: ожидается JSON-объект",
            )
            return
        try:
            self._apply_profile_payload(payload)
            module.QMessageBox.information(
                self,
                "Профиль",
                f"Профиль загружен:\n{path}",
            )
        except Exception as exc:
            module.QMessageBox.critical(
                self,
                "Профиль",
                f"Ошибка загрузки профиля: {exc}",
            )

    module.MicroscopeWindow._find_manifest_for_image = _find_manifest_for_image
    module.MicroscopeWindow._load_profile_on_start = _load_profile_on_start
    module.MicroscopeWindow._load_profile_from_ui = _load_profile_from_ui
    module._ui_runtime_patched = True
