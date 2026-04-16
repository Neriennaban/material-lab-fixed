from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _profiles_root(custom_root: str | Path | None = None) -> Path:
    if custom_root is not None:
        return Path(custom_root)
    return Path(__file__).resolve().parents[2] / "profiles_v3"


def _to_gray_u8(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8)


def compute_style_descriptors(image_gray: np.ndarray) -> dict[str, float]:
    arr = image_gray.astype(np.float32)
    p10 = float(np.quantile(arr, 0.10))
    p90 = float(np.quantile(arr, 0.90))
    grad_x = np.abs(np.diff(arr, axis=1))
    grad_y = np.abs(np.diff(arr, axis=0))
    grad_mean = float((grad_x.mean() + grad_y.mean()) * 0.5)
    hist, _ = np.histogram(arr, bins=32, range=(0, 255), density=True)
    hist = hist + 1e-9
    entropy = float(-(hist * np.log(hist)).sum())
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": p10,
        "p90": p90,
        "dynamic_range": float(max(1e-6, p90 - p10)),
        "gradient_mean": grad_mean,
        "entropy": entropy,
    }


def load_builtin_profiles(profiles_root: str | Path | None = None) -> dict[str, Any]:
    root = _profiles_root(profiles_root)
    path = root / "metallography_profiles.json"
    if not path.exists():
        return {"profiles": {}}
    cache_key = _path_signature(path)
    if cache_key is None:
        return {"profiles": {}}
    return dict(_load_builtin_profiles_cached(*cache_key))


def resolve_reference_style(
    *,
    profile_id: str,
    profiles_root: str | Path | None = None,
) -> dict[str, Any] | None:
    data = load_builtin_profiles(profiles_root)
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        return None
    payload = profiles.get(profile_id)
    if isinstance(payload, dict):
        return dict(payload)
    return None


def import_reference_profile(
    *,
    image_path: str | Path,
    profile_id: str,
    profiles_root: str | Path | None = None,
) -> Path:
    root = _profiles_root(profiles_root)
    root.mkdir(parents=True, exist_ok=True)
    ref_dir = root / "reference_profiles"
    ref_dir.mkdir(parents=True, exist_ok=True)
    image = _to_gray_u8(Path(image_path))
    desc = compute_style_descriptors(image)
    out = ref_dir / f"{profile_id}.json"
    payload = {"profile_id": profile_id, "descriptor": desc, "source_image": str(image_path)}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_reference_profile(
    *,
    profile_id: str,
    profiles_root: str | Path | None = None,
) -> dict[str, Any] | None:
    root = _profiles_root(profiles_root)
    path = root / "reference_profiles" / f"{profile_id}.json"
    if not path.exists():
        return None
    cache_key = _path_signature(path)
    if cache_key is None:
        return None
    cached = _load_reference_profile_cached(*cache_key)
    return None if cached is None else dict(cached)


def _path_signature(path: Path) -> tuple[str, int, int] | None:
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return None
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=8)
def _load_builtin_profiles_cached(
    path_str: str,
    mtime_ns: int,
    file_size: int,
) -> dict[str, Any]:
    del mtime_ns, file_size
    try:
        payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    except Exception:
        return {"profiles": {}}
    return payload if isinstance(payload, dict) else {"profiles": {}}


@lru_cache(maxsize=64)
def _load_reference_profile_cached(
    path_str: str,
    mtime_ns: int,
    file_size: int,
) -> dict[str, Any] | None:
    del mtime_ns, file_size
    try:
        payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    desc = payload.get("descriptor")
    if not isinstance(desc, dict):
        return None
    return dict(desc)
