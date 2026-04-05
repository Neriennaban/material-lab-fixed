from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization

from core.app_paths import get_app_base_dir


def get_teacher_config_path() -> Path:
    return get_app_base_dir() / "profiles" / "teacher_config.json"


def validate_teacher_private_key(path: Path) -> None:
    key_data = path.read_bytes()
    serialization.load_pem_private_key(key_data, password=None)


def load_saved_teacher_key_path() -> Path | None:
    config_path = get_teacher_config_path()
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    raw = payload.get("private_key_path")
    if not raw:
        return None

    path = Path(str(raw))
    if not path.exists():
        return None

    try:
        validate_teacher_private_key(path)
    except Exception:
        return None
    return path


def save_teacher_key_path(path: Path) -> None:
    config_path = get_teacher_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"private_key_path": str(path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_teacher_key_path(prefer_saved: bool = True) -> Path | None:
    if prefer_saved:
        saved = load_saved_teacher_key_path()
        if saved is not None:
            return saved

    fallback = get_app_base_dir() / "keys" / "teacher_private_key.pem"
    if not fallback.exists():
        return None

    try:
        validate_teacher_private_key(fallback)
    except Exception:
        return None
    return fallback


def activate_teacher_mode_with_prompt(parent) -> Path | None:
    from PySide6.QtWidgets import QFileDialog

    resolved = resolve_teacher_key_path(prefer_saved=True)
    if resolved is not None:
        save_teacher_key_path(resolved)
        return resolved

    key_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Выберите приватный ключ преподавателя",
        str(Path.home()),
        "PEM Files (*.pem);;All Files (*)",
    )
    if not key_path:
        return None

    selected = Path(key_path)
    validate_teacher_private_key(selected)
    save_teacher_key_path(selected)
    return selected
