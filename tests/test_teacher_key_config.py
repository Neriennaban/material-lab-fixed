import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.app_paths import get_app_base_dir
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from core.security.teacher_mode import (
    load_saved_teacher_key_path,
    resolve_teacher_key_path,
    save_teacher_key_path,
    validate_teacher_private_key,
)


def _generate_private_key_pem() -> bytes:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )


class TeacherKeyConfigTests(unittest.TestCase):
    def test_get_app_base_dir_returns_repo_root_in_dev(self) -> None:
        self.assertEqual(get_app_base_dir(), Path(__file__).resolve().parents[1])

    def test_save_and_load_shared_teacher_key_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "profiles" / "teacher_config.json"
            key_path = root / "teacher_private_key.pem"
            key_path.write_bytes(_generate_private_key_pem())

            with patch("core.security.teacher_mode.get_teacher_config_path", return_value=config_path):
                save_teacher_key_path(key_path)
                loaded = load_saved_teacher_key_path()

            self.assertEqual(loaded, key_path)

    def test_resolve_teacher_key_falls_back_to_app_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fallback = root / "keys" / "teacher_private_key.pem"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_bytes(_generate_private_key_pem())

            with patch("core.security.teacher_mode.get_teacher_config_path", return_value=root / "profiles" / "teacher_config.json"):
                with patch("core.security.teacher_mode.get_app_base_dir", return_value=root):
                    resolved = resolve_teacher_key_path(prefer_saved=True)

            self.assertEqual(resolved, fallback)

    def test_validate_teacher_private_key_accepts_valid_pem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "teacher_private_key.pem"
            key_path.write_bytes(_generate_private_key_pem())
            validate_teacher_private_key(key_path)
