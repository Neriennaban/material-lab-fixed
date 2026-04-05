"""Security module for cryptographic operations and verification."""

from .crypto_manager import (
    compute_image_hash,
    sign_data,
    verify_signature,
    encrypt_answers,
    decrypt_answers,
)
from .verification import (
    VerificationResult,
    verify_sample_integrity,
)
from .teacher_mode import (
    activate_teacher_mode_with_prompt,
    get_teacher_config_path,
    load_saved_teacher_key_path,
    resolve_teacher_key_path,
    save_teacher_key_path,
    validate_teacher_private_key,
)

__all__ = [
    'compute_image_hash',
    'sign_data',
    'verify_signature',
    'encrypt_answers',
    'decrypt_answers',
    'VerificationResult',
    'verify_sample_integrity',
    'activate_teacher_mode_with_prompt',
    'get_teacher_config_path',
    'load_saved_teacher_key_path',
    'resolve_teacher_key_path',
    'save_teacher_key_path',
    'validate_teacher_private_key',
]
