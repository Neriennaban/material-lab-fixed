from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class VerificationResult:
    """Result of sample integrity verification."""
    is_valid: bool
    image_authentic: bool
    data_authentic: bool
    signature_valid: bool
    hashes_match: bool
    error_message: str | None
    answers: dict[str, Any] | None

import json
from core.security.crypto_manager import (
    compute_image_hash,
    verify_signature,
    decrypt_answers
)

def verify_sample_integrity(
    image_path: Path,
    student_json_path: Path,
    answers_enc_path: Path,
    private_key_path: Path
) -> VerificationResult:
    """Verify complete integrity of lab sample package.

    Args:
        image_path: Path to lab_sample.png
        student_json_path: Path to lab_sample_student.json
        answers_enc_path: Path to lab_sample_answers.enc
        private_key_path: Path to teacher's private key

    Returns:
        VerificationResult with detailed verification status
    """
    try:
        # Step 1: Compute current image hash
        current_hash = compute_image_hash(image_path)

        # Step 2: Load student data
        with open(student_json_path, 'r', encoding='utf-8') as f:
            student_data = json.load(f)

        student_hash = student_data.get("image_sha256")
        signature = student_data.get("digital_signature")

        if not student_hash or not signature:
            return VerificationResult(
                is_valid=False,
                image_authentic=False,
                data_authentic=False,
                signature_valid=False,
                hashes_match=False,
                error_message="Отсутствуют обязательные поля в student.json",
                answers=None
            )

        # Step 3: Load public key (embedded in application)
        from core.security.crypto_manager import get_embedded_public_key
        try:
            public_key = get_embedded_public_key()
        except FileNotFoundError as e:
            return VerificationResult(
                is_valid=False,
                image_authentic=False,
                data_authentic=False,
                signature_valid=False,
                hashes_match=False,
                error_message=str(e),
                answers=None
            )

        # Step 4: Verify signature
        data_to_verify = {k: v for k, v in student_data.items() if k != "digital_signature"}
        signature_valid = verify_signature(data_to_verify, signature, public_key)

        if not signature_valid:
            return VerificationResult(
                is_valid=False,
                image_authentic=False,
                data_authentic=False,
                signature_valid=False,
                hashes_match=False,
                error_message="Цифровая подпись недействительна - данные были изменены",
                answers=None
            )

        # Step 5: Compare image hashes
        image_authentic = (current_hash == student_hash)

        if not image_authentic:
            return VerificationResult(
                is_valid=False,
                image_authentic=False,
                data_authentic=True,
                signature_valid=True,
                hashes_match=False,
                error_message="Изображение было изменено - хеш не совпадает",
                answers=None
            )

        # Step 6: Decrypt answers
        private_key = private_key_path.read_bytes()
        encrypted_answers = answers_enc_path.read_bytes()
        answers = decrypt_answers(encrypted_answers, private_key)

        # Step 7: Verify answer hash matches
        answer_hash = answers.get("image_sha256")
        hashes_match = (current_hash == student_hash == answer_hash)

        if not hashes_match:
            return VerificationResult(
                is_valid=False,
                image_authentic=image_authentic,
                data_authentic=True,
                signature_valid=True,
                hashes_match=False,
                error_message="Хеши не совпадают между файлами",
                answers=None
            )

        # All checks passed
        return VerificationResult(
            is_valid=True,
            image_authentic=True,
            data_authentic=True,
            signature_valid=True,
            hashes_match=True,
            error_message=None,
            answers=answers
        )

    except Exception as e:
        return VerificationResult(
            is_valid=False,
            image_authentic=False,
            data_authentic=False,
            signature_valid=False,
            hashes_match=False,
            error_message=f"Ошибка при проверке: {str(e)}",
            answers=None
        )
