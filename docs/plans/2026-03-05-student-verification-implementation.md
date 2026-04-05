# Student Verification System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement secure student verification system with RSA encryption, digital signatures, and teacher/student mode separation

**Architecture:** Two-file export system (student.json + answers.enc), hybrid RSA-2048 + AES-256-GCM encryption, SHA256 integrity verification, mode-based UI with protected teacher panel

**Tech Stack:** Python 3.10+, cryptography library, PySide6, existing material_lab V3 architecture

---

## Task 1: Setup Dependencies and Key Generation Utility

**Files:**
- Modify: `requirements.txt`
- Create: `scripts/generate_keys.py`
- Create: `keys/.gitignore`

**Step 1: Write test for key generation utility**

```python
# tests/test_key_generation.py
import unittest
from pathlib import Path
from scripts.generate_keys import generate_keypair, save_keys

class TestKeyGeneration(unittest.TestCase):
    def test_generate_keypair_returns_bytes(self):
        private_key, public_key = generate_keypair()
        self.assertIsInstance(private_key, bytes)
        self.assertIsInstance(public_key, bytes)
        self.assertGreater(len(private_key), 100)
        self.assertGreater(len(public_key), 100)

    def test_keys_are_different(self):
        private_key, public_key = generate_keypair()
        self.assertNotEqual(private_key, public_key)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_key_generation.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.generate_keys'"

**Step 3: Add cryptography dependency**

```txt
# requirements.txt (append)
cryptography>=41.0.0
```

**Step 4: Install dependency**

Run: `pip install cryptography>=41.0.0`
Expected: Successfully installed cryptography

**Step 5: Write minimal key generation script**

```python
# scripts/generate_keys.py
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from pathlib import Path

def generate_keypair() -> tuple[bytes, bytes]:
    """Generate RSA-2048 keypair. Returns (private_key_pem, public_key_pem)."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    return private_pem, public_pem

def save_keys(private_key: bytes, public_key: bytes, output_dir: Path) -> None:
    """Save keys to PEM files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    private_path = output_dir / "teacher_private_key.pem"
    public_path = output_dir / "public_key.pem"

    private_path.write_bytes(private_key)
    public_path.write_bytes(public_key)

    print(f"✅ Keys generated:")
    print(f"   Private: {private_path}")
    print(f"   Public: {public_path}")

if __name__ == "__main__":
    private_key, public_key = generate_keypair()
    save_keys(private_key, public_key, Path("keys"))
```

**Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_key_generation.py -v`
Expected: PASS (2 tests)

**Step 7: Create keys directory with gitignore**

```
# keys/.gitignore
# Never commit private keys
teacher_private_key.pem
*.pem
!public_key.pem
```

**Step 8: Generate actual keys**

Run: `python scripts/generate_keys.py`
Expected: Creates `keys/teacher_private_key.pem` and `keys/public_key.pem`

**Step 9: Commit**

```bash
git add requirements.txt scripts/generate_keys.py keys/.gitignore tests/test_key_generation.py
git commit -m "feat: add RSA key generation utility and cryptography dependency"
```

---

## Task 2: Create Crypto Manager Module

**Files:**
- Create: `core/security/__init__.py`
- Create: `core/security/crypto_manager.py`
- Create: `tests/test_crypto_manager.py`

**Step 1: Write test for image hash computation**

```python
# tests/test_crypto_manager.py
import unittest
from pathlib import Path
from PIL import Image
import tempfile
from core.security.crypto_manager import compute_image_hash

class TestCryptoManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_image = Path(self.temp_dir) / "test.png"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(self.test_image)

    def test_compute_image_hash_returns_hex_string(self):
        hash_value = compute_image_hash(self.test_image)
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)  # SHA256 = 64 hex chars

    def test_same_image_produces_same_hash(self):
        hash1 = compute_image_hash(self.test_image)
        hash2 = compute_image_hash(self.test_image)
        self.assertEqual(hash1, hash2)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_crypto_manager.py::TestCryptoManager::test_compute_image_hash_returns_hex_string -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'core.security'"

**Step 3: Create security module structure**

```python
# core/security/__init__.py
"""Security module for cryptographic operations and verification."""

from .crypto_manager import (
    compute_image_hash,
    sign_data,
    verify_signature,
    encrypt_answers,
    decrypt_answers,
)

__all__ = [
    'compute_image_hash',
    'sign_data',
    'verify_signature',
    'encrypt_answers',
    'decrypt_answers',
]
```

**Step 4: Implement compute_image_hash**

```python
# core/security/crypto_manager.py
import hashlib
import json
from pathlib import Path
from typing import Any

def compute_image_hash(image_path: Path) -> str:
    """Compute SHA256 hash of image file.

    Args:
        image_path: Path to image file

    Returns:
        Hex string of SHA256 hash (64 characters)
    """
    sha256 = hashlib.sha256()
    with open(image_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_crypto_manager.py::TestCryptoManager::test_compute_image_hash_returns_hex_string -v`
Expected: PASS

**Step 6: Write test for digital signatures**

```python
# tests/test_crypto_manager.py (append to TestCryptoManager)
    def test_sign_and_verify_data(self):
        from scripts.generate_keys import generate_keypair
        from core.security.crypto_manager import sign_data, verify_signature

        private_key, public_key = generate_keypair()
        test_data = {"sample_id": "test123", "carbon": 0.45}

        signature = sign_data(test_data, private_key)
        self.assertIsInstance(signature, str)

        is_valid = verify_signature(test_data, signature, public_key)
        self.assertTrue(is_valid)

    def test_verify_fails_with_modified_data(self):
        from scripts.generate_keys import generate_keypair
        from core.security.crypto_manager import sign_data, verify_signature

        private_key, public_key = generate_keypair()
        original_data = {"sample_id": "test123", "carbon": 0.45}

        signature = sign_data(original_data, private_key)

        modified_data = {"sample_id": "test123", "carbon": 0.50}
        is_valid = verify_signature(modified_data, signature, public_key)
        self.assertFalse(is_valid)
```

**Step 7: Run test to verify it fails**

Run: `python -m pytest tests/test_crypto_manager.py::TestCryptoManager::test_sign_and_verify_data -v`
Expected: FAIL with "ImportError: cannot import name 'sign_data'"

**Step 8: Implement sign_data and verify_signature**

```python
# core/security/crypto_manager.py (append)
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

def _canonicalize_json(data: dict[str, Any]) -> bytes:
    """Convert dict to canonical JSON bytes for signing."""
    return json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')

def sign_data(data: dict[str, Any], private_key_pem: bytes) -> str:
    """Create digital signature for data using RSA private key.

    Args:
        data: Dictionary to sign
        private_key_pem: PEM-encoded RSA private key

    Returns:
        Base64-encoded signature string
    """
    private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None
    )

    canonical_data = _canonicalize_json(data)

    signature = private_key.sign(
        canonical_data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    return base64.b64encode(signature).decode('utf-8')

def verify_signature(data: dict[str, Any], signature: str, public_key_pem: bytes) -> bool:
    """Verify digital signature using RSA public key.

    Args:
        data: Dictionary that was signed
        signature: Base64-encoded signature string
        public_key_pem: PEM-encoded RSA public key

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        public_key = serialization.load_pem_public_key(public_key_pem)
        canonical_data = _canonicalize_json(data)
        signature_bytes = base64.b64decode(signature)

        public_key.verify(
            signature_bytes,
            canonical_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False
```

**Step 9: Run tests to verify they pass**

Run: `python -m pytest tests/test_crypto_manager.py -v`
Expected: PASS (4 tests)

**Step 10: Commit**

```bash
git add core/security/ tests/test_crypto_manager.py
git commit -m "feat: add crypto manager with hashing and digital signatures"
```

---

## Task 3: Implement Hybrid Encryption

**Files:**
- Modify: `core/security/crypto_manager.py`
- Modify: `tests/test_crypto_manager.py`

**Step 1: Write test for encryption/decryption**

```python
# tests/test_crypto_manager.py (append to TestCryptoManager)
    def test_encrypt_and_decrypt_answers(self):
        from scripts.generate_keys import generate_keypair
        from core.security.crypto_manager import encrypt_answers, decrypt_answers

        private_key, public_key = generate_keypair()

        answers = {
            "sample_id": "test123",
            "phase_fractions": {"ferrite": 0.69, "pearlite": 0.31},
            "steel_grade": "30",
            "carbon_content": 0.32
        }

        encrypted = encrypt_answers(answers, public_key)
        self.assertIsInstance(encrypted, bytes)
        self.assertGreater(len(encrypted), 100)

        decrypted = decrypt_answers(encrypted, private_key)
        self.assertEqual(decrypted, answers)

    def test_decrypt_fails_with_wrong_key(self):
        from scripts.generate_keys import generate_keypair
        from core.security.crypto_manager import encrypt_answers, decrypt_answers

        private_key1, public_key1 = generate_keypair()
        private_key2, _ = generate_keypair()

        answers = {"sample_id": "test123"}
        encrypted = encrypt_answers(answers, public_key1)

        with self.assertRaises(Exception):
            decrypt_answers(encrypted, private_key2)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_crypto_manager.py::TestCryptoManager::test_encrypt_and_decrypt_answers -v`
Expected: FAIL with "ImportError: cannot import name 'encrypt_answers'"

**Step 3: Implement hybrid encryption**

```python
# core/security/crypto_manager.py (append)
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def encrypt_answers(answers: dict[str, Any], public_key_pem: bytes) -> bytes:
    """Encrypt answers using hybrid RSA + AES encryption.

    Args:
        answers: Dictionary containing teacher answers
        public_key_pem: PEM-encoded RSA public key

    Returns:
        Encrypted bytes (RSA-encrypted AES key + AES-encrypted data)
    """
    public_key = serialization.load_pem_public_key(public_key_pem)

    # Generate random AES-256 key
    aes_key = os.urandom(32)

    # Encrypt AES key with RSA
    encrypted_aes_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Encrypt data with AES-GCM
    nonce = os.urandom(12)
    cipher = Cipher(algorithms.AES(aes_key), modes.GCM(nonce))
    encryptor = cipher.encryptor()

    plaintext = json.dumps(answers, sort_keys=True).encode('utf-8')
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()

    # Format: [encrypted_aes_key_length(2)] + [encrypted_aes_key] + [nonce(12)] + [tag(16)] + [ciphertext]
    result = (
        len(encrypted_aes_key).to_bytes(2, 'big') +
        encrypted_aes_key +
        nonce +
        encryptor.tag +
        ciphertext
    )

    return result

def decrypt_answers(encrypted: bytes, private_key_pem: bytes) -> dict[str, Any]:
    """Decrypt answers using hybrid RSA + AES decryption.

    Args:
        encrypted: Encrypted bytes from encrypt_answers()
        private_key_pem: PEM-encoded RSA private key

    Returns:
        Decrypted answers dictionary

    Raises:
        Exception: If decryption fails (wrong key, corrupted data, etc.)
    """
    private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None
    )

    # Parse encrypted data
    key_length = int.from_bytes(encrypted[0:2], 'big')
    encrypted_aes_key = encrypted[2:2+key_length]
    nonce = encrypted[2+key_length:2+key_length+12]
    tag = encrypted[2+key_length+12:2+key_length+28]
    ciphertext = encrypted[2+key_length+28:]

    # Decrypt AES key with RSA
    aes_key = private_key.decrypt(
        encrypted_aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Decrypt data with AES-GCM
    cipher = Cipher(algorithms.AES(aes_key), modes.GCM(nonce, tag))
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    return json.loads(plaintext.decode('utf-8'))
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_crypto_manager.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add core/security/crypto_manager.py tests/test_crypto_manager.py
git commit -m "feat: add hybrid RSA+AES encryption for answer protection"
```

---

## Task 4: Create Verification Module

**Files:**
- Create: `core/security/verification.py`
- Create: `tests/test_verification.py`

**Step 1: Write test for verification result dataclass**

```python
# tests/test_verification.py
import unittest
from core.security.verification import VerificationResult

class TestVerification(unittest.TestCase):
    def test_verification_result_creation(self):
        result = VerificationResult(
            is_valid=True,
            image_authentic=True,
            data_authentic=True,
            signature_valid=True,
            hashes_match=True,
            error_message=None,
            answers={"steel_grade": "30"}
        )
        self.assertTrue(result.is_valid)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.answers["steel_grade"], "30")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_verification.py::TestVerification::test_verification_result_creation -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'core.security.verification'"

**Step 3: Create verification module with dataclass**

```python
# core/security/verification.py
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_verification.py::TestVerification::test_verification_result_creation -v`
Expected: PASS

**Step 5: Write test for full verification workflow**

```python
# tests/test_verification.py (append to TestVerification)
    def test_verify_sample_integrity_success(self):
        import tempfile
        import json
        from PIL import Image
        from pathlib import Path
        from scripts.generate_keys import generate_keypair
        from core.security.crypto_manager import (
            compute_image_hash,
            sign_data,
            encrypt_answers
        )
        from core.security.verification import verify_sample_integrity

        # Setup temp directory
        temp_dir = Path(tempfile.mkdtemp())

        # Generate keys
        private_key, public_key = generate_keypair()
        private_key_path = temp_dir / "private.pem"
        private_key_path.write_bytes(private_key)

        # Create test image
        image_path = temp_dir / "test.png"
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(image_path)

        # Compute hash
        image_hash = compute_image_hash(image_path)

        # Create student data
        student_data = {
            "sample_id": "test123",
            "composition_wt": {"Fe": 99.68, "C": 0.32},
            "image_sha256": image_hash
        }
        signature = sign_data(student_data, private_key)
        student_data["digital_signature"] = signature

        student_json_path = temp_dir / "student.json"
        student_json_path.write_text(json.dumps(student_data, indent=2))

        # Create encrypted answers
        answers = {
            "sample_id": "test123",
            "image_sha256": image_hash,
            "steel_grade": "30",
            "carbon_content": 0.32
        }
        encrypted = encrypt_answers(answers, public_key)

        answers_path = temp_dir / "answers.enc"
        answers_path.write_bytes(encrypted)

        # Verify
        result = verify_sample_integrity(
            image_path,
            student_json_path,
            answers_path,
            private_key_path
        )

        self.assertTrue(result.is_valid)
        self.assertTrue(result.image_authentic)
        self.assertTrue(result.data_authentic)
        self.assertTrue(result.signature_valid)
        self.assertTrue(result.hashes_match)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.answers["steel_grade"], "30")
```

**Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_verification.py::TestVerification::test_verify_sample_integrity_success -v`
Expected: FAIL with "ImportError: cannot import name 'verify_sample_integrity'"

**Step 7: Implement verify_sample_integrity**

```python
# core/security/verification.py (append)
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
        # For now, extract from student data verification
        # In production, this would be hardcoded public key
        public_key_path = private_key_path.parent / "public_key.pem"
        if not public_key_path.exists():
            return VerificationResult(
                is_valid=False,
                image_authentic=False,
                data_authentic=False,
                signature_valid=False,
                hashes_match=False,
                error_message="Публичный ключ не найден",
                answers=None
            )

        public_key = public_key_path.read_bytes()

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
```

**Step 8: Run test to verify it passes**

Run: `python -m pytest tests/test_verification.py -v`
Expected: PASS (2 tests)

**Step 9: Update security module __init__**

```python
# core/security/__init__.py (replace)
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

__all__ = [
    'compute_image_hash',
    'sign_data',
    'verify_signature',
    'encrypt_answers',
    'decrypt_answers',
    'VerificationResult',
    'verify_sample_integrity',
]
```

**Step 10: Commit**

```bash
git add core/security/verification.py core/security/__init__.py tests/test_verification.py
git commit -m "feat: add sample integrity verification module"
```

---

## Task 5: Add Data Contracts for Student/Teacher Data

**Files:**
- Modify: `core/contracts_v3.py`
- Create: `tests/test_contracts_security.py`

**Step 1: Write test for StudentDataV3 dataclass**

```python
# tests/test_contracts_security.py
import unittest
from core.contracts_v3 import StudentDataV3, TeacherAnswersV3

class TestSecurityContracts(unittest.TestCase):
    def test_student_data_creation(self):
        student_data = StudentDataV3(
            sample_id="test123",
            timestamp="2026-03-05T10:00:00",
            composition_wt={"Fe": 99.68, "C": 0.32},
            thermal_program={},
            prep_route={},
            etch_profile={},
            seed=42,
            resolution=(1024, 768),
            image_sha256="abc123def456"
        )
        self.assertEqual(student_data.sample_id, "test123")
        self.assertEqual(student_data.image_sha256, "abc123def456")

    def test_teacher_answers_creation(self):
        answers = TeacherAnswersV3(
            sample_id="test123",
            image_sha256="abc123def456",
            phase_fractions={"ferrite": 0.69, "pearlite": 0.31},
            inferred_system="Fe-C",
            steel_grade="30",
            carbon_content_calculated=0.32
        )
        self.assertEqual(answers.steel_grade, "30")
        self.assertEqual(answers.carbon_content_calculated, 0.32)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_contracts_security.py -v`
Expected: FAIL with "ImportError: cannot import name 'StudentDataV3'"

**Step 3: Add dataclasses to contracts_v3.py**

```python
# core/contracts_v3.py (append at end of file)

@dataclass
class StudentDataV3:
    """Data visible to students in lab package."""
    sample_id: str
    timestamp: str
    composition_wt: dict[str, float]
    thermal_program: dict[str, Any]
    prep_route: dict[str, Any]
    etch_profile: dict[str, Any]
    seed: int
    resolution: tuple[int, int]
    image_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "timestamp": self.timestamp,
            "composition_wt": self.composition_wt,
            "thermal_program": self.thermal_program,
            "prep_route": self.prep_route,
            "etch_profile": self.etch_profile,
            "seed": self.seed,
            "resolution": list(self.resolution),
            "image_sha256": self.image_sha256,
        }


@dataclass
class TeacherAnswersV3:
    """Protected answers for teachers only."""
    sample_id: str
    image_sha256: str
    phase_fractions: dict[str, float]
    inferred_system: str
    steel_grade: str | None
    carbon_content_calculated: float | None
    verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "image_sha256": self.image_sha256,
            "phase_fractions": self.phase_fractions,
            "inferred_system": self.inferred_system,
            "steel_grade": self.steel_grade,
            "carbon_content_calculated": self.carbon_content_calculated,
            "verification": self.verification or {},
        }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_contracts_security.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add core/contracts_v3.py tests/test_contracts_security.py
git commit -m "feat: add StudentDataV3 and TeacherAnswersV3 contracts"
```

---

## Task 6: Modify Export Function to Generate Secure Package

**Files:**
- Modify: `ui_qt/sample_factory_window_v3.py:3179-3259`
- Create: `tests/test_export_security.py`

**Step 1: Write test for secure export**

```python
# tests/test_export_security.py
import unittest
import tempfile
import json
from pathlib import Path
from PIL import Image

class TestSecureExport(unittest.TestCase):
    def test_export_creates_three_files(self):
        """Test that export creates image, student.json, and answers.enc"""
        # This is an integration test that will be implemented
        # after modifying the export function
        pass

    def test_student_json_contains_no_answers(self):
        """Test that student.json does not contain phase fractions or steel grade"""
        pass

    def test_answers_enc_is_encrypted(self):
        """Test that answers.enc cannot be read without private key"""
        pass
```

**Step 2: Read current export function**

Run: Read `ui_qt/sample_factory_window_v3.py` lines 3179-3259

**Step 3: Create helper methods for steel grade calculation**

```python
# ui_qt/sample_factory_window_v3.py (add before _export_lab_package method)

    def _calculate_carbon_content(self) -> float | None:
        """Calculate carbon content from phase fractions for Fe-C system."""
        if not self.current_output or not self.current_output.metadata:
            return None

        system = self.current_output.metadata.get("inferred_system")
        if system != "Fe-C":
            return None

        phase_report = self.current_output.metadata.get("phase_model_report", {})
        phase_fractions = phase_report.get("blended_phase_fractions", {})

        # For hypoeutectoid steels: C = pearlite_fraction * 0.8
        pearlite = phase_fractions.get("PEARLITE", 0.0)
        if pearlite > 0:
            return round(pearlite * 0.8, 4)

        return None

    def _calculate_steel_grade(self) -> str | None:
        """Calculate steel grade from carbon content."""
        carbon = self._calculate_carbon_content()
        if carbon is None:
            return None

        # Round to nearest standard grade
        grade_number = round(carbon * 100)

        # Standard grades: 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85
        standard_grades = [10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
        closest_grade = min(standard_grades, key=lambda x: abs(x - grade_number))

        return str(closest_grade)
```

**Step 4: Modify _export_lab_package to create secure files**

```python
# ui_qt/sample_factory_window_v3.py (replace _export_lab_package method)

    def _export_lab_package(self) -> None:
        """Export complete lab package with security features."""
        from datetime import datetime
        from core.security import (
            compute_image_hash,
            sign_data,
            encrypt_answers
        )

        if not self.current_output or not self.current_output.image:
            QMessageBox.warning(self, "Ошибка", "Нет изображения для экспорта")
            return

        # Get export directory
        package_dir = Path(
            QFileDialog.getExistingDirectory(
                self,
                "Выберите папку для экспорта пакета ЛР",
                str(Path.home() / "Documents")
            )
        )

        if not package_dir:
            return

        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"lab_sample_{timestamp}"
        export_dir = package_dir / prefix
        export_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save image
            image_path = export_dir / f"{prefix}.png"
            self.current_output.image.save(str(image_path))

            # 2. Compute image hash
            image_hash = compute_image_hash(image_path)

            # 3. Load public key (embedded in application)
            public_key_path = Path(__file__).parent.parent / "keys" / "public_key.pem"
            if not public_key_path.exists():
                QMessageBox.critical(
                    self,
                    "Ошибка",
                    "Публичный ключ не найден. Запустите scripts/generate_keys.py"
                )
                return

            public_key = public_key_path.read_bytes()

            # 4. Create student data
            from core.contracts_v3 import StudentDataV3

            student_data = StudentDataV3(
                sample_id=self.current_request.sample_id,
                timestamp=datetime.now().isoformat(),
                composition_wt=self.current_request.composition_wt.copy(),
                thermal_program=self.current_request.thermal_program.to_dict(),
                prep_route=self.current_request.prep_route.to_dict(),
                etch_profile=self.current_request.etch_profile.to_dict(),
                seed=self.current_request.seed,
                resolution=self.current_request.resolution,
                image_sha256=image_hash
            )

            student_dict = student_data.to_dict()

            # 5. Sign student data
            from core.security import sign_data
            # Load private key for signing (in production, use embedded key)
            private_key_path = Path(__file__).parent.parent / "keys" / "teacher_private_key.pem"
            if not private_key_path.exists():
                QMessageBox.critical(
                    self,
                    "Ошибка",
                    "Приватный ключ не найден для подписи"
                )
                return

            private_key = private_key_path.read_bytes()
            signature = sign_data(student_dict, private_key)
            student_dict["digital_signature"] = signature

            # 6. Save student.json
            student_json_path = export_dir / f"{prefix}_student.json"
            with open(student_json_path, 'w', encoding='utf-8') as f:
                json.dump(student_dict, f, indent=2, ensure_ascii=False)

            # 7. Create teacher answers
            from core.contracts_v3 import TeacherAnswersV3

            phase_report = self.current_output.metadata.get("phase_model_report", {})
            phase_fractions = phase_report.get("blended_phase_fractions", {})

            answers = TeacherAnswersV3(
                sample_id=self.current_request.sample_id,
                image_sha256=image_hash,
                phase_fractions=phase_fractions,
                inferred_system=self.current_output.metadata.get("inferred_system"),
                steel_grade=self._calculate_steel_grade(),
                carbon_content_calculated=self._calculate_carbon_content(),
                verification={
                    "seed": self.current_request.seed,
                    "timestamp": datetime.now().isoformat(),
                    "generator_version": "v3.0.0"
                }
            )

            # 8. Encrypt and save answers
            encrypted_answers = encrypt_answers(answers.to_dict(), public_key)
            answers_path = export_dir / f"{prefix}_answers.enc"
            answers_path.write_bytes(encrypted_answers)

            # 9. Save full metadata (for reference, not for students)
            meta_path = export_dir / f"{prefix}_metadata.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(
                    self.current_output.metadata_json_safe(),
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            # 10. Create manifest
            manifest = {
                "package_version": "3.0.0",
                "created_at": datetime.now().isoformat(),
                "files": {
                    "image": f"{prefix}.png",
                    "student_data": f"{prefix}_student.json",
                    "teacher_answers": f"{prefix}_answers.enc",
                    "metadata": f"{prefix}_metadata.json"
                },
                "security": {
                    "encryption": "RSA-2048 + AES-256-GCM",
                    "signature": "RSA-PSS with SHA256",
                    "image_hash_algorithm": "SHA256"
                }
            }

            manifest_path = export_dir / "manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            QMessageBox.information(
                self,
                "Успех",
                f"Пакет ЛР экспортирован:\n{export_dir}\n\n"
                f"Файлы:\n"
                f"• {prefix}.png - изображение\n"
                f"• {prefix}_student.json - данные для студента\n"
                f"• {prefix}_answers.enc - ответы (зашифрованы)\n"
                f"• {prefix}_metadata.json - полные метаданные\n"
                f"• manifest.json - описание пакета"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка экспорта",
                f"Не удалось экспортировать пакет:\n{str(e)}"
            )
```

**Step 5: Test export manually**

Run: `python run_generator_app_v3.py`
- Generate a sample
- Click "Экспортировать пакет ЛР"
- Verify 5 files are created
- Open student.json and verify no phase_fractions or steel_grade
- Verify answers.enc is binary and unreadable

**Step 6: Commit**

```bash
git add ui_qt/sample_factory_window_v3.py
git commit -m "feat: implement secure lab package export with encryption"
```

---

## Task 7: Embed Public Key in Application

**Files:**
- Create: `keys/public_key_embedded.py`
- Modify: `core/security/crypto_manager.py`

**Step 1: Create embedded public key module**

```python
# keys/public_key_embedded.py
"""Embedded public key for student application distribution.

This file contains the public key used to verify digital signatures
in student.json files. The corresponding private key is kept secure
by teachers.

Generated: 2026-03-05
"""

PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
[PASTE ACTUAL PUBLIC KEY HERE AFTER GENERATION]
-----END PUBLIC KEY-----
"""

def get_public_key() -> bytes:
    """Get embedded public key for signature verification."""
    return PUBLIC_KEY_PEM
```

**Step 2: Add helper function to crypto_manager**

```python
# core/security/crypto_manager.py (append)

def get_embedded_public_key() -> bytes:
    """Get embedded public key from application.

    Returns:
        PEM-encoded public key bytes
    """
    try:
        from keys.public_key_embedded import get_public_key
        return get_public_key()
    except ImportError:
        # Fallback: load from file (development mode)
        from pathlib import Path
        key_path = Path(__file__).parent.parent / "keys" / "public_key.pem"
        if key_path.exists():
            return key_path.read_bytes()
        raise FileNotFoundError(
            "Public key not found. Run scripts/generate_keys.py first."
        )
```

**Step 3: Update verification to use embedded key**

```python
# core/security/verification.py (modify verify_sample_integrity)
# Replace the public key loading section with:

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
```

**Step 4: Update export to use embedded key**

```python
# ui_qt/sample_factory_window_v3.py (modify _export_lab_package)
# Replace public key loading with:

            # 3. Load public key (embedded in application)
            from core.security.crypto_manager import get_embedded_public_key
            try:
                public_key = get_embedded_public_key()
            except FileNotFoundError as e:
                QMessageBox.critical(
                    self,
                    "Ошибка",
                    f"Публичный ключ не найден:\n{str(e)}"
                )
                return
```

**Step 5: Generate keys and embed**

Run: `python scripts/generate_keys.py`
Then: Copy content of `keys/public_key.pem` into `keys/public_key_embedded.py`

**Step 6: Commit**

```bash
git add keys/public_key_embedded.py core/security/crypto_manager.py core/security/verification.py ui_qt/sample_factory_window_v3.py
git commit -m "feat: embed public key in application for distribution"
```

---

## Task 8: Add Teacher/Student Mode UI to Microscope

**Files:**
- Modify: `ui_qt/microscope_window.py`
- Create: `profiles/teacher_config.json` (template)
- Create: `tests/test_microscope_modes.py`

**Step 1: Write test for mode switching**

```python
# tests/test_microscope_modes.py
import unittest
from PySide6.QtWidgets import QApplication
import sys

class TestMicroscopeModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)

    def test_default_mode_is_student(self):
        """Test that microscope starts in student mode by default"""
        pass

    def test_switch_to_teacher_mode_requires_key(self):
        """Test that switching to teacher mode prompts for private key"""
        pass
```

**Step 2: Add mode state to microscope window**

```python
# ui_qt/microscope_window.py (add to __init__ method)

        # Mode state
        self.current_mode = "student"  # "student" or "teacher"
        self.teacher_private_key_path: Path | None = None
        self.teacher_answers: dict | None = None
```

**Step 3: Create mode switcher widget**

```python
# ui_qt/microscope_window.py (add new method)

    def _create_mode_switcher(self) -> QWidget:
        """Create mode switcher widget for toolbar."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Mode label
        self.mode_label = QLabel("👨‍🎓 СТУДЕНТ")
        self.mode_label.setStyleSheet("""
            QLabel {
                background-color: #4CAF50;
                color: white;
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)

        # Switch button
        self.mode_switch_btn = QPushButton("Режим преподавателя")
        self.mode_switch_btn.clicked.connect(self._switch_mode)

        layout.addWidget(self.mode_label)
        layout.addWidget(self.mode_switch_btn)

        return widget

    def _switch_mode(self) -> None:
        """Switch between student and teacher modes."""
        if self.current_mode == "student":
            # Switch to teacher mode
            self._activate_teacher_mode()
        else:
            # Switch back to student mode
            self._activate_student_mode()

    def _activate_teacher_mode(self) -> None:
        """Activate teacher mode with private key."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        # Check if key path is already configured
        config_path = Path("profiles/teacher_config.json")
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                saved_key_path = config.get("private_key_path")
                if saved_key_path and Path(saved_key_path).exists():
                    self.teacher_private_key_path = Path(saved_key_path)

        # If not configured, ask for key
        if not self.teacher_private_key_path:
            key_path, _ = QFileDialog.getOpenFileName(
                self,
                "Выберите приватный ключ преподавателя",
                str(Path.home()),
                "PEM Files (*.pem);;All Files (*)"
            )

            if not key_path:
                return

            self.teacher_private_key_path = Path(key_path)

            # Save to config
            config_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(config_path, 'w') as f:
                json.dump({"private_key_path": str(self.teacher_private_key_path)}, f)

        # Verify key works
        try:
            key_data = self.teacher_private_key_path.read_bytes()
            from cryptography.hazmat.primitives import serialization
            serialization.load_pem_private_key(key_data, password=None)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось загрузить приватный ключ:\n{str(e)}"
            )
            return

        # Switch mode
        self.current_mode = "teacher"
        self.mode_label.setText("👨‍🏫 ПРЕПОДАВАТЕЛЬ")
        self.mode_label.setStyleSheet("""
            QLabel {
                background-color: #2196F3;
                color: white;
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self.mode_switch_btn.setText("Режим студента")

        # Show teacher panel if sample is loaded
        if hasattr(self, 'teacher_panel'):
            self.teacher_panel.setVisible(True)

        # Load answers if sample is loaded
        if self.current_source_metadata:
            self._load_teacher_answers()

    def _activate_student_mode(self) -> None:
        """Activate student mode."""
        self.current_mode = "student"
        self.mode_label.setText("👨‍🎓 СТУДЕНТ")
        self.mode_label.setStyleSheet("""
            QLabel {
                background-color: #4CAF50;
                color: white;
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self.mode_switch_btn.setText("Режим преподавателя")
        self.teacher_answers = None

        # Hide teacher panel
        if hasattr(self, 'teacher_panel'):
            self.teacher_panel.setVisible(False)
```

**Step 4: Add mode switcher to toolbar**

```python
# ui_qt/microscope_window.py (modify _create_toolbar method)
# Add after existing toolbar items:

        toolbar.addSeparator()
        mode_switcher = self._create_mode_switcher()
        toolbar.addWidget(mode_switcher)
```

**Step 5: Test mode switching manually**

Run: `python run_app_v2.py`
- Verify "👨‍🎓 СТУДЕНТ" label appears
- Click "Режим преподавателя"
- Verify file dialog appears
- Select teacher_private_key.pem
- Verify label changes to "👨‍🏫 ПРЕПОДАВАТЕЛЬ"

**Step 6: Commit**

```bash
git add ui_qt/microscope_window.py
git commit -m "feat: add teacher/student mode switcher to microscope"
```

---

