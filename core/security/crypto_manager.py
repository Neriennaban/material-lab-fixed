import hashlib
import json
import base64
import os
from pathlib import Path
from typing import Any
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

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
