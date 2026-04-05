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

    print(f"Keys generated:")
    print(f"   Private: {private_path}")
    print(f"   Public: {public_path}")

if __name__ == "__main__":
    private_key, public_key = generate_keypair()
    save_keys(private_key, public_key, Path("keys"))
