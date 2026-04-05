"""Embedded public key for student application distribution.

This file contains the public key used to verify digital signatures
in student.json files. The corresponding private key is kept secure
by teachers.

Generated: 2026-03-05
"""

PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA03aNx54ktYoLtqC3xzCR
zVVBlj3abP2vGXPG/GgcTA65sCQpNgjDXtHUPHtMwKMXEer0Oir34jkUB+XgKjps
3LeH47a2XoBRUwkIpdGESpHoO13Lgj0a2d0OsTiKoIgxY/0jyF6iZo/9tnqwAyXF
zl84BranX13GgSoRpY/bTRmWUf3XnJpRhbi/mvQ6QnAUaY2PPbFDav1MldUwQuqy
odpmUa06V6DyaET5EBshgvUU26DTq4e0KGMGGnmGJXjZJQLA0F+H9yRYWnyCJtmd
tRjQ0HepFlKVADL54GiTdgCo7mM4WIAXcb79oHhZgdRfCS05TV2TbRX+eR58e1Mm
ewIDAQAB
-----END PUBLIC KEY-----
"""

def get_public_key() -> bytes:
    """Get embedded public key for signature verification."""
    return PUBLIC_KEY_PEM
