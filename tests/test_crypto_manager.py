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
