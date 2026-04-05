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
