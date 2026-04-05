from __future__ import annotations

import unittest
from pathlib import Path

from core.calphad.db_manager import (
    CalphadDBReference,
    resolve_database_reference,
    validate_database_reference,
)
from core.contracts_v2 import ThermoBackendConfig


class CalphadDBManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = Path("profiles") / "calphad_profile_v2.json"
        self.assertTrue(self.profile.exists())
        self.thermo = ThermoBackendConfig(db_profile_path=str(self.profile))

    def test_resolve_profile_database(self) -> None:
        ref = resolve_database_reference(system="fe-c", thermo=self.thermo, profile_path=self.profile)
        self.assertIsInstance(ref, CalphadDBReference)
        self.assertTrue(ref.path.exists())
        self.assertEqual(ref.system, "fe-c")
        self.assertEqual(len(ref.sha256), 64)

    def test_validate_database_reference(self) -> None:
        ref = resolve_database_reference(system="al-si", thermo=self.thermo, profile_path=self.profile)
        validate_database_reference(ref)

    def test_override_missing_db_raises(self) -> None:
        bad = ThermoBackendConfig(
            db_profile_path=str(self.profile),
            db_overrides={"fe-c": str(Path("missing") / "fe_c.tdb")},
        )
        with self.assertRaises(ValueError) as ctx:
            resolve_database_reference(system="fe-c", thermo=bad, profile_path=self.profile)
        self.assertIn("DB_MISSING", str(ctx.exception))

    def test_unsupported_system_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_database_reference(system="ni-cr", thermo=self.thermo, profile_path=self.profile)
        self.assertIn("SYSTEM_UNSUPPORTED", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
