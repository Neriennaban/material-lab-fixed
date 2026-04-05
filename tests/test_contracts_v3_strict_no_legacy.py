from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3


class ContractsV3StrictNoLegacyTests(unittest.TestCase):
    def test_from_dict_rejects_thermo(self) -> None:
        payload = {
            "sample_id": "bad_thermo",
            "composition_wt": {"Fe": 99.5, "C": 0.5},
            "thermo": {"backend": "calphad_py"},
        }
        with self.assertRaises(ValueError) as ctx:
            MetallographyRequestV3.from_dict(payload)
        self.assertIn("LEGACY_FIELD_REMOVED", str(ctx.exception))

    def test_from_dict_rejects_process_route(self) -> None:
        payload = {
            "sample_id": "bad_route",
            "composition_wt": {"Fe": 99.5, "C": 0.5},
            "process_route": {"operations": []},
        }
        with self.assertRaises(ValueError) as ctx:
            MetallographyRequestV3.from_dict(payload)
        self.assertIn("LEGACY_FIELD_REMOVED", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

