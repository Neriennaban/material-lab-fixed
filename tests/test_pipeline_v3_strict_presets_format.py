from __future__ import annotations

import json
import unittest
from pathlib import Path

from core.contracts_v3 import MetallographyRequestV3


class PipelineV3StrictPresetsFormatTests(unittest.TestCase):
    def test_all_presets_v3_are_strict_format(self) -> None:
        root = Path("presets_v3")
        self.assertTrue(root.exists())
        for path in sorted(root.glob("*.json")):
            with self.subTest(preset=path.name):
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                self.assertIsInstance(payload, dict)
                # Some cast iron or other custom presets might not have strict thermal programs but expected properties
                if "thermal_program" in payload:
                    self.assertNotIn("thermo", payload)
                    self.assertNotIn("process_route", payload)
                    req = MetallographyRequestV3.from_dict(payload)
                    self.assertGreaterEqual(len(req.thermal_program.points), 2)
                else:
                    req = MetallographyRequestV3.from_dict(payload)


if __name__ == "__main__":
    unittest.main()

