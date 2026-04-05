from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.diagram_engine import render_diagram_snapshot


class DiagramEngineTextbookFeCV3Tests(unittest.TestCase):
    def test_textbook_style_applies_for_fe_c(self) -> None:
        snapshot = render_diagram_snapshot(
            composition={"Fe": 99.2, "C": 0.8},
            processing=ProcessingState(temperature_c=760.0),
            requested_system="fe-c",
            style_profile="textbook_fe_c",
            size=(820, 460),
        )
        self.assertEqual(snapshot["image"].size, (820, 460))
        style = dict(snapshot.get("diagram_style", {}))
        self.assertEqual(style.get("profile_id"), "textbook_fe_c")
        self.assertTrue(bool(style.get("applied", False)))
        report = dict(snapshot.get("diagram_style_report", {}))
        self.assertTrue(bool(report.get("has_reference_isotherms", False)))
        self.assertTrue(bool(report.get("has_reference_verticals", False)))


if __name__ == "__main__":
    unittest.main()

