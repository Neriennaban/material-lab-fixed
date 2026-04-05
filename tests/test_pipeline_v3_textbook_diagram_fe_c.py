from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class PipelineV3TextbookDiagramFeCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = MetallographyPipelineV3()

    def test_fe_c_sets_deprecated_diagram_style_stub(self) -> None:
        req = MetallographyRequestV3(
            sample_id="diag_fe_c",
            composition_wt={"Fe": 99.2, "C": 0.8},
            system_hint="fe-c",
            resolution=(96, 96),
            seed=710,
        )
        req.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=220.0, temperature_c=780.0),
            ThermalPointV3(time_s=420.0, temperature_c=780.0),
            ThermalPointV3(time_s=740.0, temperature_c=30.0),
        ]
        out = self.pipeline.generate(req)
        style = dict(out.metadata.get("diagram_style", {}))
        self.assertTrue(bool(style.get("deprecated", False)))
        self.assertTrue(bool(style.get("removed", False)))
        self.assertIsNone(style.get("value"))
        report = dict(out.metadata.get("diagram_style_report", {}))
        self.assertTrue(bool(report.get("deprecated", False)))
        self.assertTrue(bool(report.get("removed", False)))
        self.assertIsNone(report.get("value"))

    def test_non_fe_c_sets_deprecated_diagram_style_stub(self) -> None:
        req = MetallographyRequestV3(
            sample_id="diag_al_si",
            composition_wt={"Al": 87.4, "Si": 12.6},
            system_hint="al-si",
            resolution=(96, 96),
            seed=711,
        )
        req.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=220.0, temperature_c=730.0),
            ThermalPointV3(time_s=280.0, temperature_c=730.0),
            ThermalPointV3(time_s=760.0, temperature_c=30.0),
        ]
        out = self.pipeline.generate(req)
        style = dict(out.metadata.get("diagram_style", {}))
        self.assertTrue(bool(style.get("deprecated", False)))
        self.assertTrue(bool(style.get("removed", False)))
        self.assertIsNone(style.get("value"))


if __name__ == "__main__":
    unittest.main()
