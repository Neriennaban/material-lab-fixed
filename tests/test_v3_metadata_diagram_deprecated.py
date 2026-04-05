from __future__ import annotations

import unittest

from core.contracts_v3 import MetallographyRequestV3, ThermalPointV3
from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


class V3MetadataDiagramDeprecatedTests(unittest.TestCase):
    def test_diagram_metadata_is_deprecated_stub(self) -> None:
        pipeline = MetallographyPipelineV3()
        req = MetallographyRequestV3(
            sample_id="meta_diag_deprecated",
            composition_wt={"Fe": 99.1, "C": 0.9},
            system_hint="fe-c",
            resolution=(96, 96),
            seed=933,
        )
        req.thermal_program.points = [
            ThermalPointV3(time_s=0.0, temperature_c=20.0),
            ThermalPointV3(time_s=180.0, temperature_c=820.0),
            ThermalPointV3(time_s=320.0, temperature_c=820.0),
            ThermalPointV3(time_s=620.0, temperature_c=30.0),
        ]
        out = pipeline.generate(req)

        style = dict(out.metadata.get("diagram_style", {}))
        report = dict(out.metadata.get("diagram_style_report", {}))

        self.assertTrue(bool(style.get("deprecated", False)))
        self.assertTrue(bool(style.get("removed", False)))
        self.assertIsNone(style.get("value"))

        self.assertTrue(bool(report.get("deprecated", False)))
        self.assertTrue(bool(report.get("removed", False)))
        self.assertIsNone(report.get("value"))


if __name__ == "__main__":
    unittest.main()

