from __future__ import annotations

import unittest

from core.contracts_v2 import ProcessingState
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.metallography_v3.system_generators.fe_c_unified import render_fe_c_unified


class FeCAliasNormalizationTests(unittest.TestCase):
    def test_aliases_are_normalized_to_canonical_names(self) -> None:
        ctx = SystemGenerationContext(
            size=(96, 96),
            seed=42,
            inferred_system="fe-c",
            stage="martensite_tetragonal",
            phase_fractions={"MARTENSITE_T": 0.82, "Fe3C": 0.18},
            composition_wt={"Fe": 99.0, "C": 1.0},
            processing=ProcessingState(temperature_c=120.0, cooling_mode="quenched"),
        )
        out = render_fe_c_unified(ctx)
        render_meta = dict(out.metadata.get("fe_c_phase_render", {}))
        norm = dict(render_meta.get("normalized_phase_fractions", {}))
        self.assertIn("MARTENSITE_TETRAGONAL", norm)
        self.assertIn("CEMENTITE", norm)
        self.assertNotIn("MARTENSITE_T", norm)
        self.assertNotIn("FE3C", norm)


if __name__ == "__main__":
    unittest.main()
