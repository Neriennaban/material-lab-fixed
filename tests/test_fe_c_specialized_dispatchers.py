"""Preset integration tests для специализированных семейств.

После Phase 9 cleanup старые `_build_white_cast_iron_render` и
`_build_bainitic_render_split` удалены — их прямые unit-тесты
переехали в семейственные тесты:
  - `tests/renderers/test_white_cast_iron_family.py`
  - `tests/renderers/test_bainite_family.py`

Здесь остаются только end-to-end проверки маршрутизации через
`MetallographyPipelineV3` — что пресет, указывающий на соответствующую
стадию, действительно попадает в нужный renderer и `morphology_trace.
family` совпадает с ожидаемым.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = REPO_ROOT / "presets_v3"
PROFILES_DIR = REPO_ROOT / "profiles_v3"


class PresetIntegrationTest(unittest.TestCase):
    """End-to-end pipeline check: presets routing to the right family
    trace via модульные renderer'ы Phase 2-8."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = MetallographyPipelineV3(
            presets_dir=PRESETS_DIR,
            profiles_dir=PROFILES_DIR,
        )

    def _render(self, name: str):
        payload = self.pipeline.load_preset(name)
        payload["resolution"] = [128, 128]
        return self.pipeline.generate(self.pipeline.request_from_preset(payload))

    def _family(self, output) -> str:
        return str(
            output.metadata.get("fe_c_phase_render", {})
            .get("morphology_trace", {})
            .get("family", "")
        )

    def test_hypoeutectic_preset_routes_to_dispatcher(self) -> None:
        out = self._render("cast_iron_white_hypoeutectic_v3")
        self.assertEqual(self._family(out), "white_cast_iron_hypoeutectic")
        masks = {n.upper() for n in out.phase_masks.keys()}
        self.assertIn("LEDEBURITE", masks)
        self.assertIn("PEARLITE", masks)

    def test_eutectic_preset_routes_to_dispatcher(self) -> None:
        out = self._render("cast_iron_white_eutectic_v3")
        self.assertEqual(self._family(out), "white_cast_iron_eutectic")

    def test_hypereutectic_preset_routes_to_dispatcher(self) -> None:
        out = self._render("cast_iron_white_hypereutectic_v3")
        self.assertEqual(self._family(out), "white_cast_iron_hypereutectic")
        masks = {n.upper() for n in out.phase_masks.keys()}
        self.assertIn("CEMENTITE_PRIMARY", masks)


if __name__ == "__main__":
    unittest.main()
