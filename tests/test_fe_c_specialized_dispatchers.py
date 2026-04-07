"""Tests for the white_cast_iron and bainite_upper/lower stage
dispatchers in ``render_fe_c_unified``.

The plan adds specialised render functions
``_build_white_cast_iron_render`` (A1+A2+A3) and
``_build_bainitic_render_split`` (A6) and wires them through the
``requested_stage`` override that the new presets use to opt in.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3
from core.metallography_v3.system_generators.fe_c_unified import (
    _build_bainitic_render_split,
    _build_white_cast_iron_render,
    _SPECIALIZED_BAINITIC_STAGES,
    _SPECIALIZED_CAST_IRON_STAGES,
)
from core.metallography_v3.system_generators.base import SystemGenerationContext
from core.contracts_v2 import ProcessingState

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = REPO_ROOT / "presets_v3"
PROFILES_DIR = REPO_ROOT / "profiles_v3"


def _make_context(
    *,
    stage: str,
    c_wt: float,
    size: tuple[int, int] = (128, 128),
    seed: int = 42,
) -> SystemGenerationContext:
    return SystemGenerationContext(
        size=size,
        seed=seed,
        inferred_system="fe-c",
        stage=stage,
        phase_fractions={"LEDEBURITE": 1.0},
        composition_wt={"Fe": 100.0 - c_wt, "C": c_wt},
        processing=ProcessingState(),
        thermal_summary={"max_effective_cooling_rate_c_per_s": 5.0},
    )


class WhiteCastIronDispatcherTest(unittest.TestCase):
    def test_eutectic_uses_pure_leopard(self) -> None:
        ctx = _make_context(stage="white_cast_iron_eutectic", c_wt=4.3)
        image, masks, layers, _, trace = _build_white_cast_iron_render(
            context=ctx,
            stage="white_cast_iron_eutectic",
            phase_fractions={"LEDEBURITE": 1.0},
            seed_split={"seed_topology": 42},
        )
        self.assertEqual(image.shape, (128, 128))
        self.assertEqual(layers, ["LEDEBURITE"])
        self.assertIn("LEDEBURITE", masks)
        self.assertEqual(trace["family"], "white_cast_iron_eutectic")

    def test_hypoeutectic_adds_pearlite_dendrites(self) -> None:
        ctx = _make_context(stage="white_cast_iron_hypoeutectic", c_wt=3.0)
        image, masks, layers, _, trace = _build_white_cast_iron_render(
            context=ctx,
            stage="white_cast_iron_hypoeutectic",
            phase_fractions={"LEDEBURITE": 1.0},
            seed_split={"seed_topology": 42},
        )
        self.assertIn("LEDEBURITE", masks)
        self.assertIn("PEARLITE", masks)
        self.assertEqual(trace["family"], "white_cast_iron_hypoeutectic")
        # The pearlite mask must cover *some* pixels — dendrites
        # always paint on the leopard background.
        self.assertGreater(int(masks["PEARLITE"].sum()), 0)

    def test_hypereutectic_adds_primary_cementite(self) -> None:
        ctx = _make_context(stage="white_cast_iron_hypereutectic", c_wt=5.5)
        image, masks, layers, _, trace = _build_white_cast_iron_render(
            context=ctx,
            stage="white_cast_iron_hypereutectic",
            phase_fractions={"LEDEBURITE": 1.0},
            seed_split={"seed_topology": 42},
        )
        self.assertIn("LEDEBURITE", masks)
        self.assertIn("CEMENTITE_PRIMARY", masks)
        self.assertEqual(trace["family"], "white_cast_iron_hypereutectic")
        # Primary cementite needles must be brighter than the
        # leopard ledeburite background — verifies the renderer
        # actually paints on top.
        bright = int((image > 230).sum())
        self.assertGreater(bright, 200)

    def test_constants_match_dispatcher_stages(self) -> None:
        self.assertEqual(
            _SPECIALIZED_CAST_IRON_STAGES,
            {
                "white_cast_iron_hypoeutectic",
                "white_cast_iron_eutectic",
                "white_cast_iron_hypereutectic",
            },
        )


class BainiteSplitDispatcherTest(unittest.TestCase):
    def test_upper_bainite_uses_feathery_renderer(self) -> None:
        ctx = _make_context(stage="bainite_upper", c_wt=0.45)
        image, masks, layers, _, trace = _build_bainitic_render_split(
            context=ctx,
            stage="bainite_upper",
            phase_fractions={"BAINITE": 1.0},
            seed_split={"seed_topology": 42},
        )
        self.assertEqual(image.shape, (128, 128))
        self.assertIn("BAINITE", masks)
        self.assertEqual(trace["family"], "upper_bainite_feathery")

    def test_lower_bainite_uses_lath_renderer(self) -> None:
        ctx = _make_context(stage="bainite_lower", c_wt=0.45)
        image, masks, layers, _, trace = _build_bainitic_render_split(
            context=ctx,
            stage="bainite_lower",
            phase_fractions={"BAINITE": 1.0},
            seed_split={"seed_topology": 42},
        )
        self.assertEqual(image.shape, (128, 128))
        self.assertEqual(trace["family"], "lower_bainite_lath")

    def test_upper_and_lower_produce_different_output(self) -> None:
        ctx = _make_context(stage="bainite_upper", c_wt=0.45)
        upper, *_ = _build_bainitic_render_split(
            context=ctx,
            stage="bainite_upper",
            phase_fractions={"BAINITE": 1.0},
            seed_split={"seed_topology": 42},
        )
        ctx2 = _make_context(stage="bainite_lower", c_wt=0.45)
        lower, *_ = _build_bainitic_render_split(
            context=ctx2,
            stage="bainite_lower",
            phase_fractions={"BAINITE": 1.0},
            seed_split={"seed_topology": 42},
        )
        self.assertFalse(np.array_equal(upper, lower))

    def test_constants_match(self) -> None:
        self.assertEqual(
            _SPECIALIZED_BAINITIC_STAGES, {"bainite_upper", "bainite_lower"}
        )


class PresetIntegrationTest(unittest.TestCase):
    """End-to-end pipeline check: presets that opt in via
    ``phase_model.requested_stage`` must hit the new dispatchers."""

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
