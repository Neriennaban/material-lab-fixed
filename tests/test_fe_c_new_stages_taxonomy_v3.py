"""Tests for A0.1 — Fe-C stage taxonomy extension.

The plan introduces explicit stages for white cast iron
(``white_cast_iron_hypoeutectic`` / ``_eutectic`` / ``_hypereutectic``) and
for upper/lower bainite. These are *opt-in*: the auto resolver still
returns the legacy ``ledeburite`` / ``bainite`` stages for backward
compatibility, but the new identifiers must round-trip through aliases,
the stage order, and the phase template lookups in ``fe_c_unified``.
"""

from __future__ import annotations

import unittest

from core.generator_phase_map import (
    SYSTEM_STAGE_ORDER,
    normalize_stage,
    resolve_fe_c_stage,
    supported_stages,
)
from core.metallography_v3.system_generators.fe_c_unified import (
    _SPECIALIZED_BAINITIC_STAGES,
    _SPECIALIZED_CAST_IRON_STAGES,
    _STAGE_DEFAULT_FRACTIONS,
    _TRANSITION_STAGES,
)


NEW_CAST_IRON_STAGES = (
    "white_cast_iron_hypoeutectic",
    "white_cast_iron_eutectic",
    "white_cast_iron_hypereutectic",
)
NEW_BAINITIC_STAGES = ("bainite_upper", "bainite_lower")
NEW_STAGES = NEW_CAST_IRON_STAGES + NEW_BAINITIC_STAGES


class NewStagesRegisteredTest(unittest.TestCase):
    def test_all_new_stages_appear_in_system_order(self) -> None:
        order = SYSTEM_STAGE_ORDER["fe-c"]
        for stage in NEW_STAGES:
            self.assertIn(stage, order, f"{stage} missing from SYSTEM_STAGE_ORDER['fe-c']")

    def test_supported_stages_surface_new_taxonomy(self) -> None:
        stages = supported_stages("fe-c")
        for stage in NEW_STAGES:
            self.assertIn(stage, stages)

    def test_phase_templates_define_new_stages(self) -> None:
        for stage in NEW_STAGES:
            self.assertIn(
                stage,
                _STAGE_DEFAULT_FRACTIONS,
                f"{stage} missing from _STAGE_DEFAULT_FRACTIONS",
            )
            fractions = _STAGE_DEFAULT_FRACTIONS[stage]
            self.assertGreater(len(fractions), 0)
            total = sum(float(v) for v in fractions.values())
            self.assertAlmostEqual(total, 1.0, places=5, msg=f"{stage} fractions must sum to 1")

    def test_new_stages_listed_in_transition_set(self) -> None:
        for stage in NEW_STAGES:
            self.assertIn(stage, _TRANSITION_STAGES, f"{stage} should be a transition stage")

    def test_specialised_stage_sets_are_exact(self) -> None:
        self.assertEqual(_SPECIALIZED_CAST_IRON_STAGES, set(NEW_CAST_IRON_STAGES))
        self.assertEqual(_SPECIALIZED_BAINITIC_STAGES, set(NEW_BAINITIC_STAGES))


class StageAliasRoundTripTest(unittest.TestCase):
    def test_aliases_normalize_to_canonical_identifiers(self) -> None:
        cases = {
            "upper_bainite": "bainite_upper",
            "lower_bainite": "bainite_lower",
            "Upper Bainite": "bainite_upper",
            "верхний_бейнит": "bainite_upper",
            "нижний_бейнит": "bainite_lower",
            "hypereutectic white cast iron": "white_cast_iron_hypereutectic",
            "eutectic_white_cast_iron": "white_cast_iron_eutectic",
            "hypoeutectic_white_cast_iron": "white_cast_iron_hypoeutectic",
            "белый_чугун_заэвт": "white_cast_iron_hypereutectic",
        }
        for raw, expected in cases.items():
            self.assertEqual(
                normalize_stage(raw),
                expected,
                f"alias {raw!r} should normalize to {expected!r}",
            )


class OptInResolverTest(unittest.TestCase):
    """The auto-resolver must keep legacy stages (backward compatibility).

    New white_cast_iron_* / bainite_upper / bainite_lower stages are only
    returned when explicitly requested by the caller.
    """

    def test_high_carbon_auto_returns_legacy_ledeburite(self) -> None:
        for c_wt in (2.5, 3.2, 4.3, 5.5):
            stage = resolve_fe_c_stage(
                c_wt=c_wt,
                temperature_c=20.0,
                cooling_mode="furnace_slow",
                requested_stage="auto",
            )
            self.assertEqual(stage, "ledeburite", f"c_wt={c_wt} should auto-resolve to ledeburite")

    def test_explicit_white_cast_iron_stages_pass_through(self) -> None:
        for stage in NEW_CAST_IRON_STAGES:
            resolved = resolve_fe_c_stage(
                c_wt=3.5,
                temperature_c=20.0,
                cooling_mode="furnace_slow",
                requested_stage=stage,
            )
            self.assertEqual(resolved, stage)

    def test_generic_bainite_mode_still_returns_legacy_stage(self) -> None:
        stage = resolve_fe_c_stage(
            c_wt=0.45,
            temperature_c=400.0,
            cooling_mode="bainitic",
            requested_stage="auto",
        )
        self.assertEqual(stage, "bainite")

    def test_upper_lower_keywords_in_cooling_mode_select_split(self) -> None:
        stage_upper = resolve_fe_c_stage(
            c_wt=0.45,
            temperature_c=450.0,
            cooling_mode="bainite_upper",
            requested_stage="auto",
        )
        self.assertEqual(stage_upper, "bainite_upper")

        stage_lower = resolve_fe_c_stage(
            c_wt=0.45,
            temperature_c=260.0,
            cooling_mode="bainite_lower",
            requested_stage="auto",
        )
        self.assertEqual(stage_lower, "bainite_lower")

    def test_explicit_bainite_split_via_requested_stage(self) -> None:
        self.assertEqual(
            resolve_fe_c_stage(
                c_wt=0.45,
                temperature_c=400.0,
                cooling_mode="furnace_slow",
                requested_stage="bainite_upper",
            ),
            "bainite_upper",
        )
        self.assertEqual(
            resolve_fe_c_stage(
                c_wt=0.45,
                temperature_c=260.0,
                cooling_mode="furnace_slow",
                requested_stage="bainite_lower",
            ),
            "bainite_lower",
        )


if __name__ == "__main__":
    unittest.main()
