"""Phase 1 guard: _STAGE_TO_RENDERER покрывает все ожидаемые стадии
редизайна и каждый модуль-семейство регистрирует ровно свои стадии.
"""
from __future__ import annotations

import unittest


class DispatchTableTests(unittest.TestCase):
    def test_dispatch_table_covers_target_stages(self) -> None:
        from core.metallography_v3.system_generators.fe_c_unified import (
            _STAGE_TO_RENDERER,
        )

        expected = {
            # martensite family
            "martensite",
            "martensite_tetragonal",
            "martensite_cubic",
            # bainite family
            "bainite_upper",
            "bainite_lower",
            "carbide_free_bainite",
            # tempered family
            "tempered_low",
            "tempered_medium",
            "tempered_high",
            "troostite_temper",
            "sorbite_temper",
            # quench products
            "troostite_quench",
            "sorbite_quench",
            # white cast iron
            "ledeburite",
            "white_cast_iron_hypoeutectic",
            "white_cast_iron_eutectic",
            "white_cast_iron_hypereutectic",
            # high-temperature phases
            "austenite",
            "delta_ferrite",
            "alpha_gamma",
            "gamma_cementite",
            "liquid",
            "liquid_gamma",
            # widmanstatten
            "widmanstatten_ferrite",
            # surface layers
            "decarburized_layer",
            "carburized_layer",
            # granular pearlite
            "granular_pearlite",
        }
        missing = expected - set(_STAGE_TO_RENDERER.keys())
        self.assertFalse(
            missing,
            f"_STAGE_TO_RENDERER missing expected stages: {sorted(missing)}",
        )

    def test_each_module_exports_renderer_contract(self) -> None:
        from core.metallography_v3.renderers import (
            bainite,
            granular_pearlite,
            high_temp_phases,
            martensite,
            quench_products,
            surface_layers,
            tempered,
            white_cast_iron,
            widmanstatten,
        )

        modules = (
            bainite,
            granular_pearlite,
            high_temp_phases,
            martensite,
            quench_products,
            surface_layers,
            tempered,
            white_cast_iron,
            widmanstatten,
        )
        for mod in modules:
            self.assertTrue(
                hasattr(mod, "HANDLES_STAGES"),
                f"{mod.__name__} missing HANDLES_STAGES",
            )
            self.assertIsInstance(mod.HANDLES_STAGES, frozenset)
            self.assertGreater(
                len(mod.HANDLES_STAGES),
                0,
                f"{mod.__name__} HANDLES_STAGES is empty",
            )
            self.assertTrue(
                hasattr(mod, "render"),
                f"{mod.__name__} missing render()",
            )
            self.assertTrue(
                callable(mod.render),
                f"{mod.__name__}.render is not callable",
            )

    def test_stage_mapping_is_unique(self) -> None:
        """Каждая стадия должна принадлежать ровно одному модулю."""
        from core.metallography_v3.system_generators.fe_c_unified import (
            _RENDERER_MODULES,
        )

        all_stages: list[str] = []
        for mod in _RENDERER_MODULES:
            all_stages.extend(mod.HANDLES_STAGES)
        self.assertEqual(
            len(all_stages),
            len(set(all_stages)),
            "Stage appears in more than one renderer module: "
            f"{sorted(s for s in set(all_stages) if all_stages.count(s) > 1)}",
        )

    def test_all_renderers_implemented(self) -> None:
        """Phase 8 завершена — все 9 renderer-модулей семейств должны
        выполнять render() без NotImplementedError.
        """
        from core.metallography_v3.renderers import (
            bainite, granular_pearlite, high_temp_phases, martensite,
            quench_products, surface_layers, tempered, white_cast_iron,
            widmanstatten,
        )
        from core.metallography_v3.system_generators.base import (
            SystemGenerationContext,
        )
        from core.contracts_v2 import ProcessingState

        modules_with_sample_stage = [
            (high_temp_phases, "austenite", {"AUSTENITE": 1.0}),
            (white_cast_iron, "ledeburite", {"PEARLITE": 0.5, "CEMENTITE": 0.5}),
            (martensite, "martensite_cubic", {"MARTENSITE_CUBIC": 1.0}),
            (bainite, "bainite_upper", {"BAINITE": 0.78, "CEMENTITE": 0.22}),
            (quench_products, "troostite_quench", {"TROOSTITE": 0.88, "CEMENTITE": 0.12}),
            (tempered, "tempered_high", {"SORBITE": 0.42, "FERRITE": 0.40, "CEMENTITE": 0.18}),
            (widmanstatten, "widmanstatten_ferrite", {"FERRITE": 0.5, "PEARLITE": 0.5}),
            (surface_layers, "decarburized_layer", {"FERRITE": 0.7, "PEARLITE": 0.3}),
            (granular_pearlite, "granular_pearlite", {"FERRITE": 0.85, "CEMENTITE": 0.15}),
        ]
        seed_split = {
            "seed_topology": 1, "seed_boundary": 2, "seed_particles": 3,
            "seed_lamella": 4, "seed_noise": 5,
        }
        for mod, stage, fractions in modules_with_sample_stage:
            with self.subTest(module=mod.__name__.split(".")[-1]):
                ctx = SystemGenerationContext(
                    size=(64, 64),
                    seed=1,
                    inferred_system="fe-c",
                    stage=stage,
                    phase_fractions=fractions,
                    composition_wt={"Fe": 99.7, "C": 0.3},
                    processing=ProcessingState(temperature_c=20.0, cooling_mode="air"),
                )
                # Не должен бросить NotImplementedError.
                out = mod.render(
                    context=ctx, stage=stage, phase_fractions=fractions,
                    seed_split=seed_split,
                )
                self.assertIsNotNone(out.image_gray)


if __name__ == "__main__":
    unittest.main()
