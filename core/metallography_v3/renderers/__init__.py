"""Пакет семейственных renderer'ов микроструктур Fe-C.

Каждый модуль реализует единый контракт (см. ``_common.py``):
    HANDLES_STAGES: frozenset[str]
    def render(*, context, stage, phase_fractions, seed_split) -> RendererOutput

Phase 1: модули зарегистрированы в таблице диспетчера
``core.metallography_v3.system_generators.fe_c_unified._STAGE_TO_RENDERER``,
но в основной runtime-путь не подключены — старые ``_build_*_render``
работают как прежде. Нулевой визуальный drift.

Phase 2–8 активируют модули по очереди (см. план
``whimsical-wandering-dawn.md``).
"""
from __future__ import annotations

from core.metallography_v3.renderers import (  # noqa: F401
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
from core.metallography_v3.renderers._common import RendererOutput  # noqa: F401

__all__ = [
    "RendererOutput",
    "bainite",
    "granular_pearlite",
    "high_temp_phases",
    "martensite",
    "quench_products",
    "surface_layers",
    "tempered",
    "white_cast_iron",
    "widmanstatten",
]
