"""Общий контракт для модульных renderer'ов семейств микроструктур Fe-C.

Добавлено в Phase 1 редизайна (см. план whimsical-wandering-dawn.md).

Каждый модуль в `core/metallography_v3/renderers/` экспортирует:
    - ``HANDLES_STAGES: frozenset[str]`` — набор строк стадий, которые он
      обслуживает (из SYSTEM_STAGE_ORDER и будущих новых стадий).
    - ``render(*, context, stage, phase_fractions, seed_split) -> RendererOutput``
      — функция с единой сигнатурой.

На Phase 1 все функции ``render`` бросают ``NotImplementedError``; они
зарегистрированы в ``fe_c_unified._STAGE_TO_RENDERER``, но **не**
подключены в runtime-путь ``render_fe_c_unified``. Старые
``_build_*_render`` продолжают работать без изменений → нулевой
визуальный drift по завершении Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class RendererOutput:
    """Единый тип результата семейственного renderer'а.

    Совместим с контрактом старых ``_build_*_render`` (tuple-возврат):
    старая логика в ``render_fe_c_unified`` ожидает
    ``(image_gray, phase_masks, rendered_layers, fragment_area, trace)``
    или ``(image_gray, phase_masks, trace)``. Поля здесь покрывают оба
    варианта; поля по умолчанию заполняются диспетчером.
    """

    image_gray: np.ndarray
    phase_masks: dict[str, np.ndarray]
    morphology_trace: dict[str, Any]
    rendered_layers: list[str] = field(default_factory=list)
    fragment_area: int = 0
