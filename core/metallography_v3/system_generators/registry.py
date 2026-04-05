from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .base import SystemGenerationContext, SystemGenerationResult
from .generator_al_cu_mg import generate_al_cu_mg
from .generator_al_si import generate_al_si
from .generator_cu_zn import generate_cu_zn
from .generator_custom import generate_custom
from .generator_fe_c import generate_fe_c
from .generator_fe_si import generate_fe_si

SystemGeneratorHandler = Callable[[SystemGenerationContext], SystemGenerationResult]

_AUTO_BY_SYSTEM: dict[str, str] = {
    "fe-c": "system_fe_c",
    "fe-si": "system_fe_si",
    "al-si": "system_al_si",
    "cu-zn": "system_cu_zn",
    "al-cu-mg": "system_al_cu_mg",
}


@dataclass(slots=True)
class SystemGeneratorSelection:
    requested_mode: str
    resolved_mode: str
    resolved_system: str
    resolved_stage: str
    fallback_used: bool
    selection_reason: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "resolved_mode": self.resolved_mode,
            "resolved_system": self.resolved_system,
            "resolved_stage": self.resolved_stage,
            "fallback_used": self.fallback_used,
            "selection_reason": self.selection_reason,
            "confidence": self.confidence,
        }


class SystemGeneratorRegistryV3:
    def __init__(self) -> None:
        self._handlers: dict[str, SystemGeneratorHandler] = {
            "system_fe_c": generate_fe_c,
            "system_fe_si": generate_fe_si,
            "system_al_si": generate_al_si,
            "system_cu_zn": generate_cu_zn,
            "system_al_cu_mg": generate_al_cu_mg,
            "system_custom": generate_custom,
        }

    def available_modes(self) -> list[str]:
        return sorted(self._handlers.keys())

    def resolve_mode(self, requested_mode: str, inferred_system: str) -> tuple[str, bool, str]:
        requested = str(requested_mode or "system_auto").strip().lower()
        system = str(inferred_system or "").strip().lower()

        if requested == "system_auto":
            resolved = _AUTO_BY_SYSTEM.get(system, "system_custom")
            fallback = bool(system not in _AUTO_BY_SYSTEM)
            reason = (
                f"auto_by_inferred_system:{system}"
                if not fallback
                else f"auto_fallback_custom:{system or 'unknown'}"
            )
            return resolved, fallback, reason

        if requested in self._handlers:
            return requested, False, "manual_override"

        resolved = _AUTO_BY_SYSTEM.get(system, "system_custom")
        fallback = True
        reason = f"invalid_manual_mode_fallback:{requested}"
        return resolved, fallback, reason

    def generate(
        self,
        *,
        context: SystemGenerationContext,
        requested_mode: str,
    ) -> tuple[SystemGenerationResult, SystemGeneratorSelection]:
        resolved_mode, fallback_used, reason = self.resolve_mode(requested_mode, context.inferred_system)
        handler = self._handlers.get(resolved_mode)
        if handler is None:
            handler = self._handlers["system_custom"]
            fallback_used = True
            reason = f"{reason}|handler_missing"
            resolved_mode = "system_custom"
        result = handler(context)
        selection = SystemGeneratorSelection(
            requested_mode=str(requested_mode or "system_auto"),
            resolved_mode=str(resolved_mode),
            resolved_system=str(context.inferred_system),
            resolved_stage=str(context.stage),
            fallback_used=bool(fallback_used),
            selection_reason=str(reason),
            confidence=float(context.confidence),
        )
        return result, selection
