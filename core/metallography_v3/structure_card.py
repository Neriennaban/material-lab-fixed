"""Loader структурных карточек микроструктур Fe-C.

Карточки живут в ``datasets/structure_cards/<id>.json`` и задают
калибровочные параметры (RGB-тона, размеры, плотности, углы, пр.) из
§N металлографического справочника. Используются семейственными
renderer'ами в ``core/metallography_v3/renderers/`` как единственный
источник правды по численным характеристикам каждой структуры.

Phase 1 (см. whimsical-wandering-dawn.md):
- Создаётся loader + минимальный валидатор (без внешней зависимости
  ``jsonschema``, чтобы не менять окружение).
- Заполняются 2 smoke-карточки: ``martensite_lath``, ``bainite_upper``.
- Остальные карточки — по мере реализации sub-plan'ов Phase 2–8.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_CARDS_DIR = REPO_ROOT / "datasets" / "structure_cards"

_REQUIRED_TOP_LEVEL: tuple[str, ...] = (
    "id",
    "name_ru",
    "reference_section",
    "rgb_tones",
    "morphology",
)


class StructureCardError(ValueError):
    """Ошибка при загрузке/валидации карточки."""


@dataclass(slots=True, frozen=True)
class ReferenceImage:
    path: str
    source: str
    magnification: float | None = None
    etchant: str | None = None


@dataclass(slots=True, frozen=True)
class StructureCard:
    id: str
    name_ru: str
    name_en: str
    reference_section: str
    reference_sources: tuple[str, ...]
    reference_images: tuple[ReferenceImage, ...]
    phase_composition: dict[str, float]
    rgb_tones: dict[str, dict[str, tuple[int, int, int]]]
    morphology: dict[str, Any]
    composition_triggers: dict[str, Any]
    generation_hints: dict[str, Any]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _validate_top_level(data: dict[str, Any], source: Path) -> None:
    if not isinstance(data, dict):
        raise StructureCardError(
            f"{source.name}: top-level must be a JSON object"
        )
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in data]
    if missing:
        raise StructureCardError(
            f"{source.name}: missing required keys: {missing}"
        )


def _coerce_rgb(value: Any, *, context: str) -> tuple[int, int, int]:
    if not (isinstance(value, (list, tuple)) and len(value) == 3):
        raise StructureCardError(
            f"{context}: expected RGB triple [R,G,B], got {value!r}"
        )
    try:
        rgb = tuple(int(v) for v in value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise StructureCardError(f"{context}: non-integer RGB value") from exc
    for comp in rgb:
        if not 0 <= comp <= 255:
            raise StructureCardError(
                f"{context}: RGB component {comp} out of [0,255]"
            )
    return rgb  # type: ignore[return-value]


def _parse_rgb_tones(
    node: Any, *, source: Path
) -> dict[str, dict[str, tuple[int, int, int]]]:
    if not isinstance(node, dict):
        raise StructureCardError(f"{source.name}: rgb_tones must be an object")
    out: dict[str, dict[str, tuple[int, int, int]]] = {}
    for reagent, components in node.items():
        if not isinstance(components, dict):
            raise StructureCardError(
                f"{source.name}: rgb_tones.{reagent} must be an object"
            )
        out[str(reagent)] = {
            str(comp_name): _coerce_rgb(
                rgb, context=f"{source.name}:rgb_tones.{reagent}.{comp_name}"
            )
            for comp_name, rgb in components.items()
        }
    return out


def _parse_reference_images(node: Any) -> tuple[ReferenceImage, ...]:
    if node is None:
        return ()
    if not isinstance(node, list):
        raise StructureCardError("reference_images must be an array")
    out: list[ReferenceImage] = []
    for idx, item in enumerate(node):
        if not isinstance(item, dict):
            raise StructureCardError(
                f"reference_images[{idx}] must be an object"
            )
        if "path" not in item or "source" not in item:
            raise StructureCardError(
                f"reference_images[{idx}] missing required 'path'/'source'"
            )
        out.append(
            ReferenceImage(
                path=str(item["path"]),
                source=str(item["source"]),
                magnification=(
                    float(item["magnification"])
                    if item.get("magnification") is not None
                    else None
                ),
                etchant=(
                    str(item["etchant"])
                    if item.get("etchant") is not None
                    else None
                ),
            )
        )
    return tuple(out)


def _parse(data: dict[str, Any], *, source: Path) -> StructureCard:
    _validate_top_level(data, source)
    return StructureCard(
        id=str(data["id"]),
        name_ru=str(data["name_ru"]),
        name_en=str(data.get("name_en", "")),
        reference_section=str(data["reference_section"]),
        reference_sources=tuple(
            str(x) for x in (data.get("reference_sources") or ())
        ),
        reference_images=_parse_reference_images(data.get("reference_images")),
        phase_composition=dict(data.get("phase_composition") or {}),
        rgb_tones=_parse_rgb_tones(data["rgb_tones"], source=source),
        morphology=dict(data["morphology"]),
        composition_triggers=dict(data.get("composition_triggers") or {}),
        generation_hints=dict(data.get("generation_hints") or {}),
        raw=data,
    )


@lru_cache(maxsize=128)
def load_card(structure_id: str) -> StructureCard:
    """Загрузить карточку по её id (имя файла без .json)."""
    path = STRUCTURE_CARDS_DIR / f"{structure_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Structure card not found: {path} "
            f"(available: {sorted(p.stem for p in STRUCTURE_CARDS_DIR.glob('*.json') if not p.name.startswith('_'))})"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    card = _parse(data, source=path)
    if card.id != structure_id:
        raise StructureCardError(
            f"{path.name}: id field {card.id!r} doesn't match filename"
        )
    return card


def list_cards() -> list[str]:
    """Вернуть id всех имеющихся карточек (кроме _schema.json)."""
    return sorted(
        p.stem
        for p in STRUCTURE_CARDS_DIR.glob("*.json")
        if not p.name.startswith("_")
    )
