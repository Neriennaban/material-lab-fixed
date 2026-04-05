from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CALPHAD_SUPPORTED_SYSTEMS: tuple[str, ...] = ("fe-c", "fe-si", "al-si", "cu-zn", "al-cu-mg")

_SYSTEM_REQUIRED_ELEMENTS: dict[str, tuple[str, ...]] = {
    "fe-c": ("FE", "C"),
    "fe-si": ("FE", "SI"),
    "al-si": ("AL", "SI"),
    "cu-zn": ("CU", "ZN"),
    "al-cu-mg": ("AL", "CU", "MG"),
}

_OPEN_DB_DEFAULTS: dict[str, str] = {
    "fe-c": "fe_c.tdb",
    "fe-si": "cost507.tdb",
    "al-si": "cost507.tdb",
    "cu-zn": "cost507.tdb",
    "al-cu-mg": "cost507.tdb",
}


@dataclass(slots=True)
class CalphadDBReference:
    system: str
    path: Path
    source: str
    sha256: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 128), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_system(system: str) -> str:
    value = str(system or "").strip().lower()
    aliases = {
        "fe_c": "fe-c",
        "fec": "fe-c",
        "fe_si": "fe-si",
        "fesi": "fe-si",
        "al_si": "al-si",
        "alsi": "al-si",
        "cu_zn": "cu-zn",
        "cuzn": "cu-zn",
        "al_cu_mg": "al-cu-mg",
    }
    return aliases.get(value, value)


def open_db_dir() -> Path:
    return Path(__file__).resolve().parent / "databases" / "open"


def _load_profile(profile_path: Path | None) -> dict[str, Any]:
    if profile_path is None or not profile_path.exists():
        return {}
    try:
        return json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_database_reference(
    system: str,
    *,
    thermo: Any = None,
    profile_path: str | Path | None = None,
    tdb_dir: str | Path | None = None,
) -> CalphadDBReference:
    sys_name = _normalize_system(system)
    if sys_name not in CALPHAD_SUPPORTED_SYSTEMS:
        raise ValueError(f"SYSTEM_UNSUPPORTED: {system}")

    profile_file: Path | None = None
    if profile_path:
        profile_file = Path(profile_path)
    elif thermo is not None and getattr(thermo, "db_profile_path", None):
        profile_file = Path(str(getattr(thermo, "db_profile_path")))
    profile = _load_profile(profile_file)
    profile_db_paths = profile.get("db_paths", {}) if isinstance(profile.get("db_paths", {}), dict) else {}

    override_path = None
    if thermo is not None:
        overrides = getattr(thermo, "db_overrides", {}) or {}
        if isinstance(overrides, dict):
            override_path = overrides.get(sys_name)
    if override_path:
        candidate = Path(str(override_path)).expanduser()
        if not candidate.exists():
            raise ValueError(f"DB_MISSING: override database not found for {sys_name}: {candidate}")
        return CalphadDBReference(
            system=sys_name,
            path=candidate.resolve(),
            source="request_override",
            sha256=_sha256_file(candidate.resolve()),
        )

    if sys_name in profile_db_paths:
        candidate = Path(str(profile_db_paths[sys_name])).expanduser()
        if not candidate.exists():
            raise ValueError(f"DB_MISSING: profile database not found for {sys_name}: {candidate}")
        return CalphadDBReference(
            system=sys_name,
            path=candidate.resolve(),
            source="profile",
            sha256=_sha256_file(candidate.resolve()),
        )

    if tdb_dir is None:
        if thermo is not None and getattr(thermo, "db_overrides", None):
            pass
        candidate_dir = open_db_dir()
    else:
        candidate_dir = Path(tdb_dir)

    filename = _OPEN_DB_DEFAULTS[sys_name]
    candidate = candidate_dir / filename
    if not candidate.exists():
        raise ValueError(f"DB_MISSING: open database not found for {sys_name}: {candidate}")
    return CalphadDBReference(
        system=sys_name,
        path=candidate.resolve(),
        source="open_bundled",
        sha256=_sha256_file(candidate.resolve()),
    )


def validate_database_reference(db_ref: CalphadDBReference) -> None:
    try:
        from pycalphad import Database
    except Exception as exc:  # pragma: no cover - env dependent
        raise ValueError(f"CALPHAD_UNAVAILABLE: pycalphad import failed: {exc}") from exc

    if not db_ref.path.exists():
        raise ValueError(f"DB_MISSING: {db_ref.path}")

    try:
        db = Database(str(db_ref.path))
    except Exception as exc:
        raise ValueError(f"DB_INVALID: failed to parse TDB {db_ref.path}: {exc}") from exc

    elems = {str(el).upper() for el in getattr(db, "elements", set())}
    required = set(_SYSTEM_REQUIRED_ELEMENTS.get(db_ref.system, ()))
    missing = sorted(required - elems)
    if missing:
        raise ValueError(f"DB_INCOMPATIBLE: {db_ref.system} missing elements {missing} in {db_ref.path}")

