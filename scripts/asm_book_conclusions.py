from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONCLUSIONS_PATH = ROOT / "docs" / "Literature" / "asm_volume9_conclusions_ru.json"


def default_conclusions_journal() -> dict[str, Any]:
    return {
        "version": 1,
        "entries": [],
    }


def load_conclusions_journal(path: Path | None = None) -> dict[str, Any]:
    target = Path(path or DEFAULT_CONCLUSIONS_PATH)
    if not target.exists():
        journal = default_conclusions_journal()
        save_conclusions_journal(journal, target)
        return journal
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return default_conclusions_journal()
    if not isinstance(payload, dict):
        return default_conclusions_journal()
    entries = payload.get("entries")
    if not isinstance(entries, list):
        payload["entries"] = []
    payload.setdefault("version", 1)
    return payload


def save_conclusions_journal(journal: dict[str, Any], path: Path | None = None) -> None:
    target = Path(path or DEFAULT_CONCLUSIONS_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(journal, ensure_ascii=False, indent=2), encoding="utf-8")


def append_conclusion_entry(entry: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    journal = load_conclusions_journal(path)
    normalized = dict(entry)
    start = int(normalized.get("start_page", 0))
    end = int(normalized.get("end_page", 0))
    marker = str(normalized.get("completion_marker", ""))
    entries = list(journal.get("entries", []))

    updated = False
    for idx, existing in enumerate(entries):
        if not isinstance(existing, dict):
            continue
        same_range = int(existing.get("start_page", -1)) == start and int(existing.get("end_page", -1)) == end
        same_marker = marker and str(existing.get("completion_marker", "")) == marker
        if same_range or same_marker:
            entries[idx] = {**existing, **normalized}
            updated = True
            break
    if not updated:
        entries.append(normalized)
    entries.sort(key=lambda item: (int(item.get("start_page", 0)), int(item.get("end_page", 0))))
    journal["entries"] = entries
    save_conclusions_journal(journal, path)
    return journal


def latest_conclusion_entry(path: Path | None = None) -> dict[str, Any] | None:
    journal = load_conclusions_journal(path)
    entries = [entry for entry in journal.get("entries", []) if isinstance(entry, dict)]
    if not entries:
        return None
    entries.sort(key=lambda item: (int(item.get("end_page", 0)), int(item.get("start_page", 0))))
    return dict(entries[-1])


def load_summary_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Summary file must contain a JSON object: {path}")
    return payload
