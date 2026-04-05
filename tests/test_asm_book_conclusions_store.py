from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from scripts.asm_book_conclusions import (
    append_conclusion_entry,
    latest_conclusion_entry,
    load_conclusions_journal,
)


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / "tmp" / "test_sandbox"


def _reset_case(name: str) -> Path:
    path = TMP_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class AsmBookConclusionsStoreTests(unittest.TestCase):
    def test_append_and_latest_roundtrip(self) -> None:
        case_dir = _reset_case("conclusions_case_1")
        journal_path = case_dir / "conclusions.json"
        append_conclusion_entry(
            {
                "start_page": 141,
                "end_page": 160,
                "completed_at": "2026-03-26T02:23:41+03:00",
                "completion_marker": "LOOP_COMPLETE_0141_0160_combined_test",
                "topic_summary": "SEM/TEM prep and diffraction contrast",
                "key_takeaways": ["TEM defect physics", "guidance only"],
                "project_applicability": "No immediate renderer change",
                "recommended_action": "defer",
                "implemented_changes_summary": [],
                "next_followup_hint": "Revisit when TEM branch exists",
            },
            journal_path,
        )
        journal = load_conclusions_journal(journal_path)
        self.assertEqual(len(journal["entries"]), 1)
        latest = latest_conclusion_entry(journal_path)
        self.assertIsNotNone(latest)
        self.assertEqual(int(latest["end_page"]), 160)

    def test_duplicate_marker_updates_not_duplicates(self) -> None:
        case_dir = _reset_case("conclusions_case_2")
        journal_path = case_dir / "conclusions.json"
        base = {
            "start_page": 141,
            "end_page": 160,
            "completed_at": "2026-03-26T02:23:41+03:00",
            "completion_marker": "LOOP_COMPLETE_0141_0160_combined_test",
            "topic_summary": "A",
            "key_takeaways": [],
            "project_applicability": "X",
            "recommended_action": "defer",
            "implemented_changes_summary": [],
            "next_followup_hint": "H1",
        }
        append_conclusion_entry(base, journal_path)
        append_conclusion_entry({**base, "topic_summary": "B", "next_followup_hint": "H2"}, journal_path)
        journal = load_conclusions_journal(journal_path)
        self.assertEqual(len(journal["entries"]), 1)
        self.assertEqual(str(journal["entries"][0]["topic_summary"]), "B")

    def test_corrupted_journal_recovers(self) -> None:
        case_dir = _reset_case("conclusions_case_3")
        journal_path = case_dir / "conclusions.json"
        journal_path.write_text("{broken json", encoding="utf-8")
        journal = load_conclusions_journal(journal_path)
        self.assertEqual(journal["entries"], [])

    def test_partial_entry_list_is_normalized(self) -> None:
        case_dir = _reset_case("conclusions_case_4")
        journal_path = case_dir / "conclusions.json"
        journal_path.write_text(json.dumps({"version": 1, "entries": "oops"}, ensure_ascii=False), encoding="utf-8")
        journal = load_conclusions_journal(journal_path)
        self.assertEqual(journal["entries"], [])


if __name__ == "__main__":
    unittest.main()
