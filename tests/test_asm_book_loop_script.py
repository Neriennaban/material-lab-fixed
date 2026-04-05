from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from scripts.asm_book_conclusions import load_conclusions_journal
from scripts.asm_book_loop import (
    build_combined_prompt,
    clear_pending,
    compute_next_batch,
    default_state,
    format_status,
    load_state,
    make_completion_marker,
    mark_pending_complete,
    pending_from_state,
    queue_prompt,
    save_state,
    set_completed_page,
)


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / "tmp" / "test_sandbox"


def _reset_case(name: str) -> Path:
    path = TMP_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class AsmBookLoopScriptTests(unittest.TestCase):
    def test_compute_next_batch_uses_completed_page_and_batch_size(self) -> None:
        state = default_state()
        state["completed_through_page"] = 120
        state["batch_size"] = 20
        self.assertEqual(compute_next_batch(state), (121, 140))

    def test_prompt_contains_block_conclusion_and_marker(self) -> None:
        marker = make_completion_marker(121, 140, "combined", "2026-03-26T02:12:00+03:00")
        prompt = build_combined_prompt(
            pdf_path=Path("book.pdf"),
            extract_file=Path("tmp/pages_0121_0140.txt"),
            state_path=Path("state.json"),
            start_page=121,
            end_page=140,
            completion_marker=marker,
            latest_conclusion=None,
        )
        self.assertIn("pages 121–140", prompt)
        self.assertIn("Block Conclusion", prompt)
        self.assertIn(marker, prompt)

    def test_complete_with_summary_file_updates_state_and_journal(self) -> None:
        case_dir = _reset_case("loop_case_complete")
        state_path = case_dir / "state.json"
        summary_path = case_dir / "summary.json"
        conclusions_path = case_dir / "conclusions.json"

        state = default_state()
        state["completed_through_page"] = 140
        state["pending_batch"] = {
            "start_page": 141,
            "end_page": 160,
            "mode": "combined",
            "prompt_file": "prompt.md",
            "extract_file": "extract.txt",
            "generated_at": "2026-03-26T00:00:00+03:00",
            "completion_marker": "LOOP_COMPLETE_0141_0160_combined_test",
        }
        save_state(state_path, state)

        summary_path.write_text(
            json.dumps(
                {
                    "topic_summary": "SEM/TEM prep and diffraction-contrast block",
                    "key_takeaways": ["TEM defect physics", "No immediate renderer changes"],
                    "project_applicability": "Future guidance branch only",
                    "recommended_action": "defer",
                    "implemented_changes_summary": [],
                    "next_followup_hint": "Use for future TEM guidance",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        import scripts.asm_book_conclusions as conclusions_mod
        import scripts.asm_book_loop as loop_mod

        old_default = conclusions_mod.DEFAULT_CONCLUSIONS_PATH
        old_loop_default = loop_mod.DEFAULT_CONCLUSIONS_PATH
        conclusions_mod.DEFAULT_CONCLUSIONS_PATH = conclusions_path
        loop_mod.DEFAULT_CONCLUSIONS_PATH = conclusions_path
        try:
            updated = mark_pending_complete(state_path, summary_file=summary_path)
        finally:
            conclusions_mod.DEFAULT_CONCLUSIONS_PATH = old_default
            loop_mod.DEFAULT_CONCLUSIONS_PATH = old_loop_default

        self.assertEqual(int(updated["completed_through_page"]), 160)
        self.assertIsNone(updated["pending_batch"])
        journal = load_conclusions_journal(conclusions_path)
        self.assertEqual(len(journal["entries"]), 1)
        entry = dict(journal["entries"][0])
        self.assertEqual(int(entry["start_page"]), 141)
        self.assertEqual(str(entry["recommended_action"]), "defer")

    def test_set_page_clears_pending(self) -> None:
        case_dir = _reset_case("loop_case_set_page")
        state_path = case_dir / "state.json"
        state = default_state()
        state["pending_batch"] = {
            "start_page": 121,
            "end_page": 140,
            "mode": "combined",
            "prompt_file": "prompt.md",
            "extract_file": "extract.txt",
            "generated_at": "2026-03-26T00:00:00+03:00",
            "completion_marker": "LOOP_COMPLETE_0121_0140_combined_test",
        }
        save_state(state_path, state)
        updated = set_completed_page(state_path, 200)
        self.assertEqual(int(updated["completed_through_page"]), 200)
        self.assertIsNone(updated["pending_batch"])

    def test_clear_pending_preserves_completed_page(self) -> None:
        case_dir = _reset_case("loop_case_clear")
        state_path = case_dir / "state.json"
        state = default_state()
        state["completed_through_page"] = 140
        state["pending_batch"] = {
            "start_page": 141,
            "end_page": 160,
            "mode": "combined",
            "prompt_file": "prompt.md",
            "extract_file": "extract.txt",
            "generated_at": "2026-03-26T00:00:00+03:00",
            "completion_marker": "LOOP_COMPLETE_0141_0160_combined_test",
        }
        save_state(state_path, state)
        updated = clear_pending(state_path)
        self.assertEqual(int(updated["completed_through_page"]), 140)
        self.assertIsNone(updated["pending_batch"])

    def test_status_mentions_pending_batch_and_marker(self) -> None:
        state = default_state()
        state["pending_batch"] = {
            "start_page": 121,
            "end_page": 140,
            "mode": "combined",
            "prompt_file": "prompt.md",
            "extract_file": "extract.txt",
            "generated_at": "2026-03-26T00:00:00+03:00",
            "completion_marker": "LOOP_COMPLETE_0121_0140_combined_test",
        }
        pending = pending_from_state(state)
        self.assertIsNotNone(pending)
        status = format_status(state)
        self.assertIn("Pending batch: 121-140", status)
        self.assertIn("Completion marker:", status)

    def test_load_state_bootstraps_default_file(self) -> None:
        case_dir = _reset_case("loop_case_load")
        state_path = case_dir / "state.json"
        state = load_state(state_path)
        self.assertTrue(state_path.exists())
        self.assertEqual(int(state["completed_through_page"]), 120)

    def test_queue_prompt_sets_pending_without_advancing(self) -> None:
        case_dir = _reset_case("loop_case_queue")
        state_path = case_dir / "state.json"
        prompt_path = case_dir / "prompt.md"
        prompt_path.write_text("apply followup", encoding="utf-8")
        save_state(state_path, default_state())
        updated = queue_prompt(
            state_path,
            prompt_file=prompt_path,
            start_page=101,
            end_page=120,
            mode="apply_followup",
        )
        pending = dict(updated.get("pending_batch", {}))
        self.assertEqual(int(updated["completed_through_page"]), 120)
        self.assertEqual(int(pending["start_page"]), 101)
        self.assertEqual(str(pending["mode"]), "apply_followup")
        self.assertTrue(str(pending.get("completion_marker", "")).startswith("LOOP_COMPLETE_0101_0120"))


if __name__ == "__main__":
    unittest.main()
