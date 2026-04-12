from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "send_codex_pending.ps1"
TMP_ROOT = ROOT / "tmp" / "test_sandbox"


def _reset_case(name: str) -> Path:
    path = TMP_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class SendCodexPendingScriptTests(unittest.TestCase):
    def test_print_only_outputs_pending_prompt_and_marker(self) -> None:
        case_dir = _reset_case("send_case_1")
        prompt_path = case_dir / "prompt.md"
        state_path = case_dir / "state.json"
        prompt_text = "TEST PROMPT 123"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        state = {
            "version": 1,
            "pdf_path": "book.pdf",
            "book_end_page": 1627,
            "batch_size": 20,
            "completed_through_page": 120,
            "pending_batch": {
                "start_page": 121,
                "end_page": 140,
                "mode": "combined",
                "prompt_file": str(prompt_path),
                "extract_file": str(case_dir / "extract.txt"),
                "generated_at": "2026-03-26T00:00:00+03:00",
                "completion_marker": "LOOP_COMPLETE_TEST_121_140",
            },
            "history": [],
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        ps_cmd = "pwsh" if shutil.which("pwsh") else "powershell"
        proc = subprocess.run(
            [
                ps_cmd,
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-State",
                str(state_path),
                "-PrintOnly",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )

        self.assertIn(prompt_text, proc.stdout)
        self.assertIn("LOOP_COMPLETE_TEST_121_140", proc.stdout)

    def test_print_only_can_generate_next_prompt(self) -> None:
        case_dir = _reset_case("send_case_2")
        state_path = case_dir / "state.json"
        pdf_path = case_dir / "book.pdf"
        pdf_path.write_bytes(b"not-a-real-pdf")
        state = {
            "version": 1,
            "pdf_path": str(pdf_path),
            "book_end_page": 1627,
            "batch_size": 20,
            "completed_through_page": 120,
            "pending_batch": None,
            "history": [],
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        ps_cmd = "pwsh" if shutil.which("pwsh") else "powershell"
        proc = subprocess.run(
            [
                ps_cmd,
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-State",
                str(state_path),
                "-TmpDir",
                str(case_dir / "loop"),
                "-GenerateNext",
                "-PrintOnly",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )

        self.assertIn("121", proc.stdout)
        self.assertIn("140", proc.stdout)
        updated = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertIsNotNone(updated.get("pending_batch"))
        self.assertTrue(str(updated["pending_batch"].get("completion_marker", "")).startswith("LOOP_COMPLETE_0121_0140"))


if __name__ == "__main__":
    unittest.main()
