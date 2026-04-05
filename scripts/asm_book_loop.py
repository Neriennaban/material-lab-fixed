from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.asm_book_conclusions import (
    DEFAULT_CONCLUSIONS_PATH,
    append_conclusion_entry,
    latest_conclusion_entry,
    load_summary_file,
)

DEFAULT_PDF_PATH = Path(
    r"C:\Users\Егор\Documents\Книги\книги для material lab\1asm_handbook_volume_9_metallography_and_microstructures.pdf"
)
DEFAULT_STATE_PATH = ROOT / "docs" / "Literature" / "asm_volume9_loop_state.json"
DEFAULT_TMP_DIR = ROOT / "tmp" / "asm_book_loop"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def make_completion_marker(
    start_page: int, end_page: int, mode: str, generated_at: str
) -> str:
    compact = (
        str(generated_at)
        .replace(":", "")
        .replace("-", "")
        .replace("+", "P")
        .replace(".", "")
    )
    mode_token = "".join(ch if ch.isalnum() else "_" for ch in str(mode))
    return f"LOOP_COMPLETE_{int(start_page):04d}_{int(end_page):04d}_{mode_token}_{compact}"


@dataclass(slots=True)
class PendingBatch:
    start_page: int
    end_page: int
    mode: str
    prompt_file: str
    extract_file: str
    generated_at: str
    completion_marker: str = ""
    sent_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "start_page": int(self.start_page),
            "end_page": int(self.end_page),
            "mode": str(self.mode),
            "prompt_file": str(self.prompt_file),
            "extract_file": str(self.extract_file),
            "generated_at": str(self.generated_at),
        }
        if self.completion_marker:
            payload["completion_marker"] = str(self.completion_marker)
        if self.sent_at:
            payload["sent_at"] = str(self.sent_at)
        return payload


def default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "pdf_path": str(DEFAULT_PDF_PATH),
        "book_end_page": 1627,
        "batch_size": 20,
        "completed_through_page": 120,
        "pending_batch": None,
        "history": [],
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_pending_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if not str(normalized.get("completion_marker", "")).strip():
        normalized["completion_marker"] = make_completion_marker(
            int(normalized.get("start_page", 0)),
            int(normalized.get("end_page", 0)),
            str(normalized.get("mode", "combined")),
            str(normalized.get("generated_at", now_iso())),
        )
    return normalized


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        state = default_state()
        save_state(path, state)
        return state
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid state JSON: {path}")
    state = default_state()
    state.update(payload)
    pending = state.get("pending_batch")
    if isinstance(pending, dict):
        normalized = _normalize_pending_payload(pending)
        if normalized != pending:
            state["pending_batch"] = normalized
            save_state(path, state)
    return state


def pending_from_state(state: dict[str, Any]) -> PendingBatch | None:
    payload = state.get("pending_batch")
    if not isinstance(payload, dict):
        return None
    payload = _normalize_pending_payload(payload)
    return PendingBatch(
        start_page=int(payload.get("start_page", 0)),
        end_page=int(payload.get("end_page", 0)),
        mode=str(payload.get("mode", "combined")),
        prompt_file=str(payload.get("prompt_file", "")),
        extract_file=str(payload.get("extract_file", "")),
        generated_at=str(payload.get("generated_at", "")),
        completion_marker=str(payload.get("completion_marker", "")),
        sent_at=str(payload.get("sent_at", "")),
    )


def compute_next_batch(state: dict[str, Any]) -> tuple[int, int] | None:
    completed = int(state.get("completed_through_page", 0))
    final_page = int(state.get("book_end_page", 1627))
    batch_size = max(1, int(state.get("batch_size", 20)))
    start_page = completed + 1
    if start_page > final_page:
        return None
    end_page = min(final_page, start_page + batch_size - 1)
    return start_page, end_page


def extract_block_text(
    pdf_path: Path, start_page: int, end_page: int, out_path: Path
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "pdftotext",
                "-f",
                str(start_page),
                "-l",
                str(end_page),
                "-layout",
                str(pdf_path),
                str(out_path),
            ],
            check=True,
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        out_path.write_text(
            (
                f"Failed to extract pages {start_page}-{end_page} from PDF.\n"
                f"PDF path: {pdf_path}\n"
                f"Error: {exc}\n"
                "Use the original PDF directly in the Codex session.\n"
            ),
            encoding="utf-8",
        )
    return out_path


def build_combined_prompt(
    *,
    pdf_path: Path,
    extract_file: Path,
    state_path: Path,
    start_page: int,
    end_page: int,
    completion_marker: str,
    latest_conclusion: dict[str, Any] | None,
) -> str:
    recall_text = ""
    if isinstance(latest_conclusion, dict):
        recall_text = (
            "\nПоследний сохранённый вывод из памяти:\n"
            f"- pages: {latest_conclusion.get('start_page', '')}-{latest_conclusion.get('end_page', '')}\n"
            f"- topic_summary: {latest_conclusion.get('topic_summary', '')}\n"
            f"- project_applicability: {latest_conclusion.get('project_applicability', '')}\n"
            f"- recommended_action: {latest_conclusion.get('recommended_action', '')}\n"
            f"- next_followup_hint: {latest_conclusion.get('next_followup_hint', '')}\n"
            "Используй это как recall context и не дублируй уже принятые решения без нового основания.\n"
        )
    return (
        "Продолжай строго по книге ASM Handbook Volume 9.\n\n"
        f"Источник PDF: {pdf_path}\n"
        f"Извлечённый текст блока: {extract_file}\n"
        f"State file: {state_path}\n"
        f"Текущий блок: pages {start_page}–{end_page}\n"
        f"{recall_text}\n"
        "Сделай в одном проходе:\n"
        f"1. Изучи страницы {start_page}–{end_page} без пропусков, с page-by-page ledger, checkpoint и выводом последней обработанной страницы.\n"
        "2. После чтения оцени, какие изменения для проекта прямо следуют из этого блока книги.\n"
        "3. Если изменения оправданы и локально применимы, сразу внеси их в код проекта и прогони релевантные тесты.\n"
        "4. Если изменения пока преждевременны, явно зафиксируй почему и не выдумывай лишние правки.\n"
        "5. В конце укажи следующую страницу для продолжения.\n\n"
        "В конце добавь раздел `Block Conclusion` со структурой JSON:\n"
        "{\n"
        '  "topic_summary": "<short summary>",\n'
        '  "key_takeaways": ["...", "..."],\n'
        '  "project_applicability": "<what is usable for the project now>",\n'
        '  "recommended_action": "apply_now | defer | no_change",\n'
        '  "implemented_changes_summary": ["..."],\n'
        '  "next_followup_hint": "<what to revisit later>"\n'
        "}\n\n"
        f"В самом конце ответа добавь отдельной строкой точный маркер: {completion_marker}\n\n"
        "Цель проекта: генерация структур должна становиться максимально реалистичной и правильной по книге.\n"
        "После ответа loop-script автоматически зафиксирует завершение блока по маркеру."
    )


def generate_next_prompt(state_path: Path, tmp_dir: Path, mode: str) -> PendingBatch:
    state = load_state(state_path)
    pending = pending_from_state(state)
    if pending is not None:
        return pending
    batch = compute_next_batch(state)
    if batch is None:
        raise SystemExit("Book is already fully covered.")
    start_page, end_page = batch
    pdf_path = Path(str(state.get("pdf_path", DEFAULT_PDF_PATH)))
    extract_file = tmp_dir / f"pages_{start_page:04d}_{end_page:04d}.txt"
    prompt_file = tmp_dir / f"prompt_{start_page:04d}_{end_page:04d}_{mode}.md"
    generated_at = now_iso()
    completion_marker = make_completion_marker(start_page, end_page, mode, generated_at)
    extract_block_text(pdf_path, start_page, end_page, extract_file)
    prompt = build_combined_prompt(
        pdf_path=pdf_path,
        extract_file=extract_file,
        state_path=state_path,
        start_page=start_page,
        end_page=end_page,
        completion_marker=completion_marker,
        latest_conclusion=latest_conclusion_entry(DEFAULT_CONCLUSIONS_PATH),
    )
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(prompt, encoding="utf-8")
    pending = PendingBatch(
        start_page=start_page,
        end_page=end_page,
        mode=mode,
        prompt_file=str(prompt_file),
        extract_file=str(extract_file),
        generated_at=generated_at,
        completion_marker=completion_marker,
    )
    state["pending_batch"] = pending.to_dict()
    save_state(state_path, state)
    return pending


def mark_pending_complete(
    state_path: Path,
    *,
    completed_end_page: int | None = None,
    summary_file: Path | None = None,
) -> dict[str, Any]:
    state = load_state(state_path)
    pending = pending_from_state(state)
    if pending is None:
        raise SystemExit("No pending batch. Run `next` first.")
    end_page = int(
        completed_end_page if completed_end_page is not None else pending.end_page
    )
    if end_page < pending.start_page or end_page > pending.end_page:
        raise SystemExit(
            f"completed_end_page {end_page} is outside pending range {pending.start_page}-{pending.end_page}."
        )

    if summary_file is not None:
        summary = load_summary_file(summary_file)
        conclusion_entry = {
            "start_page": int(pending.start_page),
            "end_page": int(pending.end_page),
            "completed_at": now_iso(),
            "completion_marker": str(pending.completion_marker),
            "topic_summary": str(summary.get("topic_summary", "")).strip(),
            "key_takeaways": list(summary.get("key_takeaways", [])),
            "project_applicability": str(
                summary.get("project_applicability", "")
            ).strip(),
            "recommended_action": str(summary.get("recommended_action", "")).strip(),
            "implemented_changes_summary": list(
                summary.get("implemented_changes_summary", [])
            ),
            "next_followup_hint": str(summary.get("next_followup_hint", "")).strip(),
        }
        append_conclusion_entry(conclusion_entry, DEFAULT_CONCLUSIONS_PATH)

    state["completed_through_page"] = int(end_page)
    history = list(state.get("history", []))
    history.append(
        {
            **pending.to_dict(),
            "completed_at": now_iso(),
            "completed_through_page": int(end_page),
        }
    )
    state["history"] = history
    state["pending_batch"] = None
    save_state(state_path, state)
    return state


def clear_pending(state_path: Path) -> dict[str, Any]:
    state = load_state(state_path)
    state["pending_batch"] = None
    save_state(state_path, state)
    return state


def queue_prompt(
    state_path: Path,
    *,
    prompt_file: Path,
    start_page: int,
    end_page: int,
    mode: str,
    extract_file: Path | None = None,
    completion_marker: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    state = load_state(state_path)
    if state.get("pending_batch") is not None and not force:
        raise SystemExit(
            "Pending batch already exists. Use clear-pending first or pass --force."
        )
    generated_at = now_iso()
    marker = str(
        completion_marker
        or make_completion_marker(start_page, end_page, mode, generated_at)
    )
    pending = PendingBatch(
        start_page=int(start_page),
        end_page=int(end_page),
        mode=str(mode),
        prompt_file=str(prompt_file),
        extract_file=str(extract_file or ""),
        generated_at=generated_at,
        completion_marker=marker,
    )
    state["pending_batch"] = pending.to_dict()
    save_state(state_path, state)
    return state


def set_completed_page(state_path: Path, page: int) -> dict[str, Any]:
    state = load_state(state_path)
    final_page = int(state.get("book_end_page", 1627))
    state["completed_through_page"] = max(0, min(final_page, int(page)))
    state["pending_batch"] = None
    save_state(state_path, state)
    return state


def format_status(state: dict[str, Any]) -> str:
    pending = pending_from_state(state)
    completed = int(state.get("completed_through_page", 0))
    final_page = int(state.get("book_end_page", 1627))
    batch_size = int(state.get("batch_size", 20))
    lines = [
        f"PDF: {state.get('pdf_path', '')}",
        f"Completed through page: {completed}",
        f"Final page: {final_page}",
        f"Batch size: {batch_size}",
    ]
    next_batch = compute_next_batch(state)
    if next_batch is None:
        lines.append("Next batch: complete")
    else:
        lines.append(f"Next batch: {next_batch[0]}-{next_batch[1]}")
    if pending is not None:
        lines.append(
            f"Pending batch: {pending.start_page}-{pending.end_page} ({pending.mode})"
        )
        lines.append(f"Prompt file: {pending.prompt_file}")
        lines.append(f"Extract file: {pending.extract_file}")
        if pending.completion_marker:
            lines.append(f"Completion marker: {pending.completion_marker}")
        if pending.sent_at:
            lines.append(f"Sent at: {pending.sent_at}")
    else:
        lines.append("Pending batch: none")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loop orchestrator for ASM Handbook batch study + project-change prompts."
    )
    parser.add_argument(
        "--state", type=Path, default=DEFAULT_STATE_PATH, help="Path to loop state JSON"
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=DEFAULT_TMP_DIR,
        help="Directory for prompt/extract artifacts",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init", help="Create or reset state file")
    init_cmd.add_argument(
        "--pdf", type=Path, default=DEFAULT_PDF_PATH, help="Absolute path to source PDF"
    )
    init_cmd.add_argument(
        "--completed-through-page", type=int, default=120, help="Current completed page"
    )
    init_cmd.add_argument(
        "--book-end-page", type=int, default=1627, help="Last page of the book"
    )
    init_cmd.add_argument("--batch-size", type=int, default=20, help="Pages per batch")
    init_cmd.add_argument(
        "--force", action="store_true", help="Overwrite existing state"
    )

    sub.add_parser("status", help="Show loop status")

    next_cmd = sub.add_parser(
        "next", help="Generate next combined prompt and extraction"
    )
    next_cmd.add_argument(
        "--mode", type=str, default="combined", help="Prompt mode label"
    )

    complete_cmd = sub.add_parser(
        "complete", help="Mark current pending batch as completed"
    )
    complete_cmd.add_argument(
        "--completed-end-page",
        type=int,
        default=None,
        help="Override completed page within pending range",
    )
    complete_cmd.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional JSON summary file for conclusions journal",
    )

    clear_cmd = sub.add_parser(
        "clear-pending", help="Clear pending batch without advancing"
    )
    clear_cmd.add_argument(
        "--yes", action="store_true", help="Acknowledge clearing pending state"
    )

    set_page_cmd = sub.add_parser(
        "set-page", help="Manually set completed page and clear pending batch"
    )
    set_page_cmd.add_argument("page", type=int, help="New completed-through page")

    queue_cmd = sub.add_parser(
        "queue-prompt", help="Queue a custom pending prompt without advancing pages"
    )
    queue_cmd.add_argument(
        "--prompt-file", type=Path, required=True, help="Path to prompt markdown file"
    )
    queue_cmd.add_argument(
        "--start-page", type=int, required=True, help="Prompt start page"
    )
    queue_cmd.add_argument(
        "--end-page", type=int, required=True, help="Prompt end page"
    )
    queue_cmd.add_argument(
        "--mode", type=str, default="custom", help="Pending prompt mode"
    )
    queue_cmd.add_argument(
        "--extract-file", type=Path, default=None, help="Optional extract file path"
    )
    queue_cmd.add_argument(
        "--completion-marker",
        type=str,
        default=None,
        help="Optional explicit completion marker",
    )
    queue_cmd.add_argument(
        "--force", action="store_true", help="Overwrite current pending batch"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path: Path = args.state

    if args.command == "init":
        if state_path.exists() and not bool(args.force):
            raise SystemExit(
                f"State file already exists: {state_path}. Use --force to overwrite."
            )
        state = default_state()
        state.update(
            {
                "pdf_path": str(args.pdf),
                "completed_through_page": int(args.completed_through_page),
                "book_end_page": int(args.book_end_page),
                "batch_size": int(args.batch_size),
                "pending_batch": None,
                "history": [],
            }
        )
        save_state(state_path, state)
        print(format_status(state))
        return

    if args.command == "status":
        print(format_status(load_state(state_path)))
        return

    if args.command == "next":
        pending = generate_next_prompt(state_path, args.tmp_dir, mode=str(args.mode))
        prompt_text = Path(pending.prompt_file).read_text(encoding="utf-8")
        print(prompt_text)
        return

    if args.command == "complete":
        state = mark_pending_complete(
            state_path,
            completed_end_page=args.completed_end_page,
            summary_file=args.summary_file,
        )
        print(format_status(state))
        return

    if args.command == "clear-pending":
        if not bool(args.yes):
            raise SystemExit("Add --yes to confirm clearing the pending batch.")
        state = clear_pending(state_path)
        print(format_status(state))
        return

    if args.command == "set-page":
        state = set_completed_page(state_path, page=int(args.page))
        print(format_status(state))
        return

    if args.command == "queue-prompt":
        state = queue_prompt(
            state_path,
            prompt_file=args.prompt_file,
            start_page=int(args.start_page),
            end_page=int(args.end_page),
            mode=str(args.mode),
            extract_file=args.extract_file,
            completion_marker=args.completion_marker,
            force=bool(args.force),
        )
        print(format_status(state))
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
