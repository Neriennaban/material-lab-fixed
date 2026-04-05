from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.logger import AppLogger


class LoggerTests(unittest.TestCase):
    def test_file_logs_do_not_receive_ansi_color_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = AppLogger(
                name="logger_test", log_dir=tmp, enable_console=True, enable_file=True
            )
            logger.info("hello")

            log_files = list(Path(tmp).glob("logger_test_*.log"))
            self.assertEqual(len(log_files), 1)
            text = log_files[0].read_text(encoding="utf-8")
            self.assertIn("INFO", text)
            self.assertNotIn("\x1b[", text)
            for handler in list(logger.logger.handlers):
                handler.close()
            logger.logger.handlers.clear()
