from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        color = self.COLORS.get(record.levelname, self.RESET)
        original_levelname = record.levelname
        try:
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            return super().format(record)
        finally:
            record.levelname = original_levelname


class AppLogger:
    """Application logger with file and console output"""

    def __init__(
        self,
        name: str = "metallography",
        log_dir: str | Path | None = None,
        level: int = logging.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = ColoredFormatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if enable_file and log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_path / f"{name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message"""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message"""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log critical message"""
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback"""
        self.logger.exception(message, extra=kwargs)


# Global logger instance
_global_logger: AppLogger | None = None


def setup_logger(
    name: str = "metallography",
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    enable_console: bool = True,
    enable_file: bool = True,
) -> AppLogger:
    """Setup global logger"""
    global _global_logger
    _global_logger = AppLogger(name, log_dir, level, enable_console, enable_file)
    return _global_logger


def get_logger() -> AppLogger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AppLogger()
    return _global_logger


# Convenience functions
def debug(message: str, **kwargs: Any) -> None:
    """Log debug message"""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs: Any) -> None:
    """Log info message"""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs: Any) -> None:
    """Log warning message"""
    get_logger().warning(message, **kwargs)


def error(message: str, exc_info: bool = False, **kwargs: Any) -> None:
    """Log error message"""
    get_logger().error(message, exc_info=exc_info, **kwargs)


def critical(message: str, exc_info: bool = False, **kwargs: Any) -> None:
    """Log critical message"""
    get_logger().critical(message, exc_info=exc_info, **kwargs)


def exception(message: str, **kwargs: Any) -> None:
    """Log exception with traceback"""
    get_logger().exception(message, **kwargs)
